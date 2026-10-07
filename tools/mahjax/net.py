"""Mortal's v4 policy network in JAX, for the Mahjax throughput spike (03 §7 item 2).

Same layers as `mortal/model.py` Brain(version=4, Norm="GN") + CategoricalPolicy:
a pre-activation Conv1d ResNet over 34 tiles (GroupNorm(32), Mish, channel
attention), conv to 32 channels, Linear(32 * 34 -> 1024), then
tanh(Linear(1024 -> 256)) and a linear head over Mahjax's 87 actions.
Random weights: this measures cost only.

The input has v4's 1012 channels. JAX has no v4 encoder (`libriichi`'s
obs_repr.rs), so the input is built from Mahjax's own dict observation (hand,
reds, discard shanten, waits, per-discard waits, tiles seen, dora, scores,
round state) and tiled out to 1012 channels: the observation's cost is real,
the network sees v4's shape.

Specs: `bNcM` (N blocks, M channels), e.g. b8c192; `obs` computes only the
observation (no network).
"""
import re

import jax
import jax.numpy as jnp

IN_CHANNELS = 1012


def mish(x):
    return x * jnp.tanh(jax.nn.softplus(x))


def conv1d(x, w, b=None):
    """x (B, C, 34), w (O, C, 3), 'same' padding."""
    y = jax.lax.conv_general_dilated(x, w, (1,), [(1, 1)], dimension_numbers=('NCH', 'OIH', 'NCH'))
    return y if b is None else y + b[None, :, None]


def group_norm(x, gamma, beta, groups=32, eps=1e-3):
    n, c, length = x.shape
    xg = x.reshape(n, groups, c // groups, length).astype(jnp.float32)
    mean = xg.mean((2, 3), keepdims=True)
    var = xg.var((2, 3), keepdims=True)
    xg = (xg - mean) * jax.lax.rsqrt(var + eps)
    return (xg.reshape(n, c, length) * gamma[None, :, None] + beta[None, :, None]).astype(x.dtype)


def channel_attention(x, p):
    def mlp(v):
        return mish(v @ p['w1'] + p['b1']) @ p['w2'] + p['b2']

    weight = jax.nn.sigmoid(mlp(x.mean(-1)) + mlp(x.max(-1)))
    return x * weight[:, :, None]


def init_params(key, blocks, channels, n_actions, dtype=jnp.bfloat16):
    keys = iter(jax.random.split(key, 8 + 8 * blocks))

    def normal(shape, fan_in):
        return (jax.random.normal(next(keys), shape) / jnp.sqrt(fan_in)).astype(dtype)

    def zeros(n):
        return jnp.zeros(n, dtype)

    def ones(n):
        return jnp.ones(n, dtype)

    p = {'conv_in': normal((channels, IN_CHANNELS, 3), IN_CHANNELS * 3), 'blocks': []}
    for _ in range(blocks):
        p['blocks'].append({
            'g1': ones(channels), 'b1': zeros(channels),
            'w1': normal((channels, channels, 3), channels * 3),
            'g2': ones(channels), 'b2': zeros(channels),
            'w2': normal((channels, channels, 3), channels * 3),
            'ca': {'w1': normal((channels, channels // 16), channels), 'b1': zeros(channels // 16),
                   'w2': normal((channels // 16, channels), channels // 16), 'b2': zeros(channels)},
        })
    p.update({
        'g_out': ones(channels), 'b_out': zeros(channels),
        'conv_out': normal((32, channels, 3), channels * 3), 'conv_out_b': zeros(32),
        'fc': normal((32 * 34, 1024), 32 * 34), 'fc_b': zeros(1024),
        'fc1': normal((1024, 256), 1024), 'fc1_b': zeros(256),
        'fc2': normal((256, n_actions), 256), 'fc2_b': zeros(n_actions),
    })
    return p


def forward(p, x):
    """x (B, 1012, 34) -> logits (B, n_actions), float32."""
    x = conv1d(x, p['conv_in'])
    for blk in p['blocks']:
        h = conv1d(mish(group_norm(x, blk['g1'], blk['b1'])), blk['w1'])
        h = conv1d(mish(group_norm(h, blk['g2'], blk['b2'])), blk['w2'])
        x = x + channel_attention(h, blk['ca'])
    x = mish(group_norm(x, p['g_out'], p['b_out']))
    x = mish(conv1d(x, p['conv_out'], p['conv_out_b']))
    phi = mish(x.reshape(x.shape[0], -1) @ p['fc'] + p['fc_b'])
    phi = jnp.tanh(phi @ p['fc1'] + p['fc1_b'])
    return (phi @ p['fc2'] + p['fc2_b']).astype(jnp.float32)


def obs_planes(obs):
    """Mahjax's dict observation (one seat) as (C, 34) planes."""
    f32 = jnp.float32
    hand = obs['hand'].astype(jnp.int32)
    planes = [(hand[None, :] > jnp.arange(4)[:, None]).astype(f32)]               # 4
    planes.append(obs['is_red'].astype(f32)[None])                                # 1
    planes.append((obs['discard_shanten'].T.astype(f32) / 14.))                    # 3
    planes.append(obs['can_win'].astype(f32)[None])                                # 1
    planes.append((obs['discard_can_win'].T.astype(f32) + 1.) / 2.)               # 34
    planes.append(obs['tiles_seen'].astype(f32)[None] / 4.)                        # 1
    dora = obs['dora_indicators'].astype(jnp.int32)
    planes.append(jax.nn.one_hot(jnp.where(dora >= 0, dora % 34, -1), 34, dtype=f32))  # 5
    scalars = jnp.concatenate([
        obs['scores'].astype(f32) / 1000., obs['riichi'].astype(f32), obs['ippatsu'].astype(f32),
        jnp.stack([obs['round'], obs['honba'], obs['kyotaku'], obs['seat_wind'], obs['wall_remaining'],
                   obs['shanten_count'], obs['furiten'], obs['is_discard_node']]).astype(f32) / 10.,
    ])
    planes.append(jnp.broadcast_to(scalars[:, None], (scalars.shape[0], 34)))     # 20
    return jnp.concatenate(planes)


def make_policy(spec, env):
    observe = jax.vmap(env.observe)
    n_actions = env.num_actions

    if spec == 'obs':
        class ObsOnly:
            @staticmethod
            def init(key):
                return None

            @staticmethod
            def apply(params, state):
                x = jax.vmap(obs_planes)(observe(state))
                return jnp.broadcast_to(x.sum((1, 2))[:, None], (x.shape[0], n_actions)) * 0.
        return ObsOnly()

    m = re.fullmatch(r'b(\d+)c(\d+)', spec)
    if not m:
        raise ValueError(f'net spec {spec!r}: expected bNcM or obs')
    blocks, channels = int(m.group(1)), int(m.group(2))

    class Policy:
        @staticmethod
        def init(key):
            return init_params(key, blocks, channels, n_actions)

        @staticmethod
        def apply(params, state):
            x = jax.vmap(obs_planes)(observe(state))
            reps = -(-IN_CHANNELS // x.shape[1])
            x = jnp.tile(x, (1, reps, 1))[:, :IN_CHANNELS].astype(jnp.bfloat16)
            return forward(params, x)
    return Policy()
