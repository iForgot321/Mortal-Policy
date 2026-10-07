#!/usr/bin/env python
"""Mahjax throughput (03 §7, spike item 2): random policy, hanchan, auto reset.

Steps a batch of `red_mahjong` hanchan games with a uniform-random legal policy,
`--chunk` steps per jitted `lax.scan`, so Python is out of the loop. Reports
env steps/s, finished hanchan/h, steps per hanchan and peak GPU memory for each
batch size. Optionally adds a policy network's forward pass per step
(`--net`, see `net.py`), which is the end-to-end actor cost without learning.

Run in the `mahjax` conda env, from the repo root:

    XLA_PYTHON_CLIENT_PREALLOCATE=false ~/miniconda3/envs/mahjax/bin/python \\
        tools/mahjax/bench.py --batch 64 256 1024 --seconds 60 \\
        --out workdir/mahjax-spike/bench-<host>.json
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--batch', type=int, nargs='+', default=[64, 256, 1024])
    ap.add_argument('--chunk', type=int, default=100, help='env steps per jitted scan')
    ap.add_argument('--seconds', type=float, default=60, help='timed seconds per batch size (after warm-up)')
    ap.add_argument('--rule', default='red_mahjong', choices=['red_mahjong', 'no_red_mahjong'])
    ap.add_argument('--net', default=None, help='policy net spec for net.py, e.g. b8c192 (default: none)')
    ap.add_argument('--steps-per-hanchan', type=float, default=None,
                    help='env steps per hanchan under realistic play (from the parity replay); '
                         'hanchan/h is derived from it, since a short timed window undercounts long games')
    ap.add_argument('--upstream-cond', action='store_true',
                    help='keep lax.cond/switch as is (default: run them as selects, select_patch.py)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    if not args.upstream_cond:
        import select_patch
        select_patch.apply()
    import jax
    import jax.numpy as jnp
    import mahjax
    from mahjax.wrappers.auto_reset_wrapper import auto_reset

    env = mahjax.make(args.rule, round_mode='half', order_points=[0, 0, 0, 0])
    step = auto_reset(env.step, env.init)
    dev = jax.devices()[0]

    policy = None
    if args.net:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import net
        policy = net.make_policy(args.net, env)

    def one_step(carry, _):
        state, key, params = carry
        key, k_act, k_step = jax.random.split(key, 3)
        b = state.legal_action_mask.shape[0]
        if policy is None:
            logits = jnp.zeros(state.legal_action_mask.shape, jnp.float32)
        else:
            logits = policy.apply(params, state)
        logits = jnp.where(state.legal_action_mask, logits, -jnp.inf)
        action = jax.random.categorical(k_act, logits, axis=-1)
        state = jax.vmap(step)(state, action, jax.random.split(k_step, b))
        return (state, key, params), state.terminated.sum()

    @jax.jit
    def run_chunk(state, key, params):
        (state, key, params), done = jax.lax.scan(one_step, (state, key, params), None, length=args.chunk)
        return state, key, done.sum()

    results = []
    for b in args.batch:
        key = jax.random.PRNGKey(0)
        key, k = jax.random.split(key)
        state = jax.jit(jax.vmap(env.init))(jax.random.split(k, b))
        params = policy.init(jax.random.PRNGKey(1)) if policy else None
        t0 = time.time()
        try:
            state, key, _ = run_chunk(state, key, params)
            jax.block_until_ready(state)
        except Exception as e:  # out of memory at this batch size
            results.append({'batch': b, 'error': str(e).splitlines()[0][:300]})
            print(results[-1], flush=True)
            continue
        compile_s = time.time() - t0
        # warm-up so that games are spread over their length before timing
        t_warm = time.time()
        while time.time() - t_warm < min(20.0, args.seconds / 3):
            state, key, _ = run_chunk(state, key, params)
        jax.block_until_ready(state)
        steps = games = 0
        t0 = time.time()
        while time.time() - t0 < args.seconds:
            state, key, done = run_chunk(state, key, params)
            games += int(done)
            steps += b * args.chunk
        dt = time.time() - t0
        mem = dev.memory_stats() or {}
        r = {
            'batch': b,
            'compile_s': round(compile_s, 1),
            'seconds': round(dt, 1),
            'env_steps_per_s': round(steps / dt),
            'random_policy_games_finished': games,
            'peak_gpu_gib': round(mem.get('peak_bytes_in_use', 0) / 2**30, 2),
        }
        if args.steps_per_hanchan:
            r['hanchan_per_h_at_realistic_length'] = round(steps / dt * 3600 / args.steps_per_hanchan)
        results.append(r)
        print(r, flush=True)

    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                         capture_output=True, text=True).stdout.strip()
    out = {
        'rule': args.rule, 'net': args.net, 'cond_as_select': not args.upstream_cond, 'chunk': args.chunk, 'policy': 'uniform random legal' if not args.net else 'softmax of the net over legal actions', 'steps_per_hanchan': args.steps_per_hanchan,
        'jax': jax.__version__, 'mahjax': mahjax.__version__, 'device': str(dev), 'gpu': gpu,
        'cpu': platform.processor() or platform.machine(), 'results': results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
