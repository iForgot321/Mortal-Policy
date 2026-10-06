"""1v3 duplicate play for the lab (02 §3), on libriichi's OneVsThree.

Engines are given as specs: a checkpoint path, or `builtin:shanten-greedy`
for the frozen rule-based floor (02 §10). A checkpoint spec may carry
`?hobble=EPS`: the synthetic hobbled patch of 02 §6 (with probability EPS the
chosen action is replaced by a uniformly random legal one).
"""
import os
import sys
import time

import numpy as np

from . import MORTAL_DIR

# libriichi.so and the flat mortal modules live in mortal/
if MORTAL_DIR not in sys.path:
    sys.path.insert(0, MORTAL_DIR)

BUILTINS = ('builtin:shanten-greedy',)


class ShantenGreedyEngine:
    """Python handle for the Rust ShantenGreedy agent (no Python inference)."""
    engine_type = 'shanten-greedy'
    name = 'shanten-greedy'


class HobbledEngine:
    """Random-action injection around a MortalEngine (02 §6, D36). Temperature
    cannot hobble argmax play, so this replaces the action instead."""

    def __init__(self, inner, eps, seed=0):
        self.inner = inner
        self.eps = eps
        self.rng = np.random.default_rng(seed)
        for attr in ('engine_type', 'name', 'is_oracle', 'version', 'enable_quick_eval',
                     'enable_rule_based_agari_guard'):
            setattr(self, attr, getattr(inner, attr))

    def react_batch(self, obs, masks, invisible_obs):
        actions, q_out, masks_out, is_greedy, *_ = self.inner.react_batch(obs, masks, invisible_obs)
        for i, mask in enumerate(masks_out):
            if self.rng.random() < self.eps:
                legal = np.flatnonzero(mask)
                actions[i] = int(self.rng.choice(legal))
                is_greedy[i] = False
        # the replacements change the sampling distribution, so the inner
        # engine's behaviour probabilities no longer hold (D46)
        return actions, q_out, masks_out, is_greedy


def load_engine(spec, device='cuda:0', name=None, enable_amp=True,
                enable_rule_based_agari_guard=True):
    path, _, query = spec.partition('?')
    opts = dict(kv.split('=', 1) for kv in query.split('&') if kv)
    if opts:
        if set(opts) != {'hobble'}:
            raise ValueError(f'unknown engine options {opts}')
        inner = load_engine(path, device, name, enable_amp, enable_rule_based_agari_guard)
        return HobbledEngine(inner, float(opts['hobble']))
    if spec == 'builtin:shanten-greedy':
        return ShantenGreedyEngine()
    if spec.startswith('builtin:'):
        raise ValueError(f'unknown builtin engine {spec}; known: {BUILTINS}')

    import torch
    from engine import MortalEngine
    from model import Brain, CategoricalPolicy

    # weights_only=False: trainer checkpoints carry numpy scalars; we only
    # ever load our own checkpoints
    state = torch.load(spec, weights_only=False, map_location=torch.device('cpu'))
    cfg = state['config']
    version = cfg['control'].get('version', 1)
    # every upstream script builds Brain with GN; record otherwise in the config
    norm = cfg['resnet'].get('norm', 'GN')
    brain = Brain(
        version=version,
        num_blocks=cfg['resnet']['num_blocks'],
        conv_channels=cfg['resnet']['conv_channels'],
        Norm=norm,
    ).eval()
    policy = CategoricalPolicy().eval()
    brain.load_state_dict(state['mortal'])
    policy.load_state_dict(state['policy_net'])
    return MortalEngine(
        brain,
        policy,
        is_oracle=False,
        version=version,
        device=torch.device(device),
        enable_amp=enable_amp,
        enable_rule_based_agari_guard=enable_rule_based_agari_guard,
        name=name or os.path.basename(spec),
    )


def play_sets(challenger, champion, key, nonce_start, n_sets, log_dir=None):
    """Play n_sets duplicate sets (4 games each) and return per-game arrays:
    seeds, challenger seats, 0-based challenger ranks, final scores (n, 4)."""
    from libriichi.arena import OneVsThree

    env = OneVsThree(disable_progress_bar=True, log_dir=log_dir)
    records = env.py_vs_py_games(
        challenger=challenger,
        champion=champion,
        seed_start=(nonce_start, key),
        seed_count=n_sets,
    )
    seeds, seats, ranks, scores = zip(*records)
    return (
        np.asarray(seeds, dtype=np.uint64),
        np.asarray(seats, dtype=np.int8),
        np.asarray(ranks, dtype=np.int8),
        np.asarray(scores, dtype=np.int32),
    )


class GameLog:
    """Accumulates per-game records across chunks and saves them as .npz."""

    def __init__(self):
        self.parts = []
        self.started = time.time()

    def add(self, seeds, seats, ranks, scores):
        self.parts.append((seeds, seats, ranks, scores))

    def arrays(self):
        if not self.parts:
            return (np.zeros(0, np.uint64), np.zeros(0, np.int8), np.zeros(0, np.int8),
                    np.zeros((0, 4), np.int32))
        return tuple(np.concatenate(cols) for cols in zip(*self.parts))

    def save(self, path, set_scores):
        seeds, seats, ranks, scores = self.arrays()
        np.savez_compressed(path, seeds=seeds, seats=seats, ranks=ranks, scores=scores,
                            set_scores=np.asarray(set_scores))


def secondary_metrics(ranks, display_pts):
    """avg rank (1-based), rank histogram and avg display pt (02 §2)."""
    ranks = np.asarray(ranks)
    hist = np.bincount(ranks, minlength=4)
    n = max(len(ranks), 1)
    return {
        'games': int(len(ranks)),
        'rank_hist': hist.tolist(),
        'avg_rank': float((ranks + 1).mean()) if len(ranks) else None,
        'avg_pt': float(np.asarray(display_pts)[ranks].sum() / n),
    }
