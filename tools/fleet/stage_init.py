#!/usr/bin/env python3
"""A run's init for a new stage of training, made from another run's snapshot (D64's
anchor-release pilot, `experiments/anchor-release.md`).

A trainer snapshot carries its run's clock (`schedule_origin`) and its KL anchor's state
(`kl_anchor`: the anchor's file and the adaptive coefficient). A run started from it continues
that clock, and `train_online.py` refuses an anchor other than the snapshot's. This writes a copy

- without the clock, so the new run's schedules and the anchor's target start at t = 0;
- with `--new-anchor`, also without the anchor's state, so the run can anchor to another policy,
  and its coefficient starts from the config's `coef`. Without the flag the anchor's state is
  kept: the same anchor, at the coefficient the snapshot's run left it.

Everything else is kept: the weights, the optimizers, the critic and the global step.

With `--from-policy`, the source is a policy checkpoint that isn't a trainer's state (a student,
`train_students.py`: weights and config only), and the copy is a trainer's first state: those
weights, the network the config names, step 0 and a fresh AMP scaler; no optimizer, clock or
anchor state, so `train_online.py` starts from it as from an offline checkpoint (D68's B0s).

    python tools/fleet/stage_init.py SNAPSHOT OUT.pth [--new-anchor | --from-policy]

OUT is written read-only and never overwritten; its sha256 is printed.
"""
import argparse
import hashlib
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'mortal'))
from checkpoint import load_checkpoint  # noqa: E402


def stage(state, new_anchor=False):
    """A shallow copy of a trainer state without its run clock and, with `new_anchor`, without
    its KL anchor's state. Returns (copy, the keys dropped)."""
    drop = ['schedule_origin'] + (['kl_anchor'] if new_anchor else [])
    dropped = [k for k in drop if k in state]
    return {k: v for k, v in state.items() if k not in drop}, dropped


# torch.amp.GradScaler's defaults, as an enabled scaler saves them before its first step
FRESH_SCALER = {'scale': 65536.0, 'growth_factor': 2.0, 'backoff_factor': 0.5, 'growth_interval': 2000,
                '_growth_tracker': 0}


def from_policy(state, source_sha256=None):
    """A trainer's first state from a policy checkpoint's weights (the module docstring)."""
    cfg = state['config']
    resnet = {k: cfg['resnet'][k] for k in ('conv_channels', 'num_blocks', 'norm') if k in cfg['resnet']}
    return {'mortal': state['mortal'], 'policy_net': state['policy_net'],
            'config': {'control': {'version': cfg['control']['version'], 'online': False}, 'resnet': resnet},
            'scaler': dict(FRESH_SCALER), 'steps': 0, 'timestamp': time.time(),
            'best_perf': {'avg_rank': 4.0, 'avg_pt': -135.0}, 'from_policy': source_sha256}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(src, out, new_anchor=False, policy=False):
    """Write `src`'s staged copy (or, with `policy`, the trainer state made from it) to `out`
    (read-only; an existing `out` is refused). Returns (the state written, the keys dropped)."""
    if os.path.exists(out):
        raise FileExistsError(f'{out} exists; a stage init is never overwritten')
    if policy:
        loaded = load_checkpoint(src)
        state = from_policy(loaded, sha256(src))
        dropped = sorted(k for k in loaded if k not in state)
    else:
        state, dropped = stage(load_checkpoint(src), new_anchor)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    torch.save(state, out + '.tmp')
    os.chmod(out + '.tmp', 0o444)
    os.replace(out + '.tmp', out)
    return state, dropped


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('snapshot')
    p.add_argument('out')
    p.add_argument('--new-anchor', action='store_true',
                   help="drop the KL anchor's state too (the new run anchors to another policy)")
    p.add_argument('--from-policy', action='store_true',
                   help="the source is a policy's weights (a student), not a trainer's snapshot")
    a = p.parse_args(argv)
    if a.new_anchor and a.from_policy:
        p.error('--from-policy makes a first state: it has no anchor to drop')
    state, dropped = write(a.snapshot, a.out, a.new_anchor, a.from_policy)
    print(f'{a.out}: global step {state["steps"]:,}; dropped {", ".join(dropped) or "nothing"}')
    if 'kl_anchor' in state:
        anchor = state['kl_anchor']
        print(f'  anchor kept: {anchor["state_file"]}, coefficient {anchor["adaptive_coef"]:g}')
    print(f'  sha256 {sha256(a.out)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
