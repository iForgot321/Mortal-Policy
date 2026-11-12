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

    python tools/fleet/stage_init.py SNAPSHOT OUT.pth [--new-anchor]

OUT is written read-only and never overwritten; its sha256 is printed.
"""
import argparse
import hashlib
import os
import sys

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


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(src, out, new_anchor=False):
    """Write `src`'s staged copy to `out` (read-only; an existing `out` is refused). Returns
    (the staged state, the keys dropped)."""
    if os.path.exists(out):
        raise FileExistsError(f'{out} exists; a stage init is never overwritten')
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
    a = p.parse_args(argv)
    state, dropped = write(a.snapshot, a.out, a.new_anchor)
    print(f'{a.out}: global step {state["steps"]:,}; dropped {", ".join(dropped) or "nothing"}')
    if 'kl_anchor' in state:
        anchor = state['kl_anchor']
        print(f'  anchor kept: {anchor["state_file"]}, coefficient {anchor["adaptive_coef"]:g}')
    print(f'  sha256 {sha256(a.out)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
