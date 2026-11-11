#!/usr/bin/env python3
"""Write a train_teacher.py run's EMA weights as a plain checkpoint the lab loads (`mortal` and
`policy_net` at the top level, as lab/arena.py reads them), read-only.

    python tools/secure/export_ema.py RUN_DIR/state.pth  OUT.pth    # from the saved EMA shadow
    python tools/secure/export_ema.py RUN_DIR/final.pth  OUT.pth    # from its `ema_weights`

Until 2026-11-10 the trainer wrote the raw weights into final.pth's `ema_weights` (it saved the live
tensors after swapping the raw weights back), so the first exports, the teachers' `ema.pth`, hold
the raw weights. This tool refuses such a final.pth. The run's last state.pth, written at the same
step, holds the real EMA in its `ema` shadow: the weights are rebuilt from that.
"""
import hashlib
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'mortal'))
# teacher.py imports dataloader.py, which reads a config; nothing here uses its values
os.environ.setdefault('MORTAL_CFG', os.path.join(ROOT, 'experiments', 'configs', 'teacher-heads-quick.toml'))
from checkpoint import load_checkpoint  # noqa: E402

NAMES = ('mortal', 'policy_net', 'heads')


def same_weights(a, b):
    return all(a[n] is None and b[n] is None or
               all(torch.equal(a[n][k], b[n][k]) for k in a[n]) for n in NAMES)


def from_shadow(st):
    """State dicts of the EMA weights, from a state.pth: the raw weights with every parameter
    replaced by its EMA shadow, in the trainer's parameter order (teacher.param_groups)."""
    from model import Brain, CategoricalPolicy, TeacherHeads
    from teacher import param_groups

    cfg = st['config']
    modules = {
        'mortal': Brain(version=cfg['control']['version'], **cfg['resnet'], Norm='GN'),
        'policy_net': CategoricalPolicy(),
        'heads': (TeacherHeads(hidden=cfg['teacher']['hidden'], detach=cfg['teacher']['detach'])
                  if st.get('heads') is not None else None),
    }
    for n, m in modules.items():
        if m is not None:
            m.load_state_dict(st[n])
    decay, no_decay = param_groups([m for m in modules.values() if m is not None])
    params, shadow = decay + no_decay, st['ema']['shadow']
    if len(params) != len(shadow) or any(p.shape != s.shape for p, s in zip(params, shadow)):
        raise SystemExit('the EMA shadow does not match the networks the config builds')
    with torch.no_grad():
        for p, s in zip(params, shadow):
            p.copy_(s)
    return {n: None if m is None else m.state_dict() for n, m in modules.items()}


def main(src, dst):
    st = load_checkpoint(src)
    raw = {n: st.get(n) for n in NAMES}
    if 'ema' in st:
        ema = from_shadow(st)
    elif 'ema_weights' in st:
        ema = st['ema_weights']
    else:
        raise SystemExit(f'{src} has neither an EMA shadow (state.pth) nor ema_weights (final.pth)')
    if same_weights(raw, ema):
        raise SystemExit(f"{src}: its EMA weights equal its raw weights (train_teacher.py's bug before 2026-11-10). "
                         "Export from the run's state.pth instead.")
    with open(src, 'rb') as f:
        src_sha = hashlib.sha256(f.read()).hexdigest()
    out = {**ema, 'config': st['config'], 'steps': st['steps'], 'presentations': st['presentations'],
           'weights': 'ema', 'source': {'path': src, 'sha256': src_sha}}
    if os.path.exists(dst):
        sys.exit(f'{dst} exists')
    torch.save(out, dst)
    os.chmod(dst, 0o444)
    with open(dst, 'rb') as f:
        print(f'{dst}: EMA weights of {src} (step {st["steps"]:,}), sha256 {hashlib.sha256(f.read()).hexdigest()}')


if __name__ == '__main__':
    main(*sys.argv[1:3])
