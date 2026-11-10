#!/usr/bin/env python3
"""Score checkpoints on a fixed validation sample of a split, with the trainer's own validation
pass (mortal/teacher.py: human-move NLL and top-1, and each head's loss when the checkpoint has
heads). Workstation only: it reads Tenhou logs from ~/mj-data.

    python tools/eval/validate.py --split heldout_time CKPT [CKPT ...]
    python tools/eval/validate.py --split heldout_game --start 2015-01-01 CKPT ...

- heldout_game with --start 2015-01-01 is the trainer's in-window sample ([dataset] val_start).
  Without --start it is the teachers' sample: 47 of its 200 logs are from 2009-2014.
- heldout_time (2025-07 on) is what the model never saw in time. It isn't in the secure bundle, so
  it is scored here, after a run. Its sample leaves out the golden set's logs (`04` §4), so
  validation never reads the golden diagnostic's decisions.

A checkpoint is a train_teacher.py final.pth (raw weights, and `ema_weights` with --weights ema),
an exported ema.pth, or any trainer checkpoint with `mortal` and `policy_net` at the top level.
Exploratory by CLAUDE.md rule 1: human agreement is a diagnostic, not a strength verdict.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'mortal'))
# dataloader.py reads the GRP from the config; any config naming the frozen GRP-v2 serves
os.environ.setdefault('MORTAL_CFG', os.path.join(ROOT, 'experiments', 'configs', 'teacher-heads-quick.toml'))

LOSS_CFG = {'w_placement': 0.5, 'w_kyoku_delta': 0.5, 'w_aux': 0.2, 'huber_delta': 1.0}


def golden_logs(mj):
    import pandas as pd
    return set(pd.read_parquet(os.path.join(mj, 'splits', 'golden.parquet'), columns=['log_id']).log_id)


def load_models(path, weights, device):
    import torch
    from checkpoint import load_checkpoint
    from model import Brain, CategoricalPolicy, TeacherHeads

    st = load_checkpoint(path)
    cfg = st['config']
    used = 'raw'
    w = st
    if weights == 'ema' and 'ema_weights' in st:
        w, used = st['ema_weights'], 'ema'
        # train_teacher.py before 2026-11-10 saved the raw weights here (tools/secure/export_ema.py)
        if all(torch.equal(st['mortal'][k], w['mortal'][k]) for k in st['mortal']):
            used = 'raw: the file has no real EMA weights'
    elif st.get('weights') == 'ema':  # tools/secure/export_ema.py
        used = 'ema'
    version = cfg['control'].get('version', 1)
    mortal = Brain(version=version, conv_channels=cfg['resnet']['conv_channels'],
                   num_blocks=cfg['resnet']['num_blocks'], Norm=cfg['resnet'].get('norm', 'GN'))
    mortal.load_state_dict(w['mortal'])
    policy = CategoricalPolicy()
    policy.load_state_dict(w['policy_net'])
    heads = None
    if w.get('heads') is not None:
        heads = TeacherHeads(hidden=cfg['teacher']['hidden'], detach=cfg['teacher']['detach'])
        heads.load_state_dict(w['heads'])
    models = [m.to(torch.device(device)).eval() for m in (mortal, policy, heads) if m is not None]
    return (models + [None])[:3], version, cfg.get('teacher', LOSS_CFG), used


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('checkpoints', nargs='+')
    ap.add_argument('--split', default='heldout_time', choices=['heldout_time', 'heldout_game'])
    ap.add_argument('--games', type=int, default=200)
    ap.add_argument('--start', default=None, help='YYYY-MM-DD: only logs from this day on')
    ap.add_argument('--end', default=None, help='YYYY-MM-DD: only logs up to this day')
    ap.add_argument('--weights', default='ema', choices=['ema', 'raw'], help="a final.pth's EMA or raw weights")
    ap.add_argument('--mj', default=os.path.expanduser('~/mj-data'))
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--batch-size', type=int, default=2048)
    ap.add_argument('--no-amp', action='store_true')
    ap.add_argument('--out', default=None, help='also write the rows as JSON here')
    a = ap.parse_args(argv)

    import torch
    from config import config
    from dataloader import load_masks
    from teacher import validate, validation_files

    exclude = golden_logs(a.mj) if a.split == 'heldout_time' else ()
    files = validation_files(a.mj, a.split, a.games, start=a.start, end=a.end, exclude=exclude)
    days = sorted(os.path.basename(f)[:8] for f in files)
    print(f'{len(files)} logs of {a.split}, {days[0]} to {days[-1]}'
          + (f' ({len(exclude):,} golden-set logs left out)' if exclude else ''), flush=True)
    masks = load_masks(os.path.join(a.mj, 'splits', 'masks.jsonl'))

    rows = []
    for path in a.checkpoints:
        (mortal, policy, heads), version, loss_cfg, used = load_models(path, a.weights, a.device)
        res = validate(mortal, policy, heads, files, version=version, pts=config['env']['pts'], masks=masks,
                       loss_cfg=loss_cfg, device=torch.device(a.device), batch_size=a.batch_size,
                       enable_amp=not a.no_amp)
        rows.append({'checkpoint': path, 'weights': used, 'split': a.split, 'start': a.start, 'end': a.end,
                     'logs': len(files), 'first_day': days[0], 'last_day': days[-1], **res})
        print(f'{path} ({used}): top-1 {res["top1"]:.4%}, NLL {res["policy"]:.4f}, {res["samples"]:,} decisions'
              + (f'; placement {res["placement"]:.4f}, kyoku-Δ {res["kyoku_delta"]:.5f}, tenpai {res["tenpai"]:.4f}, '
                 f'waits {res["waits"]:.4f}' if heads is not None else ''), flush=True)
        del mortal, policy, heads
        torch.cuda.empty_cache()
    if a.out:
        with open(a.out, 'w') as f:
            json.dump(rows, f, indent=1)
            f.write('\n')
    return rows


if __name__ == '__main__':
    main()
