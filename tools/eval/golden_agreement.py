#!/usr/bin/env python3
"""Golden-set agreement (04 §4, §6; D40): for each checkpoint, top-1 and top-3 agreement with the
human move and the human move's mean log-likelihood, per category, over the frozen golden set
(`~/mj-data/splits/golden.parquet`: 120,000 decisions of the 164 golden players in heldout_time,
20,000 per category). Workstation only: it reads Tenhou logs.

    python tools/eval/golden_agreement.py CKPT [CKPT ...] [--out FILE.json]

- A decision is (log_id, seat, entry_idx): the entry_idx-th entry of that seat in GameplayLoader's
  order, as tools/data/golden.py tagged it. Every decision's label and legal mask are checked
  against the golden set's (`action`, `mask_hash`); one mismatch stops the run, since it means the
  loader no longer replays the logs as it did when the set was frozen.
- Each log is featurized once per observation version and every checkpoint of that version is
  scored on the same tensors (true FP32 with --no-amp; AMP by default, as the lab plays).
- Exploratory by CLAUDE.md rule 1. It is half of a student's qualification (04 §6): golden-set top-1
  within 1 pp of the best single teacher in every category; the other half is the lab's T-NI.

Checkpoints are anything tools/eval/validate.py loads (a lab-ready student, an exported EMA teacher,
a trainer checkpoint).
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'mortal'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('MORTAL_CFG', os.path.join(ROOT, 'experiments', 'configs', 'teacher-heads-quick.toml'))

CATEGORIES = ('routine', 'riichi_dama', 'call_pass', 'push_fold', 'oorasu', 'early')


def mask_hash(mask):
    return hashlib.sha1(np.packbits(np.asarray(mask, dtype=bool)).tobytes()).hexdigest()[:16]


def golden_decisions(g, loader, mj, chunk):
    """(obs, mask, action, category) arrays for the golden decisions, a chunk of logs at a time."""
    by_log = {lid: rows for lid, rows in g.groupby('log_id')}
    logs = sorted(by_log)
    for start in range(0, len(logs), chunk):
        part = logs[start:start + chunk]
        paths = [os.path.join(mj, 'mjai', lid[:4], lid[4:6], lid + '.json.gz') for lid in part]
        obs, masks, actions, cats = [], [], [], []
        for lid, games in zip(part, loader.load_gz_log_files(paths)):
            rows = by_log[lid]
            want = {seat: r for seat, r in rows.groupby('seat')}
            for game in games:
                seat = game.take_player_id()
                if seat not in want:
                    continue
                o, m, a = game.take_obs(), game.take_masks(), game.take_actions()
                for r in want.pop(seat).itertuples():
                    j = r.entry_idx
                    if a[j] != r.action or mask_hash(m[j]) != r.mask_hash:
                        raise SystemExit(f'{lid} seat {seat} entry {j}: the loader gives label {a[j]} / mask '
                                         f'{mask_hash(m[j])}, the golden set {r.action} / {r.mask_hash}')
                    obs.append(o[j])
                    masks.append(np.asarray(m[j], dtype=bool))
                    actions.append(r.action)
                    cats.append(r.category)
            if want:
                raise SystemExit(f'{lid}: no loader entries for golden seats {sorted(want)}')
        yield np.stack(obs), np.stack(masks), np.asarray(actions), np.asarray(cats)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('checkpoints', nargs='+')
    ap.add_argument('--weights', default='ema', choices=['ema', 'raw'], help="a final.pth's EMA or raw weights")
    ap.add_argument('--mj', default=os.path.expanduser('~/mj-data'))
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--batch-size', type=int, default=1024)
    ap.add_argument('--chunk', type=int, default=64, help='logs featurized at a time')
    ap.add_argument('--limit', type=int, default=None, help='only the first N logs (a smoke check)')
    ap.add_argument('--no-amp', action='store_true')
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)

    import pandas as pd
    import torch
    from libriichi.dataset import GameplayLoader
    from validate import load_models

    g = pd.read_parquet(os.path.join(a.mj, 'splits', 'golden.parquet'))
    if a.limit:
        g = g[g.log_id.isin(sorted(g.log_id.unique())[:a.limit])]
    players = [l.strip() for l in open(os.path.join(a.mj, 'splits', 'golden_players.txt'), encoding='utf-8') if l.strip()]
    device = torch.device(a.device)
    models = {}
    for path in a.checkpoints:
        nets, version, _, used = load_models(path, a.weights, device)
        models.setdefault(version, []).append((path, nets[0], nets[1], used))
    stats = {path: {c: {'n': 0, 'top1': 0, 'top3': 0, 'loglik': 0.0} for c in CATEGORIES} for path in a.checkpoints}
    used_of = {}
    t0 = time.time()
    for version, ms in sorted(models.items()):
        loader = GameplayLoader(version=version, oracle=False, player_names=players)
        n = 0
        for obs, mask, actions, cats in golden_decisions(g, loader, a.mj, a.chunk):
            for path, brain, policy, used in ms:
                used_of[path] = used
                for s in range(0, len(actions), a.batch_size):
                    o = torch.as_tensor(obs[s:s + a.batch_size], device=device)
                    m = torch.as_tensor(mask[s:s + a.batch_size], device=device)
                    y = torch.as_tensor(actions[s:s + a.batch_size], device=device)
                    with torch.no_grad(), torch.autocast(device.type, enabled=not a.no_amp):
                        logits = policy.logits(brain(o), m).float()
                    logp = logits.log_softmax(-1)
                    top3 = logits.topk(3, dim=-1).indices
                    hit1 = (logits.argmax(-1) == y).cpu().numpy()
                    hit3 = (top3 == y[:, None]).any(-1).cpu().numpy()
                    ll = logp.gather(1, y[:, None]).squeeze(1).cpu().numpy()
                    for c in CATEGORIES:
                        sel = cats[s:s + a.batch_size] == c
                        d = stats[path][c]
                        d['n'] += int(sel.sum())
                        d['top1'] += int(hit1[sel].sum())
                        d['top3'] += int(hit3[sel].sum())
                        d['loglik'] += float(ll[sel].sum())
            n += len(actions)
            print(f'  v{version}: {n:,} of {len(g):,} decisions, {time.time() - t0:,.0f} s', flush=True)

    rows = []
    print(f'\n{"checkpoint":<60} {"category":<12} {"n":>7} {"top-1":>8} {"top-3":>8} {"loglik":>8}')
    for path in a.checkpoints:
        tot = {'n': 0, 'top1': 0, 'top3': 0, 'loglik': 0.0}
        for c in CATEGORIES + ('all',):
            d = stats[path][c] if c != 'all' else tot
            if c != 'all':
                for k in tot:
                    tot[k] += d[k]
            row = {'checkpoint': path, 'weights': used_of.get(path), 'category': c, 'n': d['n'],
                   'top1': d['top1'] / max(d['n'], 1), 'top3': d['top3'] / max(d['n'], 1),
                   'loglik': d['loglik'] / max(d['n'], 1)}
            rows.append(row)
            print(f'{path[-60:]:<60} {c:<12} {row["n"]:>7,} {row["top1"]:>8.4f} {row["top3"]:>8.4f} {row["loglik"]:>8.4f}')
    if a.out:
        with open(a.out, 'w') as f:
            json.dump(rows, f, indent=1)
            f.write('\n')
    return rows


if __name__ == '__main__':
    main()
