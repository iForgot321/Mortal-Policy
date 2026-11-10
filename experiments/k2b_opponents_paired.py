"""Paired reading of the k2b opponent check (experiments/k2b-opponents.md).

For each table (3x shanten-greedy on block `floor`, 3x pilot B on block `aa-prime`): the per-set
difference between k2b full's and pilot A's set scores on the same deals against the same
opponents, with the lab's set-level bootstrap CI. Reads the registry and each run's games.npz;
writes workdir/lab/k2b-opponents-paired.json.

    python experiments/k2b_opponents_paired.py        (from the repo root)
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lab import gsprt, registry  # noqa: E402

K2B = 'f5a22c5c16d3d51e08abb25f77f2e46898ca512650d676d7a5e833c347cfc4fc'
PILOT_A = '15121432d2268ae7e755fe82e7f4bb0e97b3171a1fc9cf697bbdfbabaf68a284'
PILOT_B = 'b267c0267f1538c61ab962146750862fa9d4ae30bc9543ebd6f519d0a4272c3d'
FLOOR = 'builtin:shanten-greedy'
SETS = 20000
REFERENCE = 0.019508  # k2b full vs 3x pilot A, 20261025-060641-T-confirm
OUT = os.path.join(ROOT, 'workdir', 'lab', 'k2b-opponents-paired.json')


def engine_id(rec):
    return rec.get('sha256') or rec['spec']


def find(rows, challenger, champion):
    """The latest finished 20K-set family-A measure of challenger against champion."""
    hits = [r for r in rows
            if r['test'] == 'measure' and r.get('verdict') == 'measured' and r['family'] == 'A'
            and r['nonce_count'] == SETS and engine_id(r['challenger']) == challenger
            and engine_id(r['champion']) == champion]
    return hits[-1] if hits else None


def load(row):
    z = np.load(os.path.join(row['run_dir'], 'games.npz'))
    return z['seeds'].reshape(-1, 4)[:, 0], z['set_scores']


def pair(rows, table, champion):
    new, base = find(rows, K2B, champion), find(rows, PILOT_A, champion)
    if new is None or base is None:
        return {'table': table, 'missing': [n for n, r in (('k2b', new), ('pilot A', base)) if r is None]}
    (s1, x1), (s0, x0) = load(new), load(base)
    if new['nonce_start'] != base['nonce_start'] or not np.array_equal(s1, s0):
        raise SystemExit(f'{table}: the two runs are not on the same deals')
    d = x1 - x0
    lo, hi = gsprt.bootstrap_mean_ci(d)
    return {
        'table': table,
        'k2b_run': new['run_id'], 'pilot_a_run': base['run_id'],
        'sets': int(len(d)),
        'k2b_mean': float(x1.mean()), 'pilot_a_mean': float(x0.mean()),
        'diff_mean': float(d.mean()), 'diff_ci': [float(lo), float(hi)],
        'diff_sd': float(d.std(ddof=1)),
        'corr': float(np.corrcoef(x1, x0)[0, 1]),
        'share_of_reference': float(d.mean() / REFERENCE),
    }


def main():
    rows = registry._read_jsonl(registry.RUNS)
    out = {'reference_vs_pilot_a': REFERENCE,
           'tables': [pair(rows, '3x shanten-greedy', FLOOR), pair(rows, '3x pilot B', PILOT_B)]}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
