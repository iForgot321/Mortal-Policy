"""The anchor-release pilot's reading (experiments/anchor-release.md, pre-registered 2026-11-12).

For each table (3x pilot A, the training table; 3x pilot B, the held-out one): the per-set
difference between each candidate's `full` and soak-a full on the same 40,000 deals (block
`release`) against the same opponents, and between the two candidates, with the lab's set-level
bootstrap CI. Then the pre-registered reading. Reads the registry and each run's games.npz; writes
workdir/lab/release-paired.json.

    python experiments/release_paired.py        (from the repo root)
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lab import gsprt, registry  # noqa: E402

SOAK = '9c0b7b340e52652dbb6b511d370759c66986000b4904ed784ab7f27bcf9c9de7'
TABLES = {'3x pilot A': '15121432d2268ae7e755fe82e7f4bb0e97b3171a1fc9cf697bbdfbabaf68a284',
          '3x pilot B': 'b267c0267f1538c61ab962146750862fa9d4ae30bc9543ebd6f519d0a4272c3d'}
HELD_OUT = '3x pilot B'
SETS = 40000
OUT = os.path.join(ROOT, 'workdir', 'lab', 'release-paired.json')


def full_sha(run):
    """The sha256 the watcher recorded for a run's `full` snapshot, or None before it exists."""
    path = os.path.join(ROOT, 'workdir', 'fleet', 'runs', run, 'snapshots.jsonl')
    if not os.path.exists(path):
        return None
    rows = [json.loads(line) for line in open(path)]
    return next((r['sha256'] for r in rows if r['label'] == 'full'), None)


def find(rows, challenger, champion):
    """The latest finished 40K-set family-A measure of challenger against champion."""
    hits = [r for r in rows
            if r['test'] == 'measure' and r.get('verdict') == 'measured' and r['family'] == 'A'
            and r['nonce_count'] == SETS and r['challenger'].get('sha256') == challenger
            and r['champion'].get('sha256') == champion]
    return hits[-1] if hits else None


def load(row):
    # a row written on the eval box carries its paths; the run dir is workdir/lab/<run_id> on both
    z = np.load(os.path.join(ROOT, 'workdir', 'lab', row['run_id'], 'games.npz'))
    return z['seeds'].reshape(-1, 4)[:, 0], z['set_scores']


def diff(name, new, base):
    """`new` minus `base` per set, both rows on the same deals."""
    (s1, x1), (s0, x0) = load(new), load(base)
    if new['nonce_start'] != base['nonce_start'] or not np.array_equal(s1, s0):
        raise SystemExit(f'{name}: the two runs are not on the same deals')
    d = x1 - x0
    lo, hi = gsprt.bootstrap_mean_ci(d)
    return {'runs': [new['run_id'], base['run_id']], 'sets': int(len(d)),
            'means': [float(x1.mean()), float(x0.mean())],
            'diff_mean': float(d.mean()), 'diff_ci': [float(lo), float(hi)],
            'diff_sd': float(d.std(ddof=1)), 'corr': float(np.corrcoef(x1, x0)[0, 1])}


def reading(tables):
    """The pre-registered reading: who passes, and T1's mechanism."""
    passes = {x: all(t[f'{x}_vs_soak']['diff_ci'][0] > 0 for t in tables.values()) for x in 'ab'}
    if passes['a'] and passes['b']:
        lo, hi = tables[HELD_OUT]['a_vs_b']['diff_ci']
        choice = 'a' if lo > 0 else 'b'  # B unless A's lead at the held-out table excludes 0
        why = ('both pass; the candidates\' paired difference at the held-out table '
               + ('excludes 0' if lo > 0 or hi < 0 else 'includes 0, so B'))
    elif passes['a'] or passes['b']:
        choice = 'a' if passes['a'] else 'b'
        why = 'the only candidate that passes'
    else:
        choice, why = None, "neither passes: D64's reopen goes to the user"
    return {'passes': passes, 'mechanism': choice, 'why': why}


def main():
    rows = registry._read_jsonl(registry.RUNS)
    shas = {'a': full_sha('release-a'), 'b': full_sha('release-b')}
    tables, missing = {}, []
    for table, champion in TABLES.items():
        found = {'soak': find(rows, SOAK, champion)}
        found.update({x: find(rows, shas[x], champion) if shas[x] else None for x in 'ab'})
        missing += [f'{name} at {table}' for name, r in found.items() if r is None]
        if all(found.values()):
            tables[table] = {'a_vs_soak': diff(f'A at {table}', found['a'], found['soak']),
                             'b_vs_soak': diff(f'B at {table}', found['b'], found['soak']),
                             'a_vs_b': diff(f'A vs B at {table}', found['a'], found['b'])}
    out = {'full_sha256': shas, 'tables': tables, 'missing': missing}
    if not missing:
        out['reading'] = reading(tables)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
