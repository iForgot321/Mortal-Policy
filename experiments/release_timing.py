"""The stage-timing check's reading (experiments/anchor-release.md, pre-registered 2026-11-16).

At 3x pilot A on block `release`: G(s) = release-b's snapshot s minus soak-a full, per set, for
s = p25, p50, full; and R = release-b full minus release-b p50, the stage's second half. Paired on
the same deals, with the lab's set-level bootstrap CI; then the pre-registered reading. Reads the
registry and the runs' games.npz; writes workdir/lab/release-timing.json.

    python experiments/release_timing.py        (from the repo root)
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import release_paired as P  # noqa: E402
from lab import registry  # noqa: E402

OUT = os.path.join(ROOT, 'workdir', 'lab', 'release-timing.json')
PILOT_A = P.TABLES['3x pilot A']


def snapshot_sha(run, label):
    path = os.path.join(ROOT, 'workdir', 'fleet', 'runs', run, 'snapshots.jsonl')
    rows = [json.loads(line) for line in open(path)]
    return next(r['sha256'] for r in rows if r['label'] == label)


def reading(g50, r):
    if r['diff_ci'][0] <= 0 <= r['diff_ci'][1] and g50['diff_ci'][0] > 0:
        return '1: the gain was mostly in by 24 h; T1\'s pre-registration proposes 24 h stages'
    if r['diff_ci'][0] > 0:
        return '2: the gain was still coming in the second half; 48 h stays'
    return '3: grey; 48 h stays'


def main():
    rows = registry._read_jsonl(registry.RUNS)
    soak = P.find(rows, P.SOAK, PILOT_A)
    found = {s: P.find(rows, snapshot_sha('release-b', s), PILOT_A) for s in ('p25', 'p50', 'full')}
    out = {'missing': [s for s, r in found.items() if r is None]}
    for s, r in found.items():
        if r is not None:
            out[f'G_{s}'] = P.diff(f'B {s} vs soak', r, soak)
    if found['p50'] is not None:
        out['R_full_minus_p50'] = P.diff('B full vs B p50', found['full'], found['p50'])
        g50, gfull = out['G_p50']['diff_mean'], out['G_full']['diff_mean']
        out['share_by_p50'] = g50 / gfull
        out['reading'] = reading(out['G_p50'], out['R_full_minus_p50'])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v if not isinstance(v, dict) else {x: v[x] for x in ('diff_mean', 'diff_ci')}
                      for k, v in out.items()}, indent=1))


if __name__ == '__main__':
    main()
