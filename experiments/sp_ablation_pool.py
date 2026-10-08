"""SP ablation decision rule (experiments/sp-ablation.md §5.1), applied to the two fixed-N runs.

Registered 2026-10-07 22:00, before pair 2's result existed:
1. Per pair: mean Δu over its 20K set scores, SE = SD/√n, 95% bootstrap CI (as `lab measure`).
2. Consistency: d = m2 − m1, SE_d = √(SE1² + SE2²), z = d / SE_d. The pairs are consistent if
   |z| ≤ 1.96. If not, the effect varied between training runs more than measurement noise allows:
   the verdict is INCONCLUSIVE at the recipe level, whatever the pooled number says.
3. Pooled: the average of the two pair means (equal n), 95% CI by a stratified bootstrap
   (each pair resampled within itself).
4. If consistent: ADOPT v5 if the pooled lower bound > −0.005; KEEP v4 (grey zone: try a cheaper
   SP variant) if it is in (−0.01, −0.005]; KEEP v4 if it is ≤ −0.01.

Usage, from the repo root:
    python experiments/sp_ablation_pool.py <pair-1 run id> <pair-2 run id> [--out file.json]
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lab.gsprt import _atoms, bootstrap_mean_ci  # noqa: E402

RUNS_DIR = os.path.join(ROOT, 'workdir/lab')
Z = 1.96
ADOPT_ABOVE, MARGIN = -0.005, -0.01
RESAMPLES = 10_000


def set_scores(run):
    path = run if run.endswith('.npz') else os.path.join(RUNS_DIR, run, 'games.npz')
    return np.load(path)['set_scores'].astype(np.float64)


def boot_means(x, seed):
    """Bootstrap means of x, exact multinomial over its atoms (as lab.gsprt.bootstrap_mean_ci)."""
    rng = np.random.default_rng(seed)
    values, counts = _atoms(x)
    return rng.multinomial(len(x), counts / len(x), size=RESAMPLES) @ values / len(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('pair1')
    ap.add_argument('pair2')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    pairs = []
    for i, run in enumerate((args.pair1, args.pair2), 1):
        x = set_scores(run)
        lo, hi = bootstrap_mean_ci(x)
        pairs.append({'pair': i, 'run': run, 'sets': len(x), 'mean': x.mean(), 'sd': x.std(ddof=1),
                      'se': x.std(ddof=1) / np.sqrt(len(x)), 'ci95': [lo, hi], 'x': x})

    p1, p2 = pairs
    d = p2['mean'] - p1['mean']
    se_d = np.hypot(p1['se'], p2['se'])
    z = d / se_d
    consistent = abs(z) <= Z

    pooled = (p1['mean'] + p2['mean']) / 2
    boot = (boot_means(p1['x'], 1) + boot_means(p2['x'], 2)) / 2
    lo, hi = (float(q) for q in np.quantile(boot, [0.025, 0.975]))

    if not consistent:
        verdict = 'INCONCLUSIVE: the pairs disagree beyond measurement noise; more pairs needed'
    elif lo > ADOPT_ABOVE:
        verdict = 'ADOPT v5 (pooled lower bound above -0.005)'
    elif lo > MARGIN:
        verdict = 'KEEP v4: grey zone (pooled lower bound in (-0.01, -0.005]); try a cheaper SP variant'
    else:
        verdict = 'KEEP v4 (pooled lower bound at or below -0.01)'

    out = {
        'pairs': [{k: v for k, v in p.items() if k != 'x'} for p in pairs],
        'consistency': {'d': d, 'se_d': se_d, 'z': z, 'consistent': bool(consistent), 'threshold_abs_z': Z},
        'pooled': {'mean': pooled, 'ci95': [lo, hi], 'se': se_d / 2},
        'verdict': verdict,
    }
    for p in out['pairs']:
        print(f"pair {p['pair']} ({p['run']}): {p['sets']:,} sets, mean {p['mean']:+.4f}, "
              f"SE {p['se']:.4f}, 95% CI [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}]")
    print(f'consistency: d = {d:+.4f}, SE_d = {se_d:.4f}, z = {z:+.2f} -> '
          f"{'consistent' if consistent else 'INCONSISTENT'} (|z| <= {Z})")
    print(f'pooled: mean {pooled:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}]')
    print(f'verdict: {verdict}')
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(out, f, indent=1, default=float)


if __name__ == '__main__':
    main()
