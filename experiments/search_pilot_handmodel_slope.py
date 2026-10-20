# Search pilot (experiments/search-pilot.md): recovery against hand-model quality. Belief runs with
# different hand models on the decisions they have in common (pilot-a's ids follow its seed-0 shuffle,
# so any prefix is a random subset), scored on the same true worlds: each model's belief gain, recovery
# and accuracy against the oracle, and its paired difference from the first run's model.
#   python experiments/search_pilot_handmodel_slope.py workdir/search/pilot-a \
#       'weak (2.5K steps)=workdir/search/pilot-a-sub4k-hands2500' 'big (20K games, 60K steps)=workdir/search/pilot-a-handsbig'
import json, sys
sys.path.insert(0, ".")
import numpy as np
from search.pilot import PRIMARY, RIICHI, choices, clustered_se
full = sys.argv[1]
load = lambda run, n: {r['id']: r for r in map(json.loads, open(f'{run}/{n}.jsonl'))}  # noqa: E731
ds, orc = load(full, 'decisions'), load(full, 'oracle')
runs = {'current (20K steps)': load(full, 'belief')}
runs.update((label, load(path, 'belief')) for label, path in (a.split('=', 1) for a in sys.argv[2:]))
ids = sorted(set.intersection(*(set(r) for r in runs.values())) & set(orc))
g = [ds[i]['game'] for i in ids]
pol_r = np.array([ds[i]['policy'] == RIICHI for i in ids])
tr = np.array([orc[i]['true_riichi'] for i in ids])
td = np.array([orc[i]['true_dama'] for i in ids])
tpol = np.where(pol_r, tr, td)


def gain(rows):
    p = np.array([choices(ds[i], rows[i], PRIMARY) for i in ids])
    return p * tr + (1 - p) * td - tpol, (p > 0.5) != pol_r


def delta(rows):
    d = [np.asarray(rows[i]['riichi']) - np.asarray(rows[i]['dama']) for i in ids]
    m = np.array([x.mean() if len(x) else np.nan for x in d])
    half = np.array([(x[0::2].mean(), x[1::2].mean()) if len(x) > 1 else (np.nan, np.nan) for x in d])
    return m, half


go, _ = gain(orc)
O, hO = delta(orc)
print(f'{len(ids)} common decisions from {len(set(g))} games; oracle gain {go.mean():+.5f} ± {clustered_se(go, g):.5f}')
ug, inv = np.unique(g, return_inverse=True)
rng = np.random.default_rng(0)
boots = [np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug)) for _ in range(10000)]
so = np.bincount(inv, go, len(ug))
gains = {}
for name, rows in runs.items():
    gb, dev = gain(rows)
    gains[name] = gb
    sb = np.bincount(inv, gb, len(ug))
    ci = np.percentile([(k @ sb) / (k @ so) for k in boots], [2.5, 97.5])
    B, hB = delta(rows)
    ok = ~(np.isnan(O) | np.isnan(B) | np.isnan(hB[:, 0]))
    rel = lambda h: (lambda r: 2 * r / (1 + r))(np.corrcoef(*h[ok].T)[0, 1])  # noqa: E731
    r = np.corrcoef(O[ok], B[ok])[0, 1]
    late = ok & (np.array([ds[i]['until'] for i in ids]) > np.quantile([ds[i]['until'] for i in ids], 2 / 3))
    print(f'{name:26s} belief gain {gb.mean():+.5f} ± {clustered_se(gb, g):.5f} (z {gb.mean() / clustered_se(gb, g):.1f}), '
          f'switches {dev.mean():.3f}; recovery {gb.mean() / go.mean():.3f} [{ci[0]:.3f}, {ci[1]:.3f}]; '
          f'corr with oracle {r:.3f}, without noise {r / np.sqrt(rel(hO) * rel(hB)):.3f}; '
          f'riichi bias {np.mean(B[ok] - O[ok]):+.4f} (late third {np.mean(B[late] - O[late]):+.4f}); '
          f'< 64 particles {sum(len(rows[i]["riichi"]) < 64 for i in ids)}')
base = gains.pop('current (20K steps)')
for name, gb in gains.items():
    d = gb - base
    print(f'paired difference, {name} - current: {d.mean():+.5f} ± {clustered_se(d, g):.5f} (z {d.mean() / clustered_se(d, g):.1f})')
