# Search pilot (experiments/search-pilot.md): the recovery ratio's game-clustered bootstrap CI, and
# where the oracle's and the belief's gains come from. From the repo root:
#   python experiments/search_pilot_breakdown.py workdir/search/pilot-a
import json, sys
import numpy as np
sys.path.insert(0, '.')
from search.pilot import choices, RIICHI, PRIMARY
run = sys.argv[1]
load = lambda n: {r['id']: r for r in map(json.loads, open(f'{run}/{n}.jsonl'))}
ds, orc, bel = load('decisions'), load('oracle'), load('belief')
ids = sorted(set(ds) & set(orc) & set(bel))
games = np.array([ds[i]['game'] for i in ids])
pol_r = np.array([ds[i]['policy'] == RIICHI for i in ids])
tr = np.array([orc[i]['true_riichi'] for i in ids]); td = np.array([orc[i]['true_dama'] for i in ids])
tpol = np.where(pol_r, tr, td)
def gain(rows, rule):
    p = np.array([choices(ds[i], rows[i], rule) for i in ids])
    return p * tr + (1 - p) * td - tpol, (p > 0.5) != pol_r
go, devo = gain(orc, PRIMARY); gb, devb = gain(bel, PRIMARY)
print(f'primary {PRIMARY}: oracle {go.mean():+.5f} (dev {devo.mean():.3f}), belief {gb.mean():+.5f} (dev {devb.mean():.3f}), ratio {gb.mean()/go.mean():.3f}')
ug, inv = np.unique(games, return_inverse=True)
so, sb, n = (np.bincount(inv, w, len(ug)) for w in (go, gb, np.ones(len(ids))))
rng = np.random.default_rng(0)
rs = []
for _ in range(10000):
    k = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))
    rs.append((k @ sb) / (k @ so))
print('recovery 95% CI (game bootstrap, 10K):', np.percentile(rs, [2.5, 97.5]).round(3), ' P(ratio >= 0.5):', np.mean(np.array(rs) >= 0.5))
print('by policy choice (share of decisions; per-decision mean gain within the group; dev rates):')
for name, m in (('policy riichi -> search may go dama', pol_r), ('policy dama -> search may riichi', ~pol_r)):
    print(f'  {name}: share {m.mean():.3f}; oracle {go[m].mean():+.5f} dev {devo[m].mean():.3f}; belief {gb[m].mean():+.5f} dev {devb[m].mean():.3f}')
both = devo & devb
print(f'belief deviations that the oracle also makes: {both.sum()}/{devb.sum()} ({both.sum()/max(devb.sum(),1):.2f}); oracle deviations belief finds: {both.sum()}/{devo.sum()} ({both.sum()/devo.sum():.2f})')
print(f'gain on belief deviations: belief {gb[devb].mean():+.4f} per deviation; on oracle deviations: oracle {go[devo].mean():+.4f}')
