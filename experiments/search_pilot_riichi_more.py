# Search pilot (experiments/search-pilot.md): is belief search's gain more than "riichi more often"?
# A no-search rule riichis on the dama decisions with the highest policy riichi share, at the same count.
#   python experiments/search_pilot_riichi_more.py workdir/search/pilot-a
import json, sys
import numpy as np
sys.path.insert(0, '.')
from search.pilot import RIICHI, choices, PRIMARY, clustered_se
run = sys.argv[1]
load = lambda n: {r['id']: r for r in map(json.loads, open(f'{run}/{n}.jsonl'))}
ds, orc, bel = load('decisions'), load('oracle'), load('belief')
ids = sorted(set(ds) & set(orc) & set(bel))
g = [ds[i]['game'] for i in ids]
pol_r = np.array([ds[i]['policy'] == RIICHI for i in ids])
tr = np.array([orc[i]['true_riichi'] for i in ids]); td = np.array([orc[i]['true_dama'] for i in ids])
share = np.array([ds[i]['q_riichi'] / (ds[i]['q_riichi'] + ds[i]['q_dama']) for i in ids])
pb = np.array([choices(ds[i], bel[i], PRIMARY) for i in ids]) > 0.5
to_r_b = pb & ~pol_r; to_d_b = ~pb & pol_r
print(f'belief switches: dama->riichi {to_r_b.sum()}, riichi->dama {to_d_b.sum()}')
gain_b = np.where(to_r_b, tr - td, 0) + np.where(to_d_b, td - tr, 0)
print(f'belief search, primary rule: {gain_b.mean():+.5f} ± {clustered_se(gain_b, g):.5f}')
k = to_r_b.sum()
cand = np.where(~pol_r)[0]
pick = cand[np.argsort(-share[cand])[:k]]
gain_r = np.zeros(len(ids)); gain_r[pick] = (tr - td)[pick]
print(f'no search, riichi on the {k} dama decisions with the highest policy riichi share: '
      f'{gain_r.mean():+.5f} ± {clustered_se(gain_r, g):.5f}')
gain_br = np.where(to_r_b, tr - td, 0)
print(f'belief dama->riichi switches alone: {gain_br.mean():+.5f}; overlap with the no-search picks: '
      f'{np.isin(np.where(to_r_b)[0], pick).sum()}/{k}')
for t in (0.3, 0.4, 0.45):
    sw = ~pol_r & (share >= t)
    gr = np.where(sw, tr - td, 0)
    print(f'  no search, riichi when policy riichi share >= {t}: switches {sw.mean():.3f}, gain {gr.mean():+.5f} ± {clustered_se(gr, g):.5f}')
