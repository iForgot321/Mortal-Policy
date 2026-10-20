# Search pilot (experiments/search-pilot.md): how well belief search's estimate of riichi - dama tracks the
# oracle's, and how much of the gap is sampling noise vs bias. From the repo root:
#   python experiments/search_pilot_accuracy.py workdir/search/pilot-a
import json, sys
sys.path.insert(0, ".")
import numpy as np
run = sys.argv[1]
load = lambda n: {r['id']: r for r in map(json.loads, open(f'{run}/{n}.jsonl'))}
ds, orc, bel = load('decisions'), load('oracle'), load('belief')
ids = sorted(set(ds) & set(orc) & set(bel))
def stats(rows):
    d = [np.asarray(rows[i]['riichi']) - np.asarray(rows[i]['dama']) for i in ids]
    n = np.array([len(x) for x in d])
    m = np.array([x.mean() if len(x) else np.nan for x in d])
    v = np.array([x.var(ddof=1) / len(x) if len(x) > 1 else np.nan for x in d])  # variance of the mean
    half = np.array([(x[0::2].mean(), x[1::2].mean()) if len(x) > 1 else (np.nan, np.nan) for x in d])
    return m, v, half, n
O, vO, hO, nO = stats(orc); B, vB, hB, nB = stats(bel)
ok = ~(np.isnan(O) | np.isnan(B) | np.isnan(vB))
from search.pilot import RIICHI
pol_r = np.array([ds[i]['policy'] == RIICHI for i in ids])
until = np.array([ds[i]['until'] for i in ids])
def report(name, m):
    m = m & ok
    o, b = O[m], B[m]
    r = np.corrcoef(o, b)[0, 1]
    relO = np.corrcoef(*hO[m].T)[0, 1]; relO = 2 * relO / (1 + relO)
    relB = np.corrcoef(*hB[m].T)[0, 1]; relB = 2 * relB / (1 + relB)
    mse = np.mean((b - o) ** 2); noise = np.mean(vB[m]) + np.mean(vO[m])
    bias = np.mean(b - o)
    sign = np.mean(np.sign(b) == np.sign(o))
    print(f'{name:38s} n={m.sum():5d}  corr(B,O)={r:.3f}  reliab O={relO:.3f} B={relB:.3f}  '
          f'disattenuated={r/np.sqrt(relO*relB):.3f}  mean(B-O)={bias:+.4f}  '
          f'MSE={mse:.2e} of which noise {noise/mse:.0%}  same sign {sign:.3f}')
print(f'particles drawn: median {int(np.median(nB))}; decisions with < 64 belief samples: {(nB < 64).sum()}')
report('all', np.ones(len(ids), bool))
report('policy riichi (search may go dama)', pol_r)
report('policy dama (search may riichi)', ~pol_r)
t = np.quantile(until, [1/3, 2/3])
report('early in the kyoku (event index tercile 1)', until <= t[0])
report('middle (tercile 2)', (until > t[0]) & (until <= t[1]))
report('late (tercile 3)', until > t[1])
big = np.abs(O) > 2 * np.sqrt(vO)
report('decisions where the oracle is confident', big)
