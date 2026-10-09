"""How much of the horizon-H return the current kyoku can explain: an upper bound on the critic's
explained variance, since later kyoku are unpredictable from the present. Post hoc analysis for
experiments/k1-lr-check.md; reads the last 5,000 games of its pool.

Usage, from mortal/:  python ../experiments/k1_h_ceiling.py
"""
import os, sys
import numpy as np
R = '/home/iforgot/projects/Mortal-Policy'
sys.path[:0] = [f'{R}/mortal']
os.environ.setdefault('MORTAL_CFG', f'{R}/experiments/configs/p1.toml')
import prelude  # noqa: F401
import torch
import trajectory as tj
from config import config
from libriichi.dataset import GameplayLoader
from model import GRP
from rl_dataset import fixed_unit_rewards, seat_trajectory

pool = f'{R}/workdir/k1-lr-check/pool'
logs = sorted(os.listdir(f'{pool}/logs'))[-5000:]   # the last games of the stream
grp = GRP(**config['grp']['network'])
grp.load_state_dict(torch.load(config['grp']['state_file'], weights_only=True, map_location='cpu')['model'])
calc = fixed_unit_rewards(grp, config['env']['pts'])
loader = GameplayLoader(version=4, oracle=False, featurize=False, player_names=['trainee'])
ret, cur, last, frac = [], [], [], []
for i in range(0, len(logs), 500):
    batch = [f'{pool}/logs/{n}' for n in logs[i:i + 500]]
    for name, games in zip(batch, loader.load_gz_log_files(batch)):
        sidecar = tj.load_behaviour(f'{pool}/bp/{os.path.basename(name)}')
        for g in games:
            t = seat_trajectory(g, sidecar, calc, 'H')
            r = tj.monte_carlo_returns(t['reward'], t['done'])
            k = t['at_kyoku']
            ret.append(r)
            cur.append(t['kyoku_rewards'][k])
            last.append(k == k[-1])
            # position of each decision within its kyoku, 0 (first) → 1 (last)
            pos = np.zeros(len(k))
            for kk in np.unique(k):
                idx = np.flatnonzero(k == kk)
                pos[idx] = np.linspace(0, 1, len(idx)) if len(idx) > 1 else 1.0
            frac.append(pos)
ret, cur, last, frac = map(np.concatenate, (ret, cur, last, frac))
ev = lambda y, p: 1 - np.var(y - p) / np.var(y)
print(f'decisions {len(ret):,}; Var(H return) {ret.var():.4f}; Var(current-kyoku reward) {cur.var():.4f}')
print(f'EV ceiling, perfect knowledge of the current kyoku\'s reward: {ev(ret, cur):.4f}')
for name, m in (('last kyoku of the game', last), ('earlier kyoku', ~last),
                ('first third of a kyoku', frac < 1/3), ('last third of a kyoku', frac >= 2/3)):
    print(f'  {name:26} {m.mean():6.1%} of decisions: ceiling {ev(ret[m], cur[m]):.4f}')
