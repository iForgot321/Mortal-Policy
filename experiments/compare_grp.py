"""Compare GRP-v2 with the GRP-v2-nobye ablation on heldout_game, clean vs BYE games (experiments/grp-v2.md).
Run from mortal/ with PYTHONPATH=. (needs model.py, libriichi)."""
import os, random, sys
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from model import GRP
from libriichi.dataset import Grp

MJ = os.path.expanduser('~/mj-data')
ROOT = '/home/iforgot/projects/Mortal-Policy'
PTS = torch.tensor([90., 50., -10., -130.], dtype=torch.float64)  # u x 100

def load(path):
    m = GRP(hidden_size=64, num_layers=2)
    m.load_state_dict(torch.load(path, weights_only=True, map_location='cpu')['model'])
    return m.double().eval()

models = {'v2': load(f'{ROOT}/workdir/grp-v2/grp-v2-frozen.pth'),
          'nobye': load(f'{ROOT}/workdir/grp-v2-nobye/grp.pth')}

g = pd.read_parquet(f'{MJ}/meta/games.parquet', columns=['log_id', 'bye_seats'])
bye = set(g[g.bye_seats.map(lambda b: b is not None and len(b) > 0)].log_id)
held = [l.strip() for l in open(f'{MJ}/splits/heldout_game.txt') if l.strip()]
random.Random(0).shuffle(held)
lid = lambda p: os.path.basename(p)[:-8]
groups = {'clean': [p for p in held if lid(p) not in bye][:8000],
          'BYE': [p for p in held if lid(p) in bye][:4000]}

@torch.inference_mode()
def evaluate(paths):
    out = {k: {'ce': [], 'r': []} for k in models}
    for i in range(0, len(paths), 500):
        for grp in Grp.load_gz_log_files([f'{MJ}/mjai/{p}' for p in paths[i:i + 500]]):
            feat = torch.as_tensor(grp.take_feature(), dtype=torch.float64)
            rbp = torch.as_tensor(list(grp.take_rank_by_player()), dtype=torch.int64)
            seq = [feat[:j + 1] for j in range(len(feat))]
            for name, m in models.items():
                logits = m(seq)                                   # (n_kyoku, 24)
                label = m.get_label(rbp.unsqueeze(0)).expand(len(seq))
                out[name]['ce'].append(F.cross_entropy(logits, label, reduction='none'))
                prob = m.calc_matrix(logits)                      # (n_kyoku, 4 players, 4 ranks)
                final = F.one_hot(rbp, 4).to(torch.float64).unsqueeze(0)
                ev = torch.cat([prob, final]) @ PTS               # (n_kyoku + 1, 4)
                out[name]['r'].append((ev[1:] - ev[:-1]).flatten())
    return {k: {kk: torch.cat(vv).numpy() for kk, vv in v.items()} for k, v in out.items()}

for gname, paths in groups.items():
    res = evaluate(paths)
    ce = {k: v['ce'].mean() for k, v in res.items()}
    r2, r1 = res['nobye']['r'], res['v2']['r']
    d = r2 - r1
    print(f'\n{gname}: {len(paths)} games, {len(r1):,} player-kyoku rewards (u x 100)')
    print(f'  placement cross-entropy  v2 {ce["v2"]:.5f}   nobye {ce["nobye"]:.5f}   diff {ce["nobye"] - ce["v2"]:+.5f}')
    print(f'  reward SD                v2 {r1.std():.3f}   nobye {r2.std():.3f}')
    print(f'  reward diff (nobye - v2) mean {d.mean():+.4f}  mean|.| {np.abs(d).mean():.4f}  '
          f'p99|.| {np.quantile(np.abs(d), 0.99):.3f}  corr {np.corrcoef(r1, r2)[0, 1]:.5f}')
