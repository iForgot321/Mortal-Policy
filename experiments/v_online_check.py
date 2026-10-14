"""Interface checks for arm V's online data (experiments/v-tuning.md → "Results"; 05 §8, step 1),
on the lab's 1v3 logs, which the same arena writes as the training actors do:

1. Alignment: through the trainer's own loader (FileDatasetsIter with value_targets), the model
   that played a seat picks the logged action from the observation it is paired with (argmax of
   its masked Q). An off-by-one between observations and actions shows up as a low match rate.
2. Target scale: the Q targets the trainer sees (the kyoku's GRP-Δu in u) on the 1v3 seats
   against the last Tenhou heldout games, and each model's MSE on them.

Usage, from the repo root:
    python experiments/v_online_check.py --lab-run workdir/lab/<run_id> \
        --challenger CKPT --champion CKPT [--games 200] [--tenhou 100]
"""
import argparse, os, sys
import numpy as np
R = '/home/iforgot/projects/Mortal-Policy'
sys.path[:0] = [f'{R}/mortal']
os.environ.setdefault('MORTAL_CFG', f'{R}/experiments/configs/cql-qs-a.toml')
import torch
from config import config
from dataloader import FileDatasetsIter
from model import Brain, DQN

ap = argparse.ArgumentParser()
ap.add_argument('--lab-run', required=True, help='a lab measure run dir (its logs/ are 1v3 games)')
ap.add_argument('--challenger', required=True, help='the checkpoint that played "challenger"')
ap.add_argument('--champion', required=True, help='the checkpoint that played "champion"')
ap.add_argument('--games', type=int, default=200, help='1v3 logs (one seat rotation each)')
ap.add_argument('--tenhou', type=int, default=100, help='last Tenhou heldout games')
ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
a = ap.parse_args()
dev = torch.device(a.device)
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


def load(path):
    state = torch.load(path, map_location='cpu', weights_only=False)
    brain = Brain(version=4, conv_channels=192, num_blocks=8, Norm='GN').eval().to(dev)
    dqn = DQN(version=4).eval().to(dev)
    brain.load_state_dict(state['mortal'])
    dqn.load_state_dict(state['current_dqn'])
    print(f'{path}: step {state["steps"]:,}')
    return lambda obs, masks: dqn(brain(obs), masks)


nets = {'challenger': load(a.challenger), 'champion': load(a.champion)}
lab_logs = sorted(os.listdir(os.path.join(a.lab_run, 'logs')))[:a.games]
lab_logs = [os.path.join(a.lab_run, 'logs', f) for f in lab_logs]
lines = open(os.path.expanduser('~/mj-data/splits/heldout_game.txt')).read().split()
tenhou = [os.path.expanduser('~/mj-data/mjai/' + l) for l in lines[-a.tenhou:]]


def entries(files, names):
    ds = FileDatasetsIter(version=4, file_list=list(files), pts=config['env']['pts'],
                          file_batch_size=10, player_names=names, num_epochs=1,
                          compute_advantage=False, value_targets=True)
    batch = []
    for e in ds:
        batch.append(e)
        if len(batch) == 512:
            yield batch
            batch = []
    if batch:
        yield batch


def check(label, files, names, played_by=None):
    """played_by: the net whose argmax should reproduce the logged actions (None for humans)."""
    targets, n, n2 = [], 0, 0
    match = {k: [0, 0] for k in nets}          # matches, decisions with >= 2 legal actions
    clear = {k: [0, 0] for k in nets}          # the same where the net's best - second > 0.05 u
    sq = {k: 0.0 for k in nets}
    for batch in entries(files, names):
        obs = torch.as_tensor(np.stack([e[0] for e in batch])).float().to(dev)
        act = torch.as_tensor(np.array([e[1] for e in batch], dtype=np.int64)).to(dev)
        masks = torch.as_tensor(np.stack([e[2] for e in batch]).astype(bool)).to(dev)
        tgt = torch.as_tensor(np.array([e[3] for e in batch], dtype=np.float32)).to(dev)
        targets.append(tgt.cpu().numpy())
        multi = masks.sum(-1) >= 2
        n += len(batch)
        n2 += int(multi.sum())
        with torch.inference_mode():
            for k, net in nets.items():
                q = net(obs, masks)
                q_taken = q.gather(1, act[:, None])[:, 0]
                sq[k] += float(((q_taken - tgt) ** 2).sum())
                top2 = q.masked_fill(~masks, -torch.inf).topk(2, -1).values
                hit = (q.masked_fill(~masks, -torch.inf).argmax(-1) == act)
                match[k][0] += int((hit & multi).sum()); match[k][1] += int(multi.sum())
                big = multi & ((top2[:, 0] - top2[:, 1]) > 0.05)
                clear[k][0] += int((hit & big).sum()); clear[k][1] += int(big.sum())
    t = np.concatenate(targets)
    print(f'\n== {label}: {len(files)} logs, seats {names or "all"}; {n:,} decisions, '
          f'{n2:,} with >= 2 legal actions')
    print(f'Q target (u): mean {t.mean():+.4f}, sd {t.std():.4f}, '
          f'pct 1/10/50/90/99 {np.round(np.percentile(t, [1, 10, 50, 90, 99]), 3)}')
    for k in nets:
        tag = '  <- played these seats' if k == played_by else ''
        print(f'  {k:10s} argmax = logged action {match[k][0] / max(match[k][1], 1):.4f} '
              f'(gap > 0.05 u: {clear[k][0] / max(clear[k][1], 1):.4f} of {clear[k][1]:,}); '
              f'MSE(Q taken, target) {sq[k] / n:.4f}{tag}')


check('1v3 challenger seat', lab_logs, ['challenger'], 'challenger')
check('1v3 champion seats', lab_logs, ['champion'], 'champion')
check('Tenhou heldout', tenhou, [])
