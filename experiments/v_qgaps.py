"""How much a value checkpoint's Boltzmann exploration explores (experiments/v-tuning.md): the
Q-gap structure on the last N heldout games, the share of an exploration draw that lands off the
argmax for each temperature, and the ε that gives a target number of non-greedy actions per game.
Small and CPU-only, so it can run beside a training job.

With --reference, also the share of decisions whose argmax differs from another checkpoint's.

Usage, from mortal/:  python ../experiments/v_qgaps.py CHECKPOINT [--games 5] [--target TAU:RATE ...]
                         [--reference CHECKPOINT]
"""
import argparse, os, sys
import numpy as np
R = '/home/iforgot/projects/Mortal-Policy'
sys.path[:0] = [f'{R}/mortal']
os.environ.setdefault('MORTAL_CFG', f'{R}/experiments/configs/cql-qs-a.toml')
import torch
from libriichi.dataset import GameplayLoader
from model import Brain, DQN

TAUS = (0.05, 0.1, 0.2, 0.5, 1.0)

ap = argparse.ArgumentParser()
ap.add_argument('checkpoint')
ap.add_argument('--games', type=int, default=5)
ap.add_argument('--target', action='append', default=[], metavar='TAU:RATE',
                help='non-greedy actions per game wanted at temperature TAU; prints the ε for it')
ap.add_argument('--reference', help='a second value checkpoint to compare argmax with')
a = ap.parse_args()
torch.set_num_threads(2)

def load(path):
    state = torch.load(path, map_location='cpu', weights_only=False)
    brain = Brain(version=4, conv_channels=192, num_blocks=8, Norm='GN').eval()
    dqn = DQN(version=4).eval()
    brain.load_state_dict(state['mortal'])
    dqn.load_state_dict(state['current_dqn'])
    print(f'{path}: step {state["steps"]:,}')
    return lambda obs, masks: dqn(brain(obs), masks).numpy()

net = load(a.checkpoint)
ref = load(a.reference) if a.reference else None

lines = open(os.path.expanduser('~/mj-data/splits/heldout_game.txt')).read().split()
files = [os.path.expanduser('~/mj-data/mjai/' + l) for l in lines[-a.games:]]
loader = GameplayLoader(version=4, oracle=False, player_names=[], excludes=[], augmented=False)

legal_q, human_q, agree, moved, seat_games = [], [], [], [], 0
for f in files:
    for game in loader.load_gz_log_files([f])[0]:
        seat_games += 1
        obs = torch.as_tensor(game.take_obs()).float()
        actions = np.asarray(game.take_actions())
        masks = torch.as_tensor(np.asarray(game.take_masks(), dtype=bool))
        with torch.inference_mode():
            q = net(obs, masks)
            q_ref = ref(obs, masks) if ref else q
        for qi, qr, act, m in zip(q, q_ref, actions, masks.numpy()):
            if m.sum() < 2:
                continue
            legal_q.append(np.sort(qi[m])[::-1])
            human_q.append(qi[act])
            agree.append(qi.argmax() == act)
            moved.append(qi.argmax() != qr.argmax())

per_game = len(legal_q) / seat_games
gap = np.array([q[0] - q[1] for q in legal_q])
print(f'{seat_games} seat-games, {len(legal_q):,} decisions with >= 2 legal actions ({per_game:.0f} per seat-game)')
print(f'argmax Q = human action: {np.mean(agree):.3f}')
if ref:
    print(f'argmax Q differs from the reference: {np.mean(moved):.4f}')
print('Q(best), pct 10/50/90:', np.percentile([q[0] for q in legal_q], [10, 50, 90]).round(3))
print('Q of all legal actions, pct 1/50/99:', np.percentile(np.concatenate(legal_q), [1, 50, 99]).round(3))
print('best - second best, pct 10/25/50/75/90:', np.percentile(gap, [10, 25, 50, 75, 90]).round(3))

def off_share(tau):
    """Mean over decisions of 1 - softmax(Q/τ)[argmax]: an exploration draw's chance to leave the argmax."""
    return np.mean([1 - 1 / np.exp((q - q[0]) / tau).sum() for q in legal_q])

for tau in sorted(set(TAUS) | {float(t.split(':')[0]) for t in a.target}):
    print(f'tau {tau}: off-argmax share of a draw {off_share(tau):.4f}')
for t in a.target:
    tau, rate = map(float, t.split(':'))
    eps = rate / (per_game * off_share(tau))
    print(f'target {rate} non-greedy actions per game at tau {tau}: epsilon {eps:.4f}')
