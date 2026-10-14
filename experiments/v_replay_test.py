"""Replay test for arm V's online collapse (experiments/v-tuning.md → "Results"): fine-tune frozen
CQL model A on fixed data of its own greedy play (the champion seats of the sweep's measure
logs), with the online trainer's step (train_value.py: same losses, AMP, AdamW without weight
decay, clipping 0.5, constant lr, a fresh optimizer), and a chosen CQL weight. Every
--probe-every steps, the Q structure on the last 5 Tenhou heldout games (as v_qgaps.py).

The hypothesis: with min_q_weight 0, training only the taken (argmax) action shrinks the spread
of the dueling head's Q until the argmax is noise; with the CQL term it doesn't.

With --calql, the CQL push-down is floored at the kyoku's return (Cal-QL, Nakamoto et al. 2023):
an action's Q is pushed down only while it is above it.

Usage, from the repo root (all arms get the same files in the same order):
    python experiments/v_replay_test.py --min-q-weight 0 --out workdir/v-replay/mq0
    python experiments/v_replay_test.py --min-q-weight 5 --calql --out workdir/v-replay/calql5
"""
import argparse, os, random, sys, time
import numpy as np
R = '/home/iforgot/projects/Mortal-Policy'
sys.path[:0] = [f'{R}/mortal']
os.environ.setdefault('MORTAL_CFG', f'{R}/experiments/configs/cql-qs-a.toml')
import torch
from itertools import chain
from torch import nn, optim
from torch.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from config import config
from dataloader import FileDatasetsIter, worker_init_fn
from libriichi.dataset import GameplayLoader
from model import Brain, DQN, AuxNet

MEASURES = ['20261013-110149-measure', '20261013-124951-measure', '20261013-143617-measure',
            '20261013-162554-measure']
A = f'{R}/workdir/cql/qs-a/cql-qs-a-final.pth'

ap = argparse.ArgumentParser()
ap.add_argument('--min-q-weight', type=float, required=True)
ap.add_argument('--calql', action='store_true',
                help="Cal-QL: push an action's Q down only while it is above the kyoku's return (the Q target)")
ap.add_argument('--out', required=True)
ap.add_argument('--steps', type=int, default=10000)
ap.add_argument('--probe-every', type=int, default=1000)
ap.add_argument('--lr', type=float, default=2e-5)
ap.add_argument('--batch-size', type=int, default=512)
ap.add_argument('--logs', type=int, default=14000, help='measure logs drawn (champion seats used)')
ap.add_argument('--seed', type=int, default=1)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = torch.device('cuda:0')
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)

state = torch.load(A, map_location=dev, weights_only=True)
mortal = Brain(version=4, **config['resnet'], Norm='GN').to(dev)
dqn = DQN(version=4).to(dev)
aux_net = AuxNet((4,)).to(dev)
mortal.load_state_dict(state['mortal']); dqn.load_state_dict(state['current_dqn'])
aux_net.load_state_dict(state['aux_net'])
models = (mortal, dqn, aux_net)

# the trainer's param groups; weight decay 0 as in V's online configs
params = [p for m in models for p in m.parameters()]
optimizer = optim.AdamW([{'params': params, 'weight_decay': 0.0}], lr=a.lr, weight_decay=0,
                        betas=(0.9, 0.999), eps=1e-8)
scaler = GradScaler('cuda')
mse, ce = nn.MSELoss(), nn.CrossEntropyLoss()

files = []
for m in MEASURES:
    d = os.path.join(R, 'workdir', 'lab', m, 'logs')
    files += [os.path.join(d, f) for f in sorted(os.listdir(d))]
random.Random(a.seed).shuffle(files)
files = files[:a.logs]
ds = FileDatasetsIter(version=4, file_list=files, pts=config['env']['pts'], file_batch_size=16,
                      player_names=['champion'], num_epochs=1, compute_advantage=False,
                      value_targets=True, obs_fp16=True)
dl = DataLoader(ds, batch_size=a.batch_size, drop_last=True, num_workers=2, pin_memory=True,
                worker_init_fn=worker_init_fn, persistent_workers=True)
# fork the workers before the probe's GameplayLoader starts the Rust thread pool in this
# process: forked after it, they deadlock (as the trainer, which builds its loader first)
batches = iter(dl)


# probe: the last 5 Tenhou heldout games, every seat (v_qgaps.py's set), and A's argmax on it
lines = open(os.path.expanduser('~/mj-data/splits/heldout_game.txt')).read().split()
probe_files = [os.path.expanduser('~/mj-data/mjai/' + l) for l in lines[-5:]]
loader = GameplayLoader(version=4, oracle=False, player_names=[], excludes=[], augmented=False)
p_obs, p_act, p_mask = [], [], []
for f in probe_files:
    for game in loader.load_gz_log_files([f])[0]:
        p_obs.append(np.asarray(game.take_obs())); p_act.append(np.asarray(game.take_actions()))
        p_mask.append(np.asarray(game.take_masks(), dtype=bool))
p_obs = torch.as_tensor(np.concatenate(p_obs)).float()
p_act = np.concatenate(p_act); p_mask = np.concatenate(p_mask)
multi = p_mask.sum(-1) >= 2
p_obs, p_act, p_mask = p_obs[multi], p_act[multi], p_mask[multi]


def probe_q():
    for m in models:
        m.eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(p_obs), 512):
            o = p_obs[i:i + 512].to(dev)
            mk = torch.as_tensor(p_mask[i:i + 512]).to(dev)
            out.append(dqn(mortal(o), mk).float().cpu().numpy())
    for m in models:
        m.train()
    return np.concatenate(out)


q0 = probe_q()
a_argmax = q0.argmax(-1)
log = open(os.path.join(a.out, 'probe.tsv'), 'a')


def probe(step, losses):
    q = probe_q()
    legal = np.where(p_mask, q, np.nan)
    srt = -np.sort(-np.where(p_mask, q, -np.inf), -1)
    gap = srt[:, 0] - srt[:, 1]
    # off-argmax share of a Boltzmann draw at τ 0.05 (v_qgaps.py)
    z = np.where(p_mask, (q - srt[:, :1]) / 0.05, -np.inf)
    pr = np.exp(z); pr /= pr.sum(-1, keepdims=True)
    off = 1 - pr.max(-1)
    row = dict(step=step, dqn=losses[0], cql=losses[1], rank=losses[2],
               human=(q.argmax(-1) == p_act).mean(), vs_A=(q.argmax(-1) == a_argmax).mean(),
               gap_med=np.median(gap), gap_p10=np.percentile(gap, 10),
               q1=np.nanpercentile(legal, 1), q50=np.nanpercentile(legal, 50),
               q99=np.nanpercentile(legal, 99), best_med=np.median(srt[:, 0]), off05=off.mean())
    row = {k: v if k == 'step' else float(v) for k, v in row.items()}
    if step == 0:
        log.write('\t'.join(row) + '\n')
    log.write('\t'.join(f'{v:.4f}' if isinstance(v, float) else str(v) for v in row.values()) + '\n')
    log.flush()
    print(' '.join(f'{k} {v:.4f}' if isinstance(v, float) else f'{k} {v}' for k, v in row.items()),
          flush=True)


probe(0, (float('nan'),) * 3)

steps, acc, t0 = 0, np.zeros(3), time.time()
for obs, actions, masks, q_target, ranks in batches:
    obs = obs.to(dev, torch.float32); actions = actions.to(dev, torch.int64)
    masks = masks.to(dev, torch.bool); q_target = q_target.to(dev, torch.float32)
    ranks = ranks.to(dev, torch.int64)
    with torch.autocast('cuda'):
        phi = mortal(obs)
        q_out = dqn(phi, masks)
        q = q_out[range(len(actions)), actions]
        dqn_loss = 0.5 * mse(q, q_target)
        # logged as the plain CQL term in every arm, so the arms compare
        cql_loss = q_out.logsumexp(-1).mean() - q.mean()
        if a.calql:
            # Cal-QL's floor: the behaviour policy's return from this state
            floor = q_target[:, None].to(q_out.dtype)
            q_push = torch.maximum(q_out, floor).masked_fill(~masks, -torch.inf)
            push = q_push.logsumexp(-1).mean() - q.mean()
        else:
            push = cql_loss
        next_rank_loss = ce(aux_net(phi)[0], ranks)
        loss = dqn_loss + push * a.min_q_weight + next_rank_loss * 0.2
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    clip_grad_norm_(chain.from_iterable(g['params'] for g in optimizer.param_groups), 0.5)
    scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
    acc += [float(dqn_loss), float(cql_loss), float(next_rank_loss)]
    steps += 1
    if steps % a.probe_every == 0:
        print(f'{time.time() - t0:.0f} s', end=' ')
        probe(steps, tuple(acc / a.probe_every))
        acc[:] = 0
    if steps >= a.steps:
        break

torch.save({'mortal': mortal.state_dict(), 'current_dqn': dqn.state_dict(),
            'aux_net': aux_net.state_dict(), 'steps': state['steps'] + steps,
            'replay': vars(a)}, os.path.join(a.out, 'final.pth'))
print(f'done: {steps} steps in {time.time() - t0:.0f} s', flush=True)
