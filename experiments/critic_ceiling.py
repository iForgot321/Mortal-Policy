#!/usr/bin/env python3
"""D64's critic check (experiments/critic-ceiling.md; docs/design/t1-preconditions-draft.md §1): how
much of a kyoku's GRP-Δu can a critic explain with the opponents' hands, how much without, and how
far below that is K1's online critic as `k2b` left it.

Data: the K1 learning-rate check's games (experiments/k1-lr-check.md): 100,000 hanchan of pilot A's B0
play (`workdir/k1-lr-check/pool`, for training) and 4,000 held out (`.../fresh`), the trainee's
decisions, featurized as v4 with the exact hidden information (oracle, trust_seed). Targets in u
(rl_dataset's fixed units): the decision's own kyoku's GRP-Δu, and the H return for K1's own figure.

Arms, trained in one stream (each batch is featurized once; every arm still training steps on it):
- `public`: a b8c128 on the observation only;
- `privileged`: K1's critic architecture (b8c128, observation + the opponents' hands, critic.py);
- `privileged_b16c256`: the same at b16c256.
Each: AdamW (LR 1e-4, weight decay 0, D52), batch 512, MSE, AMP, gradient clip 1.0. Every 7,500 steps
each arm is scored on the first 1,000 held-out games (the selection games); an arm stops once two
evaluations in a row fail to beat its best explained variance by 0.002, and keeps its best weights;
3 passes at most. Then every arm's best, and `k2b`'s critic as it is, are scored on the other 3,000
held-out games, which no choice has seen (the report).

    python experiments/critic_ceiling.py run [--out workdir/critic-ceiling]
    python experiments/critic_ceiling.py run --pool-games 200 --fresh-games 40 --eval-every 50 --out /tmp/x   # smoke

No lab nonces and no registry rows: not a strength claim (as the K1 check).
"""
import argparse
import json
import os
import random
import sys
import time
from functools import partial

from torch.utils.data import IterableDataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'mortal'))
os.environ.setdefault('MORTAL_CFG', os.path.join(ROOT, 'experiments', 'configs', 'k2.toml'))

POOL = os.path.join(ROOT, 'workdir', 'k1-lr-check', 'pool')
FRESH = os.path.join(ROOT, 'workdir', 'k1-lr-check', 'fresh')
K2B = os.path.join(ROOT, 'workdir', 'fleet', 'runs', 'k2b', 'snapshots', 'full.pth')
GRP_FILE = os.path.join(ROOT, 'workdir', 'grp-v2', 'grp-v2-frozen.pth')
PTS = [90.0, 50.0, -10.0, -130.0]
ARMS = {'public': (128, 8, False), 'privileged': (128, 8, True), 'privileged_b16c256': (256, 16, True)}
MIN_GAIN = 0.002
PATIENCE = 2


def logs_of(d, n=None):
    files = sorted(os.path.join(d, 'logs', f) for f in os.listdir(os.path.join(d, 'logs')))
    return files[:n] if n else files


class Decisions(IterableDataset):
    """The trainee's decisions of `files` (an IterableDataset): (obs, priv, kyoku Δ, H return, last kyoku,
    third of the kyoku), fp16 observations. Workers split the files; `shuffle` mixes within a
    file batch, with a seed per (seed, epoch, worker)."""

    def __init__(self, files, behaviour_dir, file_batch_size=16, shuffle=False, seed=0):
        super().__init__()
        self.files, self.behaviour_dir = list(files), behaviour_dir
        self.file_batch_size, self.shuffle, self.seed = file_batch_size, shuffle, seed
        self.epoch = 0

    def __iter__(self):
        import numpy as np
        import torch
        from torch.utils.data import get_worker_info
        import trajectory as tj
        from critic import privileged
        from libriichi.dataset import GameplayLoader
        from model import GRP
        from rl_dataset import fixed_unit_rewards, seat_trajectory, sidecar_path

        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        files = self.files[wid::nw]
        rng = random.Random(self.seed * 1_000_003 + self.epoch * 1_009 + wid)
        if self.shuffle:
            rng.shuffle(files)
        grp = GRP(hidden_size=64, num_layers=2)
        grp.load_state_dict(torch.load(GRP_FILE, weights_only=True, map_location='cpu')['model'])
        calc = fixed_unit_rewards(grp, PTS)
        loader = GameplayLoader(version=4, oracle=True, trust_seed=True, player_names=['trainee'])
        for start in range(0, len(files), self.file_batch_size):
            part = files[start:start + self.file_batch_size]
            buf = []
            for f, games in zip(part, loader.load_gz_log_files(part)):
                sidecar = tj.load_behaviour(sidecar_path(f, self.behaviour_dir))
                for game in games:
                    t = seat_trajectory(game, sidecar, calc, 'H', oracle=True)
                    n = len(t['action'])
                    if n == 0:
                        continue
                    at = t['at_kyoku'].astype(np.int64)
                    kd = t['kyoku_rewards'][at].astype(np.float32)
                    h = tj.monte_carlo_returns(t['reward'], t['done']).astype(np.float32)
                    last = at == at[-1]
                    # position within the kyoku's decisions of this seat: thirds 0, 1, 2
                    third = np.zeros(n, dtype=np.int64)
                    for k in np.unique(at):
                        idx = np.flatnonzero(at == k)
                        third[idx] = (np.arange(len(idx)) * 3) // len(idx)
                    for i in range(n):
                        buf.append((t['obs'][i].astype(np.float16), privileged(t['invisible_obs'][i]).astype(np.float16),
                                    kd[i], h[i], bool(last[i]), int(third[i]), int(at[i])))
            if self.shuffle:
                rng.shuffle(buf)
            yield from buf


def build(conv, blocks, priv):
    import torch
    from torch import nn
    from critic import Critic, PRIVILEGED_ROWS
    from libriichi.consts import obs_shape
    from model import ResNet

    if priv:
        return Critic(conv_channels=conv, num_blocks=blocks, version=4)

    class Public(nn.Module):
        """critic.Critic without the privileged rows."""
        def __init__(self):
            super().__init__()
            self.encoder = ResNet(in_channels=obs_shape(4)[0], conv_channels=conv, num_blocks=blocks,
                                  norm_builder=partial(nn.GroupNorm, num_channels=conv, num_groups=32, eps=1e-3),
                                  actv_builder=partial(nn.Mish, inplace=True), pre_actv=True)
            self.actv = nn.Mish(inplace=True)
            self.head = nn.Linear(1024, 1)

        def forward(self, obs, priv=None):
            return self.head(self.actv(self.encoder(obs))).squeeze(-1)
    assert PRIVILEGED_ROWS == 51
    return Public()


def evaluate(nets, files, device, batch_size, workers, amp):
    """Predictions of every net on every decision of `files`, with the targets, as numpy arrays."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    preds = {k: [] for k in nets}
    cols = {'kd': [], 'h': [], 'last': [], 'third': [], 'kyoku': []}
    for m in nets.values():
        m.eval()
    with torch.no_grad():
        for obs, priv, kd, h, last, third, kyoku in DataLoader(Decisions(files, os.path.join(os.path.dirname(os.path.dirname(files[0])), 'bp')),
                                                              batch_size=batch_size, num_workers=workers):
            obs, priv = obs.to(device).float(), priv.to(device).float()
            with torch.autocast(device.type, enabled=amp):
                for k, m in nets.items():
                    preds[k].append(m(obs, priv).float().cpu().numpy())
            for c, v in zip(cols, (kd, h, last, third, kyoku)):
                cols[c].append(v.numpy())
    for m in nets.values():
        m.train()
    return {k: np.concatenate(v) for k, v in preds.items()}, {c: np.concatenate(v) for c, v in cols.items()}


def metrics(pred, cols, target='kd'):
    import numpy as np
    from critic import calibration, explained_variance

    y = cols[target]
    out = {'ev': explained_variance(pred, y), 'ev_last_kyoku': explained_variance(pred[cols['last']], y[cols['last']]),
           'ev_earlier_kyoku': explained_variance(pred[~cols['last']], y[~cols['last']]),
           'mse': float(np.mean((pred - y) ** 2)), 'monotone': calibration(pred, y)[2]}
    for t in range(3):
        sel = cols['third'] == t
        out[f'ev_third_{t}'] = explained_variance(pred[sel], y[sel])
    return out


def cmd_run(a):
    import numpy as np
    import torch
    from torch.nn.utils import clip_grad_norm_
    from torch.utils.data import DataLoader
    from checkpoint import load_checkpoint
    from critic import Critic

    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, 'log.jsonl'), 'a')

    def say(rec):
        rec = {'time': round(time.time()), **rec}
        log.write(json.dumps(rec) + '\n')
        log.flush()
        print(json.dumps(rec), flush=True)

    device = torch.device(a.device)
    torch.manual_seed(a.seed)
    nets, opts, scalers = {}, {}, {}
    for k, (conv, blocks, priv) in ARMS.items():
        nets[k] = build(conv, blocks, priv).to(device)
        opts[k] = torch.optim.AdamW(nets[k].parameters(), lr=a.lr, weight_decay=0.0, fused=device.type == 'cuda')
        scalers[k] = torch.amp.GradScaler(device.type, enabled=not a.no_amp)
    say({'event': 'start', 'arms': {k: sum(p.numel() for p in m.parameters()) for k, m in nets.items()}})

    pool = logs_of(POOL, a.pool_games)
    fresh = logs_of(FRESH, a.fresh_games)
    eval_files, report_files = fresh[:max(1, len(fresh) // 4)], fresh[max(1, len(fresh) // 4):]
    best = {k: (-float('inf'), None) for k in nets}
    misses = {k: 0 for k in nets}
    active = set(nets)
    step, t0 = 0, time.time()
    ds = Decisions(pool, os.path.join(POOL, 'bp'), shuffle=True, seed=a.seed)
    sums = {k: 0.0 for k in nets}
    for epoch in range(a.passes):
        ds.epoch = epoch
        for obs, priv, kd, *_ in DataLoader(ds, batch_size=a.batch_size, num_workers=a.workers, drop_last=True):
            obs, priv, kd = obs.to(device, non_blocking=True).float(), priv.to(device, non_blocking=True).float(), kd.to(device)
            for k in active:
                with torch.autocast(device.type, enabled=not a.no_amp):
                    pred = nets[k](obs, priv)
                loss = torch.nn.functional.mse_loss(pred.float(), kd)
                scalers[k].scale(loss).backward()
                scalers[k].unscale_(opts[k])
                clip_grad_norm_(nets[k].parameters(), 1.0)
                scalers[k].step(opts[k])
                scalers[k].update()
                opts[k].zero_grad(set_to_none=True)
                sums[k] += loss.detach()
            step += 1
            if step % a.eval_every == 0:
                preds, cols = evaluate({k: nets[k] for k in active}, eval_files, device, 1024, a.workers, not a.no_amp)
                rec = {'event': 'eval', 'step': step, 'epoch': epoch, 'samples': step * a.batch_size,
                       'hours': round((time.time() - t0) / 3600, 3)}
                for k in sorted(active):
                    m = metrics(preds[k], cols)
                    rec[k] = {**m, 'train_mse': float(sums[k]) / a.eval_every}
                    if m['ev'] > best[k][0] + MIN_GAIN:
                        best[k] = (m['ev'], {n: t.detach().cpu().clone() for n, t in nets[k].state_dict().items()})
                        misses[k] = 0
                    else:
                        misses[k] += 1
                        if m['ev'] > best[k][0]:
                            best[k] = (m['ev'], {n: t.detach().cpu().clone() for n, t in nets[k].state_dict().items()})
                    if misses[k] >= PATIENCE:
                        active.discard(k)
                        rec[k]['stopped'] = True
                for k in sums:
                    sums[k] = 0.0
                say(rec)
                if not active:
                    break
        if not active:
            break
    say({'event': 'trained', 'step': step, 'hours': round((time.time() - t0) / 3600, 3),
         'best_ev_eval_subset': {k: v[0] for k, v in best.items()}})

    # every arm's best and k2b's critic on the held-out games no choice has seen
    for k, (_, sd) in best.items():
        if sd is not None:
            nets[k].load_state_dict(sd)
            torch.save(sd, os.path.join(a.out, f'{k}.pth'))
    st = load_checkpoint(K2B)
    k1 = Critic(conv_channels=st['config']['critic']['conv_channels'], num_blocks=st['config']['critic']['num_blocks'],
                version=4).to(device)
    k1.load_state_dict(st['critic'])
    nets['k1_k2b'] = k1
    preds, cols = evaluate(nets, report_files, device, 1024, a.workers, not a.no_amp)
    report = {'games': len(report_files), 'selection_games': len(eval_files), 'decisions': int(len(cols['kd'])), 'step': step,
              'arms': {k: metrics(p, cols) for k, p in preds.items()},
              'k1_k2b_on_h_return': metrics(preds['k1_k2b'], cols, target='h'),
              'var_kd': float(np.var(cols['kd'])), 'var_h': float(np.var(cols['h']))}
    # what the later kyoku add, on average, by kyoku index (D64: a constant would serve H's GAE)
    later = cols['h'] - cols['kd']
    report['later_return_by_kyoku'] = {int(k): {'n': int((cols['kyoku'] == k).sum()), 'mean': float(later[cols['kyoku'] == k].mean()),
                                                'sd': float(later[cols['kyoku'] == k].std())}
                                       for k in np.unique(cols['kyoku'])}
    with open(os.path.join(a.out, 'report.json'), 'w') as f:
        json.dump(report, f, indent=1)
    say({'event': 'done', 'report': os.path.join(a.out, 'report.json'),
         'ev': {k: round(v['ev'], 4) for k, v in report['arms'].items()},
         'ev_last_kyoku': {k: round(v['ev_last_kyoku'], 4) for k, v in report['arms'].items()}})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('run')
    p.add_argument('--out', default=os.path.join(ROOT, 'workdir', 'critic-ceiling'))
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--passes', type=int, default=3)
    p.add_argument('--eval-every', type=int, default=7500)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--pool-games', type=int, default=None, help='only the first N pool logs (a smoke run)')
    p.add_argument('--fresh-games', type=int, default=None, help='only the first N held-out logs (a smoke run)')
    p.add_argument('--no-amp', action='store_true')
    a = ap.parse_args(argv)
    cmd_run(a)


if __name__ == '__main__':
    main()
