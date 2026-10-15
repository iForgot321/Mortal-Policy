"""A learned model of the opponents' concealed hands: the belief's proposal for search (D58).

Given the deciding seat's observation, it gives the concealed tile counts of each of the three
other seats (in turn order after the decider), kind by kind from 1m to C. A GRU carries the hand
built so far from one kind to the next, so neighbouring tiles (shapes) are modelled jointly. Each
kind's count is drawn from the counts still possible: no more than the copies the decider hasn't
seen and earlier seats haven't taken, no more than the tiles left to place, and enough to leave
room for the rest. The density is the product of those renormalized probabilities, so it is exact.
Red fives are then dealt uniformly among the fives a seat holds.

Trained on blueprint self-play, where the truth is known (`libriichi.search.hand_examples`).

    python -m search.handmodel train --logs DIR --val DIR --init CKPT --out PATH
"""
import argparse
import math
import os
import random
import time
from glob import glob

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset

from . import MORTAL_DIR  # noqa: F401  (puts mortal/ on the path)

KINDS = 34
MAX = 4
FIVES = (4, 13, 22)  # 5m, 5p, 5s


def bounds(cap, rem):
    """The counts possible at each kind, given the copies available (`cap`, (..., 34)) and the
    tiles left to place before each kind (`rem`, (..., 34)): at most min(cap, rem), and at least
    what the later kinds can't hold."""
    later = cap.flip(-1).cumsum(-1).flip(-1) - cap
    return (rem - later).clamp(min=0), torch.minimum(cap, rem)


class HandModel(nn.Module):
    def __init__(self, brain, hidden=256):
        super().__init__()
        self.brain = brain
        self.hidden = hidden
        self.ctx = nn.Linear(1024, 3 * hidden)
        self.kind = nn.Embedding(KINDS, 32)
        self.prev = nn.Embedding(MAX + 1, 16)
        self.seat = nn.Embedding(3, 8)
        self.gru = nn.GRU(32 + 16 + 8 + 2, hidden, batch_first=True)
        self.out = nn.Linear(hidden, MAX + 1)

    def _inputs(self, prev, rem, cap, seat):
        """Step inputs: the kind, the previous kind's count, the tiles left and the copies left."""
        b = prev.shape[0]
        kinds = torch.arange(prev.shape[1], device=prev.device).expand(b, -1)
        return torch.cat([
            self.kind(kinds),
            self.prev(prev),
            self.seat(seat).unsqueeze(1).expand(-1, prev.shape[1], -1),
            (rem.float() / 13).unsqueeze(-1),
            (cap.float() / 4).unsqueeze(-1),
        ], -1)

    def log_prob(self, obs, hands, unseen):
        """log q of the true `hands` (B, 3, 34), each seat's counts in turn order, given what the
        decider hasn't seen (`unseen`, (B, 34)). Returns (B, 3)."""
        b = obs.shape[0]
        phi = self.brain(obs)
        h0 = self.ctx(phi).view(b, 3, self.hidden).transpose(0, 1).reshape(1, 3 * b, self.hidden)
        hands = hands.long().transpose(0, 1).reshape(3 * b, KINDS)  # seat-major
        taken = torch.cat([torch.zeros_like(hands[:b]), hands[:b], hands[:b] + hands[b:2 * b]])
        cap = unseen.long().repeat(3, 1) - taken
        size = hands.sum(-1, keepdim=True)
        rem = size - hands.cumsum(-1) + hands
        prev = torch.cat([torch.zeros_like(hands[:, :1]), hands[:, :-1]], -1)
        seat = torch.arange(3, device=obs.device).repeat_interleave(b)
        out, _ = self.gru(self._inputs(prev, rem, cap, seat), h0)
        logits = self.out(out).float()
        lo, hi = bounds(cap, rem)
        counts = torch.arange(MAX + 1, device=obs.device)
        allowed = (counts >= lo.unsqueeze(-1)) & (counts <= hi.unsqueeze(-1))
        logp = logits.masked_fill(~allowed, -torch.inf).log_softmax(-1)
        picked = logp.gather(-1, hands.unsqueeze(-1)).squeeze(-1)
        return picked.sum(-1).view(3, b).T

    @torch.no_grad()
    def sample(self, obs, unseen, sizes, n, generator=None):
        """`n` draws of the three hands at one root: `obs` (1, C, 34), `unseen` (34,), `sizes`
        (3,). Returns hands (n, 3, 34) and their log q (n,)."""
        dev = obs.device
        phi = self.brain(obs).expand(n, -1)
        h = self.ctx(phi).view(n, 3, self.hidden)
        cap = unseen.long().to(dev).expand(n, -1).clone()
        hands = torch.zeros(n, 3, KINDS, dtype=torch.long, device=dev)
        logq = torch.zeros(n, device=dev)
        counts = torch.arange(MAX + 1, device=dev)
        for j in range(3):
            state = h[:, j].unsqueeze(0).contiguous()
            rem = torch.full((n,), int(sizes[j]), device=dev)
            prev = torch.zeros(n, dtype=torch.long, device=dev)
            seat = torch.full((n,), j, device=dev)
            for k in range(KINDS):
                c = cap[:, k:]
                lo, hi = bounds(c, rem.unsqueeze(-1).expand(-1, c.shape[1]))
                lo, hi = lo[:, 0], hi[:, 0]
                x = torch.cat([
                    self.kind(torch.full((n,), k, device=dev)),
                    self.prev(prev),
                    self.seat(seat),
                    (rem.float() / 13).unsqueeze(-1),
                    (cap[:, k].float() / 4).unsqueeze(-1),
                ], -1).unsqueeze(1)
                out, state = self.gru(x, state)
                logits = self.out(out[:, 0]).float()
                allowed = (counts >= lo.unsqueeze(-1)) & (counts <= hi.unsqueeze(-1))
                logp = logits.masked_fill(~allowed, -torch.inf).log_softmax(-1)
                pick = torch.multinomial(logp.exp(), 1, generator=generator).squeeze(-1)
                logq += logp.gather(-1, pick.unsqueeze(-1)).squeeze(-1)
                hands[:, j, k] = pick
                cap[:, k] -= pick
                rem -= pick
                prev = pick
        return hands, logq


def deal_akas(hands, unseen, aka_unseen, rng):
    """Red fives for sampled `hands` (n, 3, 34). An unseen red five is any one of the unseen
    copies of its five, uniformly, so a seat holding c of the u unseen fives has it with
    probability c / u. `unseen` (34,) counts the red ones in. Returns akas (n, 3, 3), log q (n,)."""
    n = hands.shape[0]
    akas = np.zeros((n, 3, 3), dtype=bool)
    logq = np.zeros(n)
    for f, kind in enumerate(FIVES):
        if not aka_unseen[f]:
            continue
        u = int(unseen[kind])
        for i in range(n):
            held = hands[i, :, kind].astype(np.float64)
            p = np.append(held, u - held.sum()) / u
            seat = rng.choice(4, p=p)
            logq[i] += math.log(p[seat])
            if seat < 3:
                akas[i, seat, f] = True
    return akas, logq


# ---- data -----------------------------------------------------------------

class HandExamples(IterableDataset):
    """Examples from self-play logs, every `stride`-th decision of each game, shuffled within a
    buffer. Observations are kept as float16 in the buffer."""

    def __init__(self, paths, stride=2, buffer=4096, epochs=1, seed=0):
        self.paths, self.stride, self.buffer, self.epochs, self.seed = paths, stride, buffer, epochs, seed

    def __iter__(self):
        from libriichi.search import hand_examples
        info = torch.utils.data.get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        rng = random.Random(self.seed * 1000 + wid)
        buf = []
        for _ in range(self.epochs):
            paths = self.paths[wid::nw]
            rng.shuffle(paths)
            for path in paths:
                obs, hands, _, unseen = hand_examples(path, stride=self.stride,
                                                      offset=rng.randrange(self.stride))
                buf += zip(obs.astype(np.float16), hands, unseen)
                while len(buf) >= self.buffer:
                    i = rng.randrange(len(buf))
                    buf[i], buf[-1] = buf[-1], buf[i]
                    yield buf.pop()
        rng.shuffle(buf)
        yield from buf


def load_brain(init, device):
    from model import Brain
    state = torch.load(init, weights_only=False, map_location='cpu')
    cfg = state['config']['resnet']
    brain = Brain(version=4, conv_channels=cfg['conv_channels'], num_blocks=cfg['num_blocks'],
                  Norm=cfg.get('norm', 'GN'))
    brain.load_state_dict(state['mortal'])
    return brain.to(device), cfg


def evaluate(model, paths, device, stride=4, limit=None):
    """Mean log q per seat of the true hands on held-out games, in nats per hand."""
    from libriichi.search import hand_examples
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for path in paths[:limit]:
            obs, hands, _, unseen = hand_examples(path, stride=stride)
            for i in range(0, len(obs), 256):
                o = torch.as_tensor(obs[i:i + 256], device=device)
                with torch.autocast(device.type, dtype=torch.float16):
                    lp = model.log_prob(o, torch.as_tensor(hands[i:i + 256], device=device),
                                        torch.as_tensor(unseen[i:i + 256], device=device))
                total += lp.sum().item()
                n += lp.numel()
    model.train()
    return -total / n


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    brain, cfg = load_brain(args.init, device)
    model = HandModel(brain).to(device)
    head = [p for name, p in model.named_parameters() if not name.startswith('brain.')]
    opt = torch.optim.AdamW([{'params': brain.parameters(), 'lr': args.lr_brain},
                             {'params': head, 'lr': args.lr_head}], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / 500) * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))
    scaler = torch.amp.GradScaler()

    train_paths = sorted(glob(os.path.join(args.logs, '*.json.gz')))
    val_paths = sorted(glob(os.path.join(args.val, '*.json.gz')))
    data = DataLoader(HandExamples(train_paths, stride=args.stride, epochs=1000, seed=args.seed),
                      batch_size=args.batch, num_workers=args.workers, persistent_workers=True,
                      prefetch_factor=2)
    print(f'{len(train_paths)} training games, {len(val_paths)} held out; brain b{cfg["num_blocks"]}'
          f'c{cfg["conv_channels"]} from {args.init}', flush=True)
    print(f'held-out nats per hand at step 0: {evaluate(model, val_paths, device, limit=args.val_games):.3f}',
          flush=True)

    t0, step, running = time.time(), 0, 0.0
    for obs, hands, unseen in data:
        obs = obs.to(device, non_blocking=True).float()
        with torch.autocast(device.type, dtype=torch.float16):
            loss = -model.log_prob(obs, hands.to(device), unseen.to(device)).mean()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1
        running += loss.item()
        if step % args.log_every == 0:
            print(f'step {step}: train nats per hand {running / args.log_every:.3f} '
                  f'({step * args.batch / (time.time() - t0):.0f} examples/s)', flush=True)
            running = 0.0
        if step % args.eval_every == 0 or step == args.steps:
            val = evaluate(model, val_paths, device, limit=args.val_games)
            print(f'step {step}: held-out nats per hand {val:.3f}', flush=True)
            torch.save({'model': model.state_dict(), 'resnet': cfg, 'step': step, 'val': val,
                        'init': args.init, 'args': vars(args)}, args.out)
        if step >= args.steps:
            break


def load(path, device='cuda:0'):
    from model import Brain
    state = torch.load(path, weights_only=False, map_location='cpu')
    cfg = state['resnet']
    brain = Brain(version=4, conv_channels=cfg['conv_channels'], num_blocks=cfg['num_blocks'],
                  Norm=cfg.get('norm', 'GN'))
    model = HandModel(brain)
    model.load_state_dict(state['model'])
    return model.to(device).eval()


def hands37(hands, akas):
    """(n, 3, 34) counts and (n, 3, 3) red fives to (n, 3, 37) counts by tile id."""
    out = np.zeros(hands.shape[:2] + (37,), dtype=np.uint8)
    out[..., :34] = hands
    for f, kind in enumerate(FIVES):
        out[..., kind] -= akas[..., f]
        out[..., 34 + f] += akas[..., f]
    return out


def beliefs(model, roots, n, seed=0, shuffle=200):
    """`libriichi.search.Beliefs` for `roots` ((events, until) each), from `n` hands sampled per
    root from what the deciding seat has seen. Also returns each draw's log q, per root."""
    from libriichi.search import Beliefs, root_views
    device = next(model.parameters()).device
    gen = torch.Generator(device=device).manual_seed(seed)
    rng = np.random.default_rng(seed)
    hands, logq = [], []
    for _, obs, unseen, sizes, aka_unseen in root_views(roots):
        unseen = np.frombuffer(bytes(unseen), dtype=np.uint8)  # pyo3 gives u8 arrays as bytes
        h, lq = model.sample(torch.as_tensor(obs, device=device).unsqueeze(0), torch.as_tensor(unseen),
                             torch.tensor(list(sizes)), n, generator=gen)
        h = h.cpu().numpy()
        akas, lq_akas = deal_akas(h, unseen, aka_unseen, rng)
        hands.append(hands37(h, akas).tolist())
        logq.append(lq.cpu().numpy() + lq_akas)
    return Beliefs.from_hands(roots, hands, seed=seed, shuffle=shuffle), logq


def calibrate(model, run_dir, n=64, limit=None, seed=0, device='cuda:0'):
    """D58's test on a pilot run's decisions (held-out games): for each root, `n` sampled hands
    per opponent; the share that are tenpai against the truth (lab-only: reads the seed). Scores
    non-riichi opponents (log loss, Brier) next to the base rate, and riichi opponents' tenpai
    share, which should be 1."""
    import json
    from libriichi.search import root_shanten, root_views, shanten
    from .pilot import read_game
    with open(os.path.join(run_dir, 'decisions.jsonl'), encoding='utf-8') as f:
        ds = [json.loads(line) for line in f][:limit]
    roots, oroots, riichi = [], [], []
    for d in ds:
        seed_, kyokus = read_game(os.path.join(run_dir, 'logs', d['game']))
        lines = kyokus[d['kyoku']]
        roots.append((lines, d['until']))
        oroots.append((seed_, lines, d['until']))
        riichi.append({json.loads(x)['actor'] for x in lines[:d['until']] if '"reach"' in x})
    truth = root_shanten(oroots)
    gen = torch.Generator(device=device).manual_seed(seed)
    probs, ys, riichi_tenpai = [], [], []
    for (seat, obs, unseen, sizes, _), r, t in zip(root_views(roots), riichi, truth):
        # pyo3 hands back the u8 arrays as bytes
        hands, _ = model.sample(torch.as_tensor(obs, device=device).unsqueeze(0),
                                torch.tensor(list(unseen)), torch.tensor(list(sizes)), n, generator=gen)
        hands = hands.cpu().numpy()
        for j in range(3):
            o = (seat + 1 + j) % 4
            tenpai = np.array(shanten([h.astype(np.uint8).tolist() for h in hands[:, j]])) <= 0
            if o in r:
                riichi_tenpai.append(tenpai.mean())
            else:
                probs.append(float(np.clip(tenpai.mean(), 0.01, 0.99)))
                ys.append(t[o] == 0)
    p, y = np.array(probs), np.array(ys)
    base = y.mean()
    return {
        'roots': len(roots), 'opponents': len(y), 'tenpai_rate': float(base),
        'predicted_rate': float(p.mean()),
        'log_loss': float(-np.mean(np.where(y, np.log(p), np.log(1 - p)))),
        'brier': float(np.mean((p - y) ** 2)),
        'base_log_loss': float(-(base * np.log(base) + (1 - base) * np.log(1 - base))),
        'base_brier': float(base * (1 - base)),
        'riichi_opponents': len(riichi_tenpai),
        'riichi_tenpai_share': float(np.mean(riichi_tenpai)) if riichi_tenpai else None,
    }


def main():
    p = argparse.ArgumentParser(prog='python -m search.handmodel', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('train')
    t.add_argument('--logs', required=True, help='self-play logs to train on')
    t.add_argument('--val', required=True, help='held-out self-play logs')
    t.add_argument('--init', required=True, help='the blueprint checkpoint whose encoder starts the model')
    t.add_argument('--out', required=True)
    t.add_argument('--steps', type=int, default=20_000)
    t.add_argument('--batch', type=int, default=256)
    t.add_argument('--stride', type=int, default=2)
    t.add_argument('--lr-brain', type=float, default=1e-4)
    t.add_argument('--lr-head', type=float, default=1e-3)
    t.add_argument('--workers', type=int, default=4)
    t.add_argument('--log-every', type=int, default=200)
    t.add_argument('--eval-every', type=int, default=2000)
    t.add_argument('--val-games', type=int, default=20)
    t.add_argument('--seed', type=int, default=0)
    t.add_argument('--device', default='cuda:0')
    c = sub.add_parser('calibrate', help="D58's test: tenpai calibration on a pilot run's decisions")
    c.add_argument('--model', required=True)
    c.add_argument('--run', required=True, help='a search run directory with decisions.jsonl')
    c.add_argument('--n', type=int, default=64)
    c.add_argument('--limit', type=int)
    c.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    if args.cmd == 'train':
        train(args)
    elif args.cmd == 'calibrate':
        import json
        print(json.dumps(calibrate(load(args.model, args.device), args.run, args.n, args.limit,
                                   device=args.device), indent=2))


if __name__ == '__main__':
    main()
