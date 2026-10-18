"""Diagnosis of the online arms (05 §2.3.6, §8): does RL improve the policy it samples from
while its argmax gets worse?

Every online run's trainee gains in its own training games (sampled actions against three
argmax copies of its initialization), but its final checkpoint loses to that initialization at
argmax in the lab. This plays one challenger against the initialization on fixed deals:

    python experiments/rl_diag_sampling.py play --challenger CKPT --mode sample [--temp T] \
        --champion INIT --sets N --out workdir/diag/NAME.npz
    python experiments/rl_diag_sampling.py calibrate --challenger CKPT --probe INIT \
        --champion INIT --sets N      # entropies of INIT at several temperatures on CKPT's states
    python experiments/rl_diag_sampling.py compare workdir/diag/A.npz workdir/diag/B.npz

`--mode sample` draws every action from the policy (explore_rate 1, no agari guard), as the
trainee does in training games (mortal/player.py). `--mode argmax` plays as the lab does (agari
guard on). The champion is always the frozen initialization: greedy, agari guard on, AMP, as in
training and in the lab. A temperature T samples from softmax(logits / T).

Deals come from a diagnostic key outside every lab family, so these runs leave the nonce
ledger alone. They are diagnostics, not verdicts.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'mortal'))

DIAG_KEY = 0xd1a6_5a3b_1e0f_c0de
UTILITY = np.array([0.9, 0.5, -0.1, -1.3])


def tempered(policy, temp):
    """`policy` (CategoricalPolicy) with its logits divided by `temp`."""
    import torch

    class Tempered(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = policy

        def logits(self, phi, mask):
            return self.inner.logits(phi, mask) / temp

        def forward(self, phi, mask):
            return torch.softmax(self.logits(phi, mask), dim=-1)

    return Tempered()


def entropy(probs):
    p = np.asarray(probs, dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        h = -np.where(p > 0, p * np.log(p), 0.0).sum(-1)
    return h


class Recorder:
    """Wraps an engine: the mean entropy of the distribution each decision samples from, and of
    each probe (another network at a temperature) on the same observations."""

    def __init__(self, inner, probes=()):
        self.inner = inner
        self.probes = probes  # (label, brain, policy) on inner's device
        self.engine_type = inner.engine_type
        self.name = inner.name
        self.version = inner.version
        self.is_oracle = inner.is_oracle
        self.enable_quick_eval = inner.enable_quick_eval
        self.enable_rule_based_agari_guard = inner.enable_rule_based_agari_guard
        self.n = 0
        self.multi = 0
        self.h = 0.0
        self.probe_h = {label: 0.0 for label, *_ in probes}

    def react_batch(self, obs, masks, invisible_obs):
        out = self.inner.react_batch(obs, masks, invisible_obs)
        probs = np.asarray(out[1], dtype=np.float64)
        legal = np.asarray(masks).sum(-1)
        multi = legal > 1  # entropy is 0 where only one action is legal
        self.n += len(probs)
        self.multi += int(multi.sum())
        self.h += float(entropy(probs).sum())
        if self.probes:
            import torch
            dev = self.inner.device
            o = torch.as_tensor(np.stack(obs), device=dev)
            m = torch.as_tensor(np.stack(masks), device=dev)
            with torch.autocast(dev.type, enabled=self.inner.enable_amp), torch.inference_mode():
                for label, brain, policy in self.probes:
                    p = policy(brain(o), m).float().cpu().numpy()
                    self.probe_h[label] += float(entropy(p).sum())
        return out

    def stats(self):
        n = max(self.n, 1)
        return {'decisions': self.n, 'multi_choice_frac': self.multi / n, 'entropy': self.h / n,
                'probe_entropy': {k: v / n for k, v in self.probe_h.items()}}


def engines(challenger_spec, champion_spec, mode, temp, probe_specs=(), probe_temps=()):
    import torch
    from lab.arena import load_engine
    challenger = load_engine(challenger_spec, enable_amp=True,
                             enable_rule_based_agari_guard=mode == 'argmax', name='challenger')
    if mode == 'sample':
        challenger.explore_rate = 1.0
    if temp != 1.0:
        if mode != 'sample':
            raise SystemExit('--temp needs --mode sample')
        challenger.dqn = tempered(challenger.dqn, temp).to(challenger.device).eval()
    champion = load_engine(champion_spec, enable_amp=True, enable_rule_based_agari_guard=True,
                           name='champion')
    probes = []
    for spec in probe_specs:
        probe = load_engine(spec, enable_amp=True, name='probe')
        for t in probe_temps:
            pol = probe.dqn if t == 1.0 else tempered(probe.dqn, t).to(probe.device).eval()
            probes.append((f'{os.path.basename(spec)}@{t:g}', probe.brain, pol))
    torch.backends.cudnn.benchmark = False
    return Recorder(challenger, probes), champion


def play(challenger, champion, mode='sample', temp=1.0, sets=4000, nonce_start=0, chunk=500, out=None,
         probe_specs=(), probe_temps=(), progress=True):
    """Play `sets` duplicate sets on the diagnostic key and return the summary (also written next
    to `out`, with the per-set arrays in `out`, when given)."""
    from lab.arena import play_sets
    rec, champ = engines(challenger, champion, mode, temp, probe_specs, probe_temps)
    t0 = time.time()
    parts = []
    for start in range(nonce_start, nonce_start + sets, chunk):
        n = min(chunk, nonce_start + sets - start)
        parts.append(play_sets(rec, champ, DIAG_KEY, start, n))
        if progress:
            print(f'{start + n - nonce_start}/{sets} sets, {time.time() - t0:.0f} s', flush=True)
    seeds, seats, ranks, scores = (np.concatenate(c) for c in zip(*parts))
    u = UTILITY[ranks]
    order = np.argsort(seeds, kind='stable')
    seeds, seats, ranks, u = seeds[order], seats[order], ranks[order], u[order]
    per_set = u.reshape(-1, 4).mean(1)  # 4 games per set: the challenger in each seat
    summary = {
        'challenger': os.path.abspath(challenger), 'champion': os.path.abspath(champion),
        'mode': mode, 'temp': temp, 'key': hex(DIAG_KEY), 'nonce_start': nonce_start,
        'sets': int(len(per_set)), 'du': float(per_set.mean()),
        'se': float(per_set.std(ddof=1) / np.sqrt(len(per_set))),
        'avg_rank': float((ranks + 1).mean()), 'rank_hist': np.bincount(ranks, minlength=4).tolist(),
        'seconds': round(time.time() - t0), **rec.stats(),
    }
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        np.savez_compressed(out, seeds=seeds, seats=seats, ranks=ranks, per_set=per_set)
        with open(out.replace('.npz', '.json'), 'w') as f:
            json.dump(summary, f, indent=1)
    return summary


def cmd_play(args):
    summary = play(args.challenger, args.champion, args.mode, args.temp, args.sets, args.nonce_start, args.chunk,
                   args.out, getattr(args, 'probe', None) or (), getattr(args, 'probe_temps', ()))
    print(json.dumps(summary, indent=1))


def cmd_compare(args):
    a, b = (np.load(p) for p in (args.a, args.b))
    if not np.array_equal(a['seeds'], b['seeds']):
        raise SystemExit('different deals')
    d = b['per_set'] - a['per_set']
    print(json.dumps({'b_minus_a': float(d.mean()), 'se': float(d.std(ddof=1) / np.sqrt(len(d))),
                      'sets': int(len(d)), 'corr': float(np.corrcoef(a['per_set'], b['per_set'])[0, 1])},
                     indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    for name in ('play', 'calibrate'):
        s = sub.add_parser(name)
        s.add_argument('--challenger', required=True)
        s.add_argument('--champion', required=True)
        s.add_argument('--mode', choices=('argmax', 'sample'), default='sample')
        s.add_argument('--temp', type=float, default=1.0)
        s.add_argument('--sets', type=int, required=True)
        s.add_argument('--nonce-start', type=int, default=0)
        s.add_argument('--chunk', type=int, default=500, help='sets per arena call, as the lab')
        s.add_argument('--out')
        if name == 'calibrate':
            s.add_argument('--probe', action='append', required=True)
            s.add_argument('--probe-temps', type=float, nargs='+',
                           default=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
    s = sub.add_parser('compare')
    s.add_argument('a')
    s.add_argument('b')
    args = p.parse_args()
    if args.cmd == 'compare':
        cmd_compare(args)
    else:
        cmd_play(args)


if __name__ == '__main__':
    main()
