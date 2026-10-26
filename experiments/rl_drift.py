"""How far, and where, an online run's checkpoints moved from their initialization (05 §8:
"KL too low / LR too high"). On the trainee's decisions in kept 1v3 logs (`rl_credit_check.py
--work DIR`), per checkpoint: KL(π_init ‖ π_ckpt), argmax agreement with the initialization,
and both by the kind of decision the initialization's argmax picks.

    python experiments/rl_drift.py --logs workdir/diag/credit-b --init INIT CKPT [CKPT ...]
"""
import argparse
import os
import sys
import tomllib

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'mortal'))
sys.path.insert(0, ROOT)


def kind(a):
    return ('discard' if a <= 36 else 'riichi' if a == 37 else 'chi' if a <= 40 else 'pon' if a == 41
            else 'kan' if a == 42 else 'agari' if a == 43 else 'ryukyoku' if a == 44 else 'pass')


def load_states(logs, config='experiments/configs/p1.toml', stride=2, version=4):
    """(obs fp16, masks) of the trainee's decisions with more than one legal action in a
    `rl_credit_check.py --work` dir, every `stride`-th decision, as obs `version`."""
    import torch
    from rl_dataset import OrderedDatasetsIter
    with open(config, 'rb') as f:
        cfg = tomllib.load(f)
    log_dir = os.path.join(logs, 'logs')
    files = sorted(os.path.join(log_dir, f) for f in os.listdir(log_dir))
    ds = OrderedDatasetsIter(version=version, file_list=files, behaviour_dir=os.path.join(logs, 'logs_behaviour'),
                             pts=cfg['env']['pts'], grp_state_file=cfg['grp']['state_file'],
                             grp_network=cfg['grp']['network'])
    ds.prepare()
    # fp16 and every `stride`-th decision: 137 KB per fp32 observation adds up (15 GB box)
    obs, masks, n = [], [], 0
    for t in ds.trajectories(files):
        for o, m in zip(t['obs'], t['mask']):
            if n % stride == 0:
                obs.append(o.astype(np.float16))
                masks.append(m)
            n += 1
    obs = torch.from_numpy(np.stack(obs))
    masks = torch.as_tensor(np.stack(masks)).bool()
    multi = masks.sum(-1) > 1
    return obs[multi], masks[multi], len(files)


def measure(logs, init, ckpts, config='experiments/configs/p1.toml', stride=2):
    """Per checkpoint: KL both ways against `init`, argmax agreement, entropy, and both by the
    kind of decision the initialization's argmax picks. Returns (init entropy, [dict])."""
    import torch
    from lab.arena import load_engine
    # the states are featurized as the initialization's observation version (v5 from G0 on, D61)
    version = load_engine(init, enable_amp=False, name='x').version
    obs, masks, _ = load_states(logs, config, stride, version)

    def logp(spec):
        e = load_engine(spec, enable_amp=False, name='x')
        if e.version != version:
            raise ValueError(f'{spec} is obs v{e.version}, the initialization v{version}')
        out = []
        with torch.inference_mode():
            for s in range(0, len(obs), 2048):
                o = obs[s:s + 2048].to(e.device, torch.float32)
                m = masks[s:s + 2048].to(e.device)
                out.append(torch.log_softmax(e.dqn.logits(e.brain(o), m).float(), -1).cpu())
        return torch.cat(out)

    base = logp(init)
    p0 = base.exp()
    a0 = base.argmax(-1).numpy()
    kinds = np.array([kind(a) for a in a0])
    h0 = float(-(p0 * torch.where(masks, base, 0.0)).sum(-1).mean())
    out = []
    for spec in ckpts:
        lp = logp(spec)
        kl = (p0 * torch.where(masks, base - lp, 0.0)).sum(-1).numpy()
        # the plan's anchor measure (05 §5.1): reverse KL(π_ckpt ‖ π_init) over the legal mask
        rkl = (lp.exp() * torch.where(masks, lp - base, 0.0)).sum(-1).numpy()
        a1 = lp.argmax(-1).numpy()
        out.append({
            'ckpt': spec, 'decisions': int(len(obs)),
            'kl': float(kl.mean()), 'kl_median': float(np.median(kl)),
            'rkl': float(rkl.mean()), 'rkl_median': float(np.median(rkl)),
            'agreement': float(np.mean(a0 == a1)),
            'entropy': float(-(lp.exp() * torch.where(masks, lp, 0.0)).sum(-1).mean()),
            'by_kind': {k: {'n': int((kinds == k).sum()), 'kl': float(kl[kinds == k].mean()),
                            'agreement': float(np.mean(a0[kinds == k] == a1[kinds == k]))}
                        for k in ('discard', 'riichi', 'chi', 'pon', 'kan', 'agari', 'pass') if (kinds == k).any()},
        })
    return h0, out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--logs', required=True, help='a rl_credit_check.py work dir')
    p.add_argument('--init', required=True)
    p.add_argument('--config', default='experiments/configs/p1.toml')
    p.add_argument('--stride', type=int, default=2)
    p.add_argument('ckpts', nargs='+')
    args = p.parse_args()
    h0, rows = measure(args.logs, args.init, args.ckpts, args.config, args.stride)
    print(f'{rows[0]["decisions"]} decisions with more than one legal action')
    print(f'init: entropy {h0:.4f}')
    for r in rows:
        print(f'{os.path.relpath(r["ckpt"])}: KL(init‖ckpt) mean {r["kl"]:.4f} (median {r["kl_median"]:.4f}), '
              f'KL(ckpt‖init) mean {r["rkl"]:.4f} (median {r["rkl_median"]:.4f}), '
              f'argmax agreement {r["agreement"]:.4f}, entropy {r["entropy"]:.4f}')
        for k, v in r['by_kind'].items():
            print(f'   init picks {k:8s} n {v["n"]:>6}  KL {v["kl"]:.4f}  agreement {v["agreement"]:.4f}')


if __name__ == '__main__':
    main()
