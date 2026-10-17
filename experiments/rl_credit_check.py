"""Interface check for the online arms (05 §8's first triage item): do the advantages the trainer
sees sit on the right kyoku and decisions?

Plays fresh 1v3 games as the actors do (`mortal/player.py`: the trainee samples every action,
against three greedy copies of the initialization, with behaviour sidecars), then runs them
through the trainer's own data path (`rl_dataset.OrderedDatasetsIter.trajectories`) and checks
them against the mjai events of each log:

1. each kyoku's GRP reward against the trainee's point change in that kyoku, at offsets −1, 0
   and +1 kyoku (a misaligned reward correlates best at an offset);
2. every win (action 43) sits in a kyoku the trainee won, and the trainee's last decision in each
   kyoku where it dealt in is a discard;
3. P0's advantage (horizon K, Monte-Carlo) and P1's (horizon H, GAE with the run's critic) by
   kyoku outcome, on the winning decision and on the discard that dealt in.

    python experiments/rl_credit_check.py --trainee CKPT --init INIT --sets 50 \
        --critic workdir/fleet/runs/p1-b/snapshots/full.pth
"""
import argparse
import gzip
import json
import os
import sys
import tempfile
import tomllib

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'mortal'))
sys.path.insert(0, ROOT)

DIAG_KEY = 0xd1a6_5a3b_1e0f_c0de  # the sampling diagnostic's key, at other nonces


def play(trainee, init, sets, nonce, log_dir, behaviour_dir):
    import torch
    from lab.arena import load_engine
    from libriichi.arena import OneVsThree
    chal = load_engine(trainee, enable_amp=True, enable_rule_based_agari_guard=False, name='trainee')
    chal.explore_rate = 1.0
    champ = load_engine(init, enable_amp=True, enable_rule_based_agari_guard=True, name='baseline')
    torch.backends.cudnn.benchmark = False
    OneVsThree(disable_progress_bar=True, log_dir=log_dir, behaviour_dir=behaviour_dir).py_vs_py(
        challenger=chal, champion=champ, seed_start=(nonce, DIAG_KEY), seed_count=sets)


def kyoku_truth(log, seat):
    """Per kyoku: the seat's point change, and whether it won, dealt in, or neither."""
    out = []
    with gzip.open(log, 'rt') as f:
        for line in f:
            ev = json.loads(line)
            t = ev['type']
            if t == 'start_kyoku':
                out.append({'delta': 0, 'won': False, 'dealt_in': False})
            elif t in ('hora', 'ryukyoku') and out:
                k = out[-1]
                k['delta'] += (ev.get('deltas') or [0] * 4)[seat]
                if t == 'hora':
                    if ev['actor'] == seat:
                        k['won'] = True
                    elif ev['target'] == seat and ev['actor'] != seat:
                        k['dealt_in'] = True
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--trainee', required=True)
    p.add_argument('--init', required=True)
    p.add_argument('--critic', help='a P1 checkpoint whose critic computes the GAE advantages')
    p.add_argument('--config', default='experiments/configs/p1.toml')
    p.add_argument('--sets', type=int, default=50)
    p.add_argument('--nonce-start', type=int, default=500_000)
    p.add_argument('--work', help='keep the logs here (default: a temporary directory)')
    args = p.parse_args()

    import torch
    import trajectory as tj
    from rl_dataset import OrderedDatasetsIter

    with open(args.config, 'rb') as f:
        cfg = tomllib.load(f)
    work = args.work or tempfile.mkdtemp(prefix='credit-')
    log_dir, bdir = os.path.join(work, 'logs'), os.path.join(work, 'logs_behaviour')
    if not os.path.isdir(log_dir):
        play(args.trainee, args.init, args.sets, args.nonce_start, log_dir, bdir)
    files = sorted(os.path.join(log_dir, f) for f in os.listdir(log_dir))

    def dataset(horizon, targets):
        ds = OrderedDatasetsIter(version=4, file_list=files, behaviour_dir=bdir, pts=cfg['env']['pts'],
                                 grp_state_file=cfg['grp']['state_file'], grp_network=cfg['grp']['network'],
                                 horizon=horizon, targets=targets)
        ds.prepare()
        return ds

    critic = None
    if args.critic:
        from critic import Critic, gae_targets
        state = torch.load(args.critic, weights_only=False, map_location='cpu')
        ccfg = state['config']['critic']
        critic = Critic(conv_channels=ccfg['conv_channels'], num_blocks=ccfg['num_blocks'], version=4)
        critic.load_state_dict(state['critic'])
        critic = critic.to('cuda').eval()

    mc = dataset('K', 'mc')
    gae_ds = dataset('H', 'gae') if critic is not None else None
    corr = {-1: [], 0: [], 1: []}
    rows = []  # per decision: outcome class, is-win action, is-deal-in discard, P0 adv, P1 adv
    problems = {'win_outside_won_kyoku': 0, 'deal_in_last_not_discard': 0, 'wins': 0, 'deal_ins': 0}
    for f in files:
        trajs = list(mc.trajectories([f]))
        gtrajs = list(gae_ds.trajectories([f])) if gae_ds else [None] * len(trajs)
        for traj, gtraj in zip(trajs, gtrajs):
            seat = int(traj['player_id'])
            truth = kyoku_truth(f, seat)
            kr = np.asarray(traj['kyoku_rewards'])
            delta = np.array([k['delta'] for k in truth], dtype=float)
            if len(kr) != len(delta):
                print(f'{f}: {len(kr)} kyoku rewards, {len(delta)} kyoku in the log')
                continue
            for off in corr:
                a = kr[max(0, off):len(kr) + min(0, off)]
                b = delta[max(0, -off):len(delta) - max(0, off)]
                corr[off] += list(zip(a, b))
            at = np.asarray(traj['at_kyoku'], dtype=np.int64)
            act = np.asarray(traj['action'])
            adv_k = tj.monte_carlo_returns(traj['reward'], traj['done'])
            if gtraj is not None:
                chunk = [{k: torch.as_tensor(v) for k, v in gae_ds.compact(gtraj).items()}]
                adv_h = np.asarray(gae_targets(chunk, critic, device=torch.device('cuda'), enable_amp=True)['advantage'], dtype=float)
            else:
                adv_h = np.full(len(act), np.nan)
            last = np.append(at[1:] != at[:-1], True)
            for i in range(len(act)):
                k = truth[at[i]]
                win = act[i] == 43
                deal_in = bool(last[i] and k['dealt_in'])
                if win:
                    problems['wins'] += 1
                    problems['win_outside_won_kyoku'] += not k['won']
                if deal_in:
                    problems['deal_ins'] += 1
                    problems['deal_in_last_not_discard'] += act[i] > 37
                cls = 'won' if k['won'] else 'dealt_in' if k['dealt_in'] else 'other'
                rows.append((cls, win, deal_in, adv_k[i], adv_h[i]))

    print(f'logs {len(files)}, decisions {len(rows)}; work dir {work}')
    for off, pairs in corr.items():
        a, b = np.array(pairs).T
        print(f'kyoku reward vs point change, offset {off:+d}: r = {np.corrcoef(a, b)[0, 1]:+.3f} (n {len(a)})')
    print(json.dumps({k: int(v) for k, v in problems.items()}))
    cls = np.array([r[0] for r in rows])
    win = np.array([r[1] for r in rows])
    din = np.array([r[2] for r in rows])
    for name, col in (('P0 (K, MC)', 3), ('P1 (H, GAE)', 4)):
        adv = np.array([r[col] for r in rows], dtype=float)
        if np.isnan(adv).all():
            continue
        sd = adv.std()
        print(f'{name}: mean {adv.mean():+.4f}, sd {sd:.4f}')
        for c in ('won', 'dealt_in', 'other'):
            s = cls == c
            print(f'   kyoku {c:9s}: n {s.sum():>7}  mean adv {adv[s].mean():+.4f}  ({adv[s].mean() / sd:+.2f} sd)')
        for label, s in (('the winning action', win), ('the discard that dealt in', din)):
            print(f'   {label:26s}: n {s.sum():>7}  mean adv {adv[s].mean():+.4f}  ({adv[s].mean() / sd:+.2f} sd)'
                  f'  frac > 0 {np.mean(adv[s] > 0):.2f}')


if __name__ == '__main__':
    main()
