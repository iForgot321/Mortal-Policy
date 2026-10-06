"""Evaluation-lab CLI (02 §11). Run from the repo root:

  python -m lab aa --model X.pth [--sets 2000]
      Deterministic A/A harness check (02 §6): X vs X must give identical
      games in every rotation, so every set scores exactly 0.
  python -m lab run --challenger X --champion Y --test T-std [--family B]
      Sequential GSPRT verdict; writes a registry row.
  python -m lab measure --challenger X --champion Y --sets 20000 [--family A]
      Fixed-N measurement with no pass rule (A/A', tracking, anchors).
  python -m lab confirm --challenger X --champion Y --sets 50000 --kind superiority
      Fixed-N T-confirm on fresh family-C nonces (D36).
  python -m lab calibrate [--from RUN.npz] [--test T-std]
      §4.1 simulation of the sequential procedure.
  python -m lab table
      Re-render lab/registry/ablation.md.

Engines are checkpoint paths or `builtin:shanten-greedy`.
"""
import argparse
import json
import os
import sys

import numpy as np

from . import RUNS_DIR, REGISTRY_DIR, load_config, utility
from . import gsprt, registry


def _engines(args, cfg):
    from .arena import load_engine
    a = cfg['arena']
    enable_amp = a['enable_amp'] and not args.no_amp
    if not enable_amp:
        # true FP32: Ampere runs cuDNN convs in TF32 (10-bit mantissa) by default
        import torch
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    common = dict(device=args.device, enable_amp=enable_amp,
                  enable_rule_based_agari_guard=a['enable_rule_based_agari_guard'])
    chal = load_engine(args.challenger, name='challenger', **common)
    champ = load_engine(args.champion, name='champion', **common)
    return chal, champ


def _engine_record(spec):
    if spec.startswith('builtin:'):
        return {'spec': spec}
    path, sep, query = spec.partition('?')
    return {'spec': os.path.abspath(path) + sep + query, 'sha256': registry.sha256_file(path)}


def _play(args, cfg, run_id, family, nonce_start, key, n_sets, test=None):
    """Play up to n_sets sets in chunks, feeding a SequentialTest if given.
    Returns (GameLog, set_scores, aborted)."""
    from .arena import GameLog, play_sets

    util = utility(cfg)
    chal, champ = _engines(args, cfg)
    run_dir = os.path.join(RUNS_DIR, run_id)
    log_dir = None if args.no_logs else os.path.join(run_dir, 'logs')
    os.makedirs(run_dir, exist_ok=True)

    games = GameLog()
    scores = []
    chunk = int(cfg['arena']['chunk_sets'])
    aborted = False
    try:
        while len(scores) < n_sets:
            want = min(chunk, n_sets - len(scores))
            if test is not None:
                want = min(want, max(test.sets_until_next_look(), 1))
            start = nonce_start + len(scores)
            seeds, seats, ranks, finals = play_sets(chal, champ, key, start, want, log_dir)
            games.add(seeds, seats, ranks, finals)
            _, x = gsprt.set_scores(seeds, seats, ranks, util)
            scores.extend(x.tolist())
            line = f'{len(scores):>7} sets  mean {np.mean(scores):+.5f}'
            if test is not None:
                status = test.add(x)
                line += f'  LLR {test.llr:+.3f}  [{test.spec.lower}, {test.spec.upper}]'
                if status != gsprt.CONTINUE:
                    print(line, flush=True)
                    break
            print(line, flush=True)
    except KeyboardInterrupt:
        aborted = True
        print('interrupted; recording partial run', flush=True)
    games.save(os.path.join(run_dir, 'games.npz'), scores)
    return games, scores, aborted


def _base_row(args, cfg, run_id, test_name, family, nonce_start, n_sets):
    return {
        'run_id': run_id,
        'time': registry.now(),
        'mode': getattr(args, 'mode', 'exploratory'),
        'description': args.description,
        'test': test_name,
        'family': family,
        'nonce_start': nonce_start,
        'nonce_count': n_sets,
        'challenger': _engine_record(args.challenger),
        'champion': _engine_record(args.champion),
        'utility': cfg['utility'],
        'arm': getattr(args, 'arm', None),
        'package': getattr(args, 'package', None),
        'horizon': getattr(args, 'horizon', None),
        'amp': bool(cfg['arena']['enable_amp'] and not args.no_amp),
        'calibrated': bool(cfg['gsprt']['calibrated']),
        'lab_config': registry.lab_config_hash(),
        'git': registry.git_state(),
        'run_dir': os.path.join(RUNS_DIR, run_id),
    }


def cmd_run(args):
    cfg = load_config()
    spec = gsprt.specs_from_config(cfg)[args.test]
    sched = gsprt.schedule_from_config(cfg, args.sigma_set)
    run_id = registry.new_run_id(args.test)
    start, key = registry.allocate_nonces(args.family, spec.cap, run_id, f'{args.test} verdict')
    print(f'{run_id}: {args.test} on family {args.family} nonces [{start}, +{spec.cap})'
          f'{"" if cfg["gsprt"]["calibrated"] else "  (UNCALIBRATED boundaries: provisional)"}', flush=True)

    test = gsprt.SequentialTest(spec, sched)
    games, _, aborted = _play(args, cfg, run_id, args.family, start, key, spec.cap, test)
    row = _base_row(args, cfg, run_id, args.test, args.family, start, spec.cap)
    summary = test.summary()
    row['result'] = summary
    row['verdict'] = 'aborted' if aborted else summary['status']
    row['secondary'] = _secondary(cfg, games)
    registry.record_run(row)
    print(json.dumps(row['result'], indent=1))
    print('verdict:', row['verdict'])


def _secondary(cfg, games):
    from .arena import secondary_metrics
    _, _, ranks, _ = games.arrays()
    return secondary_metrics(ranks, cfg['display_pts'])


def _fixed(args, test_name, family, purpose, block=None):
    cfg = load_config()
    run_id = registry.new_run_id(test_name)
    start, key = registry.allocate_nonces(family, args.sets, run_id, purpose, block=block)
    print(f'{run_id}: {test_name} on family {family} nonces [{start}, +{args.sets})', flush=True)
    games, scores, aborted = _play(args, cfg, run_id, family, start, key, args.sets)
    row = _base_row(args, cfg, run_id, test_name, family, start, args.sets)
    row['secondary'] = _secondary(cfg, games)
    return cfg, row, games, np.asarray(scores), aborted


def cmd_measure(args):
    _, row, _, x, aborted = _fixed(args, 'measure', args.family, args.purpose, block=args.block)
    lo, hi = gsprt.bootstrap_mean_ci(x) if len(x) > 1 else (None, None)
    row['result'] = {'sets': len(x), 'mean': float(x.mean()), 'sd': float(x.std(ddof=1)),
                     'ci': [lo, hi], 'sigma_set': float(x.std(ddof=1))}
    row['verdict'] = 'aborted' if aborted else 'measured'
    if not args.no_logs:
        row['panel'] = _panel(row['run_dir'])
    registry.record_run(row)
    print(json.dumps(row['result'], indent=1))


def _panel(run_dir):
    """Style panel and per-seat table (02 §6–7) from a run's logs; saved as panel.json."""
    from . import panel
    overall, seats = panel.panel(os.path.join(run_dir, 'logs'))
    out = {'all': overall, 'per_seat': seats}
    with open(os.path.join(run_dir, 'panel.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(panel.format_table(out['all'], out['per_seat']))
    return out


def cmd_panel(args):
    run_dir = args.run if os.path.isdir(args.run) else os.path.join(RUNS_DIR, args.run)
    _panel(run_dir)


def cmd_confirm(args):
    if args.kind == 'non-inferiority' and args.delta <= 0:
        sys.exit('non-inferiority needs --delta > 0')
    cfg, row, _, x, aborted = _fixed(args, 'T-confirm', 'C', f'T-confirm {args.kind}')
    c = cfg['confirm']
    res = gsprt.confirm(x, args.kind, args.delta, c['level'], int(c['bootstrap_resamples']))
    row['result'] = res
    row['mode'] = 'confirmed' if res['passed'] and not aborted else 'exploratory'
    row['verdict'] = 'aborted' if aborted else ('pass' if res['passed'] else 'fail')
    registry.record_run(row)
    print(json.dumps(res, indent=1))
    print('verdict:', row['verdict'])


def cmd_aa(args):
    """Deterministic A/A (02 §6): identical models and argmax play make every
    rotation of a seed the same game. Any deviation is a harness bug."""
    args.challenger = args.champion = args.model
    args.description = args.description or 'deterministic A/A harness check'
    _, row, games, x, aborted = _fixed(args, 'A/A', 'A', 'deterministic A/A', block=args.block)
    seeds, seats, ranks, finals = games.arrays()
    per_set = finals.reshape(-1, 4, 4)
    identical = (per_set == per_set[:, :1]).all(axis=(1, 2))
    row['result'] = {
        'sets': len(x),
        'mean': float(x.mean()),
        'sd': float(x.std(ddof=1)) if len(x) > 1 else 0.0,
        'max_abs_set_score': float(np.abs(x).max()),
        'sets_with_divergent_games': int((~identical).sum()),
    }
    ok = not aborted and row['result']['sets_with_divergent_games'] == 0 \
        and row['result']['max_abs_set_score'] == 0.0
    row['verdict'] = 'aborted' if aborted else ('pass' if ok else 'FAIL')
    registry.record_run(row)
    print(json.dumps(row['result'], indent=1))
    print('deterministic A/A:', row['verdict'])
    if not ok:
        bad = np.flatnonzero(~identical)[:10]
        print('first divergent sets (nonce):', seeds.reshape(-1, 4)[bad, 0].tolist())
        sys.exit(1)


def cmd_calibrate(args):
    from . import calibrate as cal
    cfg = load_config()
    specs = gsprt.specs_from_config(cfg)
    if args.source:
        x = np.load(args.source)['set_scores']
        base = cal.empirical(x, name=os.path.basename(os.path.dirname(args.source)) or 'empirical')
        sigma = args.sigma_set if args.sigma_set is not None else float(np.std(x, ddof=1))
    else:
        base = cal.independent_ranks(utility(cfg))
        sigma = args.sigma_set if args.sigma_set is not None else cfg['gsprt']['sigma_set']
    sched = gsprt.schedule_from_config(cfg, sigma)
    tests = args.test or list(specs)
    report = {'time': registry.now(), 'source': args.source or 'synthetic independent ranks',
              'n_tests': args.n_tests, 'results': []}
    for name in tests:
        res = cal.calibrate(base, specs[name], sched, args.n_tests, args.seed)
        report['results'].append(res)
        print(f'\n{name}: base {res["base"]["name"]} mean {res["base"]["mean"]:+.4f} '
              f'sd {res["base"]["sd"]:.4f}, sigma_set {sigma:.4f}')
        print(f'  {"variant":<12} {"false pass":>10} {"false fail":>10} {"E[N] H0":>9} {"E[N] H1":>9}')
        for r in res['nominal']['rows']:
            print(f'  {r["variant"]:<12} {r["false_pass"]:>10.4f} {r["false_fail"]:>10.4f} '
                  f'{r["mean_sets_h0"]:>9.0f} {r["mean_sets_h1"]:>9.0f}')
        sel = res['selected']
        print(f'  nominal ±2.944 accepted: {res["nominal"]["accepted"]}; selected: {sel and (sel["lower"], sel["upper"])}')
    out = args.out or os.path.join(REGISTRY_DIR, f'calibration-{registry.new_run_id("cal")}.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(report, f, indent=1)
    print('\nreport:', out)


def main():
    ap = argparse.ArgumentParser(prog='python -m lab')
    sub = ap.add_subparsers(dest='cmd', required=True)

    def engines(p):
        p.add_argument('--challenger', required=True)
        p.add_argument('--champion', required=True)

    def common(p):
        p.add_argument('--device', default='cuda:0')
        p.add_argument('--description', default='')
        p.add_argument('--no-logs', action='store_true', help='skip mjai logs (not for verdicts)')
        p.add_argument('--no-amp', action='store_true',
                       help='FP32 inference (no TF32) with deterministic cuDNN (used by the deterministic A/A: '
                            'AMP numerics depend on batch composition and flip near-tie argmaxes)')
        # 02 §9: what produced the challenger, for RL checkpoints
        p.add_argument('--arm', default=None, choices=['P0', 'P1', 'V', 'H'], help='engine arm (05 §2.1)')
        p.add_argument('--package', default=None, choices=['K1', 'K2', 'K3'], help='discovery package (05 §5.1)')
        p.add_argument('--horizon', default=None, choices=['K', 'H'], help='return horizon (05 §3.1)')

    p = sub.add_parser('run')
    engines(p)
    common(p)
    p.add_argument('--test', required=True, choices=['T-smoke', 'T-NI', 'T-std', 'T-fine'])
    p.add_argument('--family', default='B', choices=['A', 'B'])
    p.add_argument('--mode', default='exploratory', choices=['exploratory'])
    p.add_argument('--sigma-set', type=float, default=None, help='override lab.toml sigma_set')
    p.set_defaults(func=cmd_run)

    p = sub.add_parser('measure')
    engines(p)
    common(p)
    p.add_argument('--sets', type=int, required=True)
    p.add_argument('--family', default='A', choices=['A', 'B'])
    p.add_argument('--block', default=None, help='named reusable family-A nonce block')
    p.add_argument('--purpose', default='fixed-N measurement')
    p.set_defaults(func=cmd_measure)

    p = sub.add_parser('confirm')
    engines(p)
    common(p)
    p.add_argument('--sets', type=int, required=True)
    p.add_argument('--kind', required=True, choices=['superiority', 'non-inferiority'])
    p.add_argument('--delta', type=float, default=0.0)
    p.set_defaults(func=cmd_confirm)

    p = sub.add_parser('aa')
    p.add_argument('--model', required=True)
    common(p)
    p.add_argument('--sets', type=int, default=2000)
    p.add_argument('--block', default='aa-harness')
    p.set_defaults(func=cmd_aa)

    p = sub.add_parser('calibrate')
    p.add_argument('--from', dest='source', default=None, help='games.npz of a measure run (A/A\')')
    p.add_argument('--test', action='append', choices=['T-smoke', 'T-NI', 'T-std', 'T-fine'])
    p.add_argument('--n-tests', type=int, default=10_000)
    p.add_argument('--sigma-set', type=float, default=None)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default=None)
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser('panel')
    p.add_argument('--run', required=True, help='run id or run directory')
    p.set_defaults(func=cmd_panel)

    p = sub.add_parser('table')
    p.set_defaults(func=lambda args: registry.render_table())

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
