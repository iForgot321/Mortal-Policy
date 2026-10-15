"""The search pilot (06 §0), from the repo root:

    python -m search selfplay  --run DIR --engine CKPT (--block NAME | --dev) --games N
    python -m search decisions --run DIR --engine CKPT --count 20000
    python -m search oracle    --run DIR --engine CKPT --redeals 64
    python -m search report    --run DIR

`--block` takes a named family-A block from the lab's nonce ledger. `--dev`
uses a key outside every lab family, for smoke tests only.
"""
import argparse
import json
import os

from . import pilot

DEV_KEY = 0x5ea7c4


def load_blueprint(args):
    import torch
    from lab import load_config
    from lab.arena import load_engine
    # true FP32: a rollout must replay its self-play game exactly (D43)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    guard = load_config()['arena']['enable_rule_based_agari_guard']
    return load_engine(args.engine, device=args.device, enable_amp=False,
                       enable_rule_based_agari_guard=guard, name='blueprint')


def record(run_dir, stage, **fields):
    """Provenance of each stage in `meta.json`."""
    from lab import registry
    path = os.path.join(run_dir, 'meta.json')
    meta = {}
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            meta = json.load(f)
    meta[stage] = {**fields, 'time': registry.now(), 'git': registry.git_state()}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)


def engine_record(spec):
    from lab import registry
    return {'spec': os.path.abspath(spec), 'sha256': registry.sha256_file(spec)}


def main():
    p = argparse.ArgumentParser(prog='python -m search', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)

    def stage(name, engine=True):
        s = sub.add_parser(name)
        s.add_argument('--run', required=True, help='run directory (workdir/search/NAME)')
        if engine:
            s.add_argument('--engine', required=True, help='blueprint checkpoint')
            s.add_argument('--device', default='cuda:0')
        return s

    s = stage('selfplay')
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument('--block', help='named family-A block in the lab nonce ledger')
    g.add_argument('--dev', action='store_true', help='a key outside the lab families')
    s.add_argument('--games', type=int, required=True)

    s = stage('decisions')
    s.add_argument('--count', type=int, default=20_000)
    s.add_argument('--sample-seed', type=int, default=0)

    s = stage('oracle')
    s.add_argument('--redeals', type=int, default=64, help='rollouts per candidate')
    s.add_argument('--chunk', type=int, default=16, help='decisions per batch of rollouts')

    stage('report', engine=False)
    args = p.parse_args()
    os.makedirs(args.run, exist_ok=True)

    if args.cmd == 'report':
        print(json.dumps(pilot.report(args.run), indent=2))
        return

    engine = load_blueprint(args)
    if args.cmd == 'selfplay':
        if args.dev:
            nonce, key, run_id = 0, DEV_KEY, None
        else:
            from lab import registry
            run_id = registry.new_run_id('search-selfplay')
            nonce, key = registry.allocate_nonces(
                'A', args.games, run_id, 'search pilot self-play, one game per nonce (06 §0)',
                block=args.block)
        record(args.run, 'selfplay', engine=engine_record(args.engine), nonce_start=nonce,
               key=hex(key), games=args.games, block=args.block, run_id=run_id)
        pilot.selfplay(engine, args.run, nonce, key, args.games)
    elif args.cmd == 'decisions':
        summary = pilot.decisions(engine, args.run, args.count, args.sample_seed)
        record(args.run, 'decisions', engine=engine_record(args.engine), **summary)
        print(json.dumps(summary, indent=2))
    elif args.cmd == 'oracle':
        from .value import GRP_STATE_FILE, GrpValue
        from lab import registry
        record(args.run, 'oracle', engine=engine_record(args.engine), redeals=args.redeals,
               grp={'path': GRP_STATE_FILE, 'sha256': registry.sha256_file(GRP_STATE_FILE)})
        pilot.oracle(engine, GrpValue(), args.run, args.redeals, args.chunk)


if __name__ == '__main__':
    main()
