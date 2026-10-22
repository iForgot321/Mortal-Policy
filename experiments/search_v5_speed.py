# Search rollout speed, v4 vs v5 observation (docs/design/observation-review-d49.md, the user's go-ahead
# 2026-10-22). Times the oracle stage's rollouts (2 candidates x (64 redeals + the true world) per decision)
# on the first N decisions of a search run, for each engine, in alternating passes. The v5 engine is a
# timing device: pilot A's weights with v5 input (workdir/sp-ablation/pilot-a-as-v5.pth), so its moves mean
# nothing and rollout lengths differ; time per logged event normalizes for that. From the repo root:
#   python experiments/search_v5_speed.py workdir/search/pilot-a workdir/pilot/qs-a/pilot-qs-a-final.pth \
#       workdir/sp-ablation/pilot-a-as-v5.pth --out workdir/search/v5-speed/speed.json
import argparse, json, os, sys, time
from types import SimpleNamespace
sys.path.insert(0, '.')
import torch
from search.__main__ import load_blueprint
from search.pilot import RIICHI, read_game, redeal_seed


def jobs_for(run, rows, redeals):
    """The oracle stage's batches: (roots, jobs, until) per chunk of 16 decisions."""
    cache, out = {}, []
    for i in range(0, len(rows), 16):
        part = rows[i:i + 16]
        roots, jobs, until = [], [], []
        for j, d in enumerate(part):
            if d['game'] not in cache:
                cache[d['game']] = read_game(os.path.join(run, 'logs', d['game']))
            seed, kyokus = cache[d['game']]
            roots.append((seed, kyokus[d['kyoku']], d['until']))
            for action in (RIICHI, d['dama']):
                for r in [None, *range(redeals)]:
                    jobs.append((j, action, None if r is None else redeal_seed(d['id'], r)))
                    until.append(d['until'])
        out.append((roots, jobs, until))
    return out


def time_engine(engine, batches):
    from libriichi.search import rollouts
    torch.cuda.synchronize()
    t0, n, events = time.perf_counter(), 0, 0
    for roots, jobs, until in batches:
        outs = rollouts(engine, roots, jobs, keep_logs=True)
        n += len(outs)
        events += sum(len(o.log) - u for o, u in zip(outs, until))
    torch.cuda.synchronize()
    return time.perf_counter() - t0, n, events


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run')
    p.add_argument('engines', nargs='+')
    p.add_argument('--decisions', type=int, default=500)
    p.add_argument('--redeals', type=int, default=64)
    p.add_argument('--passes', type=int, default=2)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    with open(os.path.join(a.run, 'decisions.jsonl'), encoding='utf-8') as f:
        rows = [json.loads(line) for _, line in zip(range(a.decisions), f)]
    batches = jobs_for(a.run, rows, a.redeals)
    engines = {path: load_blueprint(SimpleNamespace(engine=path, device='cuda:0')) for path in a.engines}
    for path, engine in engines.items():  # warm-up: CUDA context, cuDNN, allocator
        time_engine(engine, batches[:1])
        print(f'{path}: version {engine.version}', flush=True)
    results = []
    for k in range(a.passes):
        for path, engine in engines.items():
            secs, n, events = time_engine(engine, batches)
            r = {'pass': k, 'engine': path, 'version': engine.version, 'decisions': len(rows), 'rollouts': n,
                 'seconds': round(secs, 2), 'rollouts_per_s': round(n / secs, 1),
                 'events_per_rollout': round(events / n, 2), 'us_per_event': round(secs / events * 1e6, 2)}
            results.append(r)
            print(json.dumps(r), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
