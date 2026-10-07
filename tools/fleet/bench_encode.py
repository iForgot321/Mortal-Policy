"""Observation encoding cost per decision, by obs version (single thread).

Feeds mjai logs to one PlayerState per seat and times, per event, the four `update` calls and,
at every decision of every seat, `encode_obs(version, False)`. v4's single-player tables are
almost all of v4's cost (2026-10-06, 200 pilot A-vs-B games, 130,710 decisions:
v3 3.4 us, v4 1,158 us mean / 105 us median / 13.8 ms p99 / 50 ms max per decision).

Usage, from the repo root (any env that can import mortal/libriichi.so):
    python tools/fleet/bench_encode.py --logs workdir/lab/20261005-124255-measure/logs --n 200 --versions 3 4
Tenhou logs: workstation or secure-tier box only (D23).
"""
import argparse
import glob
import gzip
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'mortal'))


def load(paths):
    games = []
    for p in paths:
        with gzip.open(p, 'rt') as f:
            events = [json.loads(line) for line in f]
        for ev in events:
            ev.pop('meta', None)
        games.append([json.dumps(ev) for ev in events])
    return games


def run(games, version):
    from libriichi.state import PlayerState

    t_update, n_events, times = 0.0, 0, []
    for lines in games:
        states = [PlayerState(i) for i in range(4)]
        for line in lines:
            t0 = time.perf_counter()
            cans = [st.update(line) for st in states]
            t_update += time.perf_counter() - t0
            n_events += 1
            for st, c in zip(states, cans):
                if c.can_act:
                    t0 = time.perf_counter()
                    st.encode_obs(version, False)
                    times.append(time.perf_counter() - t0)
    t = np.array(times)
    return {
        'version': version,
        'games': len(games),
        'events': n_events,
        'decisions': len(t),
        'update_us_per_event': round(t_update / n_events * 1e6, 2),
        'encode_us_mean': round(t.mean() * 1e6, 1),
        'encode_us_p50': round(float(np.median(t)) * 1e6, 1),
        'encode_us_p99': round(float(np.quantile(t, 0.99)) * 1e6, 1),
        'encode_ms_max': round(t.max() * 1e3, 2),
        'encode_cpu_s_per_game': round(t.sum() / len(games), 3),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--logs', required=True, help='directory of *.json.gz mjai logs')
    ap.add_argument('--n', type=int, default=200, help='first N logs (sorted)')
    ap.add_argument('--versions', type=int, nargs='+', default=[3, 4])
    ap.add_argument('--out', default=None, help='optional JSON output')
    args = ap.parse_args()

    games = load(sorted(glob.glob(os.path.join(args.logs, '*.json.gz')))[:args.n])
    results = []
    for v in args.versions:
        results.append(run(games, v))
        print(json.dumps(results[-1]), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump({'logs': args.logs, 'n': args.n, 'results': results}, f, indent=1)


if __name__ == '__main__':
    main()
