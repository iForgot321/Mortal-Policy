"""Where obs v4's single-player (SP) cost goes, and what the cheaper SP variants would save
(SP ablation follow-up; experiments/sp-ablation.md).

Times `encode_obs(4, False)` at every decision of every seat (single thread) and splits the total
by decision type, real-time hand shanten and draws left. The shanten is what the calculator's
threshold sees (`PlayerState.shanten` is not refreshed by a draw), read from `brief_info()` outside
the timed call. `brief_info()` panics while printing a table from the cheap path (its candidates have
no probabilities), so a panic means real-time shanten >= 4; run with stderr discarded. v4's non-SP part costs ~3.4 us (= v5), so the time is
almost all SP. Savings per variant are estimates from these timings, not measurements of a
modified encoder:

- shanten threshold 3 -> 2: 3-shanten decisions at the cost of the cheap required-tiles path that
  4+-shanten hands already take (their mean time, per decision type);
- no SP at call/pass decisions (and, separately, after the player's own riichi): those decisions
  at v5's cost;
- work cap: each decision's time capped at the corpus's p95 / p99 (time as a proxy for search size;
  a real cap would count search steps so the encoding stays deterministic);
- required tiles only: every decision at the cheap path's cost.
The horizon cap (17 -> ~8 draws) needs a modified calculator and is not estimated.

Usage, from the repo root (any env that can import mortal/libriichi.so):
    python tools/obs/sp_cost.py --logs workdir/lab/20261005-124255-measure/logs:200 \
        ~/mj-data/mjai/2024/03:100 --out workdir/sp-ablation/sp-cost.json
Tenhou logs: workstation only (D23); the output holds aggregates only.
"""
import argparse
import glob
import gzip
import json
import os
import re
import sys
import time
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'mortal'))

V5_US = 3.4  # measured v5 / v3 encode cost per decision (sp-ablation.md §3)
KINDS = ('discard', 'discard after own riichi', 'call/pass')
SHANTEN_RE = re.compile(r'shanten: -?\d+ \(actual: (-?\d+)\)')


def shanten_bucket(s):
    return str(s) if s <= 3 else '4+'


def draws_bucket(draws):
    for hi in (4, 8, 12):
        if draws <= hi:
            return f'{hi - 3 if hi > 4 else 0}-{hi}'
    return '13-17'


def real_time_shanten(st):
    try:
        return int(SHANTEN_RE.search(st.brief_info()).group(1))
    except BaseException as e:  # pyo3's PanicException derives from BaseException
        if type(e).__name__ != 'PanicException':
            raise
        return 4  # the cheap path: 4 or more


def measure(paths):
    """One row per decision: (seconds, kind, shanten, draws left)."""
    from libriichi.state import PlayerState

    rows = []
    for p in paths:
        with gzip.open(p, 'rt') as f:
            events = [json.loads(line) for line in f]
        states = [PlayerState(i) for i in range(4)]
        tiles_left = 70  # as PlayerState counts: 70 at start_kyoku, -1 per tsumo
        for ev in events:
            ev.pop('meta', None)
            if ev['type'] == 'start_kyoku':
                tiles_left = 70
            elif ev['type'] == 'tsumo':
                tiles_left -= 1
            line = json.dumps(ev)
            for st in states:
                cans = st.update(line)
                if not cans.can_act:
                    continue
                t0 = time.perf_counter()
                st.encode_obs(4, False)
                dt = time.perf_counter() - t0
                if not cans.can_discard:
                    kind = 'call/pass'
                elif st.self_riichi_accepted:
                    kind = 'discard after own riichi'
                else:
                    kind = 'discard'
                rows.append((dt, kind, real_time_shanten(st), tiles_left // 4))
    return rows


def summarize(rows):
    t = np.array([r[0] for r in rows]) * 1e6  # us
    total = t.sum()
    out = {
        'decisions': len(t),
        'mean_us': round(float(t.mean()), 1),
        'p50_us': round(float(np.median(t)), 1),
        'p95_us': round(float(np.quantile(t, 0.95)), 1),
        'p99_us': round(float(np.quantile(t, 0.99)), 1),
        'max_us': round(float(t.max()), 1),
    }
    srt = np.sort(t)[::-1]
    out['time_share_of_slowest'] = {
        f'{q}%': round(float(srt[:max(1, int(len(t) * q / 100))].sum() / total), 3) for q in (1, 5, 10, 25)
    }

    def table(key):
        acc = defaultdict(list)
        for x, r in zip(t, rows):
            acc[key(r)].append(x)
        return {
            k: {'n': len(v), 'share_n': round(len(v) / len(t), 3), 'mean_us': round(float(np.mean(v)), 1),
                'share_time': round(float(np.sum(v) / total), 3)}
            for k, v in sorted(acc.items())
        }

    out['by_kind'] = table(lambda r: r[1])
    out['by_shanten'] = table(lambda r: shanten_bucket(r[2]))
    out['by_draws_left'] = table(lambda r: draws_bucket(r[3]))
    out['by_kind_shanten'] = table(lambda r: f'{r[1]} | {shanten_bucket(r[2])}')

    # the cheap path's cost per decision type: the mean over 4+-shanten decisions
    cheap = {}
    for kind in KINDS:
        v = [x for x, r in zip(t, rows) if r[1] == kind and r[2] >= 4]
        cheap[kind] = float(np.mean(v)) if v else V5_US
    out['cheap_path_us'] = {k: round(v, 1) for k, v in cheap.items()}

    def saving(new_cost):
        return round(float(1 - sum(new_cost(x, r) for x, r in zip(t, rows)) / total), 3)

    p95, p99 = np.quantile(t, 0.95), np.quantile(t, 0.99)
    out['estimated_saving'] = {
        'shanten threshold 3 -> 2': saving(lambda x, r: min(x, cheap[r[1]]) if r[2] == 3 else x),
        'no SP at call/pass': saving(lambda x, r: V5_US if r[1] == 'call/pass' else x),
        'no SP at call/pass or after own riichi': saving(lambda x, r: V5_US if r[1] != 'discard' else x),
        'work cap at p95': saving(lambda x, r: min(x, p95)),
        'work cap at p99': saving(lambda x, r: min(x, p99)),
        'required tiles only': saving(lambda x, r: min(x, cheap[r[1]])),
        'no SP (v5)': saving(lambda x, r: V5_US),
    }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--logs', nargs='+', required=True, metavar='DIR:N',
                    help='log directories with the number of sorted logs to take from each; one corpus each')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    results = {}
    for spec in args.logs:
        d, n = spec.rsplit(':', 1)
        paths = sorted(glob.glob(os.path.join(os.path.expanduser(d), '*.json.gz')))[:int(n)]
        if len(paths) < int(n):
            sys.exit(f'{d}: {len(paths)} logs, wanted {n}')
        t0 = time.time()
        res = summarize(measure(paths))
        res['games'] = len(paths)
        res['seconds'] = round(time.time() - t0, 1)
        results[spec] = res
        print(json.dumps({spec: res}, indent=1), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=1)


if __name__ == '__main__':
    main()
