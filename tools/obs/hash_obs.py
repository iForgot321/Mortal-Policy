"""Golden hash of the observation encoder (SP ablation, M0/M1; CLAUDE.md rule 2).

Replays mjai logs through one PlayerState per seat. At every decision of every seat, and again
at kan select when an ankan or kakan is possible, it feeds the bytes of
`encode_obs(version, at_kan_select)` (obs, then mask) into a sha256, per log and overall. Run it
before and after an encoder change: equal hashes mean bit-identical output on those logs.

`--check-v5` also asserts at the same decisions that obs v5 equals v4 with the single-player
rows zeroed, and that the masks are equal.

Usage, from the repo root (any env that can import mortal/libriichi.so):
    python tools/obs/hash_obs.py --version 4 --out workdir/sp-ablation/golden-v4.json
    python tools/obs/hash_obs.py --version 4 --check-v5 --expect workdir/sp-ablation/golden-v4.json
Tenhou logs: workstation only (D23); the output holds hashes and file names only.
"""
import argparse
import glob
import gzip
import hashlib
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'mortal'))

# v4's single-player block: the last 123 of its 1012 rows
SP_START = 889

# (directory, first N sorted *.json.gz): pilot A vs B, the hobbled pilot (random actions reach
# unusual states), Tenhou houou 2024-03
DEFAULT_SETS = [
    (os.path.join(ROOT, 'workdir/lab/20261005-124255-measure/logs'), 20),
    (os.path.join(ROOT, 'workdir/lab/20261005-163506-T-std/logs'), 8),
    (os.path.expanduser('~/mj-data/mjai/2024/03'), 8),
]


def decisions(path):
    """Yields (state, at_kan_select) at every decision of every seat in one log."""
    from libriichi.state import PlayerState

    with gzip.open(path, 'rt') as f:
        events = [json.loads(line) for line in f]
    states = [PlayerState(i) for i in range(4)]
    for ev in events:
        ev.pop('meta', None)
        line = json.dumps(ev)
        for st in states:
            cans = st.update(line)
            if not cans.can_act:
                continue
            yield st, False
            if cans.can_ankan or cans.can_kakan:
                yield st, True


def hash_log(path, version, check_v5):
    h = hashlib.sha256()
    n = 0
    for st, kan in decisions(path):
        obs, mask = st.encode_obs(version, kan)
        h.update(obs.tobytes())
        h.update(mask.tobytes())
        n += 1
        if check_v5:
            obs4, mask4 = (obs, mask) if version == 4 else st.encode_obs(4, kan)
            obs5, mask5 = st.encode_obs(5, kan)
            assert obs5.shape == obs4.shape, (path, obs5.shape)
            assert np.array_equal(obs5[:SP_START], obs4[:SP_START]), (path, n)
            assert not obs5[SP_START:].any(), (path, n)
            assert np.array_equal(mask5, mask4), (path, n)
    return h.hexdigest(), n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--version', type=int, default=4)
    ap.add_argument('--logs', nargs='+', default=None, metavar='DIR:N',
                    help='log directories with the number of sorted logs to take from each (default: the M0 sets)')
    ap.add_argument('--check-v5', action='store_true', help='also assert v5 == v4 with the SP rows zeroed')
    ap.add_argument('--expect', default=None, help='a previous --out file; exit 1 unless every hash matches')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    sets = DEFAULT_SETS
    if args.logs:
        sets = [(d, int(n)) for d, n in (s.rsplit(':', 1) for s in args.logs)]

    total = hashlib.sha256()
    logs, n_decisions = {}, 0
    t0 = time.time()
    for d, n in sets:
        paths = sorted(glob.glob(os.path.join(d, '*.json.gz')))[:n]
        if len(paths) < n:
            sys.exit(f'{d}: {len(paths)} logs, wanted {n}')
        for p in paths:
            digest, k = hash_log(p, args.version, args.check_v5)
            logs[os.path.relpath(p, os.path.expanduser('~'))] = {'sha256': digest, 'decisions': k}
            total.update(digest.encode())
            n_decisions += k
    result = {
        'version': args.version,
        'sha256': total.hexdigest(),
        'logs': len(logs),
        'decisions': n_decisions,
        'v5_checked': args.check_v5,
        'seconds': round(time.time() - t0, 1),
        'per_log': logs,
    }
    print(json.dumps({k: v for k, v in result.items() if k != 'per_log'}), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(result, f, indent=1)
    if args.expect:
        with open(args.expect) as f:
            want = json.load(f)
        bad = [k for k, v in want['per_log'].items() if logs.get(k, {}).get('sha256') != v['sha256']]
        if want['sha256'] != result['sha256'] or bad:
            print(f'MISMATCH against {args.expect}: {len(bad)} logs differ, e.g. {bad[:3]}')
            sys.exit(1)
        print(f'matches {args.expect}')


if __name__ == '__main__':
    main()
