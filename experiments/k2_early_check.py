"""The G0 retry's early check at seed 1's p25 snapshot (D59, `experiments/k2-retry.md`): three
pre-registered measures against the initialization; any one firing stops the run.

1. reverse KL(π_ckpt ‖ π_SL), mean per decision: stop if > 0.10 nats;
2. argmax agreement with π_SL: stop if < 90%;
3. Δu at argmax against the initialization, 4,000 sets on the diagnostic key: stop if the 95% CI
   is entirely below 0.

Measures 1 and 2 use the diagnosis's states (`rl_credit_check.py --work` dirs, every second
decision), and 3 the diagnosis's key and nonces [0, 4000), so the numbers compare with
`experiments/rl-diagnosis.md`. If 1 or 2 fires, 3 isn't played: the run stops an hour sooner.

    python experiments/k2_early_check.py --ckpt SNAP --init INIT --states workdir/diag/credit-a \
        --out workdir/fleet/runs/RUN/early-check-p25.json

`fleet.py watch --early-check '<this command without --ckpt and --out>'` launches it once the p25
snapshot is in and downs the run on a "stop" verdict. The verdict file is written atomically and
only at the end, so its presence means the check finished.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MAX_RKL = 0.10
MIN_AGREEMENT = 0.90
SETS = 4000
Z = 1.96


def verdict(rkl, agreement, du=None, se=None):
    """(verdict, fired): 'stop' if any measure fires, else 'continue'. With du None, measure 3
    wasn't played (1 or 2 had fired)."""
    fired = []
    if rkl > MAX_RKL:
        fired.append('rkl')
    if agreement < MIN_AGREEMENT:
        fired.append('agreement')
    if du is not None and du + Z * se < 0:
        fired.append('du')
    return ('stop' if fired else 'continue'), fired


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ckpt', required=True, help='the p25 snapshot')
    p.add_argument('--init', required=True, help='the initialization (π_SL)')
    p.add_argument('--states', required=True, help="the initialization's rl_credit_check.py work dir")
    p.add_argument('--out', required=True, help='the verdict file (JSON)')
    p.add_argument('--sets', type=int, default=SETS, help='measure 3 (pre-registered: 4,000)')
    args = p.parse_args()

    import rl_diag_sampling
    import rl_drift

    t0 = time.time()
    rec = {
        'ckpt': os.path.abspath(args.ckpt), 'ckpt_sha256': sha256(args.ckpt),
        'init': os.path.abspath(args.init), 'states': os.path.abspath(args.states),
        'thresholds': {'max_rkl': MAX_RKL, 'min_agreement': MIN_AGREEMENT, 'du': 'CI entirely below 0'},
        'git': subprocess.run(['git', 'describe', '--always', '--dirty', '--abbrev=12'], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip(),
        'started': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }
    _, (drift,) = rl_drift.measure(args.states, args.init, [args.ckpt])
    rec['drift'] = {k: drift[k] for k in ('decisions', 'rkl', 'rkl_median', 'kl', 'agreement', 'entropy')}
    print(f'rkl {drift["rkl"]:.4f}, agreement {drift["agreement"]:.4f}', flush=True)
    rec['verdict'], rec['fired'] = verdict(drift['rkl'], drift['agreement'])
    if rec['verdict'] == 'continue':
        out_npz = args.out.replace('.json', '') + '-argmax.npz'
        play = rl_diag_sampling.play(args.ckpt, args.init, mode='argmax', sets=args.sets, out=out_npz)
        rec['argmax'] = {k: play[k] for k in ('sets', 'du', 'se', 'avg_rank', 'rank_hist', 'key', 'nonce_start')}
        rec['verdict'], rec['fired'] = verdict(drift['rkl'], drift['agreement'], play['du'], play['se'])
    else:
        rec['argmax'] = None  # not played: 1 or 2 fired
    rec['seconds'] = round(time.time() - t0)
    rec['finished'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    tmp = args.out + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(rec, f, indent=1)
    os.replace(tmp, args.out)
    print(json.dumps({k: rec[k] for k in ('verdict', 'fired', 'drift', 'argmax')}, indent=1))


if __name__ == '__main__':
    main()
