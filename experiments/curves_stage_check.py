"""D68's check at each stage boundary of D31's curves (`experiments/size-curves.md`, "Checks while
a run trains"), on the workstation. Any one firing stops the run:

1. **drift from the starting student**, on its own states (`experiments/rl_drift.py`): reverse
   KL(π_snapshot ‖ π_student) above 0.15 nats, argmax agreement below 88%, or entropy below 70% of
   the student's;
2. **the stop check:** 10,000 sets of the new snapshot against three copies of its stage's anchor
   (the previous snapshot; the student for the first stage), a lab `measure` on family A block
   `curves-boundary`: its 95% CI entirely below 0.

If 1 fires, 2 isn't played. 2 holds a workstation lock (`workdir/lab/stage-check.lock`), so one
lab measure plays at a time (a batch's two runs reach their boundaries together), and waits while
the eval box holds the lab registry. A measure already in the registry under the same description
is read, not replayed, so a relaunch after a reboot doesn't play it twice.

    python experiments/curves_stage_check.py --run curves-b12-s1 --start STUDENT \
        --states workdir/diag/credit-b12c256 --ckpt SNAP --anchor ANCHOR --anchor-label student \
        --label p25 --out workdir/fleet/runs/curves-b12-s1/stage-check-p25.json

`fleet.py watch --stage-check '<this command up to --states>'` adds the rest after each re-anchor
request and downs the run on a 'stop' verdict. The verdict file is written atomically and only at
the end; 'error' means the check couldn't be made (it never stops the run).
"""
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MAX_RKL = 0.15
MIN_AGREEMENT = 0.88
MIN_ENTROPY_FRACTION = 0.70
SETS = 10_000
BLOCK = 'curves-boundary'
RUNS = os.path.join(ROOT, 'lab', 'registry', 'runs.jsonl')
LEASE = os.path.join(ROOT, 'workdir', 'lab-lease.json')
LOCK = os.path.join(ROOT, 'workdir', 'lab', 'stage-check.lock')
LEASE_WAIT_S = 48 * 3600


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def say(msg):
    print(f'{now()} {msg}', flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def verdict(rkl, agreement, entropy, start_entropy, ci=None):
    """(verdict, fired): 'stop' if any limit fires, else 'continue'. `ci` is the stop check's
    95% CI, None when it wasn't played. Every limit is strict: at the limit it doesn't fire."""
    fired = []
    if rkl > MAX_RKL:
        fired.append('rkl')
    if agreement < MIN_AGREEMENT:
        fired.append('agreement')
    if entropy < MIN_ENTROPY_FRACTION * start_entropy:
        fired.append('entropy')
    if ci is not None and ci[1] < 0:
        fired.append('du')
    return ('stop' if fired else 'continue'), fired


def description(run, label, anchor_label):
    return f'curves boundary: {run} {label} at 3x {anchor_label}'


def measured_row(desc, path=RUNS):
    """The last finished measure in the registry with this description, or None."""
    row = None
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r.get('test') == 'measure' and r.get('verdict') == 'measured' and r.get('description') == desc:
                    row = r
    return row


def lab_argv(ckpt, anchor, desc, sets=SETS):
    return [sys.executable, '-m', 'lab', 'measure', '--challenger', ckpt, '--champion', anchor,
            '--sets', str(sets), '--family', 'A', '--block', BLOCK, '--arm', 'P1', '--package', 'K2',
            '--horizon', 'H', '--description', desc,
            '--purpose', 'D68 stop check at a stage boundary (experiments/size-curves.md; exploratory)']


def play_measure(argv, desc, lease=LEASE, lock=LOCK, wait_s=600, lease_wait_s=LEASE_WAIT_S, run=subprocess.run,
                 registry=RUNS):
    """The stop check's registry row: read if it exists, else played under the lock once the
    lease is free. Raises RuntimeError if it can't be had."""
    if (row := measured_row(desc, registry)) is not None:
        say(f'already measured: {row["run_id"]}')
        return row
    os.makedirs(os.path.dirname(lock), exist_ok=True)
    with open(lock, 'a') as lf:
        say(f'waiting for the lab lock {lock}')
        fcntl.flock(lf, fcntl.LOCK_EX)
        t0 = time.time()
        while os.path.exists(lease):
            if time.time() - t0 > lease_wait_s:
                raise RuntimeError(f'the eval box held the lab registry ({lease}) for {lease_wait_s / 3600:.0f} h')
            say(f'the eval box holds the lab registry ({lease}); waiting')
            time.sleep(wait_s)
        if (row := measured_row(desc, registry)) is not None:  # played while this one waited
            return row
        env = {**os.environ, 'LD_LIBRARY_PATH': os.path.join(sys.prefix, 'lib')}
        say('START ' + ' '.join(argv[2:]))
        rc = run(argv, cwd=ROOT, env=env).returncode
        say(f'END lab measure rc={rc}')
    if (row := measured_row(desc, registry)) is None:
        raise RuntimeError(f'lab measure exited {rc} without a measured row for {desc!r}')
    return row


def check(args, drift_fn=None, measure_fn=play_measure):
    """The verdict record (the module docstring)."""
    if drift_fn is None:
        import rl_drift
        drift_fn = rl_drift.measure
    rec = {
        'run': args.run, 'label': args.label, 'anchor_label': args.anchor_label,
        'ckpt': os.path.abspath(args.ckpt), 'ckpt_sha256': sha256(args.ckpt),
        'anchor': os.path.abspath(args.anchor), 'anchor_sha256': sha256(args.anchor),
        'start': os.path.abspath(args.start), 'states': os.path.abspath(args.states),
        'limits': {'max_rkl': MAX_RKL, 'min_agreement': MIN_AGREEMENT,
                   'min_entropy_fraction': MIN_ENTROPY_FRACTION, 'du': 'CI entirely below 0'},
        'git': subprocess.run(['git', 'describe', '--always', '--dirty', '--abbrev=12'], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip(),
        'started': now(),
    }
    h0, (drift,) = drift_fn(args.states, args.start, [args.ckpt])
    rec['drift'] = {k: drift[k] for k in ('decisions', 'rkl', 'rkl_median', 'kl', 'agreement', 'entropy')}
    rec['drift']['start_entropy'] = h0
    say(f'drift: rkl {drift["rkl"]:.4f}, agreement {drift["agreement"]:.4f}, entropy {drift["entropy"]:.4f} '
        f'(the student {h0:.4f})')
    rec['verdict'], rec['fired'] = verdict(drift['rkl'], drift['agreement'], drift['entropy'], h0)
    rec['measure'] = None  # not played if a drift limit fired
    if rec['verdict'] == 'continue':
        desc = description(args.run, args.label, args.anchor_label)
        row = measure_fn(lab_argv(args.ckpt, args.anchor, desc, args.sets), desc)
        res = row['result']
        rec['measure'] = {'run_id': row['run_id'], 'sets': res['sets'], 'du': res['mean'], 'ci': res['ci'],
                          'sd': res['sd'], 'nonce_start': row['nonce_start']}
        say(f'stop check: {row["run_id"]}, du {res["mean"]:+.4f} [{res["ci"][0]:+.4f}, {res["ci"][1]:+.4f}]')
        rec['verdict'], rec['fired'] = verdict(drift['rkl'], drift['agreement'], drift['entropy'], h0, res['ci'])
    return rec


def write(rec, out):
    rec['finished'] = now()
    tmp = out + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(rec, f, indent=1)
    os.replace(tmp, out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', required=True, help='the fleet run (for the registry description)')
    p.add_argument('--start', required=True, help='the starting student (the drift reference)')
    p.add_argument('--states', required=True, help="the student's rl_credit_check.py work dir")
    p.add_argument('--ckpt', required=True, help="the boundary's new snapshot")
    p.add_argument('--anchor', required=True, help="the stage's anchor (the stop check's table)")
    p.add_argument('--anchor-label', required=True, help="the anchor's snapshot label, or 'student'")
    p.add_argument('--label', required=True, help="the new snapshot's label")
    p.add_argument('--out', required=True, help='the verdict file (JSON)')
    p.add_argument('--sets', type=int, default=SETS, help='the stop check (pre-registered: 10,000)')
    args = p.parse_args(argv)
    t0 = time.time()
    try:
        rec = check(args)
    except Exception as e:  # noqa: BLE001 - any failure is reported to the watcher, never a stop
        err = f'{type(e).__name__}: {e}'[:500]
        rec = {'run': args.run, 'label': args.label, 'verdict': 'error', 'fired': [], 'error': err}
        say(f'ERROR {err}')
    rec['seconds'] = round(time.time() - t0)
    write(rec, args.out)
    print(json.dumps({k: rec.get(k) for k in ('verdict', 'fired', 'drift', 'measure', 'error')}, indent=1))
    return 0 if rec['verdict'] != 'error' else 1


if __name__ == '__main__':
    sys.exit(main())
