"""Option B of the 2026-11-22 review triage (`docs/design/review-2026-11-22.md`): free workstation
checks on curves batch 1's final snapshots before batch 2 is decided. Exploratory, outside D68's
pre-registration (`experiments/size-curves.md`); nothing here is read by `curves_reading.py`.

    python experiments/curves_diag.py drift    # drift of p75 (a reproduction) and full from the student
    python experiments/curves_diag.py play     # the five lab measures, one at a time
    python experiments/curves_diag.py report   # the paired differences, the table and the rule
    python experiments/curves_diag.py play-offtable NAME...   # off-table measures (OFFTABLE), in order
    python experiments/curves_diag.py report-offtable

The off-table measures (2026-11-23) play b12's student and snapshots at tables the runs never trained
against, 20,000 sets each on a family-A block of their own, `curves-offtable`, so every pair in it is
on the same deals. Their reading bands are fixed in `experiments/size-curves.md` before any of them
plays.

`play` runs detached and records its PID:

    setsid -f python experiments/curves_diag.py play >> workdir/lab/curves-diag.log 2>&1 < /dev/null

Every measure is 10,000 sets of family A block `curves-boundary`, the stage checks' deals, so
`report` can pair it set by set with a stage check: full@p50 - p75@p50 is stage 4's gain and
full@student - p25@student the gain after p25. Descriptions start with 'curves diag: ', which no
other script looks up ('curves: ' is the pre-registered reading's, 'curves boundary: ' the stage
checks'). A description already measured is read, not replayed. The `curves` block (the final
measures') isn't touched.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import curves_stage_check as csc  # noqa: E402

RUNS = {'curves-b12-s1': 'b12c256', 'curves-b16-s1': 'b16c256'}
STUDENTS = os.path.join(ROOT, 'workdir', 'secure', 'runs', 'students-s1', 'box', 'students-s1')
OUT_DIR = os.path.join(ROOT, 'workdir', 'lab', 'curves-diag')
DRIFT = os.path.join(OUT_DIR, 'drift.json')
REPORT = os.path.join(OUT_DIR, 'report.json')
PIDFILE = os.path.join(OUT_DIR, 'play.pid')
SETS = 10_000
PURPOSE = 'option B diagnostic before batch 2 (exploratory; not the pre-registered reading)'
PREFIX = 'curves diag: '
MIN_ENTROPY_FRACTION = csc.MIN_ENTROPY_FRACTION  # 0.70, the stage checks' limit
H2H = ('curves-b16-s1', 'curves-b12-s1')
OFF_BLOCK = 'curves-offtable'
OFF_SETS = 20_000
OFF_PURPOSE = 'off-table transfer of curves batch 1 (exploratory; not the pre-registered reading)'
TABLES = {  # label: (checkpoint, its frozen sha256)
    'No-SP A': ('workdir/sp-ablation/nosp-a/nosp-a-final.pth',
                '498aabc83c34291c0028107fc7cee6f3ac0e662bc37fba79b66b558306418054'),
    'BC-s1 EMA': ('workdir/teacher/teacher-bc-s1/ema-from-state.pth',  # STATUS's frozen table; the
                  '59d9463626125707f329355d515aa27ac2572d385906e336694047bd4b5f586f'),  # qualification's rows
}
OFF_RUN = 'curves-b12-s1'
REFERENCE = f'{PREFIX}{OFF_RUN} full at 3x student'  # the training table's total gain (+0.0294)  # b16 full at 3x b12 full: positive means b16 is stronger


def run_dir(run):
    return os.path.join(ROOT, 'workdir', 'fleet', 'runs', run)


def snap(run, label):
    return os.path.join(run_dir(run), 'snapshots', f'{label}.pth')


def student(run):
    return os.path.join(STUDENTS, f'ema-{RUNS[run]}.pth')


def states(run):
    return os.path.join(ROOT, 'workdir', 'diag', f'credit-{RUNS[run]}')


def recorded_sha(run, label):
    """The sha256 the watcher recorded for a run's snapshot."""
    with open(os.path.join(run_dir(run), 'snapshots.jsonl')) as f:
        rows = [json.loads(line) for line in f]
    return next(r['sha256'] for r in rows if r['label'] == label)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def measures():
    """(description, challenger, champion) in the order they're played: the gain since p50 (what
    batch 2's decision rests on), the head-to-head, then the total gain over the student."""
    out = [(f'{PREFIX}{r} full at 3x p50', snap(r, 'full'), snap(r, 'p50')) for r in RUNS]
    out.append((f'{PREFIX}{H2H[0]} full at 3x {H2H[1]} full', snap(H2H[0], 'full'), snap(H2H[1], 'full')))
    out += [(f'{PREFIX}{r} full at 3x student', snap(r, 'full'), student(r)) for r in RUNS]
    return out


def lab_argv(challenger, champion, desc):
    return [sys.executable, '-m', 'lab', 'measure', '--challenger', challenger, '--champion', champion,
            '--sets', str(SETS), '--family', 'A', '--block', csc.BLOCK, '--no-logs', '--arm', 'P1',
            '--package', 'K2', '--horizon', 'H', '--description', desc, '--purpose', PURPOSE]


def offtable():
    """name: (description, challenger, champion). `full`, the student and p50 of b12 at each table."""
    out = {}
    for short, label in (('nosp', 'No-SP A'), ('bc', 'BC-s1 EMA')):
        table = os.path.join(ROOT, TABLES[label][0])
        out[f'full@{short}'] = (f'{PREFIX}{OFF_RUN} full at 3x {label}', snap(OFF_RUN, 'full'), table)
        out[f'student@{short}'] = (f'{PREFIX}b12c256 student at 3x {label}', student(OFF_RUN), table)
        out[f'p50@{short}'] = (f'{PREFIX}{OFF_RUN} p50 at 3x {label}', snap(OFF_RUN, 'p50'), table)
    return out


def check_tables():
    for label, (rel, want) in TABLES.items():
        if sha256(os.path.join(ROOT, rel)) != want:
            raise SystemExit(f'{label}: {rel} does not match its frozen sha256')


def band(ci, reference):
    """The off-table bands (fixed before play): the gain carries if its 95% interval is entirely
    above half the training table's, is specific to the student if entirely below, else ambiguous."""
    half = 0.5 * reference
    return 'carries' if ci[0] > half else 'specific' if ci[1] < half else 'ambiguous'


def check_snapshots():
    for run in RUNS:
        for label in ('p25', 'p50', 'p75', 'full'):
            if sha256(snap(run, label)) != recorded_sha(run, label):
                raise SystemExit(f'{run} {label}: the file does not match its recorded sha256')


def cmd_drift(_):
    import rl_drift
    out = {}
    for run in RUNS:
        h0, (p75, full) = rl_drift.measure(states(run), student(run), [snap(run, 'p75'), snap(run, 'full')])
        keep = ('decisions', 'rkl', 'rkl_median', 'kl', 'agreement', 'entropy')
        with open(os.path.join(run_dir(run), 'stage-check-p75.json')) as f:
            check = json.load(f)['drift']
        out[run] = {'start_entropy': h0,
                    'full': {**{k: full[k] for k in keep}, 'entropy_fraction': full['entropy'] / h0,
                             'sha256': recorded_sha(run, 'full')},
                    'p75': {**{k: p75[k] for k in keep}, 'entropy_fraction': p75['entropy'] / h0},
                    'p75_reproduces_stage_check': all(abs(p75[k] - check[k]) < 1e-6
                                                      for k in ('rkl', 'agreement', 'entropy'))}
        f = out[run]['full']
        print(f'{run} full: rkl {f["rkl"]:.4f}, agreement {f["agreement"]:.4f}, entropy {f["entropy"]:.4f} '
              f'= {f["entropy_fraction"]:.1%} of the student\'s {h0:.4f}; '
              f'p75 reproduces its stage check: {out[run]["p75_reproduces_stage_check"]}', flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(DRIFT, 'w') as f:
        json.dump(out, f, indent=1)


def cmd_play(_):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(PIDFILE, 'w') as f:
        f.write(f'{os.getpid()}\n')
    check_snapshots()
    failed = []
    for desc, challenger, champion in measures():
        try:
            row = csc.play_measure(lab_argv(challenger, champion, desc), desc)
            res = row['result']
            csc.say(f'{desc}: {row["run_id"]}, du {res["mean"]:+.4f} [{res["ci"][0]:+.4f}, {res["ci"][1]:+.4f}]')
        except Exception as e:  # noqa: BLE001 - the next measure still plays
            failed.append(desc)
            csc.say(f'FAILED {desc}: {type(e).__name__}: {e}')
    csc.say(f'END play: {len(failed)} failed' + (f' ({"; ".join(failed)})' if failed else ''))
    return 1 if failed else 0


def cmd_play_offtable(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(PIDFILE, 'w') as f:
        f.write(f'{os.getpid()}\n')
    check_snapshots()
    check_tables()
    table = offtable()
    unknown = [n for n in args.names if n not in table]
    if unknown:
        raise SystemExit(f'unknown measures {unknown}; known: {sorted(table)}')
    failed = []
    for name in args.names:
        desc, challenger, champion = table[name]
        argv = [sys.executable, '-m', 'lab', 'measure', '--challenger', challenger, '--champion', champion,
                '--sets', str(OFF_SETS), '--family', 'A', '--block', OFF_BLOCK, '--no-logs', '--arm', 'P1',
                '--package', 'K2', '--horizon', 'H', '--description', desc, '--purpose', OFF_PURPOSE]
        try:
            row = csc.play_measure(argv, desc)
            res = row['result']
            csc.say(f'{desc}: {row["run_id"]}, du {res["mean"]:+.4f} [{res["ci"][0]:+.4f}, {res["ci"][1]:+.4f}]')
        except Exception as e:  # noqa: BLE001 - the next measure still plays
            failed.append(desc)
            csc.say(f'FAILED {desc}: {type(e).__name__}: {e}')
    csc.say(f'END play-offtable: {len(failed)} failed' + (f' ({"; ".join(failed)})' if failed else ''))
    return 1 if failed else 0


def cmd_report_offtable(_):
    rows = csc_rows()
    ref = rows[REFERENCE]['result']['mean']
    got = {name: rows.get(desc) for name, (desc, _, _) in offtable().items()}
    out = {'reference': {'description': REFERENCE, 'du': ref}, 'measures': {}, 'pairs': {},
           'note': 'exploratory, outside the pre-registration'}
    for name, r in got.items():
        if r is not None:
            out['measures'][name] = direct(r)
    for short in ('nosp', 'bc'):
        full, stud, p50 = got[f'full@{short}'], got[f'student@{short}'], got[f'p50@{short}']
        if full and stud:
            d = paired(full, stud)
            out['pairs'][f'full-student@{short}'] = {**d, 'share_of_training_table': d['du'] / ref,
                                                     'band': band(d['ci'], ref)}
        if full and p50:
            out['pairs'][f'full-p50@{short}'] = paired(full, p50)
        if p50 and stud:
            out['pairs'][f'p50-student@{short}'] = paired(p50, stud)
    path = os.path.join(OUT_DIR, 'offtable.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


def scores(row):
    z = np.load(os.path.join(ROOT, 'workdir', 'lab', row['run_id'], 'games.npz'))
    return z['seeds'].reshape(-1, 4)[:, 0], z['set_scores']


def direct(row):
    """A measure's own Δu and CI, as the lab recorded it."""
    res = row['result']
    return {'run_id': row['run_id'], 'sets': res['sets'], 'du': res['mean'], 'ci': res['ci']}


def paired(new, base):
    """`new` minus `base` per set: both rows on the same deals against the same opponents."""
    from lab import gsprt
    if new['champion']['sha256'] != base['champion']['sha256']:
        raise ValueError(f'{new["run_id"]} and {base["run_id"]} were played against different opponents')
    (s1, x1), (s0, x0) = scores(new), scores(base)
    if new['nonce_start'] != base['nonce_start'] or not np.array_equal(s1, s0):
        raise ValueError(f'{new["run_id"]} and {base["run_id"]} are not on the same deals')
    d = x1 - x0
    lo, hi = gsprt.bootstrap_mean_ci(d)
    return {'run_ids': [new['run_id'], base['run_id']], 'sets': int(len(d)), 'du': float(d.mean()),
            'ci': [float(lo), float(hi)], 'corr': float(np.corrcoef(x1, x0)[0, 1])}


def rule(per_run):
    """The triage's rule (the user, 2026-11-22), never acted on here: 'A' (batch 2 as
    pre-registered) if every run's entropy at full is >= 70% of the student's and its gain since
    p50 has a 95% lower bound above 0; 'C' if no run does; 'disagree' otherwise (the user's call)."""
    ok = {run: v['entropy_fraction'] >= MIN_ENTROPY_FRACTION and v['since_p50']['ci'][0] > 0
          for run, v in per_run.items()}
    option = 'A' if all(ok.values()) else 'C' if not any(ok.values()) else 'disagree'
    return {'option': option, 'meets_both_conditions': ok}


def cmd_report(_):
    rows = csc_rows()
    with open(DRIFT) as f:
        drift = json.load(f)
    per_run, missing = {}, []
    for run in RUNS:
        need = {'since_p50': f'{PREFIX}{run} full at 3x p50', 'vs_student': f'{PREFIX}{run} full at 3x student',
                'p75_at_p50': csc.description(run, 'p75', 'p50'), 'p25_at_student': csc.description(run, 'p25', 'student')}
        got = {k: rows.get(d) for k, d in need.items()}
        missing += [need[k] for k, r in got.items() if r is None]
        if any(r is None for r in got.values()):
            continue
        d = drift[run]['full']
        per_run[run] = {'drift': {k: d[k] for k in ('rkl', 'agreement', 'entropy')},
                        'entropy_fraction': d['entropy_fraction'],
                        'since_p50': direct(got['since_p50']),
                        'stage_4': paired(got['since_p50'], got['p75_at_p50']),
                        'vs_student': direct(got['vs_student']),
                        'after_p25': paired(got['vs_student'], got['p25_at_student'])}
    h2h = rows.get(f'{PREFIX}{H2H[0]} full at 3x {H2H[1]} full')
    if h2h is None:
        missing.append(f'{PREFIX}{H2H[0]} full at 3x {H2H[1]} full')
    out = {'per_run': per_run, 'head_to_head': direct(h2h) if h2h else None, 'missing': missing,
           'note': 'exploratory, outside the pre-registration'}
    if not missing:
        out['rule'] = rule(per_run)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(REPORT, 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


def csc_rows():
    """The registry's last finished measure per description."""
    out = {}
    with open(csc.RUNS) as f:
        for line in f:
            r = json.loads(line)
            if r.get('test') == 'measure' and r.get('verdict') == 'measured':
                out[r.get('description')] = r
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('cmd', choices=('drift', 'play', 'report', 'play-offtable', 'report-offtable'))
    p.add_argument('names', nargs='*', help='play-offtable: the measures, in order (offtable())')
    args = p.parse_args(argv)
    return {'drift': cmd_drift, 'play': cmd_play, 'report': cmd_report, 'play-offtable': cmd_play_offtable,
            'report-offtable': cmd_report_offtable}[args.cmd](args) or 0


if __name__ == '__main__':
    sys.exit(main())
