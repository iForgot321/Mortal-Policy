"""B0 warm-up acceptance (experiments/p1.md, pre-registered 2026-10-09).

Reads the critic diagnostics the trainer logs every window ("critic: explained variance X (last
kyoku Y), calibration monotone: Z") from a fleet run's pulled learner logs, and applies the rule:
  1. calibration (10 bins) monotone in the final window;
  2. explained variance, overall and in the last kyoku, not falling over the last third (its mean
     at least the middle third's mean).
Pass: both hold. Otherwise the rule allows one 4 box-hour extension from the run's `full` snapshot.
If that also fails, stop and report (D52's reopen condition).

Usage: python3 experiments/b0_check.py workdir/fleet/runs/p1a-b0 [more run dirs...]
"""
import glob
import re
import sys

PAT = re.compile(r'critic: explained variance (-?[\d.]+|nan) \(last kyoku (-?[\d.]+|nan)\), '
                 r'calibration monotone: (True|False)')


def windows(run_dir):
    """Diagnostic windows of every learner the run used, in time order (by log name)."""
    rows = []
    for log in sorted(glob.glob(f'{run_dir}/boxes/learner-*/logs/trainer.log')):
        with open(log, errors='replace') as f:
            text = f.read().replace('\r', '\n')
        rows += [(float(m[1]), float(m[2]), m[3] == 'True') for m in PAT.finditer(text)]
    return rows


def check(run_dir):
    rows = windows(run_dir)
    if len(rows) < 6:
        return f'{run_dir}: only {len(rows)} diagnostic windows; nothing to judge', False
    n = len(rows)
    thirds = [rows[int(a * n):int(b * n)] for a, b in ((0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1))]
    mean = lambda part, i: sum(r[i] for r in part) / len(part)  # noqa: E731
    ev = [mean(p, 0) for p in thirds]
    last = [mean(p, 1) for p in thirds]
    monotone_final = rows[-1][2]
    rising = ev[2] >= ev[1] and last[2] >= last[1]
    ok = monotone_final and rising
    msg = (f'{run_dir}: {n} windows; EV by third {ev[0]:.4f} / {ev[1]:.4f} / {ev[2]:.4f}; last-kyoku EV '
           f'{last[0]:.4f} / {last[1]:.4f} / {last[2]:.4f}; final window monotone {monotone_final} '
           f'({sum(r[2] for r in thirds[2])}/{len(thirds[2])} in the last third) -> {"PASS" if ok else "FAIL"}')
    return msg, ok


if __name__ == '__main__':
    results = [check(d) for d in sys.argv[1:]]
    for msg, _ in results:
        print(msg)
    sys.exit(0 if results and all(ok for _, ok in results) else 1)
