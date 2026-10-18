"""The K2 tuning runs' pre-registered measures (`experiments/k2-retry.md` → Tuning), from each run's
pulled `learner-run/anchor.jsonl` and `fleet.log`:

- the fraction of logged batch-KL values inside [0.013, 0.03] after the run's first hour of
  training (counted from its first anchor entry);
- their maximum, and the coefficient's trace;
- the fleet's games per hour, as its watcher logged them.

    python experiments/k2_tuning_report.py k2-tune-a k2-tune-b
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BAND = (0.02 / 1.5, 0.02 * 1.5)
MIN_IN_BAND = 0.70
TIE = 0.10


def entries(run):
    path = os.path.join(ROOT, 'workdir', 'fleet', 'runs', run, 'learner-run', 'anchor.jsonl')
    out = []
    if os.path.exists(path):
        for line in open(path):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def games_per_hour(run):
    path = os.path.join(ROOT, 'workdir', 'fleet', 'runs', run, 'fleet.log')
    pat = re.compile(rf'\[{re.escape(run)}\] ([\d,]+) games in the last hour')
    vals = [int(m.group(1).replace(',', '')) for line in open(path) if (m := pat.search(line))] if os.path.exists(path) else []
    return vals


def measure(run):
    e = entries(run)
    if not e:
        return {'run': run, 'entries': 0}
    t0 = e[0]['time']
    late = [x for x in e if x['time'] >= t0 + 3600]
    kls = [x['kl'] for x in late]
    inband = [BAND[0] <= k <= BAND[1] for k in kls]
    gph = games_per_hour(run)
    return {
        'run': run, 'entries': len(e), 'hours': round((e[-1]['time'] - t0) / 3600, 2), 'after_first_hour': len(late),
        'in_band': sum(inband) / len(inband) if inband else None,
        'kl_max': max(kls) if kls else None, 'kl_mean': sum(kls) / len(kls) if kls else None,
        'agreement_last': e[-1].get('agreement'), 'coef_first': e[0].get('coef'), 'coef_last': e[-1].get('coef'),
        'coef_min_seen': min(x['coef'] for x in e), 'coef_max_seen': max(x['coef'] for x in e),
        'games_per_hour_last': gph[-1] if gph else None, 'games_per_hour_max': max(gph) if gph else None,
    }


def choose(a, b):
    """The pre-registered rule: the larger in-band fraction if ≥ 70%; within 10 points, adaptive (a)."""
    fa, fb = a.get('in_band'), b.get('in_band')
    if fa is None or fb is None:
        return None, 'not enough data'
    if max(fa, fb) < MIN_IN_BAND:
        return None, f'neither reaches {MIN_IN_BAND:.0%}: no launch; report and adjust'
    if abs(fa - fb) <= TIE:
        return a['run'], 'within 10 points: adaptive kept'
    return (a['run'], 'larger in-band fraction') if fa > fb else (b['run'], 'larger in-band fraction')


def main():
    runs = sys.argv[1:] or ['k2-tune-a', 'k2-tune-b']
    rows = [measure(r) for r in runs]
    for r in rows:
        print(json.dumps(r))
    if len(rows) == 2:
        print('choice:', *choose(*rows))


if __name__ == '__main__':
    main()
