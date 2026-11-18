"""D31's curves, the pre-registered reading (experiments/size-curves.md, "Reading, fixed now"; D31,
D65, D68), from the final measures' registry rows (experiments/curves_measure_chain.sh).

For each run and reading (6 box-days: p50; 12: `full`), Δu per dollar is its Δu against three
copies of its starting student over the run's cost at the planning price ($13.2 per box-day).
With each size's figure the mean of its seeds', L = b16c256's figure − b12c256's, and S the larger
of the two within-size ranges, floored at 0.0043 Δu over a run's cost (the lab's noise for a
difference of two runs):

1. L > S at 12 box-days: b16c256. Otherwise b12c256 (the cheaper net wins ties).
2. Crossover: b16c256 trailing at 6 box-days but leading at 12 is read by rule 1 (the same
   condition); it is reported.
3. Close, 0.5 S < L < 1.5 S at either reading, with two seeds per size: a third seed per size
   (batch 3; the user's yes on a live quote). With three, rules 1-2 decide.

Reported, not deciding: a 95% interval for L from the seeds and the lab's measurement noise, the
held-out table (3× BC s1), each run's boundary checks (experiments/curves_stage_check.py) and its
billed cost.

    python experiments/curves_reading.py [RUN ...]     # default: batches 1 and 2's four runs
"""
import json
import math
import os
import re
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_JSONL = os.path.join(ROOT, 'lab', 'registry', 'runs.jsonl')
FLEET = os.path.join(ROOT, 'workdir', 'fleet', 'runs')
OUT = os.path.join(ROOT, 'workdir', 'lab', 'curves-reading.json')
PLANNING_USD_PER_BOX_DAY = 13.2
NOISE_DU = 0.0043  # the lab's noise for a difference of two runs (D68)
READINGS = {'p50': 6, 'full': 12}  # snapshot label: nominal box-days
SIZES = ('b12c256', 'b16c256')
DEFAULT_RUNS = ('curves-b12-s1', 'curves-b16-s1', 'curves-b12-s2', 'curves-b16-s2')
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447}


def size_of(run):
    m = re.match(r'^curves-(b12|b16)-s\d$', run)
    if not m:
        raise ValueError(f'not a curves run: {run}')
    return {'b12': 'b12c256', 'b16': 'b16c256'}[m.group(1)]


def per_dollar(du, box_days):
    return du / (box_days * PLANNING_USD_PER_BOX_DAY)


def spread(figures_by_size, box_days):
    """S: the larger within-size range, floored at the lab's noise over a run's cost."""
    ranges = [max(v) - min(v) for v in figures_by_size.values() if len(v) > 1]
    return max([*ranges, per_dollar(NOISE_DU, box_days)])


def decide(fig, nseeds):
    """The rule on {reading: {size: [Δu per dollar per seed]}} (readings 'p50', 'full').
    Returns the decision record."""
    out = {}
    for label, days in READINGS.items():
        by = fig[label]
        means = {s: statistics.fmean(by[s]) for s in SIZES}
        L, S = means['b16c256'] - means['b12c256'], spread(by, days)
        out[label] = {'means': means, 'L': L, 'S': S, 'close': 0.5 * S < L < 1.5 * S}
    full, p50 = out['full'], out['p50']
    pick = 'b16c256' if full['L'] > full['S'] else 'b12c256'
    close = full['close'] or p50['close']
    return {
        'readings': out,
        'pick': pick,
        'crossover': p50['L'] < 0 < full['L'],
        'close': close,
        # with two seeds a close reading asks for batch 3 first; with three, the rule stands
        'third_seed_needed': close and min(nseeds.values()) < 3,
        'rule': ('1: L > S at 12 box-days' if pick == 'b16c256' else '1: L <= S at 12 box-days, the cheaper net'),
    }


def interval(values_by_size, ses_by_size):
    """A 95% interval for L: each size's mean over its seeds, its variance the seeds' (which
    holds the measurement noise too) over n, never below the mean measurement variance over n;
    t on the pooled seed degrees of freedom."""
    var, df = {}, 0
    for s in SIZES:
        v, se = values_by_size[s], ses_by_size[s]
        n = len(v)
        meas = statistics.fmean(x * x for x in se)
        seeds = statistics.variance(v) if n > 1 else 0.0
        var[s] = max(seeds, meas) / n
        df += n - 1
    L = statistics.fmean(values_by_size['b16c256']) - statistics.fmean(values_by_size['b12c256'])
    se = math.sqrt(var['b12c256'] + var['b16c256'])
    t = T975.get(df, 1.96)
    return {'L': L, 'se': se, 'df': df, 'ci95': [L - t * se, L + t * se], 'ci95_normal': [L - 1.96 * se, L + 1.96 * se]}


def rows_by_description(path=None):
    out = {}
    with open(path or RUNS_JSONL) as f:
        for line in f:
            r = json.loads(line)
            if r.get('test') == 'measure' and r.get('verdict') == 'measured':
                out[r['description']] = r
    return out


def snapshot_box_hours(run, label):
    with open(os.path.join(FLEET, run, 'snapshots.jsonl')) as f:
        for line in f:
            r = json.loads(line)
            if r['label'] == label:
                return r['box_hours']
    raise KeyError(f'{run}: no snapshot {label}')


def billed(run):
    sys.path.insert(0, os.path.join(ROOT, 'tools', 'fleet'))
    import fleet
    return round(fleet.cost(fleet.Run.load(run)), 2)


def stage_checks(run):
    out = {}
    for label in ('p25', 'p50', 'p75'):
        p = os.path.join(FLEET, run, f'stage-check-{label}.json')
        if os.path.exists(p):
            with open(p) as f:
                rec = json.load(f)
            out[label] = {k: rec.get(k) for k in ('verdict', 'fired', 'anchor_label')}
            out[label]['drift'] = {k: (rec.get('drift') or {}).get(k) for k in ('rkl', 'agreement', 'entropy')}
            out[label]['vs_anchor'] = (rec.get('measure') or {}).get('du'), (rec.get('measure') or {}).get('ci')
    return out


def main(argv=None):
    runs = list(argv if argv is not None else sys.argv[1:]) or list(DEFAULT_RUNS)
    rows = rows_by_description()
    per_run, missing = {}, []
    for run in runs:
        rec = {'size': size_of(run)}
        for label, days in READINGS.items():
            r = rows.get(f'curves: {run} {label} at 3x student')
            if r is None:
                missing.append(f'{run} {label}')
                continue
            res = r['result']
            bd = snapshot_box_hours(run, label) / 24
            rec[label] = {'run_id': r['run_id'], 'du': res['mean'], 'ci': res['ci'],
                          'se': res['sd'] / math.sqrt(res['sets']), 'box_days': round(bd, 3),
                          'per_dollar': per_dollar(res['mean'], bd), 'se_per_dollar': per_dollar(res['sd'] / math.sqrt(res['sets']), bd)}
        held = rows.get(f'curves: {run} full at 3x BC-s1')
        rec['heldout_bc_s1'] = None if held is None else {'run_id': held['run_id'], 'du': held['result']['mean'],
                                                          'ci': held['result']['ci']}
        rec['stage_checks'] = stage_checks(run)
        try:
            rec['billed_usd'] = billed(run)
        except Exception as e:  # noqa: BLE001 - reported, not deciding
            rec['billed_usd'] = f'unknown ({e!r})'
        per_run[run] = rec
    report = {'runs': per_run, 'missing': missing}
    if not missing:
        nseeds = {s: sum(1 for r in per_run.values() if r['size'] == s) for s in SIZES}
        fig = {label: {s: [r[label]['per_dollar'] for r in per_run.values() if r['size'] == s] for s in SIZES}
               for label in READINGS}
        report['decision'] = decide(fig, nseeds)
        report['interval_full'] = interval(
            {s: [r['full']['per_dollar'] for r in per_run.values() if r['size'] == s] for s in SIZES},
            {s: [r['full']['se_per_dollar'] for r in per_run.values() if r['size'] == s] for s in SIZES})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(report, f, indent=1)
    for run, r in per_run.items():
        cells = [f'{lab} {r[lab]["du"]:+.4f} [{r[lab]["ci"][0]:+.4f}, {r[lab]["ci"][1]:+.4f}]' for lab in READINGS if lab in r]
        print(f'{run}: ' + '; '.join(cells) + f'; billed {r["billed_usd"]}')
    if missing:
        print('missing measures: ' + ', '.join(missing) + ' (no reading)')
        return 1
    d = report['decision']
    for label, x in d['readings'].items():
        print(f'{label} ({READINGS[label]} box-days): L {x["L"]:+.3e} per $, S {x["S"]:.3e}, close {x["close"]}')
    print(f'pick {d["pick"]} (rule {d["rule"]}); crossover {d["crossover"]}; third seed needed {d["third_seed_needed"]}')
    print(f'L at 12 box-days, 95% interval {report["interval_full"]["ci95"]} (reported, not deciding)')
    print(f'written: {OUT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
