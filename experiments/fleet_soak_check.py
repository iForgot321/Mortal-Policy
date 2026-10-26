"""The fleet soak's criteria (03 §3.7, D39, D60), judged from a fleet run's own records: the
watcher's health log (every 5 min), its events, and the learner's TensorBoard (pulled every 30 min).
Pre-registered in experiments/fleet-soak.md.

    python experiments/fleet_soak_check.py --run soak-a --mode final --out workdir/fleet/runs/soak-a/soak-final.json
    # as `fleet.py watch --early-check` (the hook adds --ckpt and --out at the p25 snapshot, ≈ 12 h in):
    python experiments/fleet_soak_check.py --run soak-a --mode early

`final` (pass/fail; windows end at the last health record):
  1. replay queue bounded: the least-squares line through the learner's queue length over the last
     24 h rises by less than 25% of the queue's capacity over those 24 h (the queue swings between
     0 and capacity as batches land and drain; `k2b`'s last 24 h rose 8%);
  2. policy lag (D60): p95 of the trainer's per-interval mean lag ≤ 2 versions over the last 24 h,
     or learning stable over the last 24 h: the PPO ratio's p99 ≤ 2 in ≥ 95% of intervals, the
     least-squares line through the mean lag rises by less than 1 version over the 24 h, and no
     `anchor_alert` during the soak;
  3. useful updates: games the learner drained ≥ 90% of games the actors produced, and actors
     blocked < 10% of wall-clock, both over the soak after its first hour;
  4. box loss: every loss is recovered (event `recovered`) within 30 min, and none failed.

`early` (stop/continue, over the last 6 h; only clear failures stop the run):
  actors blocked > 25%, or the learner drains < 75% of production, or the ratio p99 > 2 in > 20%
  of intervals, or an `anchor_alert`, or a `recovery_failed`, or a recovery after more than 60 min.
"""
import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ts(iso):
    return datetime.fromisoformat(iso).timestamp()


def health(run_dir):
    """Per health record: time, the learner's queue and drained games/h, the actors' games/h and
    blocked seconds/h (summed over their processes), and the number of actor processes reporting."""
    out = []
    with open(os.path.join(run_dir, 'health.jsonl')) as f:
        for line in f:
            r = json.loads(line)
            row = {'t': ts(r['time']), 'queue': None, 'drained': None, 'games': 0, 'blocked_s': 0, 'procs': 0}
            for b in r['boxes'].values():
                probe = b.get('probe') or {}
                for name, p in (probe.get('procs') or {}).items():
                    if name == 'server' and p.get('queue') is not None:
                        row['queue'], row['drained'] = p['queue'], p.get('drained_1h')
                    if name.startswith('client') and p.get('games_1h') is not None:
                        row['games'] += p['games_1h']
                        row['blocked_s'] += p.get('blocked_s_1h') or 0
                        row['procs'] += 1
            out.append(row)
    return out


def events(run_dir):
    with open(os.path.join(run_dir, 'events.jsonl')) as f:
        return [json.loads(line) for line in f]


def scalars(run_dir, tags):
    """The learner's TensorBoard scalars as {tag: (wall_times, values)}."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(os.path.join(run_dir, 'learner-run', 'tb'), size_guidance={'scalars': 0})
    ea.Reload()
    have = set(ea.Tags()['scalars'])
    return {t: (np.array([e.wall_time for e in ea.Scalars(t)]), np.array([e.value for e in ea.Scalars(t)]))
            for t in tags if t in have}


def rise(t, y, hours):
    """The least-squares line's rise over `hours`."""
    if len(t) < 3:
        return float('nan')
    return float(np.polyfit((t - t[0]) / 3600, y, 1)[0] * hours)


def window(rows, start, end):
    return [r for r in rows if start <= r['t'] <= end]


def judge(run_dir, mode, capacity):
    rows = health(run_dir)
    ev = events(run_dir)
    t0, t_end = rows[0]['t'], rows[-1]['t']
    span = 6 if mode == 'early' else 24
    w = window(rows, t_end - span * 3600, t_end)
    soak = window(rows, t0 + 3600, t_end) if mode == 'final' else w
    tb = scalars(run_dir, ('ppo/lag_mean', 'ppo/ratio_p99', 'anchor/kl', 'anchor/agreement'))

    def recent(tag):
        if tag not in tb:
            return np.array([]), np.array([])
        t, v = tb[tag]
        keep = t >= t_end - span * 3600
        return t[keep], v[keep]

    produced = sum(r['games'] for r in soak)
    drained = sum(r['drained'] or 0 for r in soak if r['queue'] is not None)
    proc_hours = sum(r['procs'] for r in soak)
    blocked = sum(r['blocked_s'] for r in soak) / (3600 * proc_hours) if proc_hours else float('nan')
    q = [(r['t'], r['queue']) for r in w if r['queue'] is not None]
    lag_t, lag = recent('ppo/lag_mean')
    _, ratio = recent('ppo/ratio_p99')
    alerts = [e for e in ev if e['kind'] == 'anchor_alert']
    recovered = [e for e in ev if e['kind'] == 'recovered']
    failed = [e for e in ev if e['kind'] == 'recovery_failed']
    m = {
        'hours': round((t_end - t0) / 3600, 1), 'window_h': span,
        'games_produced_per_h': round(produced / max(len(soak), 1)),
        'drained_over_produced': round(drained / produced, 3) if produced else float('nan'),
        'actors_blocked': round(blocked, 4),
        'queue_rise_over_window': round(rise(np.array([a for a, _ in q]), np.array([b for _, b in q], dtype=float), span), 1),
        'queue_capacity': capacity,
        'queue_mean': round(float(np.mean([b for _, b in q])), 1) if q else float('nan'),
        'lag_mean_p95': round(float(np.percentile(lag, 95)), 2) if len(lag) else float('nan'),
        'lag_rise_over_window': round(rise(lag_t, lag, span), 2),
        'ratio_p99_share_le_2': round(float(np.mean(ratio <= 2)), 4) if len(ratio) else float('nan'),
        'ratio_p99_max': round(float(ratio.max()), 3) if len(ratio) else float('nan'),
        'anchor_alerts': len(alerts),
        'recoveries': [e.get('after_s') for e in recovered], 'recovery_failures': len(failed),
    }
    fired = []
    if mode == 'early':
        if m['actors_blocked'] > 0.25:
            fired.append('actors blocked > 25%')
        if m['drained_over_produced'] < 0.75:
            fired.append('learner drains < 75% of production')
        if len(ratio) and m['ratio_p99_share_le_2'] < 0.80:
            fired.append('ratio p99 > 2 in > 20% of intervals')
        if alerts:
            fired.append('anchor alert')
        if failed or any((s or 0) > 3600 for s in m["recoveries"]):
            fired.append('a box loss not recovered within 60 min')
        verdict = 'stop' if fired else 'continue'
    else:
        if not m['queue_rise_over_window'] < 0.25 * capacity:
            fired.append('1: queue rises ≥ 25% of capacity over the last 24 h')
        lag_ok = m['lag_mean_p95'] <= 2
        stable = (m['ratio_p99_share_le_2'] >= 0.95 and m['lag_rise_over_window'] < 1 and not alerts)
        if not (lag_ok or stable):
            fired.append('2: lag p95 > 2 and learning not stable (D60)')
        if not m['drained_over_produced'] >= 0.90:
            fired.append('3: learner drained < 90% of production')
        if not m['actors_blocked'] < 0.10:
            fired.append('3: actors blocked ≥ 10%')
        if failed or any((s or 0) > 1800 for s in m["recoveries"]):
            fired.append('4: a box loss not recovered within 30 min')
        m['lag_criterion'] = 'p95 ≤ 2' if lag_ok else ('stable (D60)' if stable else 'failed')
        verdict = 'fail' if fired else 'pass'
    # the watcher's early-check hook records these two (as K2's check reports drift)
    _, kl = recent('anchor/kl')
    _, agree = recent('anchor/agreement')
    drift = {'rkl': float(kl.mean()) if len(kl) else float('nan'),
             'agreement': float(agree.mean()) if len(agree) else float('nan')}
    return {'run': os.path.basename(run_dir), 'mode': mode, 'verdict': verdict, 'fired': fired,
            'drift': drift, 'argmax': None, 'measures': m}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True, help='the fleet run (workdir/fleet/runs/<run>)')
    ap.add_argument('--mode', choices=('early', 'final'), required=True)
    ap.add_argument('--capacity', type=int, default=None, help="the replay queue's capacity (default: the run's)")
    ap.add_argument('--ckpt', help="ignored: the watcher's early-check hook passes the snapshot")
    ap.add_argument('--out')
    a = ap.parse_args()
    run_dir = os.path.join(ROOT, 'workdir', 'fleet', 'runs', a.run)
    with open(os.path.join(run_dir, 'fleet.json')) as f:
        capacity = a.capacity or json.load(f).get('capacity') or 1600
    res = judge(run_dir, a.mode, capacity)
    text = json.dumps(res, indent=1, ensure_ascii=False)
    print(text)
    if a.out:
        with open(a.out + '.tmp', 'w') as f:
            f.write(text)
        os.replace(a.out + '.tmp', a.out)  # the watcher reads it as soon as it exists


if __name__ == '__main__':
    sys.exit(main())
