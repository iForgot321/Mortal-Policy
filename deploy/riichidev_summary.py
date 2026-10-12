"""Daily summary of play on riichi.dev (07 §1 operations), from the runner's summaries.jsonl and
game logs: per UTC day, games, placements, mean rank and u, timeouts, chombo (actions the server
rejected or couldn't parse), other failures; and decision latency per class over the window.

The rating isn't in the game messages: read it from riichi.dev/leaderboard. Ranks come from the
final scores, ties going to the seat nearer the first dealer (Tenhou's rule).

Usage, from the repo root:
    python3 -m deploy.riichidev_summary [--mode ranked] [--days 7] [--summaries PATH]
"""
import argparse
import datetime
import json
import os
import sys

from . import RUNS_DIR
from .riichidev_bot import latency_summary

U = (0.9, 0.5, -0.1, -1.3)  # utility per rank, u (D2)
CHOMBO = ('ack_rejected', 'ack_unparseable')


def rank_of(scores, seat):
    return 1 + sum(s > scores[seat] or (s == scores[seat] and j < seat) for j, s in enumerate(scores))


def day_of(summary):
    return datetime.datetime.fromtimestamp(summary.get('end') or summary['start'], datetime.timezone.utc).date()


def latencies(summary):
    """{class: [ms]} from the game log; empty if the log is gone."""
    out = {}
    try:
        with open(summary['log'], encoding='utf-8') as f:
            for line in f:
                rec = json.loads(line)
                if rec.get('dir') == 'decision' and 'class' in rec:
                    out.setdefault(rec['class'], []).append(rec['ms'])
    except (OSError, KeyError):
        pass
    return out


def summarize(summaries):
    """Per-day rows and the window's latency, from summary dicts."""
    days, by_class = {}, {}
    for s in summaries:
        d = days.setdefault(day_of(s), {'games': 0, 'ranks': [0, 0, 0, 0], 'unfinished': 0,
                                        'timeouts': 0, 'chombo': 0, 'failures': 0})
        d['games'] += 1
        scores = (s.get('result') or {}).get('scores')
        if scores and s.get('seat') is not None:
            d['ranks'][rank_of(scores, s['seat']) - 1] += 1
        else:
            d['unfinished'] += 1
        d['timeouts'] += s.get('timeouts', 0)
        kinds = [f.get('kind') for f in s.get('failures', [])]
        d['chombo'] += sum(k in CHOMBO for k in kinds)
        d['failures'] += sum(k not in CHOMBO for k in kinds)
        for cls, ms in latencies(s).items():
            by_class.setdefault(cls, []).extend(ms)
    for d in days.values():
        n = sum(d['ranks'])
        d['mean_rank'] = sum((i + 1) * c for i, c in enumerate(d['ranks'])) / n if n else None
        d['mean_u'] = sum(U[i] * c for i, c in enumerate(d['ranks'])) / n if n else None
    return dict(sorted(days.items())), latency_summary(by_class) if by_class else {}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--summaries', default=os.path.join(RUNS_DIR, 'summaries.jsonl'))
    p.add_argument('--mode', default='ranked', help="'ranked', 'validate' or 'all'")
    p.add_argument('--days', type=int, default=7)
    a = p.parse_args(argv)
    if not os.path.exists(a.summaries):
        raise SystemExit(f'no {a.summaries}')
    with open(a.summaries, encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    since = datetime.datetime.now(datetime.timezone.utc).date() - datetime.timedelta(days=a.days - 1)
    rows = [r for r in rows if (a.mode == 'all' or r.get('mode') == a.mode) and day_of(r) >= since]
    days, latency = summarize(rows)
    print(f'riichi.dev, {a.mode}, {since} to today (UTC); the rating is on riichi.dev/leaderboard\n')
    print('| Day | Games | 1st/2nd/3rd/4th | Mean rank | Mean u | Unfinished | Timeouts | Chombo | Other failures |')
    print('|---|---|---|---|---|---|---|---|---|')
    for day, d in days.items():
        mr = '' if d['mean_rank'] is None else f'{d["mean_rank"]:.3f}'
        mu = '' if d['mean_u'] is None else f'{d["mean_u"]:+.3f}'
        print(f'| {day} | {d["games"]} | {"/".join(map(str, d["ranks"]))} | {mr} | {mu} | {d["unfinished"]} | '
              f'{d["timeouts"]} | {d["chombo"]} | {d["failures"]} |')
    if latency:
        print('\n| Decision | n | p50 ms | p99 ms | max ms |\n|---|---|---|---|---|')
        for cls, s in latency.items():
            print(f'| {cls} | {s["n"]} | {s["p50"]} | {s["p99"]} | {s["max"]} |')
    return 0


if __name__ == '__main__':
    sys.exit(main())
