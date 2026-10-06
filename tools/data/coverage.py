"""D5 (04a §5): coverage and meta audit -> reports/coverage.md.

Decides whether the default held-out-by-time window (params.toml) is a
contiguous, complete window of >= ~6 months and >= 100K games. If not, move it
and record D14b in docs/reference/08_decision_log.md.

Usage: python tools/data/coverage.py [--mj ~/mj-data]
"""
import argparse
import os

import pandas as pd
import toml

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mj', default=os.path.expanduser('~/mj-data'))
    args = ap.parse_args()
    params = toml.load(os.path.join(HERE, 'params.toml'))
    meta = os.path.join(args.mj, 'meta')

    games = pd.read_parquet(os.path.join(meta, 'games.parquet'),
                            columns=['log_id', 'year', 'month', 'n_kyoku', 'xml_found', 'bye_seats'])
    seats = pd.read_parquet(os.path.join(meta, 'seats.parquet'))
    seats = seats.merge(games[['log_id', 'year', 'n_kyoku']], on='log_id')

    out = ['# Coverage and meta audit (04a §5)\n']

    per_month = games.groupby(['year', 'month']).size().unstack(fill_value=0)
    per_month['total'] = per_month.sum(axis=1)
    out.append('## Games per month\n')
    out.append(per_month.to_markdown())
    out.append('')

    ym = games.year * 100 + games.month
    lo, hi = int(ym.min()), int(ym.max())
    all_months = [y * 100 + m for y in range(lo // 100, hi // 100 + 1) for m in range(1, 13)
                  if lo <= y * 100 + m <= hi]
    counts = ym.value_counts()
    missing = [m for m in all_months if m not in counts.index]
    out.append(f'Months covered: {lo} → {hi}; missing months: {missing or "none"}.\n')

    # held-out window check
    s = params['splits']
    start, end = s['heldout_time_start'].replace('-', ''), s['heldout_time_end'].replace('-', '')
    day = games.log_id.str[:8]
    window = games[(day >= start) & (day <= end)]
    w_months = window.groupby(window.year * 100 + window.month).size()
    prior = games[(day < start)]
    prior_months = prior.groupby(prior.year * 100 + prior.month).size().tail(12)
    median_prior = float(prior_months.median()) if len(prior_months) else float('nan')
    full_months = [m for m in w_months.index if m != int(end[:6])]  # last month may be partial by design
    thin = [int(m) for m in full_months if w_months[m] < 0.5 * median_prior]
    span_months = len(w_months)
    out.append('## Held-out-by-time window\n')
    out.append(f'- Window {s["heldout_time_start"]} → {s["heldout_time_end"]}: {len(window)} games over '
               f'{span_months} calendar months; per month {w_months.to_dict()}.')
    out.append(f'- Median monthly games over the prior 12 months: {median_prior:.0f}. '
               f'Months in the window below half of that: {thin or "none"}.')
    out.append(f'- Source-DB coverage in the window: {int(window.xml_found.sum())}/{len(window)} games '
               '(the golden set may only use games with DB coverage, 04a §7).')
    ok = len(window) >= 100_000 and span_months >= 6 and not thin and \
        all(m in w_months.index for m in all_months if start[:6] <= str(m) <= end[:6])
    out.append(f'- **Verdict: {"window OK (no D14b needed)" if ok else "window FAILS the 04a §5 rule: move it and record D14b"}.**\n')

    out.append('## Players\n')
    per_year = seats.groupby('year').agg(players=('name', 'nunique'), seats=('name', 'size'))
    gpp = seats.groupby(['year', 'name']).size()
    q = gpp.groupby(level=0).quantile([0.5, 0.9, 0.99]).unstack()
    q.columns = ['games/player p50', 'p90', 'p99']
    out.append(per_year.join(q).to_markdown())
    out.append('')

    out.append('## Per-year meta statistics (rates per player-kyoku)\n')
    agg = seats.groupby('year').agg(
        kyoku=('n_kyoku', 'sum'), riichi=('riichi', 'sum'), agari=('agari', 'sum'),
        houjuu=('houjuu', 'sum'), calls=('calls', 'sum'))
    meta_tab = pd.DataFrame({
        'riichi %': 100 * agg.riichi / agg.kyoku,
        'win %': 100 * agg.agari / agg.kyoku,
        'deal-in %': 100 * agg.houjuu / agg.kyoku,
        'calls per kyoku': agg.calls / agg.kyoku,
        'kyoku per game': games.groupby('year').n_kyoku.mean(),
        'games w/ BYE %': 100 * games.groupby('year').bye_seats.apply(
            lambda b: b.dropna().map(len).gt(0).sum() / max(len(b.dropna()), 1)),
    }).round(3)
    out.append(meta_tab.to_markdown())
    out.append('\n`calls per kyoku` counts chi/pon/daiminkan events per player-kyoku, not the share of '
               'kyoku with a call; the style panel (02 §7) defines its own call rate.\n')

    path = os.path.join(args.mj, 'reports', 'coverage.md')
    with open(path, 'w') as f:
        f.write('\n'.join(out))
    print('\n'.join(out[-12:]))
    print('->', path)


if __name__ == '__main__':
    main()
