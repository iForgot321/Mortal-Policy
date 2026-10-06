"""D6 (04a §6): cross-fitted, time-decayed empirical-Bayes player scoring.

Runs AFTER the splits are frozen (D40) and uses `train` games only. Games
containing any BYE are excluded from skill statistics for all four seats
(04a §4.1); known bot seats (Q8) are excluded too.

Method (04a §6.2):
  1. Halves E/O by sha1(salt + log_id) mod 2 (the same halves as the split lists).
  2. Within a half, pool all years with decay weight 0.5^(age / half_life);
     half_life in {1, 2, 4, inf} years, chosen by out-of-sample fit.
  3. EB skill: s_hat = (n_eff·xbar + k·mu) / (n_eff + k), k = sigma_w² / sigma_b²,
     where n_eff is Kish's effective size of the decayed weights.
  4. Combined score per seat-game, fitted on E-skills -> O outcomes:
     u ~ c + a·s_hat(other half) + b·(R_at_start - 2150)/60, applied symmetrically.

Outputs: meta/player_scores.parquet, meta/seat_scores.parquet,
weights/A{1..5}.parquet (log_id, seat, weight) for the sweep arms (04a §6.3),
reports/player_scores.md.

Usage: python tools/data/score_players.py [--mj ~/mj-data]
"""
import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd
import toml

HERE = os.path.dirname(os.path.abspath(__file__))
HALF_LIVES = (1.0, 2.0, 4.0, float('inf'))
MIN_GAMES_SIGMA_B = 50
R_CENTER, R_SCALE = 2150.0, 60.0


def read_list(path):
    with open(path) as f:
        return [os.path.basename(line.strip())[:-len('.json.gz')] for line in f if line.strip()]


def eb_stats(df, weight_col):
    """Per-player decayed stats and the EB hyperparameters for one half."""
    df = df.assign(wu=df[weight_col] * df.u, w2=df[weight_col] ** 2)
    g = df.groupby('name')
    p = pd.DataFrame({
        'n': g.size(),
        'w': g[weight_col].sum(),
        'w2': g.w2.sum(),
        'wu': g.wu.sum(),
        'raw_mean': g.u.mean(),
    })
    p['xbar'] = p.wu / p.w
    p['n_eff'] = p.w ** 2 / p.w2  # Kish effective sample size
    mu = float(p.wu.sum() / p.w.sum())
    # pooled within-player variance (unweighted, ANOVA form)
    resid = df.u - df.name.map(p.raw_mean)
    sigma2_w = float((resid ** 2).sum() / (len(df) - len(p)))
    big = p[p.n >= MIN_GAMES_SIGMA_B]
    sigma2_b = float(max((big.n_eff * ((big.xbar - mu) ** 2 - sigma2_w / big.n_eff)).sum() / big.n_eff.sum(),
                         1e-6))
    k = sigma2_w / sigma2_b
    p['s_hat'] = (p.n_eff * p.xbar + k * mu) / (p.n_eff + k)
    return p, {'mu': mu, 'sigma2_w': sigma2_w, 'sigma2_b': sigma2_b, 'sigma_b': sigma2_b ** 0.5, 'k': k}


def cross_skill(seats, half_life, ref_days):
    """s_hat for each seat-game, estimated from the other half."""
    age_years = (ref_days - seats.day) / 365.25
    w = np.ones(len(seats)) if np.isinf(half_life) else 0.5 ** (age_years / half_life)
    seats = seats.assign(w=w)
    out = np.empty(len(seats))
    hyper = {}
    for half, other in (('E', 'O'), ('O', 'E')):
        p, h = eb_stats(seats[seats.half == other], 'w')
        hyper[f'from_{other}'] = h
        mask = (seats.half == half).to_numpy()
        out[mask] = seats.loc[mask, 'name'].map(p.s_hat).fillna(h['mu']).to_numpy()
    return out, hyper


def r2(y, x):
    x = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    return 1 - resid.var() / y.var(), beta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mj', default=os.path.expanduser('~/mj-data'))
    args = ap.parse_args()
    params = toml.load(os.path.join(HERE, 'params.toml'))
    salt = params['splits']['hash_salt']
    mj = args.mj
    splits = os.path.join(mj, 'splits')
    if not os.path.exists(os.path.join(splits, 'SPLITS.sha256')):
        raise SystemExit('splits are not frozen yet (D40: freeze before scoring)')

    train_e, train_o = set(read_list(os.path.join(splits, 'train_E.txt'))), \
        set(read_list(os.path.join(splits, 'train_O.txt')))
    games = pd.read_parquet(os.path.join(mj, 'meta', 'games.parquet'), columns=['log_id', 'bye_seats'])
    seats = pd.read_parquet(os.path.join(mj, 'meta', 'seats.parquet'),
                            columns=['log_id', 'seat', 'name', 'u', 'rate_at_start'])
    q = pd.read_parquet(os.path.join(mj, 'meta', 'quarantine.parquet'))
    bots = set(zip(q[q.code == 'Q8'].log_id, q[q.code == 'Q8'].seat))

    in_train = seats.log_id.isin(train_e | train_o)
    seats = seats[in_train].copy()
    seats['half'] = np.where(seats.log_id.isin(train_e), 'E', 'O')
    # sanity: halves agree with the hash rule used by splits.py
    probe = seats.drop_duplicates('log_id').head(1000)
    expect = probe.log_id.map(lambda x: 'E' if int(hashlib.sha1((salt + x).encode()).hexdigest(), 16) % 2 == 0 else 'O')
    assert (expect == probe.half).all(), 'train halves disagree with the hash rule'
    seats['day'] = pd.to_datetime(seats.log_id.str[:8], format='%Y%m%d').map(pd.Timestamp.toordinal)
    ref_days = pd.Timestamp(params['splits']['heldout_time_start']).toordinal() - 1

    bye_games = set(games[games.bye_seats.map(lambda b: b is not None and len(b) > 0)].log_id)
    is_bot = pd.Series([(l, s) in bots for l, s in zip(seats.log_id, seats.seat)], index=seats.index)
    stat = seats[~seats.log_id.isin(bye_games) & ~is_bot].copy()

    # 2. choose the half-life by out-of-sample fit (E->O and O->E pooled)
    fits = {}
    for hl in HALF_LIVES:
        s_hat, hyper = cross_skill(stat, hl, ref_days)
        fits[hl] = (r2(stat.u.to_numpy(), s_hat)[0], hyper)
    best_hl = max(fits, key=lambda h: fits[h][0])

    # 3–4. cross-fitted skill for every train seat-game, then the combined fit
    s_stat, hyper = cross_skill(stat, best_hl, ref_days)
    stat['s_hat'] = s_stat
    has_r = stat.rate_at_start.notna()
    fit_rows = stat[(stat.half == 'O') & has_r]
    x = np.column_stack([np.ones(len(fit_rows)), fit_rows.s_hat,
                         (fit_rows.rate_at_start - R_CENTER) / R_SCALE])
    (c, a, b), *_ = np.linalg.lstsq(x, fit_rows.u.to_numpy(), rcond=None)

    # apply to ALL train seats (BYE games included: they still need a weight),
    # with skills estimated from the other half's statistics games
    all_hat = np.empty(len(seats))
    for half, other in (('E', 'O'), ('O', 'E')):
        p, h = eb_stats(stat[stat.half == other].assign(
            w=np.ones((stat.half == other).sum()) if np.isinf(best_hl) else
            0.5 ** (((ref_days - stat[stat.half == other].day) / 365.25) / best_hl)), 'w')
        m = (seats.half == half).to_numpy()
        all_hat[m] = seats.loc[m, 'name'].map(p.s_hat).fillna(h['mu']).to_numpy()
    seats['s_hat'] = all_hat
    r_term = ((seats.rate_at_start - R_CENTER) / R_SCALE).fillna(0.0)
    seats['score'] = a * seats.s_hat + b * r_term
    seats['bot'] = is_bot

    # sweep arms (04a §6.3): seat-weighted cuts over train seat-games
    human = seats[~seats.bot]
    q20 = human.score.quantile(0.20)
    q40 = human.score.quantile(0.40)
    q60 = human.score.quantile(0.60)
    r40 = human.rate_at_start.quantile(0.40)
    z = (seats.score - human.score.mean()) / human.score.std()
    soft = np.clip(1 + 0.5 * z, 0.5, 1.5)
    soft = soft / soft[~seats.bot].mean()
    arms = {
        'A1': (seats.score > q20).astype(float),
        'A2': soft,
        'A3': (seats.score >= q40).astype(float),
        'A4': (seats.score >= q60).astype(float),
        'A5': (seats.rate_at_start >= r40).astype(float),
    }
    wdir = os.path.join(mj, 'weights')
    os.makedirs(wdir, exist_ok=True)
    arm_rows = []
    for name, w in arms.items():
        w = w.where(~seats.bot, 0.0)
        out = pd.DataFrame({'log_id': seats.log_id, 'seat': seats.seat.astype('int8'), 'weight': w.astype('float32')})
        out.to_parquet(os.path.join(wdir, f'{name}.parquet'), index=False)
        hw = w[~seats.bot]
        arm_rows.append((name, float(hw.mean()), float((hw * seats.u[~seats.bot]).sum() / hw.sum())))

    # outputs
    meta = os.path.join(mj, 'meta')
    seats[['log_id', 'seat', 'half', 's_hat', 'score']].to_parquet(os.path.join(meta, 'seat_scores.parquet'), index=False)
    prow = []
    for half in ('E', 'O'):
        sub = stat[stat.half == half]
        w = np.ones(len(sub)) if np.isinf(best_hl) else 0.5 ** (((ref_days - sub.day) / 365.25) / best_hl)
        p, h = eb_stats(sub.assign(w=w), 'w')
        prow.append(p.assign(half=half, k=h['k'], half_life=best_hl).reset_index())
    pd.concat(prow).to_parquet(os.path.join(meta, 'player_scores.parquet'), index=False)

    # checks (04a §6.4)
    rep = ['# Player scoring (04a §6)\n',
           f'Train seat-games: {len(seats):,} ({len(stat):,} used for skill statistics after excluding '
           f'{len(bye_games & set(seats.log_id)):,} BYE games and {int(is_bot.sum())} bot seats).\n',
           '## Half-life choice (pooled out-of-sample R², E→O and O→E)\n',
           '| half-life (years) | R² | k (from E) | σ_b (from E) |', '|---|---|---|---|']
    for hl, (val, hy) in fits.items():
        rep.append(f'| {hl} | {val:.6f} | {hy["from_E"]["k"]:.0f} | {hy["from_E"]["sigma_b"]:.4f} |')
    rep.append(f'\nChosen half-life: **{best_hl}** years.\n')
    rep.append(f'## Combined score\n\nu ~ {c:+.5f} + {a:.4f}·ŝ + {b:.5f}·(R − 2150)/60, fitted on '
               f'{len(fit_rows):,} O-half seat-games with E-half skills.\n')

    rep.append('## Per-year EB statistics (train, unweighted, all games of the year)\n')
    rep.append('| Year | Seat-games | σ_w | σ_b | k |')
    rep.append('|---|---|---|---|---|')
    stat['year'] = stat.log_id.str[:4]
    for year, sub in stat.groupby('year'):
        _, h = eb_stats(sub.assign(w=1.0), 'w')
        rep.append(f'| {year} | {len(sub):,} | {h["sigma2_w"] ** 0.5:.4f} | {h["sigma_b"]:.4f} | {h["k"]:.0f} |')

    rep.append('\n## Split-half reliability by games per half (players, corr of raw means)\n')
    rep.append('| min games per half | players | reliability |')
    rep.append('|---|---|---|')
    pe = stat[stat.half == 'E'].groupby('name').u.agg(['size', 'mean'])
    po = stat[stat.half == 'O'].groupby('name').u.agg(['size', 'mean'])
    both = pe.join(po, lsuffix='_e', rsuffix='_o', how='inner')
    nmin = both[['size_e', 'size_o']].min(axis=1)
    for lo, hi in ((200, 500), (500, 1000), (1000, 3000), (3000, None)):
        m = (nmin >= lo) & ((nmin < hi) if hi else True)
        sub = both[m]
        val = sub['mean_e'].corr(sub['mean_o']) if len(sub) > 2 else float('nan')
        rep.append(f'| {lo}–{hi or "∞"} | {len(sub)} | {val:.3f} |')

    rep.append('\n## Sweep arms: kept volume and out-of-sample mean u of kept seats\n')
    rep.append('| Arm | Mean weight (≈ kept share) | OOS mean u (weighted) |')
    rep.append('|---|---|---|')
    rep.append(f'| A0 | 1.000 | {seats.u[~seats.bot].mean():+.4f} |')
    rep.extend(f'| {n} | {m:.3f} | {u:+.4f} |' for n, m, u in arm_rows)
    rep.append('\nScores are cross-fitted, so each kept seat-game is judged only by the other half: '
               'the mean u above is out of sample (04a §6.1 reports ≈ +0.02 for an ideal top-25%).')
    with open(os.path.join(mj, 'reports', 'player_scores.md'), 'w') as f:
        f.write('\n'.join(rep) + '\n')
    with open(os.path.join(mj, 'reports', 'player_scores.json'), 'w') as f:
        json.dump({'half_life': best_hl, 'a': a, 'b': b, 'c': c, 'hyper': hyper}, f, indent=1, default=float)
    print('\n'.join(rep))


if __name__ == '__main__':
    main()
