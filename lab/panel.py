"""Style panel (02 §7) and per-seat table (02 §6) from a lab run's mjai logs.

Uses libriichi's Stat on the challenger's seat of every game, plus a replay of
that seat with PlayerState for the metrics Stat lacks (honitsu and mangan+
shares, first-discard rates, shanten-backward rates, 4th rate as oorasu
leader). The seat comes from the log name (`<nonce>_<key>_<a|b|c|d>.json.gz`
= seat 0..3 in 1v3), so this works whatever the engines are named.

Golden-set agreement is computed separately (it needs model inference).
"""
import glob
import gzip
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

from . import MORTAL_DIR

if MORTAL_DIR not in sys.path:
    sys.path.insert(0, MORTAL_DIR)

SEATS = 'ESWN'
STAT_FIELDS = ('game', 'round', 'agari', 'houjuu', 'fuuro', 'riichi', 'dama_agari', 'tobi',
               'rank_1', 'rank_2', 'rank_3', 'rank_4', 'agari_point_oya', 'agari_point_ko',
               'houjuu_point_to_oya', 'houjuu_point_to_ko')
REPLAY_FIELDS = ('wins', 'honitsu', 'mangan_plus', 'first_discards', 'first_honor', 'first_2_8',
                 'quiet_discards', 'quiet_backward', 'facing_discards', 'facing_backward',
                 'oorasu_lead', 'oorasu_lead_4th')
FIELDS = STAT_FIELDS + REPLAY_FIELDS
HONORS = {'E', 'S', 'W', 'N', 'P', 'F', 'C'}


def _suit(tid):
    return 'mps'[tid // 9] if tid < 27 else 'z'


def _tile_suit(pai):
    return 'z' if pai in HONORS else pai[1]


def _replay(lines, seat, acc):
    """Replay one game from `seat`'s view and add the replay-only counters."""
    from libriichi.dataset import Grp
    from libriichi.state import PlayerState
    ps = PlayerState(seat)
    events = [json.loads(line) for line in lines]
    final_rank = Grp.load_log('\n'.join(lines)).take_rank_by_player()[seat]
    oorasu_seen = False
    for line, ev in zip(lines, events):
        t = ev['type']
        if t == 'start_kyoku':
            oya, honba, kyotaku = ev['oya'], ev['honba'], ev['kyotaku']
            first_done, riichi_opp, first_hora, last_pai = False, set(), True, None
            if ev['bakaze'] == 'S' and ev['kyoku'] == 4 and not oorasu_seen:
                oorasu_seen = True
                scores = ev['scores']
                leader = min(range(4), key=lambda i: (-scores[i], i))
                if leader == seat:
                    acc['oorasu_lead'] += 1
                    acc['oorasu_lead_4th'] += final_rank == 3
        elif t == 'reach_accepted' and ev['actor'] != seat:
            riichi_opp.add(ev['actor'])
        elif t in ('dahai', 'kakan'):
            last_pai = ev['pai']
        if t == 'dahai' and ev['actor'] == seat:
            if not first_done:
                first_done = True
                acc['first_discards'] += 1
                pai = ev['pai']
                acc['first_honor'] += pai in HONORS
                acc['first_2_8'] += pai not in HONORS and 2 <= int(pai[0]) <= 8
            closed = not (ps.chis or ps.pons or ps.minkans)
            if closed and not ps.self_riichi_declared:
                before = ps.shanten
                ps.update(line)
                backward = ps.shanten > before
                key = 'facing' if riichi_opp else 'quiet'
                acc[f'{key}_discards'] += 1
                acc[f'{key}_backward'] += backward
                continue
        if t == 'hora' and ev['actor'] == seat:
            acc['wins'] += 1
            suits = {_suit(tid) for tid, n in enumerate(ps.tehai) if n}
            suits |= {_suit(tid) for meld in (ps.chis, ps.pons, ps.minkans, ps.ankans) for tid in meld}
            if ev['target'] != seat and last_pai is not None:
                suits.add(_tile_suit(last_pai))
            acc['honitsu'] += 'z' in suits and len(suits - {'z'}) == 1
            gain = ev['deltas'][seat] - 300 * honba - (1000 * kyotaku if first_hora else 0)
            acc['mangan_plus'] += gain >= (12000 if oya == seat else 8000)
            first_hora = False
        ps.update(line)


def _counts(paths):
    from libriichi.stat import Stat
    acc = {s: dict.fromkeys(FIELDS, 0) for s in range(4)}
    for path in paths:
        seat = 'abcd'.index(path[-len('a.json.gz')])
        with gzip.open(path, 'rt') as f:
            raw = f.read()
        stat = Stat.from_log(raw, seat)
        for k in STAT_FIELDS:
            acc[seat][k] += getattr(stat, k)
        _replay([line for line in raw.splitlines() if line.strip()], seat, acc[seat])
    return acc


def metrics(c):
    rounds, agari, houjuu, games = max(c['round'], 1), max(c['agari'], 1), max(c['houjuu'], 1), max(c['game'], 1)
    ranks = [c['rank_1'], c['rank_2'], c['rank_3'], c['rank_4']]
    return {
        'games': int(c['game']),
        'kyoku': int(c['round']),
        'avg_rank': sum((i + 1) * r for i, r in enumerate(ranks)) / games,
        'rank_dist': [r / games for r in ranks],
        'win_pct': 100 * c['agari'] / rounds,
        'deal_in_pct': 100 * c['houjuu'] / rounds,
        'call_pct': 100 * c['fuuro'] / rounds,
        'riichi_pct': 100 * c['riichi'] / rounds,
        'dama_share_of_wins_pct': 100 * c['dama_agari'] / agari,
        'avg_win': (c['agari_point_oya'] + c['agari_point_ko']) / agari,
        'avg_deal_in': (c['houjuu_point_to_oya'] + c['houjuu_point_to_ko']) / houjuu,
        'tobi_pct': 100 * c['tobi'] / games,
        'honitsu_share_of_wins_pct': 100 * c['honitsu'] / max(c['wins'], 1),
        'mangan_plus_share_of_wins_pct': 100 * c['mangan_plus'] / max(c['wins'], 1),
        'first_discard_honor_pct': 100 * c['first_honor'] / max(c['first_discards'], 1),
        'first_discard_2_8_pct': 100 * c['first_2_8'] / max(c['first_discards'], 1),
        'shanten_backward_quiet_pct': 100 * c['quiet_backward'] / max(c['quiet_discards'], 1),
        'shanten_backward_facing_riichi_pct': 100 * c['facing_backward'] / max(c['facing_discards'], 1),
        'fourth_as_oorasu_leader_pct': 100 * c['oorasu_lead_4th'] / max(c['oorasu_lead'], 1),
    }


def panel(log_dir, workers=8):
    """(overall metrics, {seat: metrics}) for the challenger."""
    paths = sorted(glob.glob(os.path.join(log_dir, '*.json.gz')))
    chunks = [paths[i::workers] for i in range(workers)]
    total = {s: dict.fromkeys(FIELDS, 0) for s in range(4)}
    with ProcessPoolExecutor(workers) as ex:
        for acc in ex.map(_counts, chunks):
            for s in range(4):
                for k in FIELDS:
                    total[s][k] += acc[s][k]
    overall = {k: sum(total[s][k] for s in range(4)) for k in FIELDS}
    return metrics(overall), {SEATS[s]: metrics(total[s]) for s in range(4)}


def format_table(overall, seats=None):
    keys = ['games', 'avg_rank', 'win_pct', 'deal_in_pct', 'call_pct', 'riichi_pct',
            'dama_share_of_wins_pct', 'avg_win', 'avg_deal_in', 'honitsu_share_of_wins_pct',
            'mangan_plus_share_of_wins_pct', 'first_discard_honor_pct', 'first_discard_2_8_pct',
            'shanten_backward_quiet_pct', 'shanten_backward_facing_riichi_pct', 'tobi_pct',
            'fourth_as_oorasu_leader_pct']
    cols = ['all'] + (list(seats) if seats else [])
    rows = [f'| metric | {" | ".join(cols)} |', '|---' * (len(cols) + 1) + '|']
    for k in keys:
        vals = [overall[k]] + ([seats[s][k] for s in seats] if seats else [])
        rows.append(f'| {k} | ' + ' | '.join(f'{v:.4g}' if isinstance(v, float) else str(v) for v in vals) + ' |')
    return '\n'.join(rows)
