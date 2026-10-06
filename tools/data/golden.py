"""Golden set (04a §7, 04 §4): a human-agreement diagnostic on held-out games.

Order (D40):
  1. Golden players = top 5% by player-level skill among players with >= 500
     `train` games (time-decayed EB, the half-life chosen by score_players.py),
     computed from `train` only. The list is written and hashed BEFORE any
     held-out decision is read.
  2. Decisions of golden seats in `heldout_time` games with source-DB coverage,
     outside Q6 disconnect windows, are tagged by category and sampled,
     stratified, deterministically by hash.

Outputs: splits/golden_players.txt, splits/golden.parquet, splits/GOLDEN.sha256,
reports/golden.md.

A decision is identified by (log_id, seat, entry_idx): entry_idx indexes the
seat's entries from GameplayLoader, so evaluation can featurize exactly that
entry. Kan-select sub-decisions are excluded.

Usage: python tools/data/golden.py [--mj ~/mj-data] [--per-category 20000]
"""
import argparse
import gzip
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'mortal'))
sys.path.insert(0, HERE)

from score_players import eb_stats, read_list  # noqa: E402

PRIORITY = ('riichi_dama', 'call_pass', 'push_fold', 'oorasu', 'early', 'routine')
TOP_SHARE = 0.05
MIN_GAMES = 500


def sha256_file(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def golden_players(mj):
    games = pd.read_parquet(os.path.join(mj, 'meta', 'games.parquet'), columns=['log_id', 'bye_seats'])
    seats = pd.read_parquet(os.path.join(mj, 'meta', 'seats.parquet'), columns=['log_id', 'seat', 'name', 'u'])
    q = pd.read_parquet(os.path.join(mj, 'meta', 'quarantine.parquet'))
    with open(os.path.join(mj, 'reports', 'player_scores.json')) as f:
        half_life = json.load(f)['half_life']
    train = set(read_list(os.path.join(mj, 'splits', 'train_E.txt'))) | \
        set(read_list(os.path.join(mj, 'splits', 'train_O.txt')))
    bye = set(games[games.bye_seats.map(lambda b: b is not None and len(b) > 0)].log_id)
    bots = set(zip(q[q.code == 'Q8'].log_id, q[q.code == 'Q8'].seat))
    s = seats[seats.log_id.isin(train) & ~seats.log_id.isin(bye)]
    s = s[[(l, t) not in bots for l, t in zip(s.log_id, s.seat)]]
    day = pd.to_datetime(s.log_id.str[:8], format='%Y%m%d').map(pd.Timestamp.toordinal)
    ref = day.max()
    w = np.ones(len(s)) if np.isinf(half_life) else 0.5 ** (((ref - day) / 365.25) / half_life)
    p, hyper = eb_stats(s.assign(w=w.to_numpy() if hasattr(w, 'to_numpy') else w), 'w')
    eligible = p[p.n >= MIN_GAMES].sort_values('s_hat', ascending=False)
    top = eligible.head(max(1, int(round(TOP_SHARE * len(eligible)))))
    return sorted(top.index), len(eligible), hyper, half_life


def kyoku_context(events):
    """Per event index: (is_all_last, frozenset of actors in riichi)."""
    all_last = np.zeros(len(events), dtype=bool)
    riichi = [frozenset()] * len(events)
    cur_last, cur_riichi = False, frozenset()
    for i, ev in enumerate(events):
        t = ev['type']
        if t == 'start_kyoku':
            cur_last = ev['bakaze'] == 'W' or (ev['bakaze'] == 'S' and ev['kyoku'] == 4)
            cur_riichi = frozenset()
        elif t == 'reach_accepted':
            cur_riichi = cur_riichi | {ev['actor']}
        all_last[i] = cur_last
        riichi[i] = cur_riichi
    return all_last, riichi


def tag_game(path, names, loader, masked_kyoku):
    from_log = []
    with gzip.open(path, 'rt') as f:
        events = [json.loads(line) for line in f if line.strip()]
    all_last, riichi = kyoku_context(events)
    log_id = os.path.basename(path)[:-len('.json.gz')]
    for gp in loader.load_gz_log_files([path])[0]:
        seat = gp.take_player_id()
        if events[0]['names'][seat] not in names:
            continue
        actions = gp.take_actions()
        masks = [m for m in gp.take_masks()]
        at_kyoku = gp.take_at_kyoku()
        turns = gp.take_at_turns()
        shantens = gp.take_shantens()
        ev_idx = gp.take_event_idx()
        prev_ev, prev_act = None, None
        for j, (a, m, k, turn, sh, e) in enumerate(zip(actions, masks, at_kyoku, turns, shantens, ev_idx)):
            kan_select = e == prev_ev and prev_act == 42
            prev_ev, prev_act = e, a
            if kan_select or k in masked_kyoku:
                continue
            m = np.asarray(m)
            discard = bool(m[:37].any())
            opp_riichi = bool(riichi[e] - {seat})
            flags = {
                'riichi_dama': bool(m[37]),
                'call_pass': bool(m[45] and m[38:43].any()),
                'push_fold': discard and opp_riichi and sh >= 1,
                'oorasu': bool(all_last[e]),
                'early': discard and turn <= 6,
            }
            flags['routine'] = discard and not any(flags[c] for c in ('riichi_dama', 'push_fold', 'oorasu', 'early'))
            primary = next((c for c in PRIORITY if flags[c]), None)
            if primary is None:
                continue
            from_log.append({
                'log_id': log_id, 'seat': seat, 'entry_idx': j, 'event_idx': int(e),
                'category': primary, **flags, 'action': int(a), 'at_turn': int(turn), 'shanten': int(sh),
                'mask_hash': hashlib.sha1(np.packbits(m).tobytes()).hexdigest()[:16],
            })
    return from_log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mj', default=os.path.expanduser('~/mj-data'))
    ap.add_argument('--per-category', type=int, default=20_000)
    args = ap.parse_args()
    mj = args.mj
    sp = os.path.join(mj, 'splits')
    if os.path.exists(os.path.join(sp, 'GOLDEN.sha256')):
        sys.exit('golden set already frozen (GOLDEN.sha256 exists)')

    # 1. players, frozen before any held-out decision is read
    players, n_eligible, hyper, half_life = golden_players(mj)
    plist = os.path.join(sp, 'golden_players.txt')
    with open(plist, 'w', encoding='utf-8') as f:
        f.writelines(p + '\n' for p in players)
    players_hash = sha256_file(plist)
    print(f'golden players: {len(players)} of {n_eligible} eligible; list sha256 {players_hash}', flush=True)

    # 2. decisions
    from libriichi.dataset import GameplayLoader
    games = pd.read_parquet(os.path.join(mj, 'meta', 'games.parquet'), columns=['log_id', 'xml_found', 'names'])
    held = set(read_list(os.path.join(sp, 'heldout_time.txt')))
    names = set(players)
    g = games[games.log_id.isin(held) & games.xml_found]
    g = g[g.names.map(lambda ns: ns is not None and bool(names & set(ns)))]
    masks = {}
    with open(os.path.join(sp, 'masks.jsonl')) as f:
        for line in f:
            row = json.loads(line)
            masks[row['log_id']] = set(row['kyoku'])
    loader = GameplayLoader(version=4, oracle=False, player_names=sorted(names), featurize=False)
    rows = []
    for i, lid in enumerate(sorted(g.log_id)):
        path = os.path.join(mj, 'mjai', lid[:4], lid[4:6], lid + '.json.gz')
        rows.extend(tag_game(path, names, loader, masks.get(lid, set())))
        if (i + 1) % 2000 == 0:
            print(f'  tagged {i + 1}/{len(g)} games, {len(rows):,} decisions', flush=True)
    df = pd.DataFrame(rows)

    # stratified, deterministic sample
    df['h'] = [hashlib.sha1(f'golden:{a}:{b}:{c}'.encode()).hexdigest() for a, b, c in
               zip(df.log_id, df.seat, df.entry_idx)]
    sample = (df.sort_values('h').groupby('category', group_keys=False)
              .head(args.per_category).drop(columns='h').sort_values(['log_id', 'seat', 'entry_idx']))
    out = os.path.join(sp, 'golden.parquet')
    sample.to_parquet(out, index=False)
    with open(os.path.join(sp, 'GOLDEN.sha256'), 'w') as f:
        f.write(f'{players_hash}  golden_players.txt\n{sha256_file(out)}  golden.parquet\n')

    rep = ['# Golden set (04a §7)\n',
           f'- Golden players: {len(players)} (top {TOP_SHARE:.0%} of {n_eligible} players with ≥ {MIN_GAMES} '
           f'train games; time-decayed EB skill, half-life {half_life} y, k = {hyper["k"]:.0f}). '
           f'List hashed before any held-out decision was read: `{players_hash}`.',
           f'- Source: {len(g):,} heldout_time games with DB coverage and ≥ 1 golden seat.',
           f'- Tagged decisions: {len(df):,}; sampled {len(sample):,} (≤ {args.per_category:,} per category).\n',
           '| Category | Tagged | Sampled |', '|---|---|---|']
    for c in PRIORITY:
        rep.append(f'| {c} | {(df.category == c).sum():,} | {(sample.category == c).sum():,} |')
    rep.append('\nCategories are assigned by priority (riichi_dama > call_pass > push_fold > oorasu > early > '
               'routine); every row also carries all flags.')
    with open(os.path.join(mj, 'reports', 'golden.md'), 'w') as f:
        f.write('\n'.join(rep) + '\n')
    print('\n'.join(rep))


if __name__ == '__main__':
    main()
