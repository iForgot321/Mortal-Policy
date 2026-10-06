"""D7 (04a §7): freeze the splits. Runs BEFORE player scoring (D40).

Writes, under splits/:
  heldout_time.txt, heldout_game.txt, train_E.txt, train_O.txt,
  train_ablate_2009.txt   game lists, paths relative to mjai/, sorted
  masks.jsonl             per-game Q6 kyoku masks and Q8 seat exclusions for
                          every listed game that has any
  SPLITS.sha256           sha256 of every list
and reports/splits.md with the leak tests (all must pass) and split sizes.

The golden set is built later (golden.py), after scoring on train only.

Usage: python tools/data/splits.py [--mj ~/mj-data]
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import toml

HERE = os.path.dirname(os.path.abspath(__file__))


def hash_int(salt, log_id):
    return int(hashlib.sha1((salt + log_id).encode()).hexdigest(), 16)


def rel_path(log_id):
    return f'{log_id[:4]}/{log_id[4:6]}/{log_id}.json.gz'


def sha256_file(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mj', default=os.path.expanduser('~/mj-data'))
    args = ap.parse_args()
    params = toml.load(os.path.join(HERE, 'params.toml'))
    s, qp = params['splits'], params['quarantine']
    meta = os.path.join(args.mj, 'meta')
    out_dir = os.path.join(args.mj, 'splits')
    if os.path.exists(os.path.join(out_dir, 'SPLITS.sha256')):
        sys.exit(f'{out_dir} is already frozen (SPLITS.sha256 exists); refusing to overwrite')
    os.makedirs(out_dir, exist_ok=True)

    games = pd.read_parquet(os.path.join(meta, 'games.parquet'), columns=['log_id', 'n_decisions'])
    q = pd.read_parquet(os.path.join(meta, 'quarantine.parquet'))

    drop_codes = set(qp['drop_games']) | ({'Q4'} if qp['q4'] == 'drop' else set())
    game_level = q[(q.seat == -1) & (q.kyoku == -1)]
    dropped = set(game_level[game_level.code.isin(drop_codes)].log_id) \
        | set(game_level[game_level.code == 'Q6'].log_id)

    lid = games.log_id
    day = lid.str[:8]
    h = np.array([hash_int(s['hash_salt'], x) for x in lid], dtype=object)
    ok = ~lid.isin(dropped)
    start = s['heldout_time_start'].replace('-', '')
    end = s['heldout_time_end'].replace('-', '')
    in_time = (day >= start) & (day <= end)
    in_game = ~in_time & np.array([x % s['heldout_game_mod'] < s['heldout_game_lt'] for x in h])
    pre = day < start
    train = ok & pre & (day >= s['train_start'].replace('-', '')) & ~in_game
    ablate = ok & pre & (day >= s['ablate_start'].replace('-', '')) & ~in_game
    half_e = np.array([x % 2 == 0 for x in h])

    lists = {
        'heldout_time': games[ok & in_time],
        'heldout_game': games[ok & in_game],
        'train_E': games[train & half_e],
        'train_O': games[train & ~half_e],
        'train_ablate_2009': games[ablate],
    }

    listed = set().union(*(set(df.log_id) for df in lists.values()))
    masks = {}
    for row in q[q.log_id.isin(listed)].itertuples():
        if row.code == 'Q6' and row.kyoku >= 0:
            masks.setdefault(row.log_id, {'kyoku': [], 'seats': []})['kyoku'].append(int(row.kyoku))
        elif row.code == 'Q8':
            masks.setdefault(row.log_id, {'kyoku': [], 'seats': []})['seats'].append(int(row.seat))

    # leak tests (04a §7)
    tests = []
    disjoint = ['heldout_time', 'heldout_game', 'train_E', 'train_O']
    sets = {k: set(lists[k].log_id) for k in lists}
    for i, a in enumerate(disjoint):
        for b in disjoint[i + 1:]:
            tests.append((f'no log_id in both {a} and {b}', not (sets[a] & sets[b])))
    tests.append(('train_ablate_2009 contains no held-out game',
                  not (sets['train_ablate_2009'] & (sets['heldout_time'] | sets['heldout_game']))))
    train_days = pd.concat([lists['train_E'], lists['train_O'], lists['train_ablate_2009']]).log_id.str[:10]
    time_days = lists['heldout_time'].log_id.str[:10]
    tests.append(('max(train.ts) < min(heldout_time.ts)', train_days.max() < time_days.min()))
    tests.append(('no quarantined (dropped) game in any list', not (listed & dropped)))
    failed = [name for name, passed in tests if not passed]

    for name, df in lists.items():
        with open(os.path.join(out_dir, f'{name}.txt'), 'w') as f:
            f.writelines(rel_path(x) + '\n' for x in sorted(df.log_id))
    with open(os.path.join(out_dir, 'masks.jsonl'), 'w') as f:
        for k in sorted(masks):
            m = masks[k]
            f.write(json.dumps({'log_id': k, 'kyoku': sorted(set(m['kyoku'])),
                                'seats': sorted(set(m['seats']))}) + '\n')

    rep = ['# Splits (04a §7)\n', f'Frozen {pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")}, '
           'before player scoring (D40).\n', '| Split | Games | Decisions (all seats) | Date range |',
           '|---|---|---|---|']
    for name, df in lists.items():
        days = df.log_id.str[:8]
        rep.append(f'| {name} | {len(df):,} | {int(df.n_decisions.sum()):,} | '
                   f'{days.min() if len(df) else "-"} → {days.max() if len(df) else "-"} |')
    rep.append(f'\nGames dropped by quarantine: {len(dropped):,} (codes {sorted(drop_codes)} plus whole-game Q6). '
               f'Games with loader masks (Q6 kyoku / Q8 seats): {len(masks):,}.\n')
    rep.append('## Leak tests\n')
    rep.extend(f'- [{"x" if passed else " "}] {name}' for name, passed in tests)
    with open(os.path.join(args.mj, 'reports', 'splits.md'), 'w') as f:
        f.write('\n'.join(rep) + '\n')
    print('\n'.join(rep))
    if failed:
        sys.exit(f'LEAK TESTS FAILED: {failed}; splits NOT frozen')

    names = [f'{k}.txt' for k in lists] + ['masks.jsonl']
    with open(os.path.join(out_dir, 'SPLITS.sha256'), 'w') as f:
        f.writelines(f'{sha256_file(os.path.join(out_dir, n))}  {n}\n' for n in names)
    print('frozen:', os.path.join(out_dir, 'SPLITS.sha256'))


if __name__ == '__main__':
    main()
