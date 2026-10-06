"""Build trainer file indexes from frozen split lists (04a §7: "Build
file_index.pth from the lists, never from a raw glob").

  phase1:  {'file_list': [...]}                          (train_offline_phase1.py)
  grp:     {'train_file_list': [...], 'val_file_list': [...]}   (train_grp.py)

Lists hold paths relative to <mj>/mjai; the index holds absolute paths.
Optionally derives a deterministic subset (e.g. the 500K-game quick-student
subset, 04 §7) by sha1 order, and writes it as a list next to the index.

Usage:
  python tools/data/make_index.py phase1 OUT.pth --lists train_E train_O \
      [--from 2020-01-01 --to 2025-06-30 --subset 500000 --subset-key quick500k]
  python tools/data/make_index.py grp OUT.pth --lists train_E train_O --val heldout_game
"""
import argparse
import hashlib
import os

import torch


def read(mj, name):
    with open(os.path.join(mj, 'splits', f'{name}.txt')) as f:
        return [line.strip() for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('kind', choices=['phase1', 'grp'])
    ap.add_argument('out')
    ap.add_argument('--mj', default=os.path.expanduser('~/mj-data'))
    ap.add_argument('--lists', nargs='+', required=True)
    ap.add_argument('--val', nargs='*', default=[])
    ap.add_argument('--from', dest='date_from', default=None, help='inclusive, YYYY-MM-DD')
    ap.add_argument('--to', dest='date_to', default=None, help='inclusive, YYYY-MM-DD')
    ap.add_argument('--subset', type=int, default=None)
    ap.add_argument('--subset-key', default='subset')
    args = ap.parse_args()

    rel = sorted({p for name in args.lists for p in read(args.mj, name)})
    day = lambda p: os.path.basename(p)[:8]
    if args.date_from:
        rel = [p for p in rel if day(p) >= args.date_from.replace('-', '')]
    if args.date_to:
        rel = [p for p in rel if day(p) <= args.date_to.replace('-', '')]
    if args.subset is not None:
        key = lambda p: hashlib.sha1(f'{args.subset_key}:{os.path.basename(p)}'.encode()).hexdigest()
        rel = sorted(sorted(rel, key=key)[:args.subset])
        if len(rel) < args.subset:
            raise SystemExit(f'only {len(rel)} games available for a {args.subset} subset')
    absolute = lambda ps: [os.path.join(args.mj, 'mjai', p) for p in ps]
    missing = [p for p in absolute(rel[:1000]) if not os.path.exists(p)]
    if missing:
        raise SystemExit(f'missing files, e.g. {missing[:3]}')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    list_path = os.path.splitext(args.out)[0] + '.list.txt'
    with open(list_path, 'w') as f:
        f.writelines(p + '\n' for p in rel)
    digest = hashlib.sha256(open(list_path, 'rb').read()).hexdigest()

    if args.kind == 'phase1':
        torch.save({'file_list': absolute(rel)}, args.out)
    else:
        val = sorted({p for name in args.val for p in read(args.mj, name)})
        torch.save({'train_file_list': absolute(rel), 'val_file_list': absolute(val)}, args.out)
        print(f'val: {len(val):,} games')
    print(f'{args.kind}: {len(rel):,} games -> {args.out}; list sha256 {digest} ({list_path})')


if __name__ == '__main__':
    main()
