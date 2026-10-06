"""
Compute per-player stats across the raw mjai dataset.

Output: data/stats/player_stats.csv, data/stats/metadata.json

Each .mjson file is gzip-compressed newline-delimited JSON.
Score tracking: initialize from each start_kyoku, apply deltas from hora/ryukyoku.
Placement: rank by final score descending; ties broken by seat index (lower = better).
"""

import argparse
import csv
import gzip
import json
import os
import subprocess
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


def process_file(path: str) -> list[tuple[str, int, int]]:
    """Return list of (player_name, placement_1indexed, final_score) for each seat."""
    try:
        with gzip.open(path) as f:
            lines = f.read().decode().split('\n')
    except Exception:
        return []

    names = []
    scores = []
    results = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue

        t = ev['type']
        if t == 'start_game':
            names = ev['names']
        elif t == 'start_kyoku':
            scores = list(ev['scores'])
        elif t in ('hora', 'ryukyoku'):
            deltas = ev.get('deltas')
            if deltas and scores:
                for i, d in enumerate(deltas):
                    scores[i] += d
        elif t == 'end_game':
            if len(names) == 4 and len(scores) == 4:
                # Sort seats by score desc, then by seat index asc for ties
                order = sorted(range(4), key=lambda i: (-scores[i], i))
                placements = [0] * 4
                for rank, seat in enumerate(order):
                    placements[seat] = rank + 1
                for seat, name in enumerate(names):
                    results.append((name, placements[seat], scores[seat]))

    return results


def collect_files(data_dirs: list[Path]) -> list[str]:
    files = []
    for d in data_dirs:
        files.extend(str(p) for p in sorted(d.glob('*.mjson')))
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-dirs', nargs='+',
        default=[f'data/raw/{y}' for y in range(2020, 2025)],
        help='Directories to scan for .mjson files',
    )
    parser.add_argument('--out-dir', default='data/stats')
    parser.add_argument('--workers', type=int, default=os.cpu_count())
    args = parser.parse_args()

    data_dirs = [Path(d) for d in args.data_dirs]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = collect_files(data_dirs)
    print(f'Found {len(files):,} files across {len(data_dirs)} directories')
    print(f'Using {args.workers} workers')

    # Per-player accumulators
    games    = defaultdict(int)
    ranks    = defaultdict(lambda: [0, 0, 0, 0])  # rank1..rank4 counts
    score_sum = defaultdict(int)

    done = 0
    report_every = 10_000

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_file, f): f for f in files}
        for future in as_completed(futures):
            results = future.result()
            for name, placement, score in results:
                games[name] += 1
                ranks[name][placement - 1] += 1
                score_sum[name] += score
            done += 1
            if done % report_every == 0:
                print(f'  {done:,}/{len(files):,} files processed', flush=True)

    print(f'  {done:,}/{len(files):,} files processed')
    print(f'Found {len(games):,} unique players')

    # Write CSV
    csv_path = out_dir / 'player_stats.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'player', 'games',
            'avg_placement', 'avg_score',
            'rank1', 'rank2', 'rank3', 'rank4',
            'rank1_rate', 'rank2_rate', 'rank3_rate', 'rank4_rate',
        ])
        for name in sorted(games):
            g = games[name]
            r = ranks[name]
            avg_pl = sum((i + 1) * r[i] for i in range(4)) / g
            avg_sc = score_sum[name] / g
            writer.writerow([
                name, g,
                f'{avg_pl:.4f}', f'{avg_sc:.1f}',
                r[0], r[1], r[2], r[3],
                f'{r[0]/g:.4f}', f'{r[1]/g:.4f}', f'{r[2]/g:.4f}', f'{r[3]/g:.4f}',
            ])

    # Write metadata
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True
        ).strip()
    except Exception:
        commit = 'unknown'

    meta = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'commit': commit,
        'dataset_dirs': [str(d) for d in data_dirs],
        'total_files': len(files),
        'total_players': len(games),
    }
    with open(out_dir / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'Wrote {csv_path}')
    print(f'Wrote {out_dir / "metadata.json"}')

    # Quick sanity: show top 10 by avg placement (min 500 games)
    qualified = [(name, games[name], sum((i+1)*ranks[name][i] for i in range(4))/games[name])
                 for name in games if games[name] >= 500]
    qualified.sort(key=lambda x: x[2])
    print(f'\nTop 10 by avg placement (>=500 games, {len(qualified):,} total qualified):')
    for name, g, avg in qualified[:10]:
        print(f'  {name:<30} games={g:>6,}  avg_placement={avg:.4f}')


if __name__ == '__main__':
    main()