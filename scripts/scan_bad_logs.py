"""Scan the full mjai archive with the actual training loader (GameplayLoader)
to find semantically-invalid logs that gzip/JSON checks cannot catch (e.g. the
5sr-kakan converter bug found 2026-07-09, see experiments/phase1-v0-unfiltered.md).

Output: data/stats/bad_files.tsv (path<TAB>error), part of the Stage 1 hygiene
pass — dataset-v1 must exclude these files deterministically.

Usage (any cwd): python scripts/scan_bad_logs.py
"""
import os
import sys
from glob import glob
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'data', 'stats', 'bad_files.tsv')

# libriichi.so lives in mortal/; make the script cwd-independent
sys.path.insert(0, os.path.join(ROOT, 'mortal'))

loader = None

def init_worker():
    global loader
    from libriichi.dataset import GameplayLoader
    loader = GameplayLoader(version=4, oracle=False, player_names=[], excludes=None, augmented=False)

def check(path):
    try:
        data = loader.load_gz_log_files([path])
        del data
        return None
    except Exception as e:
        return path + '\t' + str(e).replace('\n', ' | ')[:300]

def main():
    files = sorted(glob(os.path.join(ROOT, 'data', 'raw', '*', '*.mjson')))
    print(f'scanning {len(files):,} files', flush=True)
    bad = []
    with Pool(12, initializer=init_worker) as pool:
        for i, result in enumerate(pool.imap_unordered(check, files, chunksize=32), 1):
            if result:
                bad.append(result)
                print('BAD:', result.split('\t')[0], flush=True)
            if i % 100000 == 0:
                print(f'progress {i:,}/{len(files):,}, bad so far: {len(bad)}', flush=True)
    with open(OUT, 'w') as fh:
        fh.writelines(line + '\n' for line in sorted(bad))
    print(f'scan complete: {len(bad)} bad of {len(files):,} -> {OUT}', flush=True)

if __name__ == '__main__':
    main()
