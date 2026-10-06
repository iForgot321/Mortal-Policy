"""D8 (04a §8) determinism test: re-run the index on a sample of years in a
scratch tree (sharing mjai/, db/, names/ by symlink) and require identical
output hashes. Splits are pure functions of games.parquet and params.toml, so
they are checked by recomputing the lists in memory.

Writes reports/determinism.md. Usage:
  python tools/data/determinism.py [--mj ~/mj-data] [--years 2009 2026]
"""
import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def sha(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mj', default=os.path.expanduser('~/mj-data'))
    ap.add_argument('--years', nargs='+', default=['2009', '2026'])
    args = ap.parse_args()
    mj = args.mj
    rows, ok = [], True
    with tempfile.TemporaryDirectory(dir=os.path.dirname(mj)) as tmp:
        for d in ('mjai', 'db', 'names'):
            os.symlink(os.path.join(mj, d), os.path.join(tmp, d))
        subprocess.run([sys.executable, os.path.join(HERE, 'index.py'), '--mj', tmp, '--workers', '8',
                        '--years', *args.years], check=True, stdout=subprocess.DEVNULL)
        for year in args.years:
            for kind in ('games', 'seats', 'quarantine'):
                a = os.path.join(mj, 'meta', 'by_year', kind, f'{year}.parquet')
                b = os.path.join(tmp, 'meta', 'by_year', kind, f'{year}.parquet')
                same = sha(a) == sha(b)
                ok &= same
                rows.append(f'| index {kind} {year} | {sha(a)[:16]} | {sha(b)[:16]} | {"✓" if same else "✗"} |')

        # splits: recompute into a scratch splits dir from the main meta
        os.makedirs(os.path.join(tmp, 'meta2'), exist_ok=True)
        tmp2 = os.path.join(tmp, 'tree2')
        os.makedirs(os.path.join(tmp2, 'reports'))
        os.symlink(os.path.join(mj, 'meta'), os.path.join(tmp2, 'meta'))
        subprocess.run([sys.executable, os.path.join(HERE, 'splits.py'), '--mj', tmp2], check=True,
                       stdout=subprocess.DEVNULL)
        for name in sorted(os.listdir(os.path.join(mj, 'splits'))):
            if not (name.endswith('.txt') or name == 'masks.jsonl') or name.startswith('golden'):
                continue
            a, b = os.path.join(mj, 'splits', name), os.path.join(tmp2, 'splits', name)
            same = os.path.exists(b) and sha(a) == sha(b)
            ok &= same
            rows.append(f'| splits {name} | {sha(a)[:16]} | {sha(b)[:16] if os.path.exists(b) else "-"} | '
                        f'{"✓" if same else "✗"} |')
        shutil.rmtree(tmp2, ignore_errors=True)

    rep = ['# Determinism test (04a §8)\n',
           f'Index re-run on years {", ".join(args.years)} in a scratch tree; splits recomputed from meta.\n',
           '| Output | Main | Re-run | Match |', '|---|---|---|---|', *rows,
           f'\n**Result: {"PASS" if ok else "FAIL"}**']
    with open(os.path.join(mj, 'reports', 'determinism.md'), 'w') as f:
        f.write('\n'.join(rep) + '\n')
    print('\n'.join(rep))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
