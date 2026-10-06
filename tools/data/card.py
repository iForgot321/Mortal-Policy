"""D8 (04a §8): assemble reports/DATASET_CARD.md and the dataset-v1 identity.

dataset-v1 = sha256(MANIFEST.id ‖ pipeline commit ‖ params.toml). The pipeline
commit is only meaningful on a clean tree, so the card also records a content
hash of tools/data/*.py and refuses to call the identity final while dirty.

The card is private (it names no players but describes Tenhou data); only the
identity and aggregate numbers are publishable (D13).

Usage: python tools/data/card.py [--mj ~/mj-data]
"""
import argparse
import glob
import hashlib
import os
import subprocess

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def sha(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def section(mj, name):
    path = os.path.join(mj, 'reports', name)
    if not os.path.exists(path):
        return f'_{name} not generated yet._\n'
    with open(path) as f:
        text = f.read()
    # demote headings one level so they nest under the card's sections
    return '\n'.join('#' + line if line.startswith('#') else line for line in text.splitlines()) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mj', default=os.path.expanduser('~/mj-data'))
    args = ap.parse_args()
    mj = args.mj

    manifest_id = open(os.path.join(mj, 'raw', 'MANIFEST.id')).read().split()[0]
    revision = open(os.path.join(mj, 'raw', 'HF_REVISION')).read().strip()
    # the last commit that changed the pipeline, so unrelated commits don't change the identity
    commit = git('log', '-1', '--format=%H', '--', 'tools/data')
    dirty = bool(git('status', '--porcelain', '--', 'tools/data', 'libriichi', 'mortal/dataloader.py'))
    params_path = os.path.join(HERE, 'params.toml')
    params = open(params_path, 'rb').read()
    code = hashlib.sha256()
    for path in sorted(glob.glob(os.path.join(HERE, '*.py'))):
        code.update(os.path.basename(path).encode() + b'\0' + open(path, 'rb').read())
    identity = hashlib.sha256(manifest_id.encode() + b'\0' + commit.encode() + b'\0' + params).hexdigest()

    games = pd.read_parquet(os.path.join(mj, 'meta', 'games.parquet'), columns=['log_id', 'n_decisions'])
    convert = pd.read_csv(os.path.join(mj, 'reports', 'convert_counts.tsv'), sep='\t')
    splits_hash = os.path.join(mj, 'splits', 'SPLITS.sha256')
    golden_hash = os.path.join(mj, 'splits', 'GOLDEN.sha256')

    lines = [
        '# DATASET CARD: dataset-v1 (private)\n',
        '## Identity\n',
        f'- HF repo `nikketryhard/tenhou-houou-mjai`, revision `{revision}`',
        f'- `MANIFEST.id`: `{manifest_id}`',
        f'- Pipeline commit: `{commit}`{" **(tree dirty: identity provisional, regenerate after committing)**" if dirty else ""}',
        f'- Pipeline code hash (tools/data/*.py): `{code.hexdigest()}`',
        f'- params.toml sha256: `{sha(params_path)}`',
        f'- **dataset-v1 identity** = sha256(MANIFEST.id ‖ commit ‖ params.toml) = `{identity}`',
        '',
        '## Counts by stage\n',
        f'- Converted (D2): {int(convert[convert.shard == "TOTAL"].games.iloc[0]):,} games, '
        f'{int(convert[convert.shard == "TOTAL"].q1_failures.iloc[0])} Q1 failures',
        f'- Indexed (D3): {len(games):,} games, {int(games.n_decisions.sum()):,} decisions (all seats)',
        '',
        '## Split hashes\n',
        '```',
        open(splits_hash).read().strip() if os.path.exists(splits_hash) else '(splits not frozen)',
        open(golden_hash).read().strip() if os.path.exists(golden_hash) else '(golden set not frozen)',
        '```',
        '',
        '## Splits\n', section(mj, 'splits.md'),
        '## Quarantine\n', section(mj, 'quarantine.md'),
        '## Coverage\n', section(mj, 'coverage.md'),
        '## Player scoring\n', section(mj, 'player_scores.md'),
        '## Golden set\n', section(mj, 'golden.md'),
        '## Determinism\n', section(mj, 'determinism.md'),
    ]
    out = os.path.join(mj, 'reports', 'DATASET_CARD.md')
    with open(out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'dataset-v1 identity {identity}{" (provisional: dirty tree)" if dirty else ""}\n-> {out}')


if __name__ == '__main__':
    main()
