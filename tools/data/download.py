"""D1 (04a §1): download and snapshot the houou dataset at a pinned HF revision.

Downloads only the Tenhou assets (per-year mjai shards + the source-DB archive),
verifies each file against the LFS sha256 that HF publishes for the pinned
revision, then writes raw/MANIFEST.sha256, raw/MANIFEST.id and raw/HF_REVISION
and makes raw/ read-only. Mahjong Soul shards are skipped (04a §9).

Resumable: re-running skips verified files and resumes partial downloads.

Usage: python tools/data/download.py [--mj ~/mj-data] [--revision <sha>]
"""
import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
import urllib.request

REPO = 'nikketryhard/tenhou-houou-mjai'
API = 'https://huggingface.co/api/datasets/{repo}/revision/{rev}?blobs=true'
URL = 'https://huggingface.co/datasets/{repo}/resolve/{rev}/{name}'


def wanted(name):
    return (name.startswith('tenhou-houou-') and name.endswith('.tar')) \
        or name == 'tenhou-original-xml.tar.zst' or name == 'README.md'


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mj', default=os.path.expanduser('~/mj-data'))
    ap.add_argument('--revision', default='main')
    args = ap.parse_args()

    raw = os.path.join(args.mj, 'raw')
    hf_dir = os.path.join(raw, 'hf')
    os.makedirs(hf_dir, exist_ok=True)

    rev_file = os.path.join(raw, 'HF_REVISION')
    if os.path.exists(rev_file):
        rev = open(rev_file).read().strip()
        if args.revision not in ('main', rev):
            sys.exit(f'raw/ is already pinned to {rev}; refusing to mix revisions')
    else:
        with urllib.request.urlopen(API.format(repo=REPO, rev=args.revision)) as r:
            rev = json.load(r)['sha']

    with urllib.request.urlopen(API.format(repo=REPO, rev=rev)) as r:
        info = json.load(r)
    assert info['sha'] == rev
    files = sorted((s for s in info['siblings'] if wanted(s['rfilename'])), key=lambda s: s['size'])
    total = sum(s['size'] for s in files)
    print(f'revision {rev}: {len(files)} files, {total / 1e9:.2f} GB', flush=True)

    for s in files:
        name = s['rfilename']
        dst = os.path.join(hf_dir, name)
        expect = s['lfs']['sha256'] if 'lfs' in s else None
        if os.path.exists(dst) and os.path.getsize(dst) == s['size'] \
                and (expect is None or sha256_of(dst) == expect):
            print(f'ok (cached) {name}', flush=True)
            continue
        print(f'downloading {name} ({s["size"] / 1e9:.2f} GB)', flush=True)
        subprocess.run([
            'curl', '-sS', '-L', '--fail', '--retry', '10', '--retry-delay', '5',
            '-C', '-', '-o', dst, URL.format(repo=REPO, rev=rev, name=name),
        ], check=True)
        if os.path.getsize(dst) != s['size']:
            sys.exit(f'size mismatch for {name}')
        if expect is not None and (got := sha256_of(dst)) != expect:
            os.remove(dst)
            sys.exit(f'sha256 mismatch for {name}: {got} != {expect} (deleted; re-run)')
        print(f'verified {name}', flush=True)

    with open(rev_file, 'w') as f:
        f.write(rev + '\n')

    # sha256sum-compatible manifest, paths relative to raw/
    lines = []
    for s in files:
        rel = os.path.join('hf', s['rfilename'])
        digest = s['lfs']['sha256'] if 'lfs' in s else sha256_of(os.path.join(raw, rel))
        lines.append(f'{digest}  {rel}\n')
    manifest = os.path.join(raw, 'MANIFEST.sha256')
    with open(manifest, 'w') as f:
        f.writelines(lines)
    with open(os.path.join(raw, 'MANIFEST.id'), 'w') as f:
        f.write(f'{sha256_of(manifest)}  MANIFEST.sha256\n')

    # raw/ is immutable from here on
    print('snapshot complete:', open(os.path.join(raw, 'MANIFEST.id')).read().strip(), flush=True)
    for root, dirs, names in os.walk(raw, topdown=False):
        for p in [os.path.join(root, n) for n in names] + [root]:
            os.chmod(p, os.stat(p).st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


if __name__ == '__main__':
    main()
