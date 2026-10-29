#!/usr/bin/env python3
"""Pack the SL training set into encrypted shards for the secure tier (docs/design/secure-data-path-draft.md, D23).

    key    create the bundle key (~/.config/mortal-secure/bundle.key, 32 random bytes, mode 600) if
           there is none. Back it up somewhere safe: without it the shards are noise.
    pack   check the frozen split lists against SPLITS.sha256, then write
           workdir/secure/bundle/<bundle id>/: meta.tar.gpg (the split lists and masks.jsonl) and
           shard-NNN.tar.gpg (the logs of train_E ∪ train_O ∪ heldout_game, in path order), each a
           deterministic tar encrypted with gpg (AES-256, no compression), each decrypted again and
           checked; and manifest.json.

The manifest holds no game ids, only counts and hashes: each shard's plaintext tar sha256 and
ciphertext sha256. The bundle id is the sha256 of the plaintext side (split hashes and shard hashes),
so the same data gives the same id however often it is encrypted. Upload with
`tools/secure/bucket.py push workdir/secure/bundle/<id> bundle/<id>`.
"""
import argparse
import hashlib
import io
import json
import os
import secrets
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MJ = Path.home() / 'mj-data'
OUT = REPO / 'workdir' / 'secure' / 'bundle'
KEY = Path.home() / '.config' / 'mortal-secure' / 'bundle.key'
GNUPGHOME = KEY.parent / 'gnupg'  # gpg's own home, away from the user's keyring
SPLITS = ('train_E', 'train_O', 'heldout_game')
META_FILES = [f'{s}.txt' for s in SPLITS] + ['masks.jsonl', 'SPLITS.sha256']
GPG = ['gpg', '--batch', '--yes', '--quiet', '--no-symkey-cache', '--pinentry-mode', 'loopback']


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def key_path():
    if not KEY.exists():
        sys.exit(f'no bundle key: run `pack.py key` first ({KEY})')
    if KEY.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        sys.exit(f'{KEY} is readable by group or others: chmod 600 it')
    return KEY


def key_id():
    """A public name for the key: 64 bits of a hash of it, safe to write in the manifest."""
    return hashlib.sha256(b'mortal-secure bundle key\0' + key_path().read_bytes()).hexdigest()[:16]


def gpg_env():
    GNUPGHOME.mkdir(mode=0o700, parents=True, exist_ok=True)
    return {**os.environ, 'GNUPGHOME': str(GNUPGHOME)}


class Tee:
    """A write-only file object that hashes and counts what passes through."""

    def __init__(self, f):
        self.f, self.h, self.n = f, hashlib.sha256(), 0

    def write(self, b):
        self.h.update(b)
        self.n += len(b)
        return self.f.write(b)


def write_shard(name, root, rels, out_dir):
    """Tar `rels` (paths under `root`) deterministically, encrypt to out_dir/name, verify by decrypting."""
    dst = out_dir / name
    proc = subprocess.Popen(GPG + ['--passphrase-file', str(key_path()), '--symmetric', '--cipher-algo', 'AES256',
                                   '--compress-algo', 'none', '-o', str(dst)],
                            stdin=subprocess.PIPE, env=gpg_env())
    tee = Tee(proc.stdin)
    with tarfile.open(fileobj=tee, mode='w|', format=tarfile.PAX_FORMAT) as tar:
        for arc, rel in rels:
            data = (root / rel).read_bytes()
            ti = tarfile.TarInfo(arc)
            ti.size, ti.mtime, ti.mode, ti.uid, ti.gid, ti.uname, ti.gname = len(data), 0, 0o444, 0, 0, '', ''
            tar.addfile(ti, fileobj=io.BytesIO(data))
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f'gpg failed on {name}')
    plain_sha = tee.h.hexdigest()
    # decrypt it again: the key works and the ciphertext holds exactly what was written
    dec = subprocess.Popen(GPG + ['--passphrase-file', str(key_path()), '--decrypt', str(dst)],
                           stdout=subprocess.PIPE, env=gpg_env())
    h = hashlib.sha256()
    for b in iter(lambda: dec.stdout.read(1 << 20), b''):
        h.update(b)
    if dec.wait() != 0 or h.hexdigest() != plain_sha:
        raise RuntimeError(f'{name}: decrypting it back does not give the tar written')
    return {'name': name, 'files': len(rels), 'plain_bytes': tee.n, 'plain_sha256': plain_sha,
            'cipher_bytes': dst.stat().st_size, 'cipher_sha256': sha256_file(dst)}


def check_splits():
    want = {}
    for line in (MJ / 'splits' / 'SPLITS.sha256').read_text().split('\n'):
        if line.strip():
            h, name = line.split()
            want[name] = h
    for name in META_FILES[:-1]:
        got = sha256_file(MJ / 'splits' / name)
        if got != want.get(name):
            sys.exit(f'{name}: sha256 {got[:12]}… does not match SPLITS.sha256')
    return want


def cmd_key(a):
    if KEY.exists():
        print(f'{KEY} exists (key id {key_id()}); not replaced')
        return
    KEY.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(KEY, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as f:
        f.write(secrets.token_hex(32) + '\n')
    print(f'created {KEY} (key id {key_id()}). Back it up somewhere safe: without it the bundle is noise.')


def cmd_pack(a):
    key_path()
    want = check_splits()
    rels = sorted({line.strip() for s in SPLITS for line in open(MJ / 'splits' / f'{s}.txt') if line.strip()})
    missing = [r for r in rels[:: max(1, len(rels) // 1000)] if not (MJ / 'mjai' / r).exists()]
    if missing:
        sys.exit(f'{len(missing)} sampled logs missing, e.g. {missing[0]}')
    need = sum((MJ / 'mjai' / r).stat().st_size for r in rels)
    free = shutil.disk_usage(REPO / 'workdir').free
    if free < 2 * need:
        sys.exit(f'need ≈ {2 * need / 2**30:.0f} GiB free in workdir, have {free / 2**30:.0f}')
    print(f'{len(rels):,} logs, {need / 2**30:.2f} GiB, into {a.shards} shards with {a.workers} workers', flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(dir=OUT, prefix='staging-'))
    chunks = [rels[i * len(rels) // a.shards:(i + 1) * len(rels) // a.shards] for i in range(a.shards)]
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(write_shard, 'meta.tar.gpg', MJ / 'splits',
                          [(f'splits/{n}', n) for n in META_FILES], stage)]
        futs += [ex.submit(write_shard, f'shard-{i:03d}.tar.gpg', MJ / 'mjai',
                           [(f'mjai/{r}', r) for r in chunk], stage) for i, chunk in enumerate(chunks)]
        shards = []
        for f in futs:
            shards.append(f.result())
            print(f'  {shards[-1]["name"]}: {shards[-1]["files"]:,} files, {shards[-1]["cipher_bytes"] / 2**20:,.0f} MiB, '
                  f'{time.time() - t0:,.0f} s', flush=True)

    ident = {'format': 1, 'splits': {n: want[n] for n in META_FILES[:-1]},
             'shards': [{k: s[k] for k in ('name', 'files', 'plain_bytes', 'plain_sha256')} for s in shards]}
    bundle_id = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()
    manifest = {**ident, 'bundle_id': bundle_id, 'key_id': key_id(), 'shards': shards,
                'created': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                'cipher': 'gpg --symmetric --cipher-algo AES256 --compress-algo none (OpenPGP, integrity-protected)',
                'logs': len(rels), 'games_per_split': {s: sum(1 for l in open(MJ / 'splits' / f'{s}.txt') if l.strip())
                                                       for s in SPLITS}}
    (stage / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    final = OUT / bundle_id[:16]
    if final.exists():
        shutil.rmtree(stage)
        sys.exit(f'{final} already exists: the same data is packed already')
    stage.rename(final)
    for p in final.iterdir():
        p.chmod(0o444)
    print(f'bundle {bundle_id[:16]} in {final} ({sum(s["cipher_bytes"] for s in shards) / 2**30:.2f} GiB, '
          f'{time.time() - t0:,.0f} s). Upload: tools/secure/bucket.py push {final.relative_to(REPO)} bundle/{bundle_id[:16]}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('key')
    s = sub.add_parser('pack')
    s.add_argument('--shards', type=int, default=64)
    s.add_argument('--workers', type=int, default=4)
    a = ap.parse_args(argv)
    {'key': cmd_key, 'pack': cmd_pack}[a.cmd](a)


if __name__ == '__main__':
    main()
