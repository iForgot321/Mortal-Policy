#!/usr/bin/env python3
"""Fetch, check, decrypt and extract the encrypted SL bundle on a secure-tier box (D23;
docs/design/secure-data-path-draft.md). Run by tools/secure/secure.py ship, which pipes in:

    line 1        the bundle key
    then, each    NAME CIPHER_SHA256 PLAIN_SHA256 URL     (URL presigned, time-limited)

For each object: download it, check the ciphertext hash, decrypt it with gpg (the key reaches gpg
through a pipe, never a file), extract the tar under DEST while hashing it, check the plaintext
hash, delete the ciphertext. Stdlib only, so the box's system Python (3.10+) runs it.

    python3 unpack.py DEST [--jobs 4] < spec
"""
import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

GPG = ['gpg', '--batch', '--quiet', '--no-symkey-cache', '--pinentry-mode', 'loopback']


class HashingReader:
    def __init__(self, f):
        self.f, self.h = f, hashlib.sha256()

    def read(self, n=-1):
        b = self.f.read(n)
        self.h.update(b)
        return b


def fetch(url, dst, tries=5):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(dst, 'wb') as f:
                shutil.copyfileobj(r, f, 1 << 20)
            return
        except OSError as e:
            if i == tries - 1:
                raise
            print(f'retry {os.path.basename(dst)}: {e}', flush=True)
            time.sleep(5 * (i + 1))


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def one(key, dest, gnupghome, name, cipher_sha, plain_sha, url):
    src = os.path.join(dest, '.incoming', name)
    fetch(url, src)
    if sha256_file(src) != cipher_sha:
        raise RuntimeError(f'{name}: ciphertext sha256 mismatch')
    r, w = os.pipe()
    os.write(w, key.encode())
    os.close(w)
    proc = subprocess.Popen(GPG + ['--passphrase-fd', str(r), '--decrypt', src], stdout=subprocess.PIPE,
                            pass_fds=(r,), env={**os.environ, 'GNUPGHOME': gnupghome})
    os.close(r)
    reader = HashingReader(proc.stdout)
    with tarfile.open(fileobj=reader, mode='r|') as tar:
        for m in tar:
            if not m.isfile():
                raise RuntimeError(f'{name}: unexpected member {m.name!r}')
            tar.extract(m, dest, filter='data')
    for b in iter(lambda: reader.read(1 << 20), b''):  # the end-of-archive padding
        pass
    if proc.wait() != 0:
        raise RuntimeError(f'{name}: gpg failed')
    if reader.h.hexdigest() != plain_sha:
        raise RuntimeError(f'{name}: plaintext sha256 mismatch')
    os.remove(src)
    return name


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dest')
    ap.add_argument('--jobs', type=int, default=4)
    a = ap.parse_args(argv)
    key = sys.stdin.readline().strip()
    specs = [line.split() for line in sys.stdin if line.strip()]
    if not key or any(len(s) != 4 for s in specs):
        sys.exit('bad spec: the key, then NAME CIPHER_SHA256 PLAIN_SHA256 URL per line')
    os.makedirs(os.path.join(a.dest, '.incoming'), exist_ok=True)
    gnupghome = tempfile.mkdtemp(prefix='gnupg-')
    t0 = time.time()
    try:
        with ThreadPoolExecutor(a.jobs) as ex:
            for i, name in enumerate(ex.map(lambda s: one(key, a.dest, gnupghome, *s), specs), 1):
                print(f'{i}/{len(specs)} {name} ok ({time.time() - t0:,.0f} s)', flush=True)
    finally:
        subprocess.run(['gpgconf', '--homedir', gnupghome, '--kill', 'gpg-agent'], capture_output=True)
        shutil.rmtree(gnupghome, ignore_errors=True)
    os.rmdir(os.path.join(a.dest, '.incoming'))
    print(f'unpacked {len(specs)} objects into {a.dest} in {time.time() - t0:,.0f} s')


if __name__ == '__main__':
    main()
