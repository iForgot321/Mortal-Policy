#!/home/iforgot/.venvs/s3/bin/python
"""The SL data bucket on Hyperstack's object storage (docs/design/secure-data-path-draft.md, D23).

Holds only ciphertext: the encrypted training bundle (tools/secure/pack.py) and, later, the
encrypted raw/ backup. Boxes never get these credentials; they fetch through presigned URLs.

    init             create the S3 access key (Infrahub API) and the private bucket, once; the
                     credentials go to ~/.config/hyperstack/s3.json (mode 600) and are never printed
    check            the bucket is private: an anonymous GET is refused, a presigned one works
    ls [PREFIX]      objects with sizes
    put FILE KEY     upload one file (multipart above 64 MB)
    push DIR PREFIX  upload a directory under PREFIX/, skipping objects already there (manifest last)
    presign KEY      a time-limited GET URL (--minutes, default 120)
    urls BUNDLE_DIR  presigned URLs for every object of a bundle, as tools/secure/unpack.py reads them
    rm KEY           delete one object

Runs with boto3 from its own venv (~/.venvs/s3, like vastai's ~/.venvs/vast). Hyperstack's object
storage is S3-compatible, in CANADA-1 only, at https://ca1.obj.nexgencloud.io.
"""
import argparse
import json
import os
import secrets
import stat
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hyperstack  # noqa: E402  (its API-key loader)

ENDPOINT = 'https://ca1.obj.nexgencloud.io'
REGION = 'CANADA-1'
CREDS = Path.home() / '.config' / 'hyperstack' / 's3.json'
BUCKET_PREFIX = 'mortal-sl-'


def load_creds():
    if not CREDS.exists():
        sys.exit(f'no {CREDS}: run `bucket.py init` first')
    if CREDS.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        sys.exit(f'{CREDS} is readable by group or others: chmod 600 it')
    return json.loads(CREDS.read_text())


def save_creds(creds):
    CREDS.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=CREDS.parent, prefix='.s3-')
    with os.fdopen(fd, 'w') as f:  # mkstemp creates it 0600
        json.dump(creds, f, indent=1)
    os.replace(tmp, CREDS)


def client(creds):
    # botocore ≥ 1.36 adds CRC checksums to every upload by default; S3-compatible stores may
    # reject them, so only send one where the operation requires it
    cfg = Config(signature_version='s3v4', retries={'max_attempts': 5, 'mode': 'standard'},
                 request_checksum_calculation='when_required', response_checksum_validation='when_required')
    return boto3.client('s3', endpoint_url=ENDPOINT, region_name='us-east-1', config=cfg,
                        aws_access_key_id=creds['access_key'], aws_secret_access_key=creds['secret_key'])


def create_access_key():
    """POST /object-storage/access-keys; the secret is returned once and goes straight to CREDS."""
    body = json.dumps({'region': REGION, 'description': 'mortal-policy SL data (D23)'}).encode()
    req = urllib.request.Request(hyperstack.API + '/object-storage/access-keys', data=body, method='POST',
                                 headers={'api_key': hyperstack.api_key(), 'Content-Type': 'application/json',
                                          'Accept': 'application/json', 'User-Agent': 'mortal-policy-secure/1'})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:
        sys.exit(f'creating the access key: HTTP {e.code}: {e.read().decode(errors="replace")[:400]}')
    save_creds({'raw': d})  # the secret is shown only once: keep it before parsing anything
    d = d.get('data', d)  # tolerate an envelope
    return {'access_key_id': d['id'], 'access_key': d['access_key'], 'secret_key': d['secret_key'],
            'region': d.get('region', REGION)}


def cmd_init(a):
    creds = json.loads(CREDS.read_text()) if CREDS.exists() else {}
    if not creds.get('secret_key'):
        creds = create_access_key()
        save_creds(creds)
        print(f'access key created: id {creds["access_key_id"]}, {creds["access_key"]} (secret saved to {CREDS})')
    s3 = client(load_creds())
    if not creds.get('bucket'):
        creds['bucket'] = BUCKET_PREFIX + secrets.token_hex(4)  # bucket names are shared across the store
        s3.create_bucket(Bucket=creds['bucket'])
        save_creds(creds)
        print(f'bucket created: {creds["bucket"]}')
    print(f'bucket {creds["bucket"]} at {ENDPOINT}')


def cmd_check(a):
    creds = load_creds()
    s3, bucket = client(creds), creds['bucket']
    key = f'check/{secrets.token_hex(4)}.txt'
    s3.put_object(Bucket=bucket, Key=key, Body=b'private?\n')
    try:
        try:
            urllib.request.urlopen(f'{ENDPOINT}/{bucket}/{key}', timeout=30)
            anon = 'ALLOWED'
        except urllib.error.HTTPError as e:
            anon = f'refused ({e.code})'
        url = s3.generate_presigned_url('get_object', Params={'Bucket': bucket, 'Key': key}, ExpiresIn=300)
        with urllib.request.urlopen(url, timeout=30) as r:
            signed = 'works' if r.read() == b'private?\n' else 'WRONG CONTENT'
    finally:
        s3.delete_object(Bucket=bucket, Key=key)
    print(f'anonymous GET: {anon}; presigned GET: {signed}')
    if anon == 'ALLOWED' or signed != 'works':
        sys.exit(1)


def cmd_ls(a):
    creds = load_creds()
    total = n = 0
    for page in client(creds).get_paginator('list_objects_v2').paginate(Bucket=creds['bucket'], Prefix=a.prefix):
        for o in page.get('Contents', []):
            print(f'{o["Size"]:>14,}  {o["LastModified"]:%Y-%m-%d %H:%M}  {o["Key"]}')
            total += o['Size']
            n += 1
    print(f'{n} objects, {total / 2**30:.2f} GiB')


def cmd_put(a):
    creds = load_creds()
    client(creds).upload_file(a.file, creds['bucket'], a.key,
                              Config=TransferConfig(multipart_threshold=64 << 20, multipart_chunksize=64 << 20))
    print(f'put {a.file} → {a.key}')


def cmd_push(a):
    """Upload every file in DIR under PREFIX/, skipping objects already there with the same size."""
    creds = load_creds()
    s3, bucket = client(creds), creds['bucket']
    have = {}
    for page in s3.get_paginator('list_objects_v2').paginate(Bucket=bucket, Prefix=a.prefix.rstrip('/') + '/'):
        have.update({o['Key']: o['Size'] for o in page.get('Contents', [])})
    files = sorted(p for p in Path(a.dir).iterdir() if p.is_file())
    # the manifest last, so a bundle prefix with a manifest is a complete upload
    files.sort(key=lambda p: p.name == 'manifest.json')
    cfg = TransferConfig(multipart_threshold=64 << 20, multipart_chunksize=64 << 20)
    for p in files:
        key = f'{a.prefix.rstrip("/")}/{p.name}'
        if have.get(key) == p.stat().st_size:
            continue
        s3.upload_file(str(p), bucket, key, Config=cfg)
        print(f'put {key} ({p.stat().st_size / 2**20:,.0f} MiB)', flush=True)
    print(f'{a.prefix}: {len(files)} files in the bucket')


def cmd_urls(a):
    """One line per bundle object, for tools/secure/unpack.py: NAME CIPHER_SHA256 PLAIN_SHA256 URL."""
    creds = load_creds()
    s3 = client(creds)
    m = json.loads((Path(a.bundle_dir) / 'manifest.json').read_text())
    prefix = f'bundle/{m["bundle_id"][:16]}'
    for s in m['shards']:
        url = s3.generate_presigned_url('get_object', Params={'Bucket': creds['bucket'], 'Key': f'{prefix}/{s["name"]}'},
                                        ExpiresIn=a.minutes * 60)
        print(s['name'], s['cipher_sha256'], s['plain_sha256'], url)


def cmd_presign(a):
    creds = load_creds()
    print(client(creds).generate_presigned_url('get_object', Params={'Bucket': creds['bucket'], 'Key': a.key},
                                               ExpiresIn=a.minutes * 60))


def cmd_rm(a):
    creds = load_creds()
    client(creds).delete_object(Bucket=creds['bucket'], Key=a.key)
    print(f'deleted {a.key}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('init')
    sub.add_parser('check')
    s = sub.add_parser('ls')
    s.add_argument('prefix', nargs='?', default='')
    s = sub.add_parser('put')
    s.add_argument('file')
    s.add_argument('key')
    s = sub.add_parser('push')
    s.add_argument('dir')
    s.add_argument('prefix')
    s = sub.add_parser('urls')
    s.add_argument('bundle_dir')
    s.add_argument('--minutes', type=int, default=180)
    s = sub.add_parser('presign')
    s.add_argument('key')
    s.add_argument('--minutes', type=int, default=120)
    s = sub.add_parser('rm')
    s.add_argument('key')
    a = ap.parse_args(argv)
    globals()['cmd_' + a.cmd.replace('-', '_')](a)


if __name__ == '__main__':
    main()
