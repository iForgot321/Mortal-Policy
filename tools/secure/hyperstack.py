#!/usr/bin/env python3
"""A read-only look at the Hyperstack account from the workstation (docs/design/secure-data-path-draft.md).

Hyperstack has no official CLI, only alpha SDKs generated from its REST API
(https://infrahub-api.nexgencloud.com/v1); this is a stdlib client for exploring it before
tools/secure/secure.py rents anything.

    credit           the account's credit balance
    regions          compute regions
    flavors          VM types: GPU, vCPU, RAM, disk, stock (--gpu A100, --in-stock)
    stock            GPU stock per region
    prices           the price book (--match A100)
    images           OS images per region (--match ubuntu)
    vms              virtual machines (anything listed here is billing)
    environments     environments (a VM is created in one; it fixes the region)
    keypairs         SSH key pairs
    volumes          volumes (billed while they exist)
    buckets          object-storage buckets
    storage-regions  object-storage regions
    get PATH         any other GET endpoint, as JSON (e.g. `get /core/gpus`)

Every command is a GET: nothing here creates, changes or deletes a resource, so nothing here bills.
The API key comes from $HYPERSTACK_API_KEY or ~/.config/hyperstack/api_key, which must not be
readable by group or others. Add --json to any listing for the raw response.
"""
import argparse
import json
import os
import stat
import sys
import urllib.error
import urllib.request
from pathlib import Path

API = 'https://infrahub-api.nexgencloud.com/v1'
KEY_FILE = Path.home() / '.config' / 'hyperstack' / 'api_key'


def api_key():
    key = os.environ.get('HYPERSTACK_API_KEY', '').strip()
    if key:
        return key
    if not KEY_FILE.exists():
        sys.exit(f'no API key: set HYPERSTACK_API_KEY or write it to {KEY_FILE} (mode 600)')
    if KEY_FILE.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        sys.exit(f'{KEY_FILE} is readable by group or others: chmod 600 it')
    return KEY_FILE.read_text().strip()


def get(path, key=None):
    """GET one endpoint; the only request this tool makes."""
    if not path.startswith('/'):
        raise ValueError(f'path must start with /: {path!r}')
    # Cloudflare in front of the API rejects urllib's default User-Agent (error 1010)
    req = urllib.request.Request(API + path, method='GET',
                                 headers={'api_key': key or api_key(), 'Accept': 'application/json',
                                          'User-Agent': 'mortal-policy-secure/1'})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors='replace')[:400]
        sys.exit(f'GET {path}: HTTP {e.code}: {body}')


def table(rows, cols):
    rows = [[('' if r.get(c) is None else str(r.get(c))) for c in cols] for r in rows]
    width = [max([len(c)] + [len(r[i]) for r in rows]) for i, c in enumerate(cols)]
    print('  '.join(c.ljust(w) for c, w in zip(cols, width)))
    for r in rows:
        print('  '.join(v.ljust(w) for v, w in zip(r, width)))


def match(text, needle):
    return needle is None or needle.lower() in (text or '').lower()


def cmd_flavors(a, data):
    rows = []
    for group in data.get('data') or []:
        for f in group.get('flavors') or []:
            if not match(f.get('gpu') or group.get('gpu'), a.gpu) or not match(f.get('region_name'), a.region):
                continue
            if a.in_stock and not f.get('stock_available'):
                continue
            rows.append(f)
    table(sorted(rows, key=lambda f: (f.get('region_name') or '', f.get('gpu') or '', f.get('gpu_count') or 0)),
          ['name', 'region_name', 'gpu', 'gpu_count', 'cpu', 'ram', 'disk', 'ephemeral', 'stock_available'])


def cmd_stock(a, data):
    rows = []
    for s in data.get('stocks') or []:
        for m in s.get('models') or []:
            if match(m.get('model'), a.gpu):
                rows.append({'region': s.get('region'), 'stock_type': s.get('stock-type'),
                             'model': m.get('model'), 'available': m.get('available'),
                             'configurations': json.dumps(m.get('configurations'))})
    table(rows, ['region', 'stock_type', 'model', 'available', 'configurations'])


def cmd_prices(a, data):
    rows = [p for p in (data if isinstance(data, list) else data.get('data') or []) if match(p.get('name'), a.match)]
    table(sorted(rows, key=lambda p: p.get('name') or ''), ['name', 'value', 'original_value', 'discount_applied'])


def cmd_images(a, data):
    rows = []
    for group in data.get('images') or []:
        for i in group.get('images') or []:
            if match(i.get('name'), a.match) and match(i.get('region_name') or group.get('region_name'), a.region):
                rows.append(i)
    table(rows, ['name', 'region_name', 'type', 'version', 'display_size', 'is_public'])


def cmd_vms(a, data):
    rows = [{'id': v.get('id'), 'name': v.get('name'), 'status': v.get('status'),
             'flavor': (v.get('flavor') or {}).get('name'), 'environment': (v.get('environment') or {}).get('name'),
             'floating_ip': v.get('floating_ip'), 'created_at': v.get('created_at')}
            for v in data.get('instances') or []]
    table(rows, ['id', 'name', 'status', 'flavor', 'environment', 'floating_ip', 'created_at'])


# command → (endpoint, printer or the list key to tabulate with its columns)
COMMANDS = {
    'credit': ('/billing/user-credit/credit', None),
    'regions': ('/core/regions', ('regions', ['id', 'name', 'country', 'description', 'green_status'])),
    'flavors': ('/core/flavors', cmd_flavors),
    'stock': ('/core/stocks', cmd_stock),
    'prices': ('/pricebook', cmd_prices),
    'images': ('/core/images', cmd_images),
    'vms': ('/core/virtual-machines', cmd_vms),
    'environments': ('/core/environments', ('environments', ['id', 'name', 'region', 'created_at'])),
    'keypairs': ('/core/keypairs', ('keypairs', ['id', 'name', 'fingerprint', 'created_at'])),
    'volumes': ('/core/volumes', ('volumes', ['id', 'name', 'size', 'status', 'volume_type', 'created_at'])),
    'buckets': ('/object-storage/buckets', ('buckets', ['name', 'region', 'endpoint', 'num_objects', 'size_bytes'])),
    'storage-regions': ('/object-storage/regions', ('regions', ['name'])),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in COMMANDS:
        s = sub.add_parser(name)
        s.add_argument('--json', action='store_true', help='print the raw response')
        if name in ('flavors', 'stock'):
            s.add_argument('--gpu', help='keep GPUs whose name contains this (case-insensitive)')
        if name in ('flavors', 'images'):
            s.add_argument('--region')
        if name == 'flavors':
            s.add_argument('--in-stock', action='store_true')
        if name in ('prices', 'images'):
            s.add_argument('--match')
    s = sub.add_parser('get')
    s.add_argument('path')
    a = ap.parse_args(argv)

    if a.cmd == 'get':
        print(json.dumps(get(a.path), indent=1))
        return
    path, show = COMMANDS[a.cmd]
    data = get(path)
    if a.json or show is None:
        print(json.dumps(data, indent=1))
    elif callable(show):
        show(a, data)
    else:
        key, cols = show
        rows = data.get(key) if isinstance(data, dict) else data
        if rows is None:  # an unexpected shape: show it rather than guess
            print(json.dumps(data, indent=1))
        else:
            table(rows, cols)


if __name__ == '__main__':
    main()
