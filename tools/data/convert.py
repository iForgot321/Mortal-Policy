"""D2 (04a §2): convert the Tenhou mjai shards to the layout Mortal's loader reads.

Each `<logid>.mjai.json.zst` in raw/hf/tenhou-houou-YYYY.tar becomes
mjai/YYYY/MM/<logid>.json.gz (gzip level 6). libriichi's GameplayLoader reads
gzip only. Tars are streamed, so no scratch extraction is needed. The gzip
header carries no name or mtime, so outputs are byte-deterministic (D8).

Decompression failures are quarantined as Q1 in reports/convert_q1.tsv; per
shard counts go to reports/convert_counts.tsv.

Usage: python tools/data/convert.py [--mj ~/mj-data] [--workers 6]
"""
import argparse
import glob
import gzip
import os
import tarfile
from multiprocessing import Pool

import zstandard

SUFFIX = '.mjai.json.zst'


def write_gz(path, data):
    tmp = path + '.tmp'
    with open(tmp, 'wb') as raw, gzip.GzipFile(filename='', mode='wb', compresslevel=6,
                                               fileobj=raw, mtime=0) as gz:
        gz.write(data)
    os.replace(tmp, path)


def convert_shard(args):
    tar_path, out_root = args
    dctx = zstandard.ZstdDecompressor()
    n_ok, n_skip, failures = 0, 0, []
    made = set()
    with tarfile.open(tar_path, mode='r|') as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(SUFFIX):
                continue
            log_id = os.path.basename(member.name)[:-len(SUFFIX)]
            out_dir = os.path.join(out_root, log_id[:4], log_id[4:6])
            out = os.path.join(out_dir, log_id + '.json.gz')
            if out_dir not in made:
                os.makedirs(out_dir, exist_ok=True)
                made.add(out_dir)
            payload = tar.extractfile(member).read()
            if os.path.exists(out):
                n_skip += 1
                continue
            try:
                with dctx.stream_reader(payload) as reader:
                    data = reader.read()
            except zstandard.ZstdError as e:
                failures.append((log_id, f'zstd: {e}'))
                continue
            write_gz(out, data)
            n_ok += 1
    return os.path.basename(tar_path), n_ok, n_skip, failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mj', default=os.path.expanduser('~/mj-data'))
    ap.add_argument('--workers', type=int, default=6)
    args = ap.parse_args()

    tars = sorted(glob.glob(os.path.join(args.mj, 'raw', 'hf', 'tenhou-houou-*.tar')),
                  key=os.path.getsize, reverse=True)
    out_root = os.path.join(args.mj, 'mjai')
    reports = os.path.join(args.mj, 'reports')
    os.makedirs(reports, exist_ok=True)

    rows, q1 = [], []
    with Pool(args.workers) as pool:
        for name, n_ok, n_skip, failures in pool.imap_unordered(
                convert_shard, [(t, out_root) for t in tars]):
            print(f'{name}: {n_ok} converted, {n_skip} already present, {len(failures)} Q1', flush=True)
            rows.append((name, n_ok + n_skip, len(failures)))
            q1.extend(failures)

    with open(os.path.join(reports, 'convert_counts.tsv'), 'w') as f:
        f.write('shard\tgames\tq1_failures\n')
        for name, n, nf in sorted(rows):
            f.write(f'{name}\t{n}\t{nf}\n')
        f.write(f'TOTAL\t{sum(r[1] for r in rows)}\t{sum(r[2] for r in rows)}\n')
    with open(os.path.join(reports, 'convert_q1.tsv'), 'w') as f:
        f.write('log_id\tdetail\n')
        f.writelines(f'{lid}\t{d}\n' for lid, d in sorted(q1))
    print(f'total {sum(r[1] for r in rows)} games, {len(q1)} Q1', flush=True)


if __name__ == '__main__':
    main()
