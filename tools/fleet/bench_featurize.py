"""Featurization rate (03 §3.4, §4.2): samples/s from Tenhou logs.

1. Loader scaling: GameplayLoader.load_gz_log_files (parse + obs v4 encoding, all four seats)
   at several rayon thread counts, each in a fresh process (the pool size is fixed at first use).
2. SL pipeline: FileDatasetsIter + DataLoader exactly as train_offline_phase1.py builds them
   (BC, fp16 obs, Q6/Q8 masks, file_batch_size 20), consumed without a GPU, by worker count.

The teacher-hardware rule (03 §4.2) needs (instance vCPU x samples/s per vCPU) >= 80% of the
GPU's training rate. Tenhou data: run only on the workstation or a secure-tier box (D23).

Usage, from the repo root:
    python tools/fleet/bench_featurize.py --out workdir/fleet/<run>/featurize.json
"""
import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MORTAL_DIR = os.path.join(ROOT, 'mortal')
WS_DATA = '/home/iforgot/mj-data'   # the root written in the workstation's file indexes


def mem_available_gb():
    with open('/proc/meminfo') as f:
        for line in f:
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) / 2**20
    return float('nan')


def sample_files(args):
    import torch
    file_list = torch.load(args.file_index, weights_only=True)['file_list']
    # evenly spaced over the sorted index, so every year of the subset is represented
    step = max(1, len(file_list) // args.files)
    files = file_list[::step][:args.files]
    if args.data_root != WS_DATA:
        files = [f.replace(WS_DATA, args.data_root, 1) for f in files]
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        sys.exit(f'{len(missing)} sampled files are missing, e.g. {missing[0]}')
    return files


def child_loader(args):
    """One loader run; RAYON_NUM_THREADS is set by the parent."""
    sys.path.insert(0, MORTAL_DIR)
    from libriichi.dataset import GameplayLoader

    files = sample_files(args)
    threads = int(os.environ['RAYON_NUM_THREADS'])
    batch = max(8, 2 * threads)
    loader = GameplayLoader(version=4, oracle=False)
    samples = 0
    t0 = time.time()
    for i in range(0, len(files), batch):
        data = loader.load_gz_log_files(files[i:i + batch])
        samples += sum(len(g.take_actions()) for f in data for g in f)
        del data
    dt = time.time() - t0
    print(json.dumps({
        'threads': threads, 'files_per_call': batch, 'files': len(files), 'samples': samples,
        'seconds': round(dt, 2), 'samples_per_s': round(samples / dt),
        'samples_per_s_per_thread': round(samples / dt / threads),
        'peak_rss_gb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2),
    }))


def child_pipeline(args):
    """One SL-pipeline run with args.workers DataLoader workers."""
    os.environ.setdefault('MORTAL_CFG', os.path.join(ROOT, 'experiments/configs/pilot-qs-a.toml'))
    sys.path.insert(0, MORTAL_DIR)
    from torch.utils.data import DataLoader
    from dataloader import FileDatasetsIter, worker_init_fn, load_masks

    files = sample_files(args)
    masks = load_masks(args.masks) if args.masks and os.path.exists(args.masks) else {}
    data = FileDatasetsIter(
        version=4,
        file_list=files,
        pts=[90, 50, -10, -130],
        file_batch_size=20,
        masks=masks,
        compute_advantage=False,
        obs_fp16=True,
    )
    loader = DataLoader(dataset=data, batch_size=2048, drop_last=False, num_workers=args.workers,
                        worker_init_fn=worker_init_fn if args.workers > 0 else None)
    samples = 0
    t0 = time.time()
    first = None
    for obs, *_ in loader:
        samples += len(obs)
        if first is None:
            first = time.time() - t0
    dt = time.time() - t0
    print(json.dumps({
        'workers': args.workers, 'files': len(files), 'masks': bool(masks), 'samples': samples,
        'seconds': round(dt, 2), 'samples_per_s': round(samples / dt), 'first_batch_s': round(first or 0, 2),
        'peak_rss_gb_main': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2),
        'peak_rss_gb_children': round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**20, 2),
    }))


def run_child(args, extra, env):
    if mem_available_gb() < args.min_free_gb:
        return {'skipped': f'MemAvailable {mem_available_gb():.1f} GB < {args.min_free_gb} GB'}
    cmd = [sys.executable, os.path.abspath(__file__), '--file-index', args.file_index, '--files', str(args.files),
           '--data-root', args.data_root, '--masks', args.masks or ''] + extra
    out = subprocess.run(cmd, env={**os.environ, **env}, capture_output=True, text=True, cwd=ROOT)
    if out.returncode != 0:
        return {'error': out.stderr.strip().splitlines()[-1] if out.stderr.strip() else f'exit {out.returncode}'}
    return json.loads(out.stdout.strip().splitlines()[-1])


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--file-index', default=os.path.join(ROOT, 'workdir/pilot/quick500k.pth'))
    p.add_argument('--files', type=int, default=256, help='log files sampled evenly from the index')
    p.add_argument('--data-root', default=os.path.expanduser('~/mj-data'))
    p.add_argument('--masks', default=os.path.expanduser('~/mj-data/splits/masks.jsonl'))
    p.add_argument('--threads', default=None, help='comma-separated rayon thread counts (default: 1,2,4,... and nproc)')
    p.add_argument('--workers', default='1,2', help='comma-separated DataLoader worker counts')
    p.add_argument('--min-free-gb', type=float, default=4.0, help='skip a run if less RAM is available')
    p.add_argument('--out', default=None)
    p.add_argument('--child-loader', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--child-pipeline', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.child_loader:
        return child_loader(args)
    if args.child_pipeline:
        args.workers = int(args.workers)
        return child_pipeline(args)

    ncpu = len(os.sched_getaffinity(0))
    if args.threads:
        threads = [int(t) for t in args.threads.split(',')]
    else:
        threads = sorted({2**k for k in range(ncpu.bit_length()) if 2**k <= ncpu} | {ncpu})
    cpu = next((line.split(':', 1)[1].strip() for line in subprocess.run(
        ['lscpu'], capture_output=True, text=True).stdout.splitlines() if line.startswith('Model name:')), None)
    commit = subprocess.run(['git', '-C', ROOT, 'describe', '--always', '--dirty', '--abbrev=12'],
                            capture_output=True, text=True).stdout.strip()
    info = {'cpu': cpu, 'cpus_affinity': ncpu, 'hostname': platform.node(), 'commit': commit,
            'file_index': args.file_index, 'files': args.files}
    print(json.dumps(info), flush=True)

    loader_runs = []
    for t in threads:
        r = run_child(args, ['--child-loader'], {'RAYON_NUM_THREADS': str(t)})
        r.setdefault('threads', t)
        print('loader', json.dumps(r), flush=True)
        loader_runs.append(r)
    pipeline_runs = []
    for w in (int(x) for x in args.workers.split(',') if x):
        r = run_child(args, ['--child-pipeline', '--workers', str(w)], {})
        r.setdefault('workers', w)
        print('pipeline', json.dumps(r), flush=True)
        pipeline_runs.append(r)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump({'info': info, 'loader': loader_runs, 'pipeline': pipeline_runs}, f, indent=1)


if __name__ == '__main__':
    main()
