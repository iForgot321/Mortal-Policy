"""Host benchmark (03 §3.1): 1v3 duplicate play with one checkpoint on both sides, argmax,
AMP, 500 sets (2,000 concurrent games) per arena call, the way the lab plays (lab/arena.py).
Reports hanchan/h per chunk and overall, the CPU used (cgroup), and GPU utilization.

Not a lab run: seeds come from a benchmark-only key outside the seed families
(lab/seeds.toml), and nothing is written to the registry or the nonce ledger.

Usage, from the repo root:
    python tools/fleet/bench_1v3.py --model workdir/pilot/qs-a/pilot-qs-a-final.pth \
        --minutes 15 --out workdir/box/bench-eval.json
"""
import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from lab.arena import load_engine, play_sets  # noqa: E402

# "bench" in ASCII; outside every seed family
BENCH_KEY = 0x62656E6368


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def cgroup_usage_usec():
    try:
        with open('/sys/fs/cgroup/cpu.stat') as f:
            for line in f:
                k, v = line.split()
                if k == 'usage_usec':
                    return int(v)
    except OSError:
        pass
    return None


def cgroup_quota_cpus():
    try:
        with open('/sys/fs/cgroup/cpu.max') as f:
            quota, period = f.read().split()
        return None if quota == 'max' else int(quota) / int(period)
    except OSError:
        return None


class GpuSampler:
    """nvidia-smi utilization samples, once a second, in a background thread."""

    def __init__(self):
        self.samples = []
        self.proc = subprocess.Popen(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits', '-lms', '1000'],
            stdout=subprocess.PIPE, text=True,
        )
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            try:
                util, mem = (float(x) for x in line.split(','))
            except ValueError:
                continue
            self.samples.append((time.time(), util, mem))

    def mean(self, t0, t1):
        s = [(u, m) for t, u, m in self.samples if t0 <= t <= t1]
        if not s:
            return None, None
        return statistics.fmean(u for u, _ in s), max(m for _, m in s)

    def stop(self):
        self.proc.terminate()


def host_info():
    def run(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    cpu = None
    for line in (run(['lscpu']) or '').splitlines():
        if line.startswith('Model name:'):
            cpu = line.split(':', 1)[1].strip()
    import torch
    return {
        'gpu': run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader']),
        'cpu': cpu,
        'cpus_affinity': len(os.sched_getaffinity(0)),
        'cpus_quota': cgroup_quota_cpus(),
        'hostname': platform.node(),
        'python': platform.python_version(),
        'torch': torch.__version__,
        'commit': run(['git', '-C', ROOT, 'describe', '--always', '--dirty', '--abbrev=12']),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--model', required=True, help='checkpoint used on both sides')
    p.add_argument('--minutes', type=float, default=15.0, help='stop after the chunk that crosses this')
    p.add_argument('--chunk-sets', type=int, default=500, help='sets per arena call (lab.toml arena.chunk_sets)')
    p.add_argument('--no-amp', action='store_true')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--out', default=None, help='write the results as JSON here')
    args = p.parse_args()

    info = host_info()
    info.update(model=args.model, model_sha256=sha256(args.model), script_sha256=sha256(os.path.abspath(__file__)),
                amp=not args.no_amp, chunk_sets=args.chunk_sets, seed_key=hex(BENCH_KEY))
    print(json.dumps(info, indent=1), flush=True)

    amp = not args.no_amp
    challenger = load_engine(args.model, args.device, 'challenger', amp, True)
    champion = load_engine(args.model, args.device, 'champion', amp, True)

    gpu = GpuSampler()
    chunks = []
    start = time.time()
    try:
        while not chunks or time.time() - start < args.minutes * 60:
            i = len(chunks)
            u0, t0 = cgroup_usage_usec(), time.time()
            play_sets(challenger, champion, BENCH_KEY, i * args.chunk_sets, args.chunk_sets)
            u1, t1 = cgroup_usage_usec(), time.time()
            games = 4 * args.chunk_sets
            util, mem = gpu.mean(t0, t1)
            c = {
                'chunk': i,
                'seconds': round(t1 - t0, 2),
                'hanchan_per_h': round(games / (t1 - t0) * 3600),
                'cpu_busy': None if u0 is None else round((u1 - u0) / 1e6 / (t1 - t0), 2),
                'gpu_util_pct': None if util is None else round(util, 1),
                'gpu_mem_max_mib': mem,
            }
            chunks.append(c)
            print(json.dumps(c), flush=True)
    finally:
        gpu.stop()

    # the first chunk carries engine warm-up (cuDNN autotuning, allocator growth)
    steady = chunks[1:] or chunks
    total_games = 4 * args.chunk_sets * len(chunks)
    summary = {
        'chunks': len(chunks),
        'games': total_games,
        'hanchan_per_h_overall': round(total_games / (time.time() - start) * 3600),
        'hanchan_per_h_steady_median': statistics.median(c['hanchan_per_h'] for c in steady),
        'cpu_busy_steady_median': statistics.median(c['cpu_busy'] for c in steady) if steady[0]['cpu_busy'] is not None else None,
        'gpu_util_steady_median': statistics.median(c['gpu_util_pct'] for c in steady) if steady[0]['gpu_util_pct'] is not None else None,
    }
    print(json.dumps(summary, indent=1), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump({'info': info, 'chunks': chunks, 'summary': summary}, f, indent=1)


if __name__ == '__main__':
    main()
