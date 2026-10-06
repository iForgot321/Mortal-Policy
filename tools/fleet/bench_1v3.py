"""Host benchmark (03 §3.1): 1v3 duplicate play with one checkpoint on both sides, argmax,
AMP, 500 sets (2,000 concurrent games) per arena call, the way the lab plays (lab/arena.py).
Reports hanchan/h per chunk and overall, the CPU used (cgroup), and GPU utilization.

Not a lab run: seeds come from a benchmark-only key outside the seed families
(lab/seeds.toml), and nothing is written to the registry or the nonce ledger.

`--shadow bNcM` prices a network of another size (the reduced inference matrix, 03 §4.1): the
checkpoint still decides, so the games stay those of a trained player, and a random-init network
of that size runs first on every batch, on the same CUDA stream, so each step costs what it would
with the bigger network. `--compile` runs the shadow network through torch.compile with CUDA
graphs and fixed batch buckets.

Usage, from the repo root:
    python tools/fleet/bench_1v3.py --model workdir/pilot/qs-a/pilot-qs-a-final.pth \
        --minutes 15 --out workdir/box/bench-eval.json
"""
import argparse
import hashlib
import json
import os
import platform
import re
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


class ShadowEngine:
    """The inner engine decides; a random-init network of another size runs first on the same
    batch and CUDA stream, and the inner engine's .tolist() waits for both. The inner forward and
    a second np.stack stay in (~2–3% of the main thread at b8c192)."""

    BUCKETS = (256, 512, 1024, 2048, 4096, 8192)

    def __init__(self, inner, arch, compile_model=False):
        import torch
        from model import Brain, CategoricalPolicy

        m = re.fullmatch(r'b(\d+)c(\d+)', arch)
        if not m:
            raise ValueError(f'--shadow wants bNcM, got {arch!r}')
        self.inner = inner
        for attr in ('engine_type', 'name', 'is_oracle', 'version', 'enable_quick_eval',
                     'enable_rule_based_agari_guard'):
            setattr(self, attr, getattr(inner, attr))
        self.device = inner.device
        self.amp = inner.enable_amp
        self.brain = Brain(version=4, num_blocks=int(m[1]), conv_channels=int(m[2]), Norm='GN').to(self.device).eval()
        self.policy = CategoricalPolicy().to(self.device).eval()
        self.compiled = compile_model
        self.forward = self._forward
        if compile_model:
            self.forward = torch.compile(self._forward, mode='reduce-overhead', dynamic=False)

    def _forward(self, obs, masks):
        return self.policy(self.brain(obs), masks)

    def react_batch(self, obs, masks, invisible_obs):
        import numpy as np
        import torch

        with torch.autocast(self.device.type, enabled=self.amp), torch.inference_mode():
            o = torch.as_tensor(np.stack(obs, axis=0), device=self.device)
            mk = torch.as_tensor(np.stack(masks, axis=0), device=self.device)
            if self.compiled:
                n = o.shape[0]
                size = next((b for b in self.BUCKETS if b >= n), n)
                if size > n:
                    o = torch.cat([o, o.new_zeros((size - n, *o.shape[1:]))])
                    mk = torch.cat([mk, mk.new_ones((size - n, *mk.shape[1:]))])
            self.forward(o, mk)
        return self.inner.react_batch(obs, masks, invisible_obs)


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
    p.add_argument('--min-chunks', type=int, default=1, help='play at least this many chunks')
    p.add_argument('--nonce-offset', type=int, default=0, help='first chunk index, so concurrent copies play different seeds')
    p.add_argument('--shadow', default=None, help='price a network of this size, e.g. b24c512 (see the module docstring)')
    p.add_argument('--compile', action='store_true', help='torch.compile the shadow network (CUDA graphs, batch buckets)')
    p.add_argument('--no-amp', action='store_true')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--out', default=None, help='write the results as JSON here')
    args = p.parse_args()

    info = host_info()
    info.update(model=args.model, model_sha256=sha256(args.model), script_sha256=sha256(os.path.abspath(__file__)),
                amp=not args.no_amp, chunk_sets=args.chunk_sets, seed_key=hex(BENCH_KEY),
                nonce_offset=args.nonce_offset, shadow=args.shadow, compile=args.compile)
    print(json.dumps(info, indent=1), flush=True)

    amp = not args.no_amp
    challenger = load_engine(args.model, args.device, 'challenger', amp, True)
    champion = load_engine(args.model, args.device, 'champion', amp, True)
    if args.shadow:
        challenger = ShadowEngine(challenger, args.shadow, args.compile)
        champion = ShadowEngine(champion, args.shadow, args.compile)

    gpu = GpuSampler()
    chunks = []
    start = time.time()
    try:
        while len(chunks) < max(1, args.min_chunks) or time.time() - start < args.minutes * 60:
            i = len(chunks)
            u0, t0 = cgroup_usage_usec(), time.time()
            play_sets(challenger, champion, BENCH_KEY, (args.nonce_offset + i) * args.chunk_sets, args.chunk_sets)
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
