#!/usr/bin/env python3
"""Where the SL trainer's GPU time goes (docs/design/teacher-trainer-draft.md): train_offline_phase1.py's
step, rebuilt on its real data pipeline, in variants that each remove one host-device wait:

    base      as train_offline_phase1.py: a blocking copy to the GPU, the label-mask assert on the
              GPU (a sync), fp16 AMP with GradScaler (a sync in step), AdamW (foreach)
    noassert  base, with the assert done on the CPU tensors before the copy
    prefetch  noassert, with the next batch copied on a side stream while this one trains
    fused     prefetch, with AdamW(fused=True)
    bf16      fused, with bf16 autocast and no GradScaler (no overflow check, so no sync in step)

Each variant runs in its own process (a fresh torch.compile), times --steps steps after --warmup,
and samples the GPU's utilization. Tenhou data: run only on the workstation or a secure-tier box (D23).

    python tools/secure/bench_loop.py --file-index workdir/pilot/quick500k.pth --batch 1024
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ('base', 'noassert', 'prefetch', 'fused', 'bf16')


def run_variant(a):
    sys.path.insert(0, str(ROOT / 'mortal'))
    os.chdir(ROOT / 'mortal')
    import torch
    from itertools import chain
    from torch.amp import GradScaler
    from torch.distributions import Categorical
    from torch.nn.utils import clip_grad_norm_
    from torch.utils.data import DataLoader
    from dataloader import FileDatasetsIter, worker_init_fn, load_masks
    from model import Brain, CategoricalPolicy
    v = a.variant
    level = VARIANTS.index(v)
    torch.manual_seed(0)
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True

    file_list = torch.load(a.file_index, weights_only=True)['file_list']
    data = FileDatasetsIter(version=5, file_list=file_list, pts=[90.0, 50.0, -10.0, -130.0],
                            file_batch_size=20, num_epochs=1, enable_augmentation=True,
                            masks=load_masks(a.masks), compute_advantage=False, obs_fp16=True)
    loader = iter(DataLoader(dataset=data, batch_size=a.batch, drop_last=True, num_workers=a.workers,
                             pin_memory=True, worker_init_fn=worker_init_fn, persistent_workers=True))

    mortal = Brain(version=5, conv_channels=512, num_blocks=a.blocks, Norm='GN').to(device)
    policy = CategoricalPolicy().to(device)
    mortal.compile()
    policy.compile()
    params = list(mortal.parameters()) + list(policy.parameters())
    opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.1, fused=level >= 3)
    bf16 = level >= 4
    scaler = GradScaler(device.type, enabled=not bf16)
    stats = torch.zeros(1, device=device)

    copy_stream = torch.cuda.Stream()
    wait = [0.0]  # seconds blocked in next(loader) during the timed steps

    def next_batch():
        t = time.perf_counter()
        b = next(loader)
        if t0 is not None:
            wait[0] += time.perf_counter() - t
        return b

    def to_gpu(batch, non_blocking):
        obs, actions, masks, _ = batch
        if level >= 1:  # the label check on the CPU: no sync
            assert masks[torch.arange(len(actions)), actions].all()
        return (obs.to(device, non_blocking=non_blocking).float(),
                actions.to(device, dtype=torch.int64, non_blocking=non_blocking),
                masks.to(device, non_blocking=non_blocking))

    def prefetch():
        b = next_batch()
        with torch.cuda.stream(copy_stream):
            g = to_gpu(b, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(copy_stream)
        return g, ev

    def step(obs, actions, masks):
        if level == 0:
            assert masks[range(len(actions)), actions].all()  # a GPU tensor: syncs, as the trainer does
        with torch.autocast(device.type, dtype=torch.bfloat16 if bf16 else torch.float16):
            logits = policy.logits(mortal(obs), masks).float()
            loss = -Categorical(logits=logits).log_prob(actions).mean()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        stats.add_(loss.detach())

    util, stop = [], threading.Event()

    def sample():
        while not stop.is_set():
            out = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                                 capture_output=True, text=True).stdout.strip()
            if out.isdigit():
                util.append(int(out))
            stop.wait(2)

    t0 = None
    nxt = prefetch() if level >= 2 else None
    for i in range(a.warmup + a.steps):
        if i == a.warmup:
            torch.cuda.synchronize()
            t0 = time.time()
            threading.Thread(target=sample, daemon=True).start()
        if level >= 2:
            (obs, actions, masks), ev = nxt
            torch.cuda.current_stream().wait_event(ev)
            for t in (obs, actions, masks):
                t.record_stream(torch.cuda.current_stream())
            nxt = prefetch()
        else:
            obs, actions, masks = to_gpu(next_batch(), non_blocking=False)
        step(obs, actions, masks)
    torch.cuda.synchronize()
    dt = time.time() - t0
    stop.set()
    print(json.dumps({'variant': v, 'blocks': a.blocks, 'batch': a.batch, 'workers': a.workers, 'steps': a.steps,
                      'samples_per_s': round(a.steps * a.batch / dt), 'ms_per_step': round(dt / a.steps * 1000, 1),
                      'data_wait_ms_per_step': round(wait[0] / a.steps * 1000, 1),
                      'gpu_util_mean': round(sum(util) / len(util), 1) if util else None,
                      'loss': round(stats.item() / (a.warmup + a.steps), 3)}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--file-index', required=True)
    ap.add_argument('--masks', default=os.path.expanduser('~/mj-data/splits/masks.jsonl'))
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--blocks', type=int, default=24,
                    help='trunk depth; fewer blocks on a slow GPU mimic a fast GPU\'s ratio of fixed costs to compute')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--warmup', type=int, default=30)
    ap.add_argument('--steps', type=int, default=150)
    ap.add_argument('--variants', default=','.join(VARIANTS))
    ap.add_argument('--out')
    ap.add_argument('--variant', help=argparse.SUPPRESS)  # the child process
    a = ap.parse_args()
    a.file_index, a.masks = os.path.abspath(a.file_index), os.path.abspath(a.masks)  # the child runs in mortal/
    if a.variant:
        os.environ.setdefault('MORTAL_CFG', str(ROOT / 'experiments/configs/sp-nosp-a.toml'))
        run_variant(a)
        return
    results = []
    for v in a.variants.split(','):
        out = subprocess.run([sys.executable, __file__, '--variant', v, '--file-index', a.file_index, '--masks', a.masks,
                              '--batch', str(a.batch), '--blocks', str(a.blocks), '--workers', str(a.workers), '--warmup', str(a.warmup),
                              '--steps', str(a.steps)], capture_output=True, text=True,
                             env={**os.environ, 'MORTAL_CFG': os.environ.get('MORTAL_CFG', str(ROOT / 'experiments/configs/sp-nosp-a.toml'))})
        lines = [l for l in out.stdout.splitlines() if l.startswith('{')]
        r = json.loads(lines[-1]) if lines else {'variant': v, 'error': (out.stderr.strip().splitlines() or ['?'])[-1]}
        results.append(r)
        print(json.dumps(r), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=1) + '\n')


if __name__ == '__main__':
    main()
