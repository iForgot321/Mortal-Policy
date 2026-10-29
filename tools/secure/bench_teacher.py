#!/usr/bin/env python3
"""The teacher-hardware measurement (D17a, `03` §4.2; docs/design/secure-data-path-draft.md), run on a
secure-tier box after `secure.py ship`:

1. GPU only: b24c512 (v5, AMP, torch.compile) forward + backward + AdamW on synthetic batches.
2. The real thing: train_offline_phase1.py on the shipped train window (b24c512, BC, augmentation,
   the box's file index and masks) for --minutes; samples/s between its checkpoint saves after the
   first (compile and warm-up excluded), with the GPU's utilization sampled alongside.
3. The SL pipeline alone: tools/fleet/bench_featurize.py on the shipped logs, by worker count.

Writes workdir/secure-box/bench/teacher.json (`secure.py pull` brings it back). Tenhou data: only
on the workstation or a secure-tier box (D23).

    python tools/secure/bench_teacher.py --data /ephemeral/mj
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'workdir' / 'secure-box' / 'bench'


def gpu_synthetic(batch, compile_, secs=60):
    import torch
    sys.path.insert(0, str(ROOT / 'mortal'))
    from model import Brain, CategoricalPolicy
    from libriichi.consts import obs_shape, ACTION_SPACE
    torch.manual_seed(0)
    dev = torch.device('cuda')
    brain = Brain(conv_channels=512, num_blocks=24, version=5, Norm='GN').to(dev)
    pol = CategoricalPolicy().to(dev)
    opt = torch.optim.AdamW(list(brain.parameters()) + list(pol.parameters()), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    f = torch.compile(brain) if compile_ else brain
    obs = torch.randn(batch, *obs_shape(5), device=dev)
    mask = torch.ones(batch, ACTION_SPACE, dtype=torch.bool, device=dev)
    act = torch.randint(0, ACTION_SPACE, (batch,), device=dev)

    def step():
        with torch.autocast('cuda'):
            loss = torch.nn.functional.cross_entropy(pol.logits(f(obs), mask).float(), act)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    for _ in range(15):
        step()
    torch.cuda.synchronize()
    t0, n = time.time(), 0
    while time.time() - t0 < secs:
        step()
        n += 1
    torch.cuda.synchronize()
    r = {'batch': batch, 'compile': compile_, 'samples_per_s': round(n * batch / (time.time() - t0)),
         'peak_mem_gib': round(torch.cuda.max_memory_allocated() / 2**30, 1)}
    del brain, pol, opt
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return r


def gpu_util_sampler(samples, stop):
    while not stop.is_set():
        out = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True).stdout.strip()
        if out.isdigit():
            samples.append(int(out))
        stop.wait(5)


def trainer(args):
    import toml
    base = toml.load(args.base_config)
    run = OUT / 'trainer'
    run.mkdir(parents=True, exist_ok=True)
    for p in run.glob('*.pth'):
        p.unlink()
    c = base['control']
    c.update(state_file=str(run / 'mortal.pth'), best_state_file=str(run / 'best.pth'),
             tensorboard_dir=str(run / 'tb'), batch_size=args.batch, save_every=args.save_every)
    base['test_play']['log_dir'] = str(run / 'test_play')
    # TestPlayer loads a baseline at start-up even though test play never runs here; the config's
    # workdir/phase1/baseline.pth is workstation-only, so use a frozen checkpoint every box has
    for k in ('train', 'test'):
        base['baseline'][k]['state_file'] = str(ROOT / 'workdir/sp-ablation/nosp-a/nosp-a-final.pth')
    base['resnet'] = {'conv_channels': 512, 'num_blocks': 24}
    d = base['dataset']
    d.update(file_index=f'{args.data}/index-train.pth', masks_file=f'{args.data}/splits/masks.jsonl',
             num_workers=args.workers, enable_augmentation=True)
    base['optim']['scheduler']['max_steps'] = 1_150_000  # ≈ 2.34B presentations / 2048
    cfg = run / 'teacher-bench.toml'
    cfg.write_text(toml.dumps(base))

    samples, stop = [], threading.Event()
    th = threading.Thread(target=gpu_util_sampler, args=(samples, stop), daemon=True)
    th.start()
    logf = run / 'train.log'
    with open(logf, 'w') as f:
        subprocess.run(['timeout', str(args.minutes * 60), sys.executable, 'train_offline_phase1.py'],
                       cwd=ROOT / 'mortal', env={**os.environ, 'MORTAL_CFG': str(cfg)}, stdout=f, stderr=subprocess.STDOUT)
    stop.set()
    saves = []
    for line in logf.read_text(errors='replace').splitlines():
        m = re.match(r'(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ .*total steps: ([\d,]+)', line)
        if m:
            saves.append((datetime.strptime(m[1], '%Y-%m-%d %H:%M:%S').timestamp(), int(m[2].replace(',', ''))))
    saves = [s for s in saves if s[1] > 0]
    r = {'batch': args.batch, 'workers': args.workers, 'minutes': args.minutes, 'saves': len(saves),
         'gpu_util_mean': round(sum(samples) / len(samples), 1) if samples else None}
    if len(saves) >= 3:  # from the first save on: compile and warm-up excluded
        (t0, s0), (t1, s1) = saves[0], saves[-1]
        r['samples_per_s'] = round((s1 - s0) * args.batch / (t1 - t0))
        r['hours_per_teacher'] = round(2.34e9 / r['samples_per_s'] / 3600, 1)
    else:
        r['error'] = f'only {len(saves)} saves in {args.minutes} min; see {logf}'
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True, help='the shipped data directory (secure.py ship)')
    ap.add_argument('--minutes', type=int, default=15)
    ap.add_argument('--batch', type=int, default=2048)
    ap.add_argument('--save-every', type=int, default=400)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--featurize-workers', default='2,4,8')
    ap.add_argument('--base-config', default=str(ROOT / 'workdir' / 'box' / 'configs' / 'sp-nosp-a.toml'),
                    help="the config the bench changes (default: setup_box.sh's box-path copy of sp-nosp-a)")
    ap.add_argument('--skip', default='', help='comma-separated parts to skip: gpu,trainer,featurize')
    args = ap.parse_args()
    skip = set(filter(None, args.skip.split(',')))
    OUT.mkdir(parents=True, exist_ok=True)
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                         capture_output=True, text=True).stdout.strip()
    res = {'gpu': gpu, 'vcpu': len(os.sched_getaffinity(0)), 'started': datetime.now().isoformat(timespec='seconds')}
    out = OUT / 'teacher.json'

    def save():
        out.write_text(json.dumps(res, indent=1) + '\n')

    if 'gpu' not in skip:
        res['gpu_synthetic'] = [gpu_synthetic(b, True) for b in (1024, 2048)]
        print('gpu:', res['gpu_synthetic'], flush=True)
        save()
    if 'trainer' not in skip:
        res['trainer'] = trainer(args)
        print('trainer:', res['trainer'], flush=True)
        save()
    if 'featurize' not in skip:
        fz = OUT / 'featurize.json'
        subprocess.run([sys.executable, str(ROOT / 'tools/fleet/bench_featurize.py'), '--file-index',
                        f'{args.data}/index-train.pth', '--data-root', args.data, '--masks',
                        f'{args.data}/splits/masks.jsonl', '--version', '5', '--workers', args.featurize_workers,
                        '--out', str(fz)], cwd=ROOT, check=False)
        res['featurize'] = json.loads(fz.read_text()) if fz.exists() else {'error': 'no output'}
        save()
    res['finished'] = datetime.now().isoformat(timespec='seconds')
    save()
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
