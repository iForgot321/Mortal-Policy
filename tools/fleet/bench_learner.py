"""P0 learner throughput by obs version (03 §3.6; SP ablation follow-up).

The P0 learner re-featurizes every drained log (`OrderedDatasetsIter`, MC targets), so obs v4's
single-player tables may bind it the way they bound SL. This replays one fixed drain the way
`train_online.py` trains it and reports samples/s three ways:

- **full:** a fresh DataLoader per pass (as per drain), featurization + PPO steps with stored
  behaviour probabilities (AMP, torch.compile, grad clip, AdamW), timed on the second pass so
  compilation and cuDNN autotuning are excluded;
- **data only:** the same DataLoader, batches consumed without any GPU work;
- **GPU only:** PPO steps on batches already on the GPU (the version doesn't change the shapes).

The drain is generated once (`--games` 1v3 games, pilot A both sides, the trainee sampling with
explore_rate 1.0, behaviour sidecars), on the benchmark-only seed key, so no nonces are used.
The cloud reference is ~3,750 samples/s on an RTX 4090 + EPYC 9684X box (v4,
experiments/fleet-host-benchmarks.md), measured on live drains of the same size.

Usage, from the repo root:
    python tools/fleet/bench_learner.py --drain workdir/sp-ablation/learner-drain \
        --versions 4 5 --out workdir/sp-ablation/learner.json
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'mortal'))
# mortal/dataloader.py reads a config at import; P0's, though the settings used here are explicit
os.environ.setdefault('MORTAL_CFG', os.path.join(ROOT, 'experiments/configs/p0.toml'))

BENCH_KEY = 0x62656E6368  # "bench": outside the seed families (lab/seeds.toml)
PILOT = os.path.join(ROOT, 'workdir/pilot/qs-a/pilot-qs-a-final.pth')
GRP_FILE = os.path.join(ROOT, 'workdir/grp-v2/grp-v2-frozen.pth')
GRP_NETWORK = {'hidden_size': 64, 'num_layers': 2}
PTS = [90.0, 50.0, -10.0, -130.0]
# experiments/configs/p0.toml
BATCH, WORKERS, FILE_BATCH = 512, 2, 50
HP = {'clip_ratio': 0.2, 'dual_clip': 3.0, 'entropy_weight': 1e-3, 'advantage_norm': True}
LR, MAX_GRAD_NORM = 2e-5, 0.5


def load_nets(version, device):
    import torch
    from model import Brain, CategoricalPolicy

    state = torch.load(PILOT, weights_only=False, map_location='cpu')
    brain = Brain(version=version, num_blocks=8, conv_channels=192, Norm='GN')
    policy = CategoricalPolicy()
    brain.load_state_dict(state['mortal'])
    policy.load_state_dict(state['policy_net'])
    return brain.to(device), policy.to(device)


def generate(drain, games):
    import torch
    from engine import MortalEngine
    from libriichi.arena import OneVsThree

    device = torch.device('cuda:0')
    brain, policy = load_nets(4, device)
    brain.eval(), policy.eval()

    def engine(explore_rate, name):
        return MortalEngine(brain, policy, is_oracle=False, version=4, device=device, enable_amp=True,
                            enable_rule_based_agari_guard=True, explore_rate=explore_rate, name=name)

    env = OneVsThree(disable_progress_bar=True, log_dir=os.path.join(drain, 'logs'),
                     behaviour_dir=os.path.join(drain, 'bp'))
    env.py_vs_py(challenger=engine(1.0, 'trainee'), champion=engine(0.0, 'baseline'),
                 seed_start=(0, BENCH_KEY), seed_count=games // 4)


def loader(version, logs, drain):
    from torch.utils.data import DataLoader
    from dataloader import worker_init_fn
    from rl_dataset import OrderedDatasetsIter

    data = OrderedDatasetsIter(
        version=version, file_list=list(logs), behaviour_dir=os.path.join(drain, 'bp'), pts=PTS,
        grp_state_file=GRP_FILE, grp_network=GRP_NETWORK, horizon='K', targets='mc',
        player_names=['trainee'], file_batch_size=FILE_BATCH, obs_fp16=True,
    )
    return DataLoader(dataset=data, batch_size=BATCH, drop_last=False, num_workers=WORKERS,
                      pin_memory=True, worker_init_fn=worker_init_fn)


class Learner:
    def __init__(self, version, device):
        import torch
        self.device = device
        self.brain, self.policy = load_nets(version, device)
        self.brain.train(), self.policy.train()
        self.brain.compile()
        self.policy.compile()
        params = list(self.brain.parameters()) + list(self.policy.parameters())
        self.params = params
        self.opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8)
        self.scaler = torch.amp.GradScaler(device.type)

    def step(self, obs, actions, masks, advantage, b):
        import torch
        from ppo import ppo_loss

        obs = obs.to(dtype=torch.float32, device=self.device, non_blocking=True)
        actions = actions.to(dtype=torch.int64, device=self.device, non_blocking=True)
        masks = masks.to(dtype=torch.bool, device=self.device, non_blocking=True)
        advantage = advantage.to(dtype=torch.float32, device=self.device, non_blocking=True)
        b = b.to(dtype=torch.float32, device=self.device, non_blocking=True)
        with torch.autocast(self.device.type):
            logits = self.policy.logits(self.brain(obs), masks)
        loss, _ = ppo_loss(logits, actions, advantage, b, **HP)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(self.params, MAX_GRAD_NORM)
        self.scaler.step(self.opt)
        self.scaler.update()
        self.opt.zero_grad(set_to_none=True)


def timed_pass(dl, learner=None):
    import torch
    t0 = time.time()
    n = 0
    for obs, actions, masks, advantage, b, _ in dl:
        if learner is not None:
            learner.step(obs, actions, masks, advantage, b)
        n += len(obs)
    torch.cuda.synchronize()
    return n, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--drain', required=True, help='dir with logs/ and bp/ (generated if missing)')
    ap.add_argument('--games', type=int, default=800)
    ap.add_argument('--versions', type=int, nargs='+', default=[4, 5])
    ap.add_argument('--gpu-steps', type=int, default=200)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    import torch
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0')

    if not glob.glob(os.path.join(args.drain, 'logs', '*.json.gz')):
        t0 = time.time()
        generate(args.drain, args.games)
        print(json.dumps({'generated_games': args.games, 'seconds': round(time.time() - t0, 1)}), flush=True)
    logs = sorted(glob.glob(os.path.join(args.drain, 'logs', '*.json.gz')))

    info = {
        'commit': subprocess.run(['git', '-C', ROOT, 'describe', '--always', '--dirty', '--abbrev=12'],
                                 capture_output=True, text=True).stdout.strip(),
        'gpu': torch.cuda.get_device_name(0), 'cpus': len(os.sched_getaffinity(0)), 'drain_games': len(logs),
        'batch': BATCH, 'num_workers': WORKERS, 'file_batch_size': FILE_BATCH,
    }
    print(json.dumps(info), flush=True)

    results = []
    gpu_batches = None
    for v in args.versions:
        n, dt = timed_pass(loader(v, logs, args.drain))
        data_only = {'version': v, 'mode': 'data only', 'samples': n, 'seconds': round(dt, 1),
                     'samples_per_s': round(n / dt)}
        print(json.dumps(data_only), flush=True)
        learner = Learner(v, device)
        timed_pass(loader(v, logs, args.drain), learner)  # warm-up: compile, autotune
        n, dt = timed_pass(loader(v, logs, args.drain), learner)
        full = {'version': v, 'mode': 'full', 'samples': n, 'seconds': round(dt, 1), 'samples_per_s': round(n / dt)}
        print(json.dumps(full), flush=True)
        results += [data_only, full]
        if gpu_batches is None:
            gpu_batches = []
            for batch in loader(v, logs, args.drain):
                gpu_batches.append([x.to(device) for x in batch[:5]])
                if len(gpu_batches) == 20:
                    break
            for i in range(20):  # warm-up
                learner.step(*gpu_batches[i % len(gpu_batches)])
            torch.cuda.synchronize()
            t0 = time.time()
            for i in range(args.gpu_steps):
                learner.step(*gpu_batches[i % len(gpu_batches)])
            torch.cuda.synchronize()
            dt = time.time() - t0
            gpu = {'mode': 'GPU only', 'steps': args.gpu_steps, 'seconds': round(dt, 2),
                   'samples_per_s': round(args.gpu_steps * BATCH / dt)}
            print(json.dumps(gpu), flush=True)
            results.append(gpu)
        del learner
        torch.cuda.empty_cache()

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump({'info': info, 'results': results}, f, indent=1)


if __name__ == '__main__':
    main()
