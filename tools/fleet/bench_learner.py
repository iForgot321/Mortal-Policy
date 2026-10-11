"""P0 and P1 learner throughput (03 §3.6; SP ablation follow-up; the P0 and P1 tuning allowances).

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

`--workers 2 4 8` repeats the full and data-only passes at each loader-worker count (P0's
tuning allowance: the learner box has ≥ 32 vCPU, p0.toml uses 2).

`--arm P1` times P1's path instead (p1.toml, D52): horizon H, and the K1 critic. The loader
re-featurizes with the oracle channels and yields whole-trajectory chunks; the learner computes
critic values and GAE on each chunk, shuffles it into minibatches, and trains the policy (PPO) and
a fresh b8c128 critic (MSE onto the λ-returns) in one backward, as `train_online.py` does. Samples
are decisions loaded, in every mode. Peak memory (PSS of the benchmark and its loader workers) is
reported per pass.

Usage, from the repo root:
    python tools/fleet/bench_learner.py --drain workdir/sp-ablation/learner-drain \
        --versions 4 5 --out workdir/sp-ablation/learner.json
    python tools/fleet/bench_learner.py --drain workdir/box/learner-drain --games 1600 \
        --versions 4 --workers 2 4 8 --out workdir/box/learner-workers.json
    python tools/fleet/bench_learner.py --arm P1 --drain workdir/box/learner-drain --games 3200 \
        --versions 4 --workers 4 8 16 --no-compile --out workdir/box/p1-learner-workers.json
"""
import argparse
import glob
import json
import os
import resource
import subprocess
import sys
import threading
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
# experiments/configs/p1.toml: horizon and targets per arm, and the [critic] section
ARMS = {'P0': {'horizon': 'K', 'targets': 'mc'}, 'P1': {'horizon': 'H', 'targets': 'gae'}}
CRITIC = {'conv_channels': 128, 'num_blocks': 8}
CRITIC_LR, CRITIC_MAX_GRAD_NORM, GAMMA, LAMBDA = 1e-4, 1.0, 1.0, 0.95
P1_KEYS = ('obs', 'action', 'mask', 'advantage', 'behaviour_prob', 'priv', 'returns')


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


def loader(version, logs, drain, workers=WORKERS, arm='P0'):
    from torch.utils.data import DataLoader
    from dataloader import worker_init_fn
    from rl_dataset import OrderedDatasetsIter

    data = OrderedDatasetsIter(
        version=version, file_list=list(logs), behaviour_dir=os.path.join(drain, 'bp'), pts=PTS,
        grp_state_file=GRP_FILE, grp_network=GRP_NETWORK, **ARMS[arm],
        player_names=['trainee'], file_batch_size=FILE_BATCH, obs_fp16=True,
    )
    # P1: whole-trajectory chunks, batched by the learner after GAE
    return DataLoader(dataset=data, batch_size=BATCH if arm == 'P0' else None, drop_last=False,
                      num_workers=workers, pin_memory=True, worker_init_fn=worker_init_fn)


class Learner:
    def __init__(self, version, device, compile=True, arm='P0'):
        import torch
        from critic import Critic
        self.device = device
        self.brain, self.policy = load_nets(version, device)
        self.brain.train(), self.policy.train()
        self.critic = None
        if arm == 'P1':
            self.critic = Critic(**CRITIC, version=version).to(device)
            self.critic.train()
        if compile:
            self.brain.compile()
            self.policy.compile()
            if self.critic is not None:
                self.critic.compile()
        params = list(self.brain.parameters()) + list(self.policy.parameters())
        self.params = params
        self.opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8)
        if self.critic is not None:
            self.critic_opt = torch.optim.AdamW(self.critic.parameters(), lr=CRITIC_LR, weight_decay=0.0,
                                                betas=(0.9, 0.999), eps=1e-8)
        self.scaler = torch.amp.GradScaler(device.type)

    def step(self, obs, actions, masks, advantage, b, priv=None, returns=None):
        import torch
        import torch.nn.functional as F
        from ppo import ppo_loss

        obs = obs.to(dtype=torch.float32, device=self.device, non_blocking=True)
        actions = actions.to(dtype=torch.int64, device=self.device, non_blocking=True)
        masks = masks.to(dtype=torch.bool, device=self.device, non_blocking=True)
        advantage = advantage.to(dtype=torch.float32, device=self.device, non_blocking=True)
        b = b.to(dtype=torch.float32, device=self.device, non_blocking=True)
        with torch.autocast(self.device.type):
            logits = self.policy.logits(self.brain(obs), masks)
        loss, _ = ppo_loss(logits, actions, advantage, b, **HP)
        if self.critic is not None:
            priv = priv.to(dtype=torch.float32, device=self.device, non_blocking=True)
            returns = returns.to(dtype=torch.float32, device=self.device, non_blocking=True)
            with torch.autocast(self.device.type):
                value = self.critic(obs, priv)
            # disjoint parameters: one backward serves both networks
            loss = loss + F.mse_loss(value.float(), returns)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(self.params, MAX_GRAD_NORM)
        self.scaler.step(self.opt)
        if self.critic is not None:
            self.scaler.unscale_(self.critic_opt)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), CRITIC_MAX_GRAD_NORM)
            self.scaler.step(self.critic_opt)
            self.critic_opt.zero_grad(set_to_none=True)
        self.scaler.update()
        self.opt.zero_grad(set_to_none=True)

    def gae(self, chunk):
        from critic import gae_targets
        return gae_targets(chunk, self.critic, device=self.device, gamma=GAMMA, lam=LAMBDA, enable_amp=True)


class PeakMemory:
    """Peak PSS of this process and its children (the loader workers), sampled every second.
    Start it only after the loader has forked its workers: a fork while this thread runs can
    leave a worker deadlocked on a lock the thread held (seen 2026-10-11)."""

    def __init__(self):
        self.peak = 0
        self.done = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    @staticmethod
    def pss(pid):
        try:
            with open(f'/proc/{pid}/smaps_rollup') as f:
                return next(int(line.split()[1]) for line in f if line.startswith('Pss:')) * 1024
        except (OSError, StopIteration):
            return 0

    def sample(self):
        me = os.getpid()
        while not self.done.wait(1.0):
            pids = [me]
            for d in os.listdir('/proc'):
                if d.isdigit():
                    try:
                        with open(f'/proc/{d}/stat') as f:
                            if int(f.read().rsplit(')', 1)[1].split()[1]) == me:
                                pids.append(int(d))
                    except (OSError, ValueError, IndexError):
                        pass
            self.peak = max(self.peak, sum(self.pss(p) for p in pids))

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.done.set()
        self.thread.join()


def timed_pass(dl, learner=None, arm='P0'):
    import torch
    from rl_dataset import shuffled_minibatches
    t0 = time.time()
    n = 0
    dl = iter(dl)  # forks the loader workers
    with PeakMemory() as mem:
        if arm == 'P0':
            for obs, actions, masks, advantage, b, _ in dl:
                if learner is not None:
                    learner.step(obs, actions, masks, advantage, b)
                n += len(obs)
        elif learner is None:
            for chunk in dl:
                n += sum(len(t['action']) for t in chunk)
        else:
            def flats():
                nonlocal n
                for chunk in dl:
                    n += sum(len(t['action']) for t in chunk)
                    yield learner.gae(chunk)
            for mb in shuffled_minibatches(flats(), BATCH, P1_KEYS):
                learner.step(*(mb[k] for k in P1_KEYS))
        torch.cuda.synchronize()
    return n, time.time() - t0, round(mem.peak / 2**30, 2)


def gpu_batches(dl, learner, arm, count=20):
    """`count` training batches, on the GPU."""
    from rl_dataset import shuffled_minibatches
    device = learner.device
    if arm == 'P0':
        batches = (batch[:5] for batch in dl)
    else:
        batches = ([mb[k] for k in P1_KEYS] for mb in shuffled_minibatches(map(learner.gae, dl), BATCH, P1_KEYS))
    out = []
    for batch in batches:
        out.append([x.to(device) for x in batch])
        if len(out) == count:
            break
    return out


def cgroup_limits():
    """The container's CPU quota and memory limit (cgroup v2), where readable."""
    out = {}
    try:
        quota, period = open('/sys/fs/cgroup/cpu.max').read().split()
        out['cpu_quota'] = None if quota == 'max' else round(int(quota) / int(period), 1)
    except (OSError, ValueError):
        pass
    try:
        limit = open('/sys/fs/cgroup/memory.max').read().strip()
        out['memory_limit_gb'] = None if limit == 'max' else round(int(limit) / 2**30, 1)
    except (OSError, ValueError):
        pass
    try:
        out['mem_total_gb'] = round(os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE') / 2**30, 1)
    except (OSError, ValueError):
        pass
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arm', choices=sorted(ARMS), default='P0', help="the learner path to time (P1: horizon H, critic)")
    ap.add_argument('--drain', required=True, help='dir with logs/ and bp/ (generated if missing)')
    ap.add_argument('--games', type=int, default=800)
    ap.add_argument('--versions', type=int, nargs='+', default=[4, 5])
    ap.add_argument('--gpu-steps', type=int, default=200)
    ap.add_argument('--workers', type=int, nargs='+', default=[WORKERS], help='loader-worker counts to time')
    ap.add_argument('--no-compile', action='store_true',
                    help='eager learner steps (the first compile hung on a Vast EPYC 7C13 box, 2026-10-09)')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    import torch
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0')
    # chunks of whole trajectories pass between processes as many shared tensors, one file
    # descriptor each; Vast's soft limit of 1,024 is too low (box_supervise.sh raises it too)
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))

    if not glob.glob(os.path.join(args.drain, 'logs', '*.json.gz')):
        t0 = time.time()
        generate(args.drain, args.games)
        print(json.dumps({'generated_games': args.games, 'seconds': round(time.time() - t0, 1)}), flush=True)
    logs = sorted(glob.glob(os.path.join(args.drain, 'logs', '*.json.gz')))

    info = {
        'commit': subprocess.run(['git', '-C', ROOT, 'describe', '--always', '--dirty', '--abbrev=12'],
                                 capture_output=True, text=True).stdout.strip(),
        'arm': args.arm, **ARMS[args.arm],
        'gpu': torch.cuda.get_device_name(0), 'cpus': len(os.sched_getaffinity(0)), **cgroup_limits(),
        'drain_games': len(logs), 'batch': BATCH, 'num_workers': args.workers, 'file_batch_size': FILE_BATCH,
        'compile': not args.no_compile, 'nofile': hard,
    }
    print(json.dumps(info), flush=True)

    arm = args.arm
    results = []
    timed_gpu = False
    for v in args.versions:
        learner = Learner(v, device, compile=not args.no_compile, arm=arm)
        timed_pass(loader(v, logs, args.drain, args.workers[0], arm), learner, arm)  # warm-up: compile, autotune
        for w in args.workers:
            n, dt, mem = timed_pass(loader(v, logs, args.drain, w, arm), arm=arm)
            data_only = {'version': v, 'workers': w, 'mode': 'data only', 'samples': n, 'seconds': round(dt, 1),
                         'samples_per_s': round(n / dt), 'peak_pss_gb': mem}
            print(json.dumps(data_only), flush=True)
            n, dt, mem = timed_pass(loader(v, logs, args.drain, w, arm), learner, arm)
            full = {'version': v, 'workers': w, 'mode': 'full', 'samples': n, 'seconds': round(dt, 1),
                    'samples_per_s': round(n / dt), 'peak_pss_gb': mem}
            print(json.dumps(full), flush=True)
            results += [data_only, full]
        if not timed_gpu:
            # training steps only; P1's batches already carry their GAE targets
            timed_gpu = True
            batches = gpu_batches(loader(v, logs, args.drain, args.workers[0], arm), learner, arm)
            for i in range(20):  # warm-up
                learner.step(*batches[i % len(batches)])
            torch.cuda.synchronize()
            t0 = time.time()
            for i in range(args.gpu_steps):
                learner.step(*batches[i % len(batches)])
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
