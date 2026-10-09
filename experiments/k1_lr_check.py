"""K1 critic learning-rate check (D52), pre-registered in experiments/k1-lr-check.md.

    generate   play the fixed B0 game set once: pilot A frozen, the trainee sampling (ε = 1)
               against three pilot-A baselines, as P1's actors do in B0; logs + behaviour sidecars
    run        train a fresh critic in B0 at one learning rate through the real trainer
               (mortal/train_online.py), fed the fixed set in fixed drains by a replay server
    report     read the runs' critic diagnostics and apply the pre-registered rule
    evaluate   score the saved critics (and an untrained control) on fresh games: explained
               variance overall and in the game's last kyoku, where under H it can approach 1

From the repo root, in the `mortal` env, one step at a time (never alongside training or the
lab: 15 GB of RAM):
    python experiments/k1_lr_check.py generate --games 100000 --procs 2
    python experiments/k1_lr_check.py run --lr 1e-4 --workers 1 --file-batch 16
    python experiments/k1_lr_check.py run --lr 3e-4 --workers 1 --file-batch 16
    python experiments/k1_lr_check.py report
    python experiments/k1_lr_check.py evaluate --games 4000
"""
import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from socketserver import BaseRequestHandler, ThreadingTCPServer

ROOT = Path(__file__).resolve().parents[1]
P1 = ROOT / 'experiments' / 'configs' / 'p1.toml'
PILOT_A = ROOT / 'workdir' / 'pilot' / 'qs-a' / 'pilot-qs-a-final.pth'
KEY = int.from_bytes(b'k1lrchk', 'big')  # outside the lab's seed families
SEED = 1
LRS = ('1e-4', '3e-4')


def mortal_path():
    sys.path.insert(0, str(ROOT / 'mortal'))
    os.environ.setdefault('MORTAL_CFG', str(P1))


def git_commit():
    return subprocess.run(['git', 'describe', '--always', '--dirty', '--abbrev=12'], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()


# ---------------------------------------------------------------- generate

def engines(device):
    """The trainee and the baseline exactly as TrainPlayer builds them, both from pilot A."""
    import torch
    from engine import MortalEngine
    from model import Brain, CategoricalPolicy

    def nets():
        state = torch.load(PILOT_A, weights_only=False, map_location='cpu')
        cfg = state['config']
        brain = Brain(version=cfg['control'].get('version', 1), conv_channels=cfg['resnet']['conv_channels'],
                      num_blocks=cfg['resnet']['num_blocks'], Norm='GN').eval()
        policy = CategoricalPolicy().eval()
        brain.load_state_dict(state['mortal'])
        policy.load_state_dict(state['policy_net'])
        return brain.to(device), policy.to(device), cfg['control'].get('version', 1)

    brain, policy, version = nets()
    trainee = MortalEngine(brain, policy, is_oracle=False, version=version, explore_rate=1.0,
                           device=device, enable_amp=True, name='trainee')
    brain, policy, version = nets()
    baseline = MortalEngine(brain, policy, is_oracle=False, version=version, device=device, enable_amp=True,
                            enable_rule_based_agari_guard=True, name='baseline')
    return trainee, baseline


def cmd_gen_worker(a):
    mortal_path()
    import prelude  # noqa: F401
    import torch
    from libriichi.arena import OneVsThree
    trainee, baseline = engines(torch.device('cuda:0'))
    pool = Path(a.out) / a.pool_name
    done = 0
    while done < a.sets:
        n = min(a.chunk, a.sets - done)
        OneVsThree(disable_progress_bar=True, log_dir=str(pool / 'logs'), behaviour_dir=str(pool / 'bp')).py_vs_py(
            challenger=trainee, champion=baseline, seed_start=(a.start + done, KEY), seed_count=n)
        done += n
        print(f'{time.strftime("%H:%M:%S")} worker {a.start}: {done}/{a.sets} sets', flush=True)
    return 0


def cmd_generate(a):
    out = Path(a.out)
    pool = out / a.pool_name
    if pool.exists():
        raise SystemExit(f'{pool} exists; the game set is generated once')
    pool.mkdir(parents=True)
    sets = a.games // 4
    per = math.ceil(sets / a.procs)
    t0 = time.time()
    procs = []
    for k in range(a.procs):
        n = min(per, sets - k * per)
        log = open(out / f'gen-{a.pool_name}-{k}.log', 'w')
        procs.append(subprocess.Popen(
            [sys.executable, __file__, '--out', str(out), 'gen-worker', '--start', str(a.start + k * per),
             '--sets', str(n), '--chunk', str(a.chunk), '--pool-name', a.pool_name], cwd=ROOT, stdout=log,
            stderr=subprocess.STDOUT))
    codes = [p.wait() for p in procs]
    if any(codes):
        raise SystemExit(f'generation workers exited {codes}; see {out}/gen-*.log')
    logs = sorted(os.listdir(pool / 'logs'))
    sidecars = set(os.listdir(pool / 'bp'))
    missing = [x for x in logs if x not in sidecars]
    manifest = {'games': len(logs), 'missing_sidecars': len(missing), 'key': KEY, 'sets': sets,
                'procs': a.procs, 'seconds': round(time.time() - t0), 'commit': git_commit(), 'policy': str(PILOT_A)}
    (out / f'{a.pool_name}.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(json.dumps(manifest))
    return 0 if len(logs) == a.games and not missing else 1


# ---------------------------------------------------------------- run

class Replay:
    """Stands in for server.py: each drain hands the trainer the next `size` logs of the pool,
    in a fixed order, with their sidecars; parameter submissions are accepted and dropped."""

    def __init__(self, logs, pool, drain_dir, size):
        self.logs, self.pool, self.size = logs, pool, size
        self.drain_dir, self.bp_dir = drain_dir, Path(str(drain_dir) + '_behaviour')
        self.pos = 0
        self.version = 0
        self.drains = 0
        self.done = threading.Event()

    def drain(self):
        for d in (self.drain_dir, self.bp_dir):
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True)
        batch = self.logs[self.pos:self.pos + self.size]
        if not batch:
            self.done.set()  # the trainer has finished every drain it was given
        for name in batch:
            os.link(self.pool / 'logs' / name, self.drain_dir / name)
            os.link(self.pool / 'bp' / name, self.bp_dir / name)
        self.pos += len(batch)
        self.drains += bool(batch)
        return {'count': len(batch), 'drain_dir': str(self.drain_dir), 'behaviour_drain_dir': str(self.bp_dir),
                'param_versions': {name: 0 for name in batch}, 'param_version': self.version}


def serve(replay, port):
    from common import recv_msg, send_msg

    class Handler(BaseRequestHandler):
        def handle(self):
            msg = recv_msg(self.request)
            if msg['type'] == 'submit_param':
                replay.version += 1
            elif msg['type'] == 'drain':
                send_msg(self.request, replay.drain())

    server = ThreadingTCPServer(('127.0.0.1', port), Handler, bind_and_activate=False)
    server.allow_reuse_address = True
    server.daemon_threads = True
    server.server_bind()
    server.server_activate()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def run_config(run_dir, lr, port, window, workers=None, file_batch=None):
    import toml
    cfg = toml.load(P1)
    # data loading only: the workstation's 15 GB can't hold P1's cloud settings (2 workers ×
    # 50 logs); both runs get the same values
    if workers:
        cfg['dataset']['num_workers'] = workers
    if file_batch:
        cfg['dataset']['file_batch_size'] = file_batch
    c = cfg['control']
    c['state_file'] = str(run_dir / 'mortal.pth')
    c['best_state_file'] = str(run_dir / 'best.pth')
    c['tensorboard_dir'] = str(run_dir / 'tb')
    c['seed'] = SEED
    c['save_every'] = c['submit_every'] = window
    cfg['critic']['warmup'] = True
    cfg['critic']['optim']['scheduler'].update(peak=float(lr), final=float(lr))
    cfg['online']['remote']['port'] = port
    cfg['online']['server']['buffer_dir'] = str(run_dir / 'buffer')
    cfg['online']['server']['drain_dir'] = str(run_dir / 'drain')
    cfg['train_play']['default']['log_dir'] = str(run_dir / 'train_play')
    cfg['test_play']['log_dir'] = str(run_dir / 'test_play')
    return cfg


def mem_available_gb():
    with open('/proc/meminfo') as f:
        info = dict(line.split(':', 1) for line in f)
    return int(info['MemAvailable'].split()[0]) / 2**20


def session_pss_gb(sid):
    """Proportional set size of every process in the trainer's session (respawned child,
    loader workers): pages shared after fork are split between them, not counted per process."""
    total = 0
    for pid in filter(str.isdigit, os.listdir('/proc')):
        try:
            if os.getsid(int(pid)) != sid:
                continue
            with open(f'/proc/{pid}/smaps_rollup') as f:
                total += next(int(line.split()[1]) for line in f if line.startswith('Pss:'))
        except (OSError, ProcessLookupError, StopIteration):
            continue
    return total / 2**20


def cmd_run(a):
    out = Path(a.out)
    run_dir = out / f'lr-{a.lr}'
    if run_dir.exists():
        raise SystemExit(f'{run_dir} exists; runs are never overwritten')
    run_dir.mkdir(parents=True)
    import toml
    cfg = run_config(run_dir, a.lr, a.port, a.window, a.workers, a.file_batch)
    cfg_path = run_dir / 'config.toml'
    cfg_path.write_text(toml.dumps(cfg))
    os.environ['MORTAL_CFG'] = str(cfg_path)
    mortal_path()
    shutil.copy(PILOT_A, run_dir / 'mortal.pth')
    os.chmod(run_dir / 'mortal.pth', 0o644)

    logs = sorted(os.listdir(out / 'pool' / 'logs'))[:a.limit or None]
    replay = Replay(logs, out / 'pool', run_dir / 'drain', a.drain)
    server = serve(replay, a.port)
    t0 = time.time()
    with open(run_dir / 'trainer.log', 'w') as log:
        trainer = subprocess.Popen([sys.executable, 'train_online.py'], cwd=ROOT / 'mortal', stdout=log,
                                   stderr=subprocess.STDOUT, env={**os.environ, 'MORTAL_CFG': str(cfg_path)},
                                   start_new_session=True)
        peak_pss, min_avail, guard = 0.0, mem_available_gb(), False
        while not replay.done.is_set() and trainer.poll() is None:
            time.sleep(5)
            peak_pss = max(peak_pss, session_pss_gb(trainer.pid))
            min_avail = min(min_avail, mem_available_gb())
            if min_avail < a.min_available_gb:
                guard = True  # stop it ourselves, cleanly, before the system starts killing
                break
    code = trainer.poll()
    if code is None:  # finished every drain (or the guard fired): stop it, its child and loader workers
        os.killpg(trainer.pid, signal.SIGTERM)
        try:
            trainer.wait(60)
        except subprocess.TimeoutExpired:
            os.killpg(trainer.pid, signal.SIGKILL)
            trainer.wait()
    server.shutdown()
    rec = {'lr': a.lr, 'logs': len(logs), 'drains': replay.drains, 'drain_size': a.drain, 'window': a.window,
           'seed': SEED, 'finished': replay.done.is_set(), 'trainer_exit': code,
           'workers': cfg['dataset']['num_workers'], 'file_batch_size': cfg['dataset']['file_batch_size'],
           'peak_pss_gb': round(peak_pss, 2), 'min_available_gb': round(min_avail, 2), 'memory_guard': guard,
           'seconds': round(time.time() - t0), 'commit': git_commit()}
    (run_dir / 'run.json').write_text(json.dumps(rec, indent=1) + '\n')
    print(json.dumps(rec))
    return 0 if rec['finished'] else 1


# ---------------------------------------------------------------- report

def scalars(run_dir):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(str(run_dir / 'tb'), size_guidance={'scalars': 0})
    ea.Reload()
    return {tag: [(e.step, e.value) for e in ea.Scalars(tag)] for tag in
            ('critic/explained_variance', 'critic/calibration_monotone', 'critic/loss') if tag in ea.Tags()['scalars']}


def summarize(s):
    ev = s['critic/explained_variance']
    steps = [x for x, _ in ev]
    lo, hi = steps[0], steps[-1]
    frac = lambda f: lo + f * (hi - lo)  # noqa: E731
    mean = lambda a, b: sum(v for x, v in ev if frac(a) <= x <= frac(b)) / max(1, sum(1 for x, _ in ev if frac(a) <= x <= frac(b)))  # noqa: E731
    values = [v for _, v in ev] + [v for _, v in s.get('critic/loss', [])]
    return {
        'windows': len(ev),
        'ev_last20': mean(0.8, 1.0),
        'ev_40_60': mean(0.4, 0.6),
        'ev_final': ev[-1][1],
        'calibration_monotone_final': bool(s['critic/calibration_monotone'][-1][1]),
        'finite': all(math.isfinite(v) for v in values),
        'loss_final': s['critic/loss'][-1][1] if s.get('critic/loss') else None,
    }


def cmd_report(a):
    out = Path(a.out)
    res = {lr: summarize(scalars(out / f'lr-{lr}')) for lr in LRS if (out / f'lr-{lr}' / 'tb').exists()}
    for lr, r in res.items():
        print(f'lr {lr}: ' + ', '.join(f'{k} {v:.4f}' if isinstance(v, float) else f'{k} {v}' for k, v in r.items()))
    if set(res) != set(LRS):
        print('both runs are needed for the decision')
        return 1
    lo, hi = res['1e-4'], res['3e-4']
    stable = hi['finite'] and hi['ev_last20'] >= hi['ev_40_60'] - 0.02
    pick = '3e-4' if (hi['ev_last20'] - lo['ev_last20'] >= 0.01 and hi['calibration_monotone_final'] and stable) else '1e-4'
    decision = {'gap_last20': hi['ev_last20'] - lo['ev_last20'], 'stable_3e-4': stable, 'choice': pick, 'runs': res}
    (out / 'report.json').write_text(json.dumps(decision, indent=1) + '\n')
    print(f'gap (3e-4 − 1e-4) over the last 20%: {decision["gap_last20"]:+.4f}; 3e-4 stable: {stable}; choice: {pick}')
    return 0


# ---------------------------------------------------------------- evaluate

def cmd_evaluate(a):
    """The saved critics on fresh B0 games (a nonce range the check never used), through the
    trainer's own gae_targets; an untrained critic with the runs' seed is the control."""
    out = Path(a.out)
    fresh = out / 'fresh'
    if not fresh.exists():
        g = argparse.Namespace(out=a.out, games=a.games, procs=a.procs, chunk=500, start=100_000, pool_name='fresh')
        if cmd_generate(g):
            raise SystemExit('fresh generation failed')
    mortal_path()
    import prelude  # noqa: F401
    import numpy as np
    import toml
    import torch
    from critic import Critic, calibration, explained_variance, gae_targets
    from rl_dataset import OrderedDatasetsIter
    cfg = toml.load(P1)
    device = torch.device('cuda:0')
    critics = {}
    torch.manual_seed(SEED)
    critics['untrained'] = Critic(conv_channels=cfg['critic']['conv_channels'], num_blocks=cfg['critic']['num_blocks'])
    for lr in LRS:
        state = torch.load(out / f'lr-{lr}' / 'mortal.pth', weights_only=False, map_location='cpu')
        c = Critic(conv_channels=cfg['critic']['conv_channels'], num_blocks=cfg['critic']['num_blocks'])
        c.load_state_dict(state['critic'])
        critics[lr] = c
    for c in critics.values():
        c.to(device).eval()
    logs = sorted(str(fresh / 'logs' / n) for n in os.listdir(fresh / 'logs'))
    data = OrderedDatasetsIter(version=cfg['control']['version'], file_list=logs, behaviour_dir=str(fresh / 'bp'),
                               pts=cfg['env']['pts'], grp_state_file=cfg['grp']['state_file'],
                               grp_network=cfg['grp']['network'], horizon='H', targets='gae',
                               file_batch_size=16, obs_fp16=True)
    vals = {k: [] for k in critics}
    mc, last = [], []
    for chunk in data:
        for k, c in critics.items():
            flat = gae_targets(chunk, c, device=device, gamma=cfg['critic']['gamma'],
                               lam=cfg['critic']['gae_lambda'], enable_amp=True)
            vals[k].append(flat['value'].numpy())
        mc.append(flat['mc_return'].numpy())
        last.append(flat['last_kyoku'].numpy())
    mc, last = np.concatenate(mc), np.concatenate(last)
    res = {'decisions': int(len(mc)), 'last_kyoku_share': float(last.mean()), 'critics': {}}
    for k, v in vals.items():
        v = np.concatenate(v)
        res['critics'][k] = {
            'ev': explained_variance(v, mc),
            'ev_last_kyoku': explained_variance(v[last], mc[last]),
            'ev_earlier_kyoku': explained_variance(v[~last], mc[~last]),
            'calibration_monotone': calibration(v, mc)[2],
            'calibration_monotone_last_kyoku': calibration(v[last], mc[last])[2],
        }
    (out / 'evaluate.json').write_text(json.dumps(res, indent=1) + '\n')
    print(f'{res["decisions"]:,} decisions, {res["last_kyoku_share"]:.1%} in the last kyoku')
    for k, r in res['critics'].items():
        print(f'{k:>9}: EV {r["ev"]:+.4f}, last kyoku {r["ev_last_kyoku"]:+.4f}, earlier {r["ev_earlier_kyoku"]:+.4f}, '
              f'calibration monotone {r["calibration_monotone"]} (last kyoku {r["calibration_monotone_last_kyoku"]})')
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=str(ROOT / 'workdir' / 'k1-lr-check'))
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('generate')
    s.add_argument('--games', type=int, default=100_000)
    s.add_argument('--procs', type=int, default=2)
    s.add_argument('--chunk', type=int, default=500, help='duplicate sets per arena call')
    s.add_argument('--start', type=int, default=10_000, help='first nonce')
    s.add_argument('--pool-name', default='pool')
    s.set_defaults(fn=cmd_generate)
    s = sub.add_parser('gen-worker')
    s.add_argument('--start', type=int, required=True)
    s.add_argument('--sets', type=int, required=True)
    s.add_argument('--chunk', type=int, default=500)
    s.add_argument('--pool-name', default='pool')
    s.set_defaults(fn=cmd_gen_worker)
    s = sub.add_parser('run')
    s.add_argument('--lr', required=True, choices=LRS)
    s.add_argument('--drain', type=int, default=800, help='logs per drain (P1: 800-game client batches)')
    s.add_argument('--window', type=int, default=100, help='steps between diagnostics (save_every)')
    s.add_argument('--port', type=int, default=5101)
    s.add_argument('--limit', type=int, default=0, help='use only the first N logs (smoke tests)')
    s.add_argument('--workers', type=int, help='dataset.num_workers (default: p1.toml)')
    s.add_argument('--file-batch', type=int, help='dataset.file_batch_size (default: p1.toml)')
    s.add_argument('--min-available-gb', type=float, default=1.5,
                   help='stop the run cleanly if MemAvailable falls below this')
    s.set_defaults(fn=cmd_run)
    s = sub.add_parser('report')
    s.set_defaults(fn=cmd_report)
    s = sub.add_parser('evaluate')
    s.add_argument('--games', type=int, default=4000)
    s.add_argument('--procs', type=int, default=2)
    s.set_defaults(fn=cmd_evaluate)
    a = ap.parse_args()
    a.out = str(Path(a.out).resolve())  # the trainer runs from mortal/
    return a.fn(a)


if __name__ == '__main__':
    sys.exit(main())
