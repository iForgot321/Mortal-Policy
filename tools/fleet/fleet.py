#!/usr/bin/env python3
"""Run one RL run's fleet of rented Vast boxes from the workstation (03 §2, §3.7; D48).

    up         rent a learner and N actors, set them up, join them, start the run
    provision  rent and set up one box (--start also joins and starts it)
    join       actors ↔ learner over restricted SSH tunnels (D48); --verify checks the restrictions
    start      learner: server.py + train_online.py; actor: N client.py processes (idempotent)
    stop       stop a box's processes; the box keeps billing
    health     probe every box; exit 1 if any is not ok/warn
    pull       copy the learner's run dir (checkpoints, tensorboard) and every box's logs back here
    snapshot   freeze the learner's checkpoint under a label, pull it, make it read-only
    watch      health and pull in a loop; restart dead processes; --replace replaces lost boxes;
               --budget-box-hours: budget snapshots (D49), then down at the budget
    down       pull, revoke and destroy boxes (all, or --name); --after delays it (the watchdog)
    deadline   arm every box's own hard deadline (box_deadline.py), an hour after the watchdog's
    status     the run state and the cost so far
    ssh        print the ssh command for a box

State lives in workdir/fleet/runs/<run>/: fleet.json (boxes, retired boxes, config), events.jsonl,
health.jsonl, fleet.log, known_hosts/, boxes/<name>-<instance>/ (logs), learner-run/ (the pulled
run dir), snapshots/ (frozen checkpoints, read-only; snapshots.jsonl). Commands that change state hold a lock on the run, so `watch` and a manual `down` don't
interleave. Rules (tools/fleet/README.md): no credentials or Tenhou data on boxes, no published
ports beyond SSH, commit before syncing, destroy boxes as soon as their results are back.
"""
import argparse
import base64
import fcntl
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / 'workdir' / 'fleet' / 'runs'
WS_ROOT = '/home/iforgot/projects/Mortal-Policy'  # the absolute root written in experiments/configs
BOX_ROOT = '/workspace/Mortal-Policy'
BOX_PY = '/venv/main/bin/python'
BOX_DIR = 'workdir/fleet-box'  # on a box, under BOX_ROOT: cfg/, logs/, manifest.json, tunnel.env
KEY = Path.home() / '.ssh' / 'vast_ed25519'
TUNNEL_USER = 'fleet'
TUNNEL_KEY = '/root/.ssh/fleet_tunnel'
TUNNEL_KNOWN_HOSTS = '/root/.ssh/known_hosts_fleet'
IMAGE = 'vastai/pytorch:2.6.0-cuda-12.4.1-py312-22.04'
# the Vast PyTorch template minus every -p port mapping (README: Creating an instance)
VAST_ENV = ('-e OPEN_BUTTON_PORT="1111" -e OPEN_BUTTON_TOKEN="1" -e JUPYTER_DIR="/" '
            '-e DATA_DIRECTORY="/workspace/" -e PORTAL_CONFIG="localhost:1111:11111:/:Instance Portal|'
            'localhost:8080:18080:/:Jupyter|localhost:8080:8080:/terminals/1:Jupyter Terminal|'
            'localhost:8384:18384:/:Syncthing|localhost:6006:16006:/:Tensorboard"')
OFFER_QUERY = ('gpu_name=RTX_4090 num_gpus=1 cpu_cores_effective>=32 cpu_ram>=64 cuda_vers>=12.4 '
               'reliability>0.98 disk_space>=50 direct_port_count>=1 inet_down>=500')
# eval-mode hanchan/h by CPU generation (experiments/fleet-host-benchmarks.md). Offers are ranked
# by $ per hanchan; other CPUs are skipped, Zen 2 unless --allow-zen2.
CPU_RATES = (
    (re.compile(r'EPYC 9\d\d[45]|Ryzen [579] [79]\d\d\d|Threadripper.* 7\d\d\d'), 50_000),  # Zen 4/5
    # Zen 3; custom parts carry a letter in the model number (7V13, 7B13)
    (re.compile(r'EPYC 7[\dA-Z]\d3|Ryzen [579] 5\d\d\d|Threadripper.* 5\d\d\d'), 40_000),
)
ZEN2 = (re.compile(r'EPYC 7[\dA-Z]\d2'), 31_000)  # e.g. 7742, 7B12, 7K62
WAIT_RUNNING_S = 25 * 60
NAME_RE = re.compile(r'^[a-z0-9][a-z0-9-]{0,31}$')
# health thresholds, seconds unless noted
LIMITS = {'submit_s': 1800, 'param_s': 1800, 'checkpoint_s': 3600, 'restarts_1h': 3,
          'unreachable_15m': 5, 'disk_gb': 5, 'mem_gb': 1.5, 'stall_s': 900}
SEVERITY = ('ok', 'warn', 'fail', 'unreachable', 'lost')
# a box stops (learner) or destroys (actor) itself this long after the run's hard deadline, in
# case the workstation is off by then (box_deadline.py)
BOX_DEADLINE_GRACE_S = 3600


class FleetError(Exception):
    pass


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


# ---------------------------------------------------------------- run state

class Run:
    def __init__(self, name):
        if not NAME_RE.match(name):
            raise FleetError(f'run name {name!r}: lowercase letters, digits and dashes only')
        self.name = name
        self.dir = RUNS / name
        self.path = self.dir / 'fleet.json'
        self.lock = threading.RLock()
        self.data = json.loads(self.path.read_text()) if self.path.exists() else None

    @classmethod
    def load(cls, name):
        run = cls(name)
        if run.data is None:
            raise FleetError(f'no run {name} ({run.path})')
        return run

    @property
    def boxes(self):
        return self.data['boxes']

    def save(self):
        with self.lock:
            tmp = self.path.with_suffix('.tmp')
            tmp.write_text(json.dumps(self.data, indent=1) + '\n')
            os.replace(tmp, self.path)

    @contextmanager
    def locked(self, wait=None):
        """Exclusive across processes; reloads the state once held. With `wait` (seconds), gives
        up waiting after that long and goes on unlocked (the watchdog must not hang)."""
        with open(self.dir / 'lock', 'a') as f:
            t0 = time.time()
            while True:
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | (fcntl.LOCK_NB if wait is not None else 0))
                    held = True
                    break
                except BlockingIOError:
                    if time.time() - t0 > wait:
                        self.log(f'run still locked after {wait} s; going on without the lock')
                        held = False
                        break
                    time.sleep(5)
            self.data = json.loads(self.path.read_text())
            try:
                yield self
            finally:
                if held:
                    fcntl.flock(f, fcntl.LOCK_UN)

    def log(self, msg, box=None):
        line = f'{datetime.now(timezone.utc):%H:%M:%S} [{box or self.name}] {msg}'
        print(line, flush=True)
        with self.lock, open(self.dir / 'fleet.log', 'a') as f:
            f.write(line + '\n')

    def event(self, kind, box=None, **kw):
        rec = {'time': now_iso(), 'kind': kind, **({'box': box} if box else {}), **kw}
        with self.lock, open(self.dir / 'events.jsonl', 'a') as f:
            f.write(json.dumps(rec) + '\n')
        self.log(f'{kind} ' + ' '.join(f'{k}={v}' for k, v in kw.items()), box)

    def known_hosts(self, box):
        return self.dir / 'known_hosts' / str(box['instance'])

    def box_dir(self, box):
        return self.dir / 'boxes' / f'{box["name"]}-{box["instance"]}'

    def live(self, role=None):
        """Boxes in play: not a learner `kept` for a late pull (see down())."""
        return [b for b in self.boxes.values()
                if b.get('status') != 'kept' and (role is None or b['role'] == role)]


def hard_deadline(run):
    """The run's hard deadline (`up --hard-hours`) as unix time, or None."""
    iso = run.data.get('hard_deadline') or (run.data.get('watchdog') or {}).get('deadline')
    return datetime.fromisoformat(iso).timestamp() if iso else None


def billed(run, until=None):
    """(box, hours billed up to `until`, default now) for every box the run has rented."""
    until = until or time.time()
    return [(b, max(0.0, min(b.get('destroyed_ts') or until, until) - b['created_ts']) / 3600)
            for b in [*run.boxes.values(), *run.data['retired']]
            if b.get('instance') and b.get('created_ts') and b['created_ts'] < until]


def cost(run):
    return sum(b['dph'] * h for b, h in billed(run))


def box_hours(run):
    """The run's compute so far in billed box-hours, failed attempts and replacements included:
    what an arm-seed's budget counts (05 §2.2, 3 boxes × 6 days = 432). Hours before a `rebase`
    (training discarded by a fault) don't count."""
    spent = sum(h for _, h in billed(run))
    rebase = run.data.get('budget_rebase')
    return spent - rebase['box_hours'] if rebase else spent


def snapshot_label(fraction):
    """`p25`, `p50`, ... for a budget fraction; `full` for the whole budget."""
    return 'full' if fraction >= 1 else f'p{round(fraction * 100)}'


# ---------------------------------------------------------------- the run's config

def load_config(path):
    """The run's TOML, checked for what the fleet relies on."""
    text = Path(path).read_text()
    cfg = tomllib.loads(text)
    if not cfg['control'].get('online'):
        raise FleetError(f'{path}: control.online must be true')
    if cfg['online']['remote']['host'] != '127.0.0.1':
        raise FleetError(f'{path}: online.remote.host must be 127.0.0.1 (D48)')
    if not cfg['control']['state_file'].startswith(WS_ROOT + '/workdir/'):
        raise FleetError(f'{path}: control.state_file must be under {WS_ROOT}/workdir/')
    return text, cfg


def set_toml_value(text, key, value):
    """`section.name = value` in a TOML text, on the line that sets it (a run's override; the
    checked-in config stays as it is). `value` is a TOML literal."""
    section, _, name = key.rpartition('.')
    pat = re.compile(rf"^(\[{re.escape(section)}\]\n(?:(?!\[).*\n)*?){re.escape(name)} = [^\n]*$", re.M)
    text, n = pat.subn(lambda m: f'{m.group(1)}{name} = {value}', text)
    if n != 1:
        raise FleetError(f'override {key}: no such key in the run\'s config')
    return text


def box_config(text, actor=None, overrides=None):
    """The run's config as a box sees it: the workstation root rewritten to BOX_ROOT, the run's
    overrides applied and, for actor process k, its own train_play dirs (client.py clears them
    before every batch)."""
    text = text.replace(WS_ROOT, BOX_ROOT)
    for key, value in (overrides or {}).items():
        text = set_toml_value(text, key, value)
    if actor is not None:
        pat = re.compile(r"^(\[train_play\.default\]\n(?:(?!\[).*\n)*?log_dir = '[^']*)(')", re.M)
        text, n = pat.subn(lambda m: f'{m.group(1)}-{actor}{m.group(2)}', text)
        if n != 1:
            raise FleetError("no log_dir under [train_play.default] in the run's config")
    return text


def frozen_inputs():
    """Every frozen checkpoint in tools/fleet/frozen.sha256. setup_box.sh checks their hashes, and
    the tests expect them, whichever pilot a run starts from."""
    lines = (REPO / 'tools' / 'fleet' / 'frozen.sha256').read_text().split('\n')
    return {line.split()[1] for line in lines if line.strip()}


def inputs(cfg, role, init=None):
    """Repo-relative files a box needs besides the code: the frozen checkpoints, the GRP and the
    baseline checkpoints, plus, for the learner, the run's initial checkpoint."""
    paths = {cfg['grp']['state_file'], cfg['baseline']['train']['state_file'],
             cfg['baseline']['test']['state_file']}
    rel = sorted({os.path.relpath(p, WS_ROOT) for p in paths} | frozen_inputs())
    if role == 'learner' and init:
        rel.append(init)
    for p in rel:
        if p.startswith('..') or not (REPO / p).is_file():
            raise FleetError(f'input {p} is not a file under the repo')
    return rel


# ---------------------------------------------------------------- vast

def vast(*args, timeout=120):
    try:
        p = subprocess.run(['vastai', *map(str, args), '--raw'], capture_output=True, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        raise FleetError(f'vastai {args[0]} {args[1]}: timed out') from None
    if p.returncode:
        raise FleetError(f'vastai {args[0]} {args[1]}: {(p.stderr or p.stdout).strip()[-300:]}')
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError:
        raise FleetError(f'vastai {args[0]} {args[1]}: not JSON: {p.stdout[:200]!r}') from None


def instances():
    """{instance id: record} of every instance on the account, or None if Vast can't say."""
    try:
        return {i['id']: i for i in vast('show', 'instances')}
    except FleetError:
        return None


def sweep(run):
    """Destroy any instance labelled for this run, recorded or not (the watchdog's backstop)."""
    listed = instances() or {}
    kept = {b['instance'] for b in run.boxes.values() if b.get('status') == 'kept'}
    for iid, inst in listed.items():
        if iid in kept:
            continue
        if (inst.get('label') or '').startswith(f'mp-{run.name}-'):
            ok = destroy_instance(iid)
            run.event('swept', instance=iid, label=inst.get('label'), destroyed=ok)


def cpu_rate(cpu_name, allow_zen2=False):
    for pat, rate in (*CPU_RATES, *((ZEN2,) if allow_zen2 else ())):
        if pat.search(cpu_name):
            return rate
    return None


def rank_offers(offers, exclude=(), allow_zen2=False):
    """Allowlisted offers, cheapest per hanchan first, each with est_rate and usd_per_m."""
    out = []
    for o in offers:
        rate = cpu_rate(o.get('cpu_name') or '', allow_zen2)
        if rate and o.get('machine_id') not in exclude:
            out.append({**o, 'est_rate': rate, 'usd_per_m': o['dph_total'] / rate * 1e6})
    return sorted(out, key=lambda o: o['usd_per_m'])


def pick_offer(run, machine=None, allow_zen2=False):
    """The best offer for a new box: on `machine` if given, else on a machine this run doesn't use
    and hasn't marked bad. Call with run.lock held, so parallel provisions pick different ones."""
    if machine:
        found = vast('search', 'offers', f'machine_id={machine} num_gpus=1 gpu_name=RTX_4090',
                     '--storage', 50)
        if not found:
            raise FleetError(f'no RTX 4090 offer on machine {machine}')
        return found[0]
    taken = {b.get('machine_id') for b in run.boxes.values()} | set(run.data['bad_machines'])
    ranked = rank_offers(vast('search', 'offers', OFFER_QUERY, '--storage', 50, '-o', 'dph_total'),
                         taken, allow_zen2)
    if not ranked:
        raise FleetError('no allowlisted offer')
    return ranked[0]


def destroy_instance(iid):
    """Destroy and confirm it is gone from the account; False if it is still there."""
    for _ in range(5):
        subprocess.run(['vastai', 'destroy', 'instance', str(iid), '-y'], capture_output=True,
                       text=True, timeout=120)
        for _ in range(6):
            listed = instances()
            if listed is not None and iid not in listed:
                return True
            time.sleep(10)
    return False


# ---------------------------------------------------------------- ssh

def ssh_opts(run, box):
    return ['-i', str(KEY), '-p', str(box['ssh_port']), '-o', 'IdentitiesOnly=yes',
            '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=accept-new',
            '-o', f'UserKnownHostsFile={run.known_hosts(box)}', '-o', 'ConnectTimeout=20',
            '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=4', '-o', 'LogLevel=ERROR']


PRELUDE = f'set -euo pipefail\ncd {BOX_ROOT} 2>/dev/null || true\n'


def remote(run, box, script, timeout=300, check=True):
    """Run a bash script on the box as root, from BOX_ROOT once it exists."""
    cmd = ['ssh', *ssh_opts(run, box), f'root@{box["ip"]}', 'bash -s']
    try:
        p = subprocess.run(cmd, input=PRELUDE + script, capture_output=True, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        p = subprocess.CompletedProcess(cmd, 124, '', f'timed out after {timeout} s')
    if check and p.returncode:
        raise FleetError(f'{box["name"]}: remote script exited {p.returncode}: '
                         f'{(p.stderr or p.stdout).strip()[-600:]}')
    return p


def rsync(run, box, args, timeout=1800):
    e = shlex.join(['ssh', *ssh_opts(run, box)])
    try:
        p = subprocess.run(['rsync', '-az', '-e', e, *args], cwd=REPO, capture_output=True,
                           text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise FleetError(f'{box["name"]}: rsync timed out') from None
    if p.returncode:
        raise FleetError(f'{box["name"]}: rsync exited {p.returncode}: {p.stderr.strip()[-400:]}')


def on_box(box, path=''):
    return f'root@{box["ip"]}:{BOX_ROOT}/{path}'


def put(run, box, path, text):
    """Write a file on the box (path relative to BOX_ROOT)."""
    b64 = base64.b64encode(text.encode()).decode()
    remote(run, box, f'mkdir -p "$(dirname {path})"\necho {b64} | base64 -d > {path}.tmp\n'
                     f'mv {path}.tmp {path}\n')


def remote_step(run, box, name, cmd, max_min):
    """A long command in a tmux session on the box, polled until it exits (survives ssh drops)."""
    inner = f'cd {BOX_ROOT} && ({cmd}) > workdir/box/{name}.log 2>&1; echo $? > workdir/box/{name}.exit'
    remote(run, box, f'mkdir -p workdir/box && rm -f workdir/box/{name}.exit\n'
                     f'tmux kill-session -t {name} 2>/dev/null || true\n'
                     f'tmux new-session -d -s {name} {shlex.quote(inner)}\n')
    fails = 0
    for _ in range(max_min * 2):
        time.sleep(30)
        out = remote(run, box, f'if [ -f workdir/box/{name}.exit ]; then echo EXIT $(cat workdir/box/{name}.exit); '
                               f'elif tmux has-session -t {name} 2>/dev/null; then echo RUNNING; else echo DEAD; fi\n',
                     timeout=60, check=False).stdout.strip()
        if out.startswith('EXIT '):
            if out != 'EXIT 0':
                raise FleetError(f'{box["name"]}: {name} exited {out[5:]} (workdir/box/{name}.log)')
            return
        if out == 'DEAD':
            raise FleetError(f'{box["name"]}: {name} died without an exit code')
        fails = 0 if out == 'RUNNING' else fails + 1
        if fails >= 10:
            raise FleetError(f'{box["name"]}: unreachable during {name}')
    raise FleetError(f'{box["name"]}: {name} still running after {max_min} min')


# ---------------------------------------------------------------- provisioning

def wait_running(run, box):
    t0 = time.time()
    while True:
        try:
            j = vast('show', 'instance', box['instance'])
        except FleetError:
            j = {}
        st = j.get('actual_status')
        if st == 'running' and j.get('ports', {}).get('22/tcp'):
            box['ip'] = j['public_ipaddr'].strip()
            box['ssh_port'] = int(j['ports']['22/tcp'][0]['HostPort'])
            break
        if st in ('exited', 'offline'):
            raise FleetError(f'{box["name"]}: instance is {st}')
        if time.time() - t0 > WAIT_RUNNING_S:
            raise FleetError(f'{box["name"]}: not running after {WAIT_RUNNING_S // 60} min '
                             f'({st}: {j.get("status_msg")})')
        time.sleep(15)
    run.save()
    # two answers 20 s apart: the image's entrypoint restarts sshd shortly after it first
    # answers, which broke a sync on 2026-10-08
    ok = 0
    for _ in range(40):
        ok = ok + 1 if remote(run, box, 'true\n', timeout=40, check=False).returncode == 0 else 0
        if ok == 2:
            return
        time.sleep(20 if ok else 10)
    raise FleetError(f'{box["name"]}: ssh never came up ({box["ip"]}:{box["ssh_port"]})')


def retry(fn, *args, tries=3, wait=15, **kw):
    """For idempotent steps (rsync, probes): a dropped connection shouldn't cost a box."""
    for i in range(tries):
        try:
            return fn(*args, **kw)
        except FleetError:
            if i == tries - 1:
                raise
            time.sleep(wait)


def sync(run, box):
    _, cfg = load_config(REPO / run.data['config'])
    retry(remote, run, box, 'command -v rsync >/dev/null || { apt-get update -qq && '
                            'DEBIAN_FRONTEND=noninteractive apt-get install -y -qq rsync >/dev/null; }\n',
          timeout=900)
    retry(rsync, run, box, ['--filter=:- .gitignore', '--exclude=.claude/', '--exclude=.pytest_cache/',
                            './', on_box(box)])
    retry(rsync, run, box, ['-R', *inputs(cfg, box['role'], run.data.get('init')), on_box(box)])
    # what the box actually got: a replacement syncs the working tree as it is now
    with run.lock:
        box['commit'] = git_commit(allow_dirty=True)
        run.save()


def provision(run, name, role, machine=None, skip_tests=False, allow_zen2=False, tries=3):
    """Rent, wait for, sync and set up one box under `name`. A failed attempt destroys its box,
    marks the machine bad for this run and, unless the machine was pinned, tries another."""
    for attempt in range(1, tries + 1):
        with run.lock:
            if name in run.boxes:
                raise FleetError(f'{name} exists (instance {run.boxes[name]["instance"]}); down it first')
            offer = pick_offer(run, machine, allow_zen2)
            box = {'name': name, 'role': role, 'instance': None, 'machine_id': offer['machine_id'],
                   'cpu_name': offer.get('cpu_name'), 'dph': offer['dph_total'], 'status': 'creating',
                   'created': now_iso(), 'created_ts': time.time()}
            run.boxes[name] = box
            run.save()
        try:
            # the reply also holds an instance API key: never log it
            reply = vast('create', 'instance', offer['id'], '--image', IMAGE, '--env', VAST_ENV,
                         '--onstart-cmd', 'entrypoint.sh', '--disk', 50, '--ssh', '--direct',
                         '--label', f'mp-{run.name}-{name}')
            if not reply.get('new_contract'):
                raise FleetError('create instance: no new_contract in the reply')
        except FleetError:
            with run.lock:
                del run.boxes[name]
                run.save()
            raise
        with run.lock:
            box['instance'] = reply['new_contract']
            run.save()
        run.event('created', name, instance=box['instance'], machine=box['machine_id'],
                  cpu=repr(box['cpu_name']), dph=round(box['dph'], 3), attempt=attempt)
        try:
            wait_running(run, box)
            run.log(f'running at {box["ip"]}:{box["ssh_port"]}', name)
            sync(run, box)
            run.log(f'synced {box["commit"]}; setting up', name)
            remote_step(run, box, 'setup', 'tools/fleet/setup_box.sh' + (' --skip-tests' if skip_tests else ''), 40)
            with run.lock:
                box['status'] = 'ready'
                run.save()
            run.event('ready', name, instance=box['instance'])
            return box
        except FleetError as e:
            run.event('provision_failed', name, instance=box['instance'], error=repr(str(e)[:300]))
            if box.get('ip'):
                try:
                    rsync(run, box, [f'root@{box["ip"]}:{BOX_ROOT}/workdir/box/', f'{run.box_dir(box)}/box/'], timeout=120)
                except FleetError:
                    pass
            with run.lock:
                run.data['bad_machines'].append(box['machine_id'])
            destroy_box(run, box)
            if machine:
                raise
    raise FleetError(f'{name}: provisioning failed {tries} times')


def destroy_box(run, box):
    iid = box.get('instance')
    if iid and not destroy_instance(iid):
        with run.lock:
            box['status'] = 'destroy-failed'
            run.save()
        raise FleetError(f'{box["name"]}: instance {iid} still listed; run: vastai destroy instance {iid} -y')
    with run.lock:
        box['status'] = 'destroyed'
        box['destroyed'], box['destroyed_ts'] = now_iso(), time.time()
        if run.boxes.get(box['name']) is box:
            del run.boxes[box['name']]
        run.data['retired'].append(box)
        run.save()
    run.event('destroyed', box['name'], instance=iid)


# ---------------------------------------------------------------- private network (D48)

def sshd_match_block(user, port):
    """What the learner's sshd allows the tunnel user: local forwards to server.py, nothing else."""
    return (f'Match User {user}\n'
            f'    AllowTcpForwarding local\n'
            f'    PermitOpen 127.0.0.1:{port}\n'
            f'    AllowAgentForwarding no\n'
            f'    AllowStreamLocalForwarding no\n'
            f'    X11Forwarding no\n'
            f'    PermitTTY no\n'
            f'    PermitTunnel no\n'
            f'    GatewayPorts no\n'
            f'    ForceCommand /bin/false\n')


def learner_sshd_script(user, port):
    """Idempotent: the tunnel user, its sshd Match block, a check of the effective settings;
    prints the learner's ed25519 host key."""
    return f'''U={user}; PORT={port}; MARK='# fleet tunnels (D48)'
PID=$(cat /run/sshd.pid 2>/dev/null || pgrep -o -x sshd)
# the config the running daemon was started with (-f), else the default
CONF=$(tr '\\0' '\\n' < /proc/$PID/cmdline | sed -n '/^-f$/{{n;p;q}}')
CONF=${{CONF:-/etc/ssh/sshd_config}}
id -u "$U" >/dev/null 2>&1 || useradd --create-home --shell /usr/sbin/nologin "$U"
usermod -p '*' "$U"   # no password, but not locked: sshd refuses locked accounts without PAM
install -d -m 700 -o "$U" -g "$U" "/home/$U/.ssh"
touch "/home/$U/.ssh/authorized_keys"
chown "$U:$U" "/home/$U/.ssh/authorized_keys"; chmod 600 "/home/$U/.ssh/authorized_keys"
if ! grep -qxF "$MARK" "$CONF"; then
    cp "$CONF" "$CONF.pre-fleet"
    printf '\\n%s\\n' "$MARK" >> "$CONF"
    echo {base64.b64encode(sshd_match_block(user, port).encode()).decode()} | base64 -d >> "$CONF"
    /usr/sbin/sshd -t -f "$CONF" || {{ cp "$CONF.pre-fleet" "$CONF"; echo "sshd rejected the fleet block" >&2; exit 1; }}
    kill -HUP "$PID"   # the daemon re-executes itself with the new config; open sessions stay
    sleep 1
fi
eff=$(/usr/sbin/sshd -T -f "$CONF" -C user=$U,host=actor,addr=192.0.2.1)
for want in 'allowtcpforwarding local' "permitopen 127.0.0.1:$PORT" 'permittty no' 'forcecommand /bin/false'; do
    grep -qxF "$want" <<<"$eff" || {{ echo "effective sshd config for $U lacks: $want" >&2; exit 1; }}
done
cat /etc/ssh/ssh_host_ed25519_key.pub
'''


def tunnel_comment(run, box):
    return f'fleet-{run.name}-{box["name"]}-{box["instance"]}'


def authorized_line(pub, port, comment):
    """The learner's authorized_keys line for one actor key: forwarding to server.py only."""
    kind, blob = pub.split()[:2]
    if kind != 'ssh-ed25519' or not re.fullmatch(r'[A-Za-z0-9+/]+={0,2}', blob):
        raise FleetError(f'unexpected tunnel key {pub[:40]!r}')
    return f'restrict,port-forwarding,permitopen="127.0.0.1:{port}",command="/bin/false" {kind} {blob} {comment}'


def tunnel_verify_script(ssh, port):
    """Run on an actor: its key authenticates, runs no command, forwards to server.py and to
    nothing else, and can't open a remote forward. `ssh` is the actor's ssh command line to the
    learner, without a command."""
    return f'''S={shlex.quote(ssh)}
fail=0
out=$(timeout 30 $S echo FLEET-SHELL-OPEN 2>&1 </dev/null) && rc=0 || rc=$?
if [ $rc -eq 1 ] && ! grep -q FLEET-SHELL-OPEN <<<"$out"; then echo "ok   key authenticates; commands don't run";
else echo "FAIL command check: exit $rc: $out"; fail=1; fi
out=$(timeout 30 $S -W 127.0.0.1:22 2>&1 </dev/null) || true
if grep -q 'administratively prohibited' <<<"$out"; then echo "ok   forwarding to :22 refused";
else echo "FAIL forwarding to :22 not refused: $out"; fail=1; fi
out=$(timeout 30 $S -W 127.0.0.1:{port} 2>&1 </dev/null) || true
if grep -qE 'administratively prohibited|Permission denied|Host key' <<<"$out"; then echo "FAIL forwarding to :{port}: $out"; fail=1;
else echo "ok   forwarding to :{port} allowed"; fi
out=$(timeout 20 $S -N -o ExitOnForwardFailure=yes -R 127.0.0.1:15999:127.0.0.1:22 2>&1 </dev/null) && rc=0 || rc=$?
if [ $rc -ne 124 ] && grep -q 'remote port forwarding failed' <<<"$out"; then echo "ok   remote forwarding refused";
else echo "FAIL remote forwarding: exit $rc: $out"; fail=1; fi
exit $fail
'''


def actor_ssh_command(learner):
    return (f'ssh -i {TUNNEL_KEY} -p {learner["ssh_port"]} -o IdentitiesOnly=yes -o BatchMode=yes '
            f'-o StrictHostKeyChecking=yes -o UserKnownHostsFile={TUNNEL_KNOWN_HOSTS} '
            f'-o HostKeyAlgorithms=ssh-ed25519 -o ConnectTimeout=20 {TUNNEL_USER}@{learner["ip"]}')


def learner_host_key(run, learner):
    """The learner's ed25519 host key, proven by connecting with only that key trusted."""
    port = run.data['port']
    out = remote(run, learner, learner_sshd_script(TUNNEL_USER, port), timeout=120).stdout.strip()
    kind, blob = out.splitlines()[-1].split()[:2]
    if kind != 'ssh-ed25519':
        raise FleetError(f'learner host key: unexpected {kind!r}')
    with tempfile.NamedTemporaryFile('w', suffix='.known_hosts') as f:
        f.write(f'[{learner["ip"]}]:{learner["ssh_port"]} {kind} {blob}\n')
        f.flush()
        opts = ssh_opts(run, learner)
        opts[opts.index(f'UserKnownHostsFile={run.known_hosts(learner)}')] = f'UserKnownHostsFile={f.name}'
        opts[opts.index('StrictHostKeyChecking=accept-new')] = 'StrictHostKeyChecking=yes'
        p = subprocess.run(['ssh', *opts, '-o', 'HostKeyAlgorithms=ssh-ed25519',
                            f'root@{learner["ip"]}', 'true'], capture_output=True, text=True, timeout=60)
    if p.returncode:
        raise FleetError(f'learner did not prove the ed25519 host key it reported: {p.stderr.strip()}')
    return blob


def join(run, names=None, verify=False):
    """Give each actor (all, or `names`) a tunnel to the current learner; repoints running ones."""
    learner = run.boxes.get('learner')
    if not learner or learner['status'] not in ('ready', 'running'):
        raise FleetError('no ready learner')
    port = run.data['port']
    blob = learner_host_key(run, learner)
    actors = [run.boxes[n] for n in names] if names else run.live('actor')
    for actor in actors:
        comment = tunnel_comment(run, actor)
        pub = remote(run, actor, f'[ -f {TUNNEL_KEY} ] || ssh-keygen -q -t ed25519 -N "" -C {comment} -f {TUNNEL_KEY}\n'
                                 f'cat {TUNNEL_KEY}.pub\n').stdout.strip()
        line = authorized_line(pub, port, comment)
        remote(run, learner, f"F=/home/{TUNNEL_USER}/.ssh/authorized_keys\nsed -i '/ {comment}$/d' \"$F\"\n"
                             f'echo {shlex.quote(line)} >> "$F"\n')
        env = (f'LEARNER_IP={learner["ip"]}\nLEARNER_SSH_PORT={learner["ssh_port"]}\nTUNNEL_USER={TUNNEL_USER}\n'
               f'LOCAL_PORT={port}\nREMOTE_PORT={port}\nKEY={TUNNEL_KEY}\nKNOWN_HOSTS={TUNNEL_KNOWN_HOSTS}\n')
        put(run, actor, f'{BOX_DIR}/tunnel.env', env)
        remote(run, actor, f'''mkdir -p {BOX_DIR}/logs
printf '%s\\n' '[{learner["ip"]}]:{learner["ssh_port"]} ssh-ed25519 {blob}' > {TUNNEL_KNOWN_HOSTS}
old=$(cat {BOX_DIR}/tunnel.pid 2>/dev/null || true)
if tmux has-session -t tunnel 2>/dev/null; then
    [ -n "$old" ] && kill "$old" 2>/dev/null || true
else
    tmux new-session -d -s tunnel "{BOX_ROOT}/tools/fleet/box_tunnel.sh {BOX_ROOT}/{BOX_DIR}/tunnel.env >> {BOX_ROOT}/{BOX_DIR}/logs/tunnel.log 2>&1"
fi
for i in $(seq 1 60); do
    new=$(cat {BOX_DIR}/tunnel.pid 2>/dev/null || true)
    if [ -n "$new" ] && [ "$new" != "$old" ] && grep -q '0100007F:{port:04X} 00000000:0000 0A' /proc/net/tcp; then exit 0; fi
    sleep 1
done
echo "tunnel not up after 60 s:" >&2; tail -5 {BOX_DIR}/logs/tunnel.log >&2; exit 1
''', timeout=120)
        with run.lock:
            actor['joined_to'] = learner['instance']
            run.save()
        run.event('joined', actor['name'], learner=learner['instance'])
        if verify:
            p = remote(run, actor, tunnel_verify_script(actor_ssh_command(learner), port), timeout=180, check=False)
            for line in p.stdout.splitlines():
                run.log(f'verify: {line}', actor['name'])
            if p.returncode:
                raise FleetError(f'{actor["name"]}: tunnel restrictions failed verification')


def revoke(run, learner, actor):
    remote(run, learner, f"F=/home/{TUNNEL_USER}/.ssh/authorized_keys\n"
                         f"[ -f \"$F\" ] && sed -i '/ {tunnel_comment(run, actor)}$/d' \"$F\" || true\n", timeout=60)
    run.event('revoked', actor['name'], instance=actor['instance'])


# ---------------------------------------------------------------- processes

def box_paths(run):
    _, cfg = load_config(REPO / run.data['config'])
    return {k: cfg['control'][k].replace(WS_ROOT, BOX_ROOT) for k in ('state_file',)}


def start(run, box):
    """Start the box's fleet processes under box_supervise.sh; running ones are left alone."""
    text, cfg = load_config(REPO / run.data['config'])
    port = run.data['port']
    cfg_dir, logs = f'{BOX_ROOT}/{BOX_DIR}/cfg', f'{BOX_ROOT}/{BOX_DIR}/logs'
    state_file = cfg['control']['state_file'].replace(WS_ROOT, BOX_ROOT)
    if box['role'] == 'learner':
        configs = {'learner.toml': box_config(text, overrides=run.data.get('overrides'))}
        # name: (kind, tmux session, script, config)
        procs = {'server': ('server', 'learner', 'server.py', 'learner.toml'),
                 'trainer': ('trainer', 'learner', 'train_online.py', 'learner.toml')}
        sessions = {'learner': f'{BOX_ROOT}/tools/fleet/box_learner.sh {BOX_PY} {cfg_dir}/learner.toml {port} {logs}'}
    elif box['role'] == 'actor':
        ks = range(1, run.data['procs'] + 1)
        configs = {f'actor-{k}.toml': box_config(text, actor=k, overrides=run.data.get('overrides')) for k in ks}
        procs = {f'client-{k}': ('client', f'client-{k}', 'client.py', f'actor-{k}.toml') for k in ks}
        sessions = {f'client-{k}': f'env MORTAL_CFG={cfg_dir}/actor-{k}.toml {BOX_PY} client.py' for k in ks}
    else:
        raise FleetError(f'{box["name"]}: nothing to start for role {box["role"]}')
    for name, body in configs.items():
        put(run, box, f'{BOX_DIR}/cfg/{name}', body)
    put(run, box, f'{BOX_DIR}/manifest.json', json.dumps({
        'run': run.name, 'name': box['name'], 'role': box['role'], 'port': port, 'started': time.time(),
        'state_file': state_file if box['role'] == 'learner' else None,
        'procs': {n: {'kind': k, 'session': s, 'script': sc, 'cfg': f'{cfg_dir}/{c}', 'log': f'{BOX_DIR}/logs/{n}.log'}
                  for n, (k, s, sc, c) in procs.items()},
    }, indent=1))
    lines = [f'mkdir -p {BOX_DIR}/logs',
             'start() {  # session, command',
             '    if tmux has-session -t "$1" 2>/dev/null; then echo "$1 already running"; return; fi',
             f'    tmux new-session -d -s "$1" -c {BOX_ROOT}/mortal "{BOX_ROOT}/tools/fleet/box_supervise.sh $1 {logs}/$1.log $2"',
             '    echo "$1 started"',
             '}']
    if box['role'] == 'learner':
        init = run.data.get('init')
        lines += [learner_init_line(state_file, f'{BOX_ROOT}/{init}' if init else None)]
    else:
        lines += ['tmux has-session -t tunnel 2>/dev/null || { echo "no tunnel: join first" >&2; exit 1; }']
    lines += [f'start {s} {shlex.quote(cmd)}' for s, cmd in sessions.items()]
    out = remote(run, box, '\n'.join(lines) + '\n', timeout=180).stdout.strip()
    with run.lock:
        box['status'] = 'running'
        run.save()
    run.event('started', box['name'], procs=','.join(procs), detail=repr(out.replace('\n', '; ')))
    try:
        arm_deadline(run, box)
    except FleetError as e:
        # the workstation watchdog still holds the deadline; say so and carry on
        run.event('deadline_failed', box['name'], error=repr(str(e)[:300]))


def deadline_spec(run, box):
    """deadline.json for box_deadline.py, or None when the run has no hard deadline."""
    hard = hard_deadline(run)
    if hard is None:
        return None
    return {'instance': box['instance'], 'deadline': hard + BOX_DEADLINE_GRACE_S,
            # a stopped learner keeps its checkpoints for a late pull; actors hold nothing to keep
            'action': 'stop' if box['role'] == 'learner' else 'destroy'}


def arm_deadline(run, box):
    """Write box_deadline.py and deadline.json into BOX_DIR and start it if it isn't running.
    Both go outside the synced tree, so a live run's code is untouched. Idempotent."""
    spec = deadline_spec(run, box)
    if spec is None:
        return None
    put(run, box, f'{BOX_DIR}/box_deadline.py', (Path(__file__).parent / 'box_deadline.py').read_text())
    put(run, box, f'{BOX_DIR}/deadline.json', json.dumps(spec) + '\n')
    cmd = f'{BOX_PY} {BOX_DIR}/box_deadline.py >> {BOX_DIR}/logs/deadline.log 2>&1'
    remote(run, box, f'mkdir -p {BOX_DIR}/logs\n'
                     f'tmux has-session -t deadline 2>/dev/null || '
                     f'tmux new-session -d -s deadline -c {BOX_ROOT} {shlex.quote(cmd)}\n'
                     'sleep 2\ntmux has-session -t deadline\n', timeout=60)
    at = datetime.fromtimestamp(spec['deadline'], timezone.utc).isoformat(timespec='seconds')
    run.event('deadline_armed', box['name'], instance=box['instance'], action=spec['action'], at=at)
    return spec


def learner_init_line(state_file, init):
    """Bash: copy the run's init to the learner's state file only if that file is missing.

    The copy is grouped in braces: `[ -f s ] || [ -f i ] && cp …` parses as
    `([ -f s ] || [ -f i ]) && cp …` and overwrote a trained checkpoint on every learner start
    (run p0-a, 2026-10-10)."""
    copy = (f'{{ [ -f {init} ] && mkdir -p {os.path.dirname(state_file)} && cp {init} {state_file} '
            f'&& chmod u+w {state_file}; }}' if init else 'false')
    return f'[ -f {state_file} ] || {copy} || {{ echo "no {state_file} and no init to copy" >&2; exit 1; }}'


def stop(run, box, tunnel=False):
    names = 'learner' if box['role'] == 'learner' else ' '.join(f'client-{k}' for k in range(1, run.data['procs'] + 1))
    if tunnel:
        names += ' tunnel'
    remote(run, box, f'for s in {names}; do tmux kill-session -t $s 2>/dev/null && echo "stopped $s" || true; done\n', timeout=60)
    with run.lock:
        box['status'] = 'ready'
        run.save()
    run.event('stopped', box['name'], procs=names.replace(' ', ','))


# ---------------------------------------------------------------- results

PULL_EXCLUDES = ['--exclude=buffer*/', '--exclude=drain*/', '--exclude=train_play*/', '--exclude=test_play*/',
                 '--exclude=*.tmp']


def pull(run, boxes=None):
    """Logs of every box, and the learner's run dir (checkpoints, tensorboard) into learner-run/."""
    errors = []
    for box in boxes or list(run.boxes.values()):
        if not box.get('ip'):
            continue
        try:
            if box.get('status') == 'kept':
                refresh_address(run, box)
            dst = run.box_dir(box)
            dst.mkdir(parents=True, exist_ok=True)
            rsync(run, box, [f'root@{box["ip"]}:{BOX_ROOT}/{BOX_DIR}/logs/', f'{dst}/logs/'], timeout=600)
            rsync(run, box, ['--include=*.log', '--include=info.txt', '--exclude=*', f'root@{box["ip"]}:{BOX_ROOT}/workdir/box/', f'{dst}/box/'], timeout=120)
            if box['role'] == 'learner':
                src = os.path.dirname(box_paths(run)['state_file'])
                (run.dir / 'learner-run').mkdir(exist_ok=True)
                rsync(run, box, [*PULL_EXCLUDES, f'root@{box["ip"]}:{src}/', f'{run.dir}/learner-run/'], timeout=1800)
        except FleetError as e:
            errors.append(str(e))
    if errors:
        raise FleetError('; '.join(errors))
    run.log('pulled ' + ', '.join(b['name'] for b in (boxes or run.boxes.values())))


def refresh_address(run, box):
    """A restarted instance may come back on another address: look it up, or fail if it isn't
    running (a kept learner is started by hand: `vastai start instance ID`)."""
    j = vast('show', 'instance', box['instance'])
    if j.get('actual_status') != 'running' or not j.get('ports', {}).get('22/tcp'):
        raise FleetError(f'{box["name"]}: {j.get("actual_status")}; start it first '
                         f'(vastai start instance {box["instance"]})')
    with run.lock:
        box['ip'] = j['public_ipaddr'].strip()
        box['ssh_port'] = int(j['ports']['22/tcp'][0]['HostPort'])
        run.save()


def snapshot(run, label):
    """Freeze the learner's current checkpoint as `label` (05 §2.2, D49). The copy is made on the
    box (the trainer replaces its state file atomically, so the copy is whole), pulled to
    snapshots/<label>.pth, checked against the box's sha256 and made read-only. A label is never
    overwritten."""
    if not NAME_RE.match(label):
        raise FleetError(f'snapshot label {label!r}: lowercase letters, digits and dashes only')
    if label in run.data.get('snapshots', {}):
        raise FleetError(f'snapshot {label} exists; labels are never overwritten')
    learner = run.boxes.get('learner')
    if not learner or not learner.get('ip'):
        raise FleetError('no learner to snapshot')
    state_file = box_paths(run)['state_file']
    snap = f'{BOX_ROOT}/{BOX_DIR}/snapshots/{run.name}-{label}.pth'
    out = remote(run, learner, f'''mkdir -p {BOX_DIR}/snapshots
[ -f {snap} ] || {{ cp {state_file} {snap}.tmp && mv {snap}.tmp {snap}; }}
sha256sum {snap} | cut -d' ' -f1
{BOX_PY} -c "import torch; print(torch.load('{snap}', weights_only=False, map_location='cpu')['steps'])"
''', timeout=300).stdout.split()
    sha, steps = out[-2], int(out[-1])
    local = run.dir / 'snapshots' / f'{label}.pth'
    local.parent.mkdir(exist_ok=True)
    if local.exists():
        raise FleetError(f'{local} exists but is not in the run state: move it away by hand')
    retry(rsync, run, learner, [f'root@{learner["ip"]}:{snap}', str(local)])
    got = subprocess.run(['sha256sum', str(local)], capture_output=True, text=True).stdout.split()[0]
    if got != sha:
        local.unlink()
        raise FleetError(f'snapshot {label}: sha256 {got[:12]} here, {sha[:12]} on the learner')
    os.chmod(local, 0o444)
    rec = {'label': label, 'time': now_iso(), 'sha256': sha, 'steps': steps, 'box_hours': round(box_hours(run), 2),
           'cost': round(cost(run), 2), 'learner': learner['instance'], 'path': os.path.relpath(local, REPO)}
    with run.lock:
        run.data.setdefault('snapshots', {})[label] = rec
        run.save()
        with open(run.dir / 'snapshots.jsonl', 'a') as f:
            f.write(json.dumps(rec) + '\n')
    run.event('snapshot', label=label, steps=steps, box_hours=rec['box_hours'], sha256=sha[:16])
    return rec


def push_state(run, learner):
    """A replacement learner continues from the last pulled checkpoint, if there is one."""
    state_file = box_paths(run)['state_file']
    local = run.dir / 'learner-run' / os.path.basename(state_file)
    if not local.exists():
        run.log('no pulled checkpoint; the learner starts from the run init', learner['name'])
        return
    remote(run, learner, f'mkdir -p {os.path.dirname(state_file)}\n')
    rsync(run, learner, [str(local), f'root@{learner["ip"]}:{state_file}'])
    run.event('pushed_state', learner['name'], age_s=round(time.time() - local.stat().st_mtime))


# ---------------------------------------------------------------- health

def assess(role, vast_status, probe, capacity=None, limits=LIMITS):
    """(verdict, problems): ok, warn, fail (box up, processes broken), unreachable, lost."""
    if vast_status not in ('running', '?'):
        return 'lost', [f'instance {vast_status or "gone"}']
    if probe is None:
        return 'unreachable', ['ssh probe failed']
    if not probe.get('started'):
        return 'fail', ['not started']
    fail, warn = [], []
    uptime = probe['time'] - probe.get('manifest_started', probe['time'])
    for name, p in probe['procs'].items():
        if not p['alive']:
            fail.append(f'{name} not running')
        if p['restarts_1h'] >= limits['restarts_1h']:
            warn.append(f'{name} restarted {p["restarts_1h"]}x in 1 h')
        if 'last_submit_age_s' in p:
            a = p['last_submit_age_s']
            if (a is None and uptime > limits['submit_s']) or (a is not None and a > limits['submit_s']):
                warn.append(f'{name} no submission for {a if a is not None else round(uptime)} s')
            if p['unreachable_15m'] >= limits['unreachable_15m']:
                warn.append(f'{name} server unreachable {p["unreachable_15m"]}x in 15 min')
        if 'last_param_age_s' in p:
            a = p['last_param_age_s']
            if (a is None and uptime > limits['param_s']) or (a is not None and a > limits['param_s']):
                warn.append(f'{name} no parameter submission for {a if a is not None else round(uptime)} s')
            if p['checkpoint_age_s'] is not None and p['checkpoint_age_s'] > limits['checkpoint_s']:
                warn.append(f'checkpoint {p["checkpoint_age_s"]} s old')
        if p.get('queue') is not None and capacity and p['queue'] >= capacity:
            warn.append(f'replay queue full ({p["queue"]}): actors blocked')
    # a trainer that is alive but has stopped submitting parameters while games pile up is wedged
    # (e.g. a DataLoader worker died): restartable, so `fail`, not `warn`
    trainer, server = probe['procs'].get('trainer', {}), probe['procs'].get('server', {})
    if (trainer.get('alive') and (trainer.get('last_param_age_s') or 0) > limits['stall_s']
            and capacity and (server.get('queue') or 0) >= capacity):
        fail.append('trainer stalled')
    if role == 'actor' and not (probe['tunnel']['alive'] and probe['tunnel']['listening']):
        fail.append('tunnel down')
    if role == 'learner' and not probe.get('server_listening'):
        fail.append('server.py not listening')
    if probe['disk_free_gb'] < limits['disk_gb']:
        warn.append(f'disk {probe["disk_free_gb"]} GB free')
    mem = probe.get('cgroup_mem_avail_gb', probe['mem_avail_gb'])
    if mem < limits['mem_gb']:
        warn.append(f'{mem} GB memory available')
    return ('fail' if fail else 'warn' if warn else 'ok'), fail + warn


def summary(role, probe):
    if not probe or not probe.get('started'):
        return ''
    gpu = probe['gpu'].get('util')
    common = f'GPU {gpu}%, CPU {probe.get("cpu_busy", "?")}, mem {probe.get("cgroup_mem_avail_gb", probe["mem_avail_gb"])} GB'
    procs = probe['procs']
    if role == 'actor':
        games = sum(p.get('games_1h', 0) for p in procs.values())
        blocked = sum(p.get('blocked_s_1h', 0) for p in procs.values()) / max(1, len(procs)) / 36
        ages = [p['last_submit_age_s'] for p in procs.values() if p.get('last_submit_age_s') is not None]
        return (f'{games:,} games in the last hour ({len(procs)} procs), blocked {blocked:.0f}%, '
                f'last submit {min(ages) if ages else "-"} s ago; {common}')
    s, t = procs.get('server', {}), procs.get('trainer', {})
    return (f'queue {s.get("queue")}, drained {s.get("drained_1h", 0):,}/h, params {t.get("last_param_age_s")} s ago, '
            f'checkpoint {t.get("checkpoint_age_s")} s ago; {common}')


def health(run, quiet=False):
    """{name: (verdict, problems, probe)}; appends a record to health.jsonl."""
    listed = instances()
    capacity = run.data.get('capacity')

    def one(box):
        if box['status'] == 'creating':
            return box['name'], ('ok', ['provisioning'], None)
        status = '?' if listed is None else (listed.get(box['instance']) or {}).get('actual_status')
        probe = None
        if box.get('ip'):
            p = remote(run, box, f'{BOX_PY} tools/fleet/box_health.py 2>/dev/null || python3 tools/fleet/box_health.py\n',
                       timeout=90, check=False)
            if p.returncode == 0:
                try:
                    probe = json.loads(p.stdout.strip().splitlines()[-1])
                except (json.JSONDecodeError, IndexError):
                    probe = None
        verdict, problems = assess(box['role'], status, probe, capacity)
        if box['status'] in ('creating', 'ready') and verdict == 'fail' and problems == ['not started']:
            verdict, problems = 'ok', ['not started']
        return box['name'], (verdict, problems, probe)

    boxes = run.live()
    with ThreadPoolExecutor(max(1, len(boxes))) as ex:
        report = dict(ex.map(one, boxes))
    with open(run.dir / 'health.jsonl', 'a') as f:
        f.write(json.dumps({'time': now_iso(), 'cost': round(cost(run), 3), 'boxes': {
            n: {'verdict': v, 'problems': pr, 'probe': pb} for n, (v, pr, pb) in report.items()}}) + '\n')
    if not quiet:
        for box in boxes:
            v, pr, pb = report[box['name']]
            run.log(f'{v:<11} {summary(box["role"], pb)}' + (f' | {"; ".join(pr)}' if pr else ''), box['name'])
        games = sum(p.get('games_1h', 0) for _, _, pb in report.values() if pb and pb.get('started')
                    for p in pb['procs'].values())
        run.log(f'{games:,} games in the last hour; ${sum(b["dph"] for b in boxes):.2f}/h; ${cost(run):.2f} so far')
    return report


# ---------------------------------------------------------------- recovery

def replace(run, name, since, allow_zen2=False):
    """Replace a lost box with a new one under the same name and bring it back into the run."""
    old = run.boxes[name]
    role = old['role']
    run.event('replacing', name, instance=old['instance'], lost_for_s=round(time.time() - since))
    learner = run.boxes.get('learner')
    if role == 'learner':
        try:
            pull(run, [old])
        except FleetError as e:
            run.log(f'pull from the lost learner failed: {e}', name)
    elif learner and learner is not old:
        try:
            revoke(run, learner, old)
        except FleetError as e:
            run.log(f'revoke failed: {e}', name)
    try:
        destroy_box(run, old)
    except FleetError as e:
        run.log(str(e), name)
        with run.lock:
            run.boxes.pop(name, None)
            run.data['retired'].append(old)
            run.save()
    # Zen 2 only for actors: the learner featurizes, and per-core speed matters there
    new = provision(run, name, role, allow_zen2=allow_zen2 and role == 'actor')
    if role == 'learner':
        push_state(run, new)
        start(run, new)
        if run.live('actor'):
            join(run)
    else:
        join(run, [name])
        start(run, new)
    run.event('recovered', name, instance=new['instance'], after_s=round(time.time() - since))


def kill_watchdog(run):
    wd = run.data.get('watchdog')
    if not wd or wd['pid'] == os.getpid():
        return
    try:
        cmdline = Path(f'/proc/{wd["pid"]}/cmdline').read_bytes().split(b'\0')
    except OSError:
        cmdline = []
    if b'down' in cmdline and run.name.encode() in cmdline:
        os.kill(wd['pid'], signal.SIGTERM)
        run.log(f'watchdog {wd["pid"]} stopped')
    run.data['watchdog'] = None
    run.save()


def spawn_watchdog(run, until):
    """A detached `down --after`: tears the run down at `until` (unix time) whatever else happens."""
    with open(run.dir / 'watchdog.log', 'a') as log:
        p = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'down', '--run', run.name,
                              '--after', str(int(until - time.time()))], cwd=REPO, start_new_session=True,
                             stdin=subprocess.DEVNULL, stdout=log, stderr=log)
    deadline = datetime.fromtimestamp(until, timezone.utc).isoformat(timespec='seconds')
    run.data['watchdog'] = {'pid': p.pid, 'deadline': deadline}
    run.data['hard_deadline'] = deadline
    run.save()
    run.event('watchdog', pid=p.pid, deadline=run.data['watchdog']['deadline'])


def down(run, names=None, do_pull=True):
    names = names or list(run.boxes)
    boxes = [run.boxes[n] for n in names if n in run.boxes]
    errors = []
    if do_pull:
        try:
            pull(run, boxes)
        except FleetError as e:
            run.log(f'pull failed: {e}')
            errors += keep_unpulled(run, boxes)
    learner = run.boxes.get('learner')
    if learner and learner not in boxes:
        for box in boxes:
            if box['role'] == 'actor':
                try:
                    revoke(run, learner, box)
                except FleetError as e:
                    run.log(f'revoke failed: {e}', box['name'])

    def one(box):
        try:
            destroy_box(run, box)
        except FleetError as e:
            errors.append(str(e))
    with ThreadPoolExecutor(max(1, len(boxes))) as ex:
        list(ex.map(one, boxes))
    if not run.live():
        sweep(run)
        kill_watchdog(run)
    run.log(f'down: {", ".join(b["name"] for b in boxes) or "nothing"}; ${cost(run):.2f} so far')
    if errors:
        raise FleetError('; '.join(errors))


def keep_unpulled(run, boxes):
    """After a failed pull: a learner that isn't running (stopped by its box-side deadline,
    exited, offline, or Vast can't say) still has the run's checkpoints on its disk, so it is
    taken out of `boxes` and kept for a human to start, pull and down. Returns the messages."""
    listed = instances()
    out = []
    for box in [b for b in boxes if b['role'] == 'learner']:
        status = '?' if listed is None else (listed.get(box['instance']) or {}).get('actual_status')
        if status is None or (status == 'running' and box.get('status') != 'kept'):
            continue  # gone, or up and merely unreachable: destroyed as before
        boxes.remove(box)
        with run.lock:
            box['status'] = 'kept'
            run.save()
        run.event('kept_unpulled', box['name'], instance=box['instance'], vast_status=status)
        out.append(f'{box["name"]}: not pulled and {status}; kept. vastai start instance {box["instance"]}, '
                   f'then fleet.py pull and down --run {run.name}')
    return out


# ---------------------------------------------------------------- commands

def git_commit(allow_dirty):
    dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=REPO, capture_output=True, text=True).stdout
    if dirty and not allow_dirty:
        raise FleetError('the working tree is dirty and boxes get it as is: commit first (or --allow-dirty)')
    return subprocess.run(['git', 'describe', '--always', '--dirty', '--abbrev=12'], cwd=REPO,
                          capture_output=True, text=True).stdout.strip()


def cmd_up(a):
    run = Run(a.run)
    if run.data is not None:
        raise FleetError(f'run {a.run} exists; use provision/join/start, or another name')
    config = os.path.relpath(Path(a.config).resolve(), REPO)
    _, cfg = load_config(REPO / config)
    if a.init:
        a.init = os.path.relpath(Path(a.init).resolve(), REPO)
    inputs(cfg, 'learner', a.init)
    commit = git_commit(a.allow_dirty)
    ranked = rank_offers(vast('search', 'offers', OFFER_QUERY, '--storage', 50, '-o', 'dph_total'),
                         allow_zen2=a.allow_zen2)
    n = 1 + a.actors
    seen = set()
    ranked = [o for o in ranked if not (o['machine_id'] in seen or seen.add(o['machine_id']))]
    print(f'{len(ranked)} allowlisted machines; the {n} best:')
    for o in ranked[:n]:
        print(f'  machine {o["machine_id"]:>6}  ${o["dph_total"]:.3f}/h  ~${o["usd_per_m"]:.1f}/M hanchan  {o["cpu_name"]}')
    if len(ranked) < n:
        raise FleetError(f'only {len(ranked)} allowlisted machines for {n} boxes')
    print(f'≈ ${sum(o["dph_total"] for o in ranked[:n]):.2f}/h for the fleet')
    if not a.yes:
        print('rerun with --yes to rent')
        return 0
    run.dir.mkdir(parents=True, exist_ok=True)
    (run.dir / 'known_hosts').mkdir(exist_ok=True)
    overrides = dict(kv.split('=', 1) for kv in a.set)
    for key, value in overrides.items():
        set_toml_value(Path(REPO / config).read_text(), key, value)  # fail before renting
    run.data = {'run': a.run, 'created': now_iso(), 'config': config, 'init': a.init, 'commit': commit,
                'overrides': overrides,
                'actors': a.actors, 'procs': a.procs, 'port': cfg['online']['remote']['port'],
                'capacity': cfg['online']['server']['capacity'], 'boxes': {}, 'retired': [],
                'bad_machines': [], 'watchdog': None}
    run.save()
    with run.locked():
        if a.hard_hours:
            spawn_watchdog(run, time.time() + a.hard_hours * 3600)
        names = [('learner', 'learner')] + [(f'actor{k}', 'actor') for k in range(1, a.actors + 1)]

        def prov(nr):
            try:
                return provision(run, nr[0], nr[1], skip_tests=a.skip_tests, allow_zen2=a.allow_zen2)
            except FleetError as e:
                run.log(f'gave up: {e}', nr[0])
                return None
        with ThreadPoolExecutor(len(names)) as ex:
            got = list(ex.map(prov, names))
        if got[0] is None:
            run.log('no learner: tearing down')
            down(run, do_pull=False)
            return 1
        if a.no_start:
            run.log(f'up (not started): {len(run.live("actor"))} of {a.actors} actors; ${cost(run):.2f} so far')
            return 0
        start(run, run.boxes['learner'])
        if run.live('actor'):
            join(run, verify=a.verify)
            for box in run.live('actor'):
                start(run, box)
        run.log(f'up: {len(run.live("actor"))} of {a.actors} actors; ${cost(run):.2f} so far')
    return 0 if len(run.live('actor')) == a.actors else 1


def cmd_provision(a):
    run = Run.load(a.run)
    with run.locked():
        run.data['commit'] = git_commit(a.allow_dirty)
        box = provision(run, a.name, a.role, a.machine, a.skip_tests, a.allow_zen2)
        if a.start:
            if a.role == 'learner':
                start(run, box)
                if run.live('actor'):
                    join(run)
            else:
                join(run, [a.name])
                start(run, box)
    return 0


def cmd_join(a):
    run = Run.load(a.run)
    with run.locked():
        join(run, a.name, verify=a.verify)
    return 0


def cmd_start(a):
    run = Run.load(a.run)
    with run.locked():
        for name in a.name or sorted(run.boxes, key=lambda n: n != 'learner'):
            start(run, run.boxes[name])
    return 0


def cmd_stop(a):
    run = Run.load(a.run)
    with run.locked():
        for name in a.name or list(run.boxes):
            stop(run, run.boxes[name], a.tunnel)
    return 0


def cmd_health(a):
    run = Run.load(a.run)
    report = health(run, quiet=a.json)
    if a.json:
        print(json.dumps({n: {'verdict': v, 'problems': p, 'probe': pb} for n, (v, p, pb) in report.items()}, indent=1))
    return 0 if all(v in ('ok', 'warn') for v, _, _ in report.values()) else 1


def cmd_pull(a):
    pull(Run.load(a.run))
    return 0


def cmd_snapshot(a):
    run = Run.load(a.run)
    with run.locked():
        snapshot(run, a.label)
    return 0


def cmd_watch(a):
    run = Run.load(a.run)
    deadline = time.time() + a.max_hours * 3600 if a.max_hours else None
    bad_since = {}
    last_pull = 0.0
    fractions = sorted(float(f) for f in a.snapshot_at.split(',')) if a.budget_box_hours else []
    full_failures = 0
    run.event('watch', interval=a.interval, replace=a.replace, max_hours=a.max_hours, max_dollars=a.max_dollars,
              budget_box_hours=a.budget_box_hours, snapshot_at=a.snapshot_at if a.budget_box_hours else None)
    while True:
        with run.locked():
            if not run.live():
                run.log('no boxes left; watch ends')
                return 0
            spent = cost(run)
            hard = hard_deadline(run)
            past_hard = hard is not None and time.time() > hard
            if (deadline and time.time() > deadline) or (a.max_dollars and spent > a.max_dollars) or past_hard:
                # past the hard deadline (the workstation slept through the watchdog): never replace
                run.event('watch_limit', spent=round(spent, 2), hard_deadline=past_hard)
                down(run)
                return 0
            report = health(run)
            learner_ok = report.get('learner', ('ok',))[0] in ('ok', 'warn')
            for name, (verdict, problems, _) in report.items():
                if verdict in ('ok', 'warn') or (name != 'learner' and not learner_ok):
                    # actors can't be judged while the learner is down
                    bad_since.pop(name, None)
                    continue
                t0 = bad_since.setdefault(name, time.time())
                waited = time.time() - t0
                box = run.boxes[name]
                try:
                    if verdict == 'fail' and waited < 2 * a.grace * 60:
                        # box up, processes not: a reboot, a dead tunnel or a wedged trainer
                        if box['role'] == 'actor' and 'tunnel down' in problems:
                            join(run, [name])
                        if 'trainer stalled' in problems:
                            stop(run, box)
                        start(run, box)
                    elif verdict in ('unreachable', 'lost', 'fail') and waited >= a.grace * 60:
                        if a.replace:
                            replace(run, name, t0, allow_zen2=a.allow_zen2)
                            bad_since.pop(name, None)
                        else:
                            run.log(f'{verdict} for {waited / 60:.0f} min; not replacing (no --replace)', name)
                except FleetError as e:
                    run.event('recovery_failed', name, error=repr(str(e)[:300]))
            if a.budget_box_hours:
                # D49: frozen checkpoints at fractions of the arm-seed's budget, then the final one
                used = box_hours(run)
                for f in fractions:
                    label = snapshot_label(f)
                    if used >= f * a.budget_box_hours and label not in run.data.get('snapshots', {}):
                        try:
                            snapshot(run, label)
                        except FleetError as e:
                            run.event('snapshot_failed', label=label, error=repr(str(e)[:300]))
                if used >= a.budget_box_hours and 'full' not in run.data.get('snapshots', {}):
                    try:
                        snapshot(run, 'full')
                    except FleetError as e:
                        full_failures += 1
                        run.event('snapshot_failed', label='full', error=repr(str(e)[:300]))
                # after 3 failed tries, down anyway: its pull still brings the final state file back
                if used >= a.budget_box_hours and ('full' in run.data.get('snapshots', {}) or full_failures >= 3):
                    run.event('budget_reached', box_hours=round(used, 1), cost=round(cost(run), 2))
                    down(run)
                    return 0
            if time.time() - last_pull >= a.pull_every * 60:
                try:
                    pull(run)
                except FleetError as e:
                    run.log(f'pull failed: {e}')
                last_pull = time.time()
        time.sleep(a.interval)


def cmd_down(a):
    if not a.after:
        run = Run.load(a.run)
        with run.locked():
            down(run, a.name, not a.no_pull)
        return 0
    # the watchdog: down everything at the deadline; if another command still holds the run
    # (a hung `up` or `watch`), go on without the lock and keep sweeping labelled instances
    # until that command is gone, so nothing it creates afterwards survives
    # wait on the wall clock: time.sleep runs on the monotonic clock, which can stop while the
    # workstation sleeps, and one long sleep would then fire late by the time asleep
    until = time.time() + a.after
    while (left := until - time.time()) > 0:
        time.sleep(min(left, 60))
    run = Run.load(a.run)
    run.data['watchdog'] = None  # don't let down() kill this process
    run.event('watchdog_fired')
    with run.locked(wait=600):
        run.data['watchdog'] = None
        try:
            down(run, do_pull=not a.no_pull)
        except FleetError as e:
            run.log(f'watchdog down: {e}')
    for _ in range(24):
        with open(run.dir / 'lock', 'a') as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                sweep(run)
                return 0
            except BlockingIOError:
                pass
        sweep(run)
        time.sleep(300)
    return 0


def cmd_deadline(a):
    """Arm (or --disarm) every live box's own deadline; --respawn-watchdog first restarts the
    workstation watchdog at the same deadline, so it runs the current code."""
    run = Run.load(a.run)
    with run.locked():
        hard = hard_deadline(run)
        if hard is None:
            raise FleetError(f'run {run.name} has no hard deadline (up --hard-hours)')
        if a.respawn_watchdog:
            if hard <= time.time():
                raise FleetError('the hard deadline has passed')
            kill_watchdog(run)
            spawn_watchdog(run, hard)
        for box in run.live():
            if not box.get('ip'):
                continue
            if a.disarm:
                remote(run, box, f'rm -f {BOX_DIR}/deadline.json\n', timeout=60)
                run.event('deadline_disarmed', box['name'], instance=box['instance'])
            else:
                arm_deadline(run, box)
    return 0


def cmd_status(a):
    run = Run.load(a.run)
    d = run.data
    print(f'run {d["run"]}: {d["config"]}, init {d["init"]}, commit {d["commit"]}, {d["procs"]} procs per actor, '
          f'created {d["created"]}')
    for b in run.boxes.values():
        up = (time.time() - b['created_ts']) / 3600
        print(f'  {b["name"]:<8} {b["status"]:<9} instance {b["instance"]}  machine {b["machine_id"]}  '
              f'${b["dph"]:.3f}/h  {up:.1f} h  {b.get("ip", "-")}:{b.get("ssh_port", "-")}  {b.get("commit", "-")}  '
              f'{b["cpu_name"]}')
    print(f'  retired: {len(d["retired"])}; bad machines: {d["bad_machines"]}')
    if d.get('budget_rebase'):
        r = d['budget_rebase']
        print(f'  budget rebased at {r["at"]}: {r["box_hours"]:.1f} box-hours before it excluded ({r["reason"]})')
    for s in d.get('snapshots', {}).values():
        print(f'  snapshot {s["label"]:<5} step {s["steps"]:,} at {s["box_hours"]} box-hours: {s["path"]} ({s["sha256"][:12]})')
    if d.get('watchdog'):
        print(f'  watchdog pid {d["watchdog"]["pid"]}, deadline {d["watchdog"]["deadline"]}')
    print(f'  {box_hours(run):.1f} box-hours, ${cost(run):.2f} so far (+ bandwidth); '
          f'${sum(b["dph"] for b in run.boxes.values()):.2f}/h now')
    return 0


def cmd_sync(a):
    """Push the working tree to running boxes (a hotfix); restart their processes to use it."""
    run = Run.load(a.run)
    with run.locked():
        commit = git_commit(a.allow_dirty)
        for name in a.name or list(run.boxes):
            box = run.boxes[name]
            retry(rsync, run, box, ['--filter=:- .gitignore', '--exclude=.claude/', '--exclude=.pytest_cache/',
                                    './', on_box(box)])
            with run.lock:
                box['commit'] = commit
                run.save()
            run.event('synced', name, commit=commit)
    return 0


def cmd_rebase(a):
    """Restart the run's compute budget at a time when training restarted from its init (a fault
    discarded the training before it); recorded with its reason."""
    run = Run.load(a.run)
    with run.locked():
        if run.data.get('budget_rebase'):
            raise FleetError('the budget was already rebased; a second rebase needs a decision')
        at = datetime.fromisoformat(a.at).timestamp()
        excluded = sum(h for _, h in billed(run, until=at))
        run.data['budget_rebase'] = {'at': a.at, 'box_hours': round(excluded, 3), 'reason': a.reason}
        run.save()
        run.event('budget_rebase', at=a.at, excluded_box_hours=round(excluded, 2), reason=repr(a.reason))
    return 0


def cmd_ssh(a):
    run = Run.load(a.run)
    box = run.boxes[a.name]
    print(shlex.join(['ssh', *ssh_opts(run, box)[:-2], f'root@{box["ip"]}']))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    def p(name, fn, help):
        s = sub.add_parser(name, help=help)
        s.add_argument('--run', required=True)
        s.set_defaults(fn=fn)
        return s
    s = p('up', cmd_up, 'rent and start a learner and actors')
    s.add_argument('--config', required=True, help='the run config, e.g. experiments/configs/p0.toml')
    s.add_argument('--init', help="checkpoint the learner copies to control.state_file if that is absent")
    s.add_argument('--actors', type=int, default=2)
    s.add_argument('--procs', type=int, default=3, help='client.py processes per actor box')
    s.add_argument('--hard-hours', type=float, help='detached watchdog: down everything after this long')
    s.add_argument('--verify', action='store_true', help='check the tunnel restrictions after joining')
    s.add_argument('--set', action='append', default=[], metavar='SECTION.KEY=VALUE',
                   help="override a config value for this run's boxes (TOML literal), e.g. dataset.num_workers=8")
    s.add_argument('--skip-tests', action='store_true')
    s.add_argument('--allow-zen2', action='store_true')
    s.add_argument('--allow-dirty', action='store_true')
    s.add_argument('--no-start', action='store_true', help='set the boxes up, but neither join nor start them')
    s.add_argument('--yes', action='store_true', help='rent; without it, only show the offers')
    s = p('provision', cmd_provision, 'rent and set up one box')
    s.add_argument('--name', required=True)
    s.add_argument('--role', required=True, choices=('learner', 'actor', 'eval'))
    s.add_argument('--machine', type=int)
    s.add_argument('--start', action='store_true', help='then join and start it')
    s.add_argument('--skip-tests', action='store_true')
    s.add_argument('--allow-zen2', action='store_true')
    s.add_argument('--allow-dirty', action='store_true')
    s = p('join', cmd_join, 'connect actors to the learner')
    s.add_argument('--name', action='append')
    s.add_argument('--verify', action='store_true')
    s = p('start', cmd_start, 'start fleet processes')
    s.add_argument('--name', action='append')
    s = p('stop', cmd_stop, 'stop fleet processes')
    s.add_argument('--name', action='append')
    s.add_argument('--tunnel', action='store_true', help='stop actor tunnels too')
    s = p('health', cmd_health, 'probe every box')
    s.add_argument('--json', action='store_true')
    p('pull', cmd_pull, 'copy results back')
    s = p('snapshot', cmd_snapshot, "freeze the learner's checkpoint under a label")
    s.add_argument('--label', required=True)
    s = p('watch', cmd_watch, 'health, pull and recovery in a loop')
    s.add_argument('--interval', type=int, default=300, help='seconds between checks')
    s.add_argument('--grace', type=float, default=10, help='minutes a box may be lost before replacement')
    s.add_argument('--pull-every', type=float, default=30, help='minutes')
    s.add_argument('--replace', action='store_true', help='replace lost boxes (rents new ones)')
    s.add_argument('--allow-zen2', action='store_true', help='replacement actors may be Zen 2 when supply is thin')
    s.add_argument('--max-hours', type=float)
    s.add_argument('--max-dollars', type=float)
    s.add_argument('--budget-box-hours', type=float,
                   help="the arm-seed's compute budget (05 §2.2: 3 boxes x 6 days = 432): snapshot at "
                        "--snapshot-at fractions, then `full`, then down")
    s.add_argument('--snapshot-at', default='0.25,0.5', help='budget fractions to snapshot at (D49)')
    s = p('down', cmd_down, 'pull, revoke and destroy')
    s.add_argument('--name', action='append')
    s.add_argument('--no-pull', action='store_true')
    s.add_argument('--after', type=int, default=0, help=argparse.SUPPRESS)
    s = p('deadline', cmd_deadline, "arm every box's own hard deadline (box_deadline.py)")
    s.add_argument('--disarm', action='store_true', help='remove it instead')
    s.add_argument('--respawn-watchdog', action='store_true',
                   help='first restart the workstation watchdog at the same deadline (to run current code)')
    p('status', cmd_status, 'run state and cost')
    s = p('sync', cmd_sync, 'push the working tree to running boxes (restart their processes to use it)')
    s.add_argument('--name', action='append')
    s.add_argument('--allow-dirty', action='store_true')
    s = p('rebase', cmd_rebase, 'restart the compute budget at a time training restarted from its init')
    s.add_argument('--at', required=True, help='ISO time, e.g. 2026-10-10T02:14:11+00:00')
    s.add_argument('--reason', required=True)
    s = p('ssh', cmd_ssh, 'print the ssh command for a box')
    s.add_argument('--name', required=True)
    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except FleetError as e:
        print(f'fleet: {e}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
