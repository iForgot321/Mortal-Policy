#!/usr/bin/env python3
"""The eval box (02 §5, 03 §2): one rented RTX 4090 that runs lab jobs from a queue, so evaluation
runs beside training instead of after it. In Phase 1 it also hosts the riichi.dev bot (07 §1).

    up      rent and set it up (fleet.py's offer ranking and setup_box.sh); takes the lab lease
    submit  copy the engines' checkpoints, sync the committed code if it moved, queue one lab job:
              evalbox.py submit -- measure --challenger workdir/x.pth --champion workdir/y.pth ...
            or, with --module and --python, another module; --priority 9 for background work
    status  the queue, the running job's last lines, GPU and disk
    pull    the registry rows (checked append-only), the lab run dirs, workdir/riichidev/ and
            the job logs
    down    pull, destroy, release the lease (refuses while jobs are queued or running, unless --force)

While the box is up it holds the lab registry. `lab/registry/*.jsonl` flow from the box to the
workstation only, and the lease (workdir/lab-lease.json) stops `python -m lab` on the workstation
from allocating nonces, so the two copies of the ledger can't fork. Code syncs leave the box's
registry alone. Rows written on the box carry its paths (/workspace/Mortal-Policy/...); the run dir
is workdir/lab/<run_id> on both machines, and engines are identified by their sha256.

State: workdir/fleet/runs/eval/ (fleet.json, fleet.log, events.jsonl, registry/ and labq/ as last
pulled). The box has no hard deadline: it bills until `down`.
"""
import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fleet as F  # noqa: E402

RUN = NAME = 'eval'
# only for fleet's input list (the GRP, the pilots and every frozen checkpoint go to the box)
CONFIG = 'experiments/configs/p0.toml'
LEASE = F.REPO / 'workdir' / 'lab-lease.json'
REGISTRY = F.REPO / 'lab' / 'registry'
LEDGERS = ('runs.jsonl', 'nonces.jsonl')
QDIR = 'workdir/labq'
RIICHIDEV_PY = '/workspace/venvs/riichidev/bin/python'  # riichienv on top of the image's Python
LAB_COMMANDS = ('measure', 'run', 'confirm', 'aa')
ENGINE_FLAGS = ('--challenger', '--champion', '--model')


def head_commit():
    """HEAD, refusing a working tree that is dirty outside lab/registry/ (the box gets the tree)."""
    out = subprocess.run(['git', 'status', '--porcelain'], cwd=F.REPO, capture_output=True, text=True).stdout
    dirty = [line for line in out.splitlines() if not line[3:].startswith('lab/registry/')]
    if dirty:
        raise F.FleetError('the working tree is dirty outside lab/registry/: commit first\n' + '\n'.join(dirty[:10]))
    return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=F.REPO, capture_output=True, text=True).stdout.strip()


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def resolve(rev):
    p = subprocess.run(['git', 'rev-parse', '--verify', '--quiet', f'{rev.removesuffix("-dirty")}^{{commit}}'],
                       cwd=F.REPO, capture_output=True, text=True)
    return p.stdout.strip() or None


def merge_ledgers(local_dir, box_dir):
    """Copy the box's ledgers over the local ones if each local file is a prefix of the box's.
    Checks every file before writing any. Returns {name: new lines}."""
    new = {}
    for name in LEDGERS:
        local, box = Path(local_dir) / name, Path(box_dir) / name
        lb = local.read_bytes() if local.exists() else b''
        bb = box.read_bytes() if box.exists() else b''
        if not bb.startswith(lb):
            raise F.FleetError(f'{name}: the workstation copy is not a prefix of the box copy, so the '
                               f'ledgers forked. Nothing copied; compare {local} with {box}.')
        new[name] = bb[len(lb):]
    for name, extra in new.items():
        if extra:
            local = Path(local_dir) / name
            tmp = local.with_suffix('.tmp')
            tmp.write_bytes((Path(box_dir) / name).read_bytes())
            os.replace(tmp, local)
    return {name: extra.decode().count('\n') for name, extra in new.items()}


def engine_files(lab_args):
    """The checkpoint files named by the job's engine flags (repo-relative; `builtin:` skipped)."""
    files = []
    for flag, value in zip(lab_args, lab_args[1:]):
        if flag in ENGINE_FLAGS and not value.startswith('builtin:'):
            path = value.partition('?')[0]
            if os.path.isabs(path):
                raise F.FleetError(f'{flag} {value}: give a repo-relative path (the box has another root)')
            if not (F.REPO / path).is_file():
                raise F.FleetError(f'{flag} {value}: no such file here')
            files.append(path)
    return files


def ready_box(run):
    box = run.boxes.get(NAME)
    if not box or box.get('status') != 'ready':
        raise F.FleetError(f'no ready eval box in {run.path}')
    return box


def ensure_runner(run, box):
    out = F.remote(run, box, f'mkdir -p {QDIR}\n'
                             f'if {F.BOX_PY} tools/fleet/labq.py status --json | grep -q \'"runner": true\'; then echo running\n'
                             f'else setsid -f {F.BOX_PY} tools/fleet/labq.py >> {QDIR}/runner.log 2>&1 < /dev/null; echo started; fi\n')
    return out.stdout.strip()


def sync_code(run, box):
    """Push the committed tree, minus the registry, if the box's commit differs; then rerun
    setup_box.sh (idempotent; rebuilds libriichi)."""
    head = head_commit()
    old = resolve(box.get('commit', ''))
    if old == head:
        return
    if old and not subprocess.run(['git', 'diff', '--quiet', old, head, '--', '.', ':(exclude)*.md',
                                   ':(exclude)lab/registry', ':(exclude)tools/fleet/evalbox.py'],
                                  cwd=F.REPO).returncode:
        return  # only notes, registry rows or this workstation-side script moved
    running = [j['id'] for j in box_status(run, box, as_json=True)['jobs'] if j['status'] == 'running']
    if running:
        # a lab row records the commit at its end: a sync under a running job would misstate it
        raise F.FleetError(f'HEAD moved, but job {running[0]} is running: submit again when it ends, '
                           f'or with --no-sync to queue on the box\'s commit {box.get("commit")}')
    F.retry(F.rsync, run, box, ['--filter=:- .gitignore', '--exclude=.claude/', '--exclude=.pytest_cache/',
                                '--exclude=/lab/registry/', './', F.on_box(box)])
    F.remote_step(run, box, 'setup', 'tools/fleet/setup_box.sh --skip-tests', 40)
    with run.lock:
        box['commit'] = head[:12]
        run.save()
    run.event('synced', NAME, commit=head[:12])
    if not old or subprocess.run(['git', 'diff', '--quiet', old, head, '--', 'tools/fleet/labq.py'],
                                 cwd=F.REPO).returncode:
        restart_runner(run, box)


def restart_runner(run, box):
    """Restart the queue runner so it loads new code, only while no job runs: a job outlives its
    runner, and the next runner would count it as dead and start another beside it."""
    st = box_status(run, box, as_json=True)
    if any(j['status'] == 'running' for j in st['jobs']):
        run.log('labq.py changed while a job runs: run `evalbox.py restart-runner` once it is idle', NAME)
        return False
    F.remote(run, box, f'pid=$(cat {QDIR}/runner.pid 2>/dev/null || true)\n'
                       'if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then kill "$pid"\n'
                       '    for i in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done\n'
                       'fi\n')
    run.log(f'runner restarted: {ensure_runner(run, box)}', NAME)
    return True


def pull(run, box):
    staged = run.dir / 'registry'
    staged.mkdir(exist_ok=True)
    F.retry(F.rsync, run, box, [F.on_box(box, 'lab/registry/'), f'{staged}/'])
    added = merge_ledgers(REGISTRY, staged)
    if any(added.values()):
        (REGISTRY / 'ablation.md').write_bytes((staged / 'ablation.md').read_bytes())
    (F.REPO / 'workdir' / 'lab').mkdir(parents=True, exist_ok=True)
    F.retry(F.rsync, run, box, [F.on_box(box, 'workdir/lab/'), str(F.REPO / 'workdir' / 'lab') + '/'], timeout=3600)
    (F.REPO / 'workdir' / 'riichidev').mkdir(parents=True, exist_ok=True)
    F.retry(F.rsync, run, box, ['--exclude=STOP', F.on_box(box, 'workdir/riichidev/'),
                                str(F.REPO / 'workdir' / 'riichidev') + '/'])
    F.retry(F.rsync, run, box, [F.on_box(box, f'{QDIR}/'), f'{run.dir}/labq/'])
    run.log(f'pulled: {added["runs.jsonl"]} new run rows, {added["nonces.jsonl"]} new nonce rows', NAME)
    return added


def box_status(run, box, as_json=False):
    p = F.remote(run, box, f'{F.BOX_PY} tools/fleet/labq.py status{" --json" if as_json else ""}\n', timeout=60)
    return json.loads(p.stdout) if as_json else p.stdout


def cmd_up(a):
    run = F.Run(RUN)
    if run.data is not None and run.boxes:
        raise F.FleetError(f'the eval box exists: {sorted(run.boxes)} in {run.path}')
    if LEASE.exists():
        raise F.FleetError(f'{LEASE} exists with no eval box; check, then remove it by hand')
    _, cfg = F.load_config(F.REPO / CONFIG)
    F.inputs(cfg, 'eval')
    commit = head_commit()
    if run.data is None:
        run.data = {'run': RUN, 'created': F.now_iso(), 'config': CONFIG, 'init': None, 'commit': commit[:12],
                    'overrides': {}, 'actors': 0, 'procs': 0, 'port': None, 'capacity': None,
                    'boxes': {}, 'retired': [], 'bad_machines': [], 'watchdog': None}
    offer = F.pick_offer(run, a.machine, a.allow_zen2)
    print(f'machine {offer["machine_id"]}  ${offer["dph_total"]:.3f}/h  {offer.get("geolocation")}  '
          f'{offer.get("cpu_name")}  ({offer.get("cpu_cores_effective", 0):.0f} cores)')
    if not a.yes:
        print('rerun with --yes to rent')
        return 0
    run.dir.mkdir(parents=True, exist_ok=True)
    (run.dir / 'known_hosts').mkdir(exist_ok=True)
    run.save()
    LEASE.write_text(json.dumps({'run': RUN, 'since': F.now_iso()}) + '\n')
    try:
        with run.locked():
            box = F.provision(run, NAME, 'eval', a.machine, a.skip_tests, a.allow_zen2)
            print(ensure_runner(run, box))
    finally:
        if not F.Run.load(RUN).boxes:
            LEASE.unlink()
            print('no box: lease released')
    return 0


def cmd_submit(a):
    args = a.lab_args[1:] if a.lab_args[:1] == ['--'] else a.lab_args
    if a.module == 'lab' and (not args or args[0] not in LAB_COMMANDS):
        raise F.FleetError(f'the job is a lab command: one of {", ".join(LAB_COMMANDS)}')
    files = sorted(set(engine_files(args)) | set(a.file))
    run = F.Run.load(RUN)
    with run.locked():
        box = ready_box(run)
        if not a.no_sync:
            sync_code(run, box)
        if files:
            F.retry(F.rsync, run, box, ['-R', *files, F.on_box(box)])
            out = F.remote(run, box, 'sha256sum ' + ' '.join(files) + '\n').stdout
            got = {line.split()[1]: line.split()[0] for line in out.splitlines() if line.strip()}
            for f in files:
                if got.get(f) != sha256(F.REPO / f):
                    raise F.FleetError(f'{f}: the box copy has another sha256')
        job = {'id': time.strftime('%Y%m%d-%H%M%S', time.gmtime()) + f'-{os.getpid() % 1000:03d}',
               'args': args, 'submitted': F.now_iso(), 'commit': box['commit'], 'priority': a.priority}
        if a.module != 'lab':
            job.update(module=a.module, python=a.python or RIICHIDEV_PY)
        b64 = base64.b64encode((json.dumps(job) + '\n').encode()).decode()
        F.remote(run, box, f'mkdir -p {QDIR}\necho {b64} | base64 -d >> {QDIR}/queue.jsonl\n')
        run.event('submitted', NAME, job=job['id'], args=repr(' '.join(args)))
        print(ensure_runner(run, box))
    return 0


def cmd_status(a):
    run = F.Run.load(RUN)
    box = ready_box(run)
    print(f'{NAME}: instance {box["instance"]}, machine {box["machine_id"]}, ${box["dph"]:.3f}/h, '
          f'commit {box.get("commit")}, ${F.cost(run):.2f} so far')
    print(box_status(run, box), end='')
    print(F.remote(run, box, 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader\n'
                             'df -h /workspace | tail -1\n', timeout=60).stdout, end='')
    return 0


def cmd_restart_runner(a):
    run = F.Run.load(RUN)
    with run.locked():
        return 0 if restart_runner(run, ready_box(run)) else 1


def cmd_pull(a):
    run = F.Run.load(RUN)
    with run.locked():
        pull(run, ready_box(run))
    return 0


def cmd_down(a):
    run = F.Run.load(RUN)
    with run.locked():
        box = run.boxes.get(NAME)
        if box:
            if box.get('status') == 'ready':
                st = box_status(run, box, as_json=True)
                busy = [j['id'] for j in st['jobs'] if j['status'] in ('queued', 'running')]
                if busy and not a.force:
                    raise F.FleetError(f'jobs queued or running: {busy}; wait, or --force')
                pull(run, box)
            elif not a.force:
                raise F.FleetError(f'the box is {box.get("status")}: nothing pulled; --force destroys it anyway')
            F.destroy_box(run, box)
        if LEASE.exists():
            LEASE.unlink()
            print('lease released: lab jobs can run on the workstation again')
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('up', help='rent and set up the eval box')
    s.add_argument('--machine', type=int)
    s.add_argument('--skip-tests', action='store_true')
    s.add_argument('--allow-zen2', action='store_true')
    s.add_argument('--yes', action='store_true', help='rent; without it, only show the offer')
    s.set_defaults(fn=cmd_up)
    s = sub.add_parser('submit', help='queue one lab job: submit [--file F] -- measure ...')
    s.add_argument('--file', action='append', default=[], help='another repo-relative file the job needs')
    s.add_argument('--module', default='lab', help='the module to run (default: the lab)')
    s.add_argument('--python', help=f'interpreter for --module (default: {RIICHIDEV_PY})')
    s.add_argument('--priority', type=int, default=0, help='lower runs first; 9 for background work')
    s.add_argument('--no-sync', action='store_true', help="queue on the box's commit even if HEAD moved")
    s.add_argument('lab_args', nargs=argparse.REMAINDER)
    s.set_defaults(fn=cmd_submit)
    sub.add_parser('status', help='the queue, GPU and disk').set_defaults(fn=cmd_status)
    sub.add_parser('pull', help='registry rows, lab run dirs, job logs').set_defaults(fn=cmd_pull)
    sub.add_parser('restart-runner', help='reload labq.py on the box (refuses while a job runs)'
                   ).set_defaults(fn=cmd_restart_runner)
    s = sub.add_parser('down', help='pull, destroy, release the lease')
    s.add_argument('--force', action='store_true')
    s.set_defaults(fn=cmd_down)
    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except F.FleetError as e:
        print(f'error: {e}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
