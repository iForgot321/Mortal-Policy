#!/usr/bin/env python3
"""The workstation smoke of D68's in-run re-anchoring (`experiments/size-curves.md`, "Before the
first rental", item 1). Plumbing, not an experiment: no registry rows, no strength claims.

server.py, train_online.py and one client.py from `experiments/configs/curves-smoke.toml`
(b12c256's EMA weights as a trainer's first state, saves every 5 steps), then, as `fleet.py
reanchor` would on a box (a copy of the state file, its sha256, `reanchor.json`):

1. at step ≥ 20, snapshot p25 and ask for it: the trainer re-anchors (stage 2) at its next save;
2. stop the trainer, move p25's file away and restart it: it resumes on stage 2 from its own
   checkpoint;
3. a request with a wrong sha256: refused, training goes on;
4. snapshot p50 and ask for it: stage 3;
5. a `submit_param` without the server's token, as a tunnel peer would send it: denied.

    python experiments/curves_smoke.py            # ≈ 15-25 min; writes workdir/curves-smoke/

Refuses to start with less than 8 GB of RAM available (no lab or training job beside it).
"""
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / 'experiments' / 'configs' / 'curves-smoke.toml'
RUN = ROOT / 'workdir' / 'curves-smoke'
STUDENT = ROOT / 'workdir' / 'secure' / 'runs' / 'students-s1' / 'box' / 'students-s1' / 'ema-b12c256.pth'
PY = sys.executable
PORT = 5099


def say(msg):
    print(f'{time.strftime("%H:%M:%S")} {msg}', flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def mem_available_gb():
    with open('/proc/meminfo') as f:
        info = dict(line.split(':', 1) for line in f)
    return int(info['MemAvailable'].split()[0]) / 2**20


class Proc:
    def __init__(self, name, script):
        self.name, self.log = name, RUN / f'{name}.log'
        env = {**os.environ, 'MORTAL_CFG': str(CFG)}
        self.p = subprocess.Popen([PY, script], cwd=ROOT / 'mortal', env=env, start_new_session=True,
                                  stdout=open(self.log, 'a'), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)

    def alive(self):
        return self.p.poll() is None

    def stop(self):
        if self.alive():
            os.killpg(self.p.pid, signal.SIGTERM)
            try:
                self.p.wait(30)
            except subprocess.TimeoutExpired:
                os.killpg(self.p.pid, signal.SIGKILL)
                self.p.wait()

    def text(self):
        return self.log.read_text(errors='replace')


def wait_for(what, cond, procs, timeout):
    """Poll `cond()` until true; fail if it times out or a process dies first."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if (got := cond()):
            return got
        for p in procs:
            if not p.alive():
                raise SystemExit(f'{p.name} died while waiting for {what}:\n{p.text()[-3000:]}')
        time.sleep(3)
    raise SystemExit(f'timed out after {timeout} s waiting for {what}')


def steps_of(trainer):
    found = re.findall(r'total steps: ([\d,]+)', trainer.text())
    return int(found[-1].replace(',', '')) if found else -1


def snapshot(label):
    """As fleet.py's snapshot() on a box: copy the state file whole, hash the copy."""
    snaps = RUN / 'snapshots'
    snaps.mkdir(exist_ok=True)
    dst = snaps / f'curves-smoke-{label}.pth'
    shutil.copy(RUN / 'mortal.pth', str(dst) + '.tmp')
    os.replace(str(dst) + '.tmp', dst)
    os.chmod(dst, 0o444)
    return dst, sha256(dst)


def request(label, file, sha):
    req = {'label': label, 'file': str(file), 'sha256': sha, 'issued': time.strftime('%Y-%m-%dT%H:%M:%S')}
    (RUN / 'reanchor.json.tmp').write_text(json.dumps(req) + '\n')
    os.replace(RUN / 'reanchor.json.tmp', RUN / 'reanchor.json')
    say(f'requested {label} ({sha[:12]})')


def tokenless_submit():
    sys.path.insert(0, str(ROOT / 'mortal'))
    os.environ['MORTAL_CFG'] = str(CFG)
    from common import recv_msg, send_msg
    with socket.create_connection(('127.0.0.1', PORT), timeout=30) as c:
        send_msg(c, {'type': 'submit_param', 'mortal': {}, 'dqn': {}, 'is_idle': True})
        return recv_msg(c)


def main():
    if (gb := mem_available_gb()) < 8:
        raise SystemExit(f'{gb:.1f} GB available: stop other jobs first (CLAUDE.md, Machine)')
    if (RUN / 'mortal.pth').exists():
        raise SystemExit(f'{RUN} has a run already; move it away first')
    RUN.mkdir(parents=True, exist_ok=True)
    init = RUN / 'init.pth'
    if not init.exists():
        subprocess.run([PY, str(ROOT / 'tools' / 'fleet' / 'stage_init.py'), '--from-policy', str(STUDENT), str(init)],
                       check=True)
    shutil.copy(init, RUN / 'mortal.pth')
    os.chmod(RUN / 'mortal.pth', 0o644)

    server = Proc('server', 'server.py')
    wait_for('the server', lambda: 'listening on' in server.text(), [server], 120)
    trainer = Proc('trainer', 'train_online.py')
    client = Proc('client', 'client.py')
    procs = [server, trainer, client]
    try:
        wait_for('the trainer at step 20', lambda: steps_of(trainer) >= 20, procs, 1800)
        say(f'trainer at step {steps_of(trainer)}')

        p25, sha25 = snapshot('p25')
        request('p25', p25, sha25)
        wait_for('stage 2', lambda: 're-anchored, stage 2' in trainer.text(), procs, 600)
        say('picked up: ' + re.findall(r're-anchored, stage 2.*', trainer.text())[-1])
        at = steps_of(trainer)
        wait_for('15 more steps', lambda: steps_of(trainer) >= at + 15, procs, 1200)

        say('restarting the trainer without p25 on disk')
        trainer.stop()
        os.chmod(p25, 0o644)
        os.replace(p25, RUN / 'p25-moved-away.pth')
        trainer = Proc('trainer', 'train_online.py')
        procs = [server, trainer, client]
        n = trainer.text().count('kl anchor stage 2')
        wait_for('the restart on stage 2', lambda: trainer.text().count('kl anchor stage 2') > n, procs, 600)
        at = steps_of(trainer)
        wait_for('10 steps after the restart', lambda: steps_of(trainer) >= at + 10, procs, 1200)

        p50, sha50 = snapshot('p50')
        request('p50', p50, '0' * 64)
        wait_for('the refusal', lambda: 're-anchor request refused' in trainer.text(), procs, 600)
        say('bad request refused')
        at = steps_of(trainer)
        wait_for('training past the refusal', lambda: steps_of(trainer) >= at + 5, procs, 600)
        request('p50', p50, sha50)
        wait_for('stage 3', lambda: 're-anchored, stage 3' in trainer.text(), procs, 600)
        say('picked up: ' + re.findall(r're-anchored, stage 3.*', trainer.text())[-1])
        at = steps_of(trainer)
        wait_for('10 more steps', lambda: steps_of(trainer) >= at + 10, procs, 1200)

        rsp = tokenless_submit()
        say(f'tokenless submit_param: {rsp}')
        if rsp != {'status': 'denied'}:
            raise SystemExit('the server took a submit_param without its token')
    finally:
        for p in (client, trainer, server):
            p.stop()

    print('\n--- reanchor.jsonl')
    print((RUN / 'reanchor.jsonl').read_text())
    print('--- anchor.jsonl (t, kl, agreement, coef)')
    for line in (RUN / 'anchor.jsonl').read_text().splitlines():
        r = json.loads(line)
        print(f'  {r["steps"]:>5} {r["kl"]:.5f} {r["agreement"]:.4f} {r["coef"]:.4f}')
    print('--- trainer.log (anchor lines)')
    for line in trainer.text().splitlines():
        if re.search(r'anchor|run clock|refused|denied|Traceback|Error', line):
            print('  ' + line[-200:])
    return 0


if __name__ == '__main__':
    sys.exit(main())
