#!/usr/bin/env python3
"""Box side of tools/fleet/evalbox.py: run the lab jobs queued in workdir/labq/queue.jsonl one at a
time, in order, from the repo root. Per job: <id>.log (its output) and <id>.json (running, done or
failed, the exit code, times). One runner at a time (a lock); it polls the queue for new jobs.

    labq.py                 the runner (evalbox.py starts it detached)
    labq.py status [--json] the queue and the running job's last lines

One job at a time also keeps the nonce ledger safe: `allocate_nonces` reads it and appends without
a lock.
"""
import fcntl
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
Q = os.path.join(ROOT, 'workdir', 'labq')
POLL_S = 30


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def jobs():
    try:
        with open(os.path.join(Q, 'queue.jsonl')) as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        return []


def state(job_id):
    try:
        with open(os.path.join(Q, f'{job_id}.json')) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def write_state(job_id, st):
    path = os.path.join(Q, f'{job_id}.json')
    with open(path + '.tmp', 'w') as f:
        json.dump(st, f)
    os.replace(path + '.tmp', path)


def lock(block):
    f = open(os.path.join(Q, 'runner.lock'), 'a')
    try:
        fcntl.flock(f, fcntl.LOCK_EX | (0 if block else fcntl.LOCK_NB))
        return f
    except BlockingIOError:
        f.close()
        return None


def runner():
    os.makedirs(Q, exist_ok=True)
    held = lock(block=False)
    if held is None:
        print('another runner holds the lock')
        return 0
    with open(os.path.join(Q, 'runner.pid'), 'w') as f:
        f.write(f'{os.getpid()}\n')
    print(f'{now()} runner {os.getpid()} started', flush=True)
    while True:
        for job in jobs():
            st = state(job['id'])
            if st and st['status'] in ('done', 'failed'):
                continue
            if st and st['status'] == 'running':
                # left by a runner that died: the job died with it
                write_state(job['id'], {**st, 'status': 'failed', 'note': 'its runner died', 'ended': now()})
                continue
            st = {'status': 'running', 'started': now(), 'args': job['args']}
            write_state(job['id'], st)
            print(f'{now()} {job["id"]}: lab {" ".join(job["args"])}', flush=True)
            with open(os.path.join(Q, f'{job["id"]}.log'), 'ab') as log:
                rc = subprocess.run([sys.executable, '-u', '-m', 'lab', *job['args']], cwd=ROOT,
                                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT).returncode
            write_state(job['id'], {**st, 'status': 'done' if rc == 0 else 'failed', 'exit': rc, 'ended': now()})
            print(f'{now()} {job["id"]}: exit {rc}', flush=True)
            break  # re-read the queue
        else:
            time.sleep(POLL_S)


def status(as_json):
    rows = []
    for job in jobs():
        st = state(job['id']) or {'status': 'queued'}
        rows.append({'id': job['id'], 'status': st['status'], 'exit': st.get('exit'),
                     'started': st.get('started'), 'ended': st.get('ended'), 'args': ' '.join(job['args'])})
    os.makedirs(Q, exist_ok=True)
    probe = lock(block=False)
    alive = probe is None
    if probe:
        probe.close()
    if as_json:
        print(json.dumps({'runner': alive, 'jobs': rows}))
        return 0
    print(f'runner: {"running" if alive else "not running"}')
    for r in rows:
        print(f'{r["id"]}  {r["status"]:<7} {"" if r["exit"] is None else r["exit"]:>3}  '
              f'{r["started"] or "":<20}  {r["ended"] or "":<20}  {r["args"]}')
    for r in rows:
        if r['status'] == 'running':
            with open(os.path.join(Q, f'{r["id"]}.log'), errors='replace') as f:
                tail = f.read().splitlines()[-3:]
            print(f'{r["id"]}, last lines:', *tail, sep='\n  ')
    return 0


if __name__ == '__main__':
    if sys.argv[1:2] == ['status']:
        sys.exit(status('--json' in sys.argv))
    sys.exit(runner())
