#!/usr/bin/env python3
"""A box's own hard deadline, so a run's billing doesn't depend only on the workstation.

`fleet.py start` (and `fleet.py deadline`) copies this script and deadline.json into
workdir/fleet-box/ and runs it in the tmux session `deadline`. deadline.json is
{"instance": id, "deadline": unix time, "action": "stop" | "destroy"}; it is re-read every
minute, and removing it disarms the box.

At the deadline the box asks Vast to stop (the learner, which keeps its checkpoints on a stopped
box) or destroy (actors) its own instance. It uses the key Vast injects into every container,
CONTAINER_API_KEY, which can only start, stop or destroy that one instance
(docs.vast.ai/guides/instances/docker-environment). No credential of ours is on the box.

The workstation's `down --after` watchdog fires an hour earlier and pulls results first, so this
only acts when the workstation is off or asleep by then. Stdlib only; the key is never logged.
"""
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone

BOX = 'workdir/fleet-box'
API = 'https://console.vast.ai/api/v0/instances/{}/'
ACTIONS = ('stop', 'destroy')


def log(msg):
    print(f'{datetime.now(timezone.utc).isoformat(timespec="seconds")} {msg}', flush=True)


def container_env(environ=os.environ, proc1='/proc/1/environ'):
    """CONTAINER_ID and CONTAINER_API_KEY from our environment, else from the container's PID 1."""
    env = {k: environ.get(k) for k in ('CONTAINER_ID', 'CONTAINER_API_KEY')}
    if not all(env.values()):
        try:
            with open(proc1, 'rb') as f:
                for item in f.read().split(b'\0'):
                    k, _, v = item.decode(errors='replace').partition('=')
                    if k in env and not env[k]:
                        env[k] = v
        except OSError:
            pass
    return env


def read_deadline(path):
    """The armed deadline, or None if disarmed (no file). Raises on a malformed file."""
    try:
        with open(path) as f:
            d = json.load(f)
    except FileNotFoundError:
        return None
    if d.get('action') not in ACTIONS:
        raise ValueError(f'unknown action {d.get("action")!r}')
    return {'instance': int(d['instance']), 'deadline': float(d['deadline']), 'action': d['action']}


def request(action, instance, key):
    """The HTTP request the Vast CLI sends for `vastai stop/destroy instance ID`."""
    if action == 'stop':
        method, body = 'PUT', {'state': 'stopped'}
    else:
        method, body = 'DELETE', {}
    return urllib.request.Request(API.format(instance), data=json.dumps(body).encode(), method=method,
                                  headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'})


def act(action, instance, key, urlopen=urllib.request.urlopen):
    with urlopen(request(action, instance, key), timeout=60) as r:
        return r.status, json.loads(r.read() or b'{}')


def main(path=f'{BOX}/deadline.json', poll_s=60, urlopen=urllib.request.urlopen, now=time.time, sleep=time.sleep):
    env = container_env()
    if not env['CONTAINER_ID'] or not env['CONTAINER_API_KEY']:
        log('no CONTAINER_ID / CONTAINER_API_KEY in the container: cannot arm')
        return 2
    announced = None
    while True:
        try:
            d = read_deadline(path)
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            log(f'bad {path}: {e}; disarmed')
            return 2
        if d is None:
            log(f'{path} removed: disarmed')
            return 0
        if d['instance'] != int(env['CONTAINER_ID']):
            log(f'{path} names instance {d["instance"]}, this is {env["CONTAINER_ID"]}: disarmed')
            return 2
        if (d['deadline'], d['action']) != announced:
            at = datetime.fromtimestamp(d['deadline'], timezone.utc).isoformat(timespec='seconds')
            log(f'armed: {d["action"]} instance {d["instance"]} at {at}')
            announced = (d['deadline'], d['action'])
        if now() >= d['deadline']:
            try:
                status, reply = act(d['action'], d['instance'], env['CONTAINER_API_KEY'], urlopen)
                log(f'{d["action"]}: HTTP {status}, success={reply.get("success")}')
                if d['action'] == 'destroy' and reply.get('success'):
                    return 0
            except Exception as e:  # noqa: BLE001 - keep trying until the box is gone or stopped
                log(f'{d["action"]} failed: {type(e).__name__}: {str(e)[:200]}')
            sleep(120)
            continue
        sleep(min(poll_s, max(1.0, d['deadline'] - now())))


if __name__ == '__main__':
    sys.exit(main())
