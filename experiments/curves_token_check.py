#!/usr/bin/env python3
"""The box check of server.py's trainer token (experiments/size-curves.md, "Before the first
rental", item 2), on a live fleet run:

1. the actors get the trainer's parameters (`param has been updated` in actor1's client log), so
   the trainer's own requests, which carry the token, are answered;
2. from actor1, through its tunnel, a `submit_param` and a `drain` without the token are both
   answered `denied`;
3. the learner's server.log shows both refused.

    python3 experiments/curves_token_check.py RUN [--wait-min 30]

Exit 0 if all three hold. It sends nothing a running fleet acts on: both requests are refused.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools' / 'fleet'))
import fleet as F  # noqa: E402

PROBE = '''cd {root}/mortal && MORTAL_CFG={cfg} {py} - <<'EOF'
import socket
from common import recv_msg, send_msg
from config import config
remote = ('127.0.0.1', config['online']['remote']['port'])
for msg in ({{'type': 'submit_param', 'mortal': {{}}, 'dqn': {{}}, 'is_idle': True}}, {{'type': 'drain'}}):
    with socket.create_connection(remote, timeout=60) as c:
        send_msg(c, msg)
        print('REPLY', msg['type'], recv_msg(c).get('status'))
EOF
'''


def updated(run, actor):
    out = F.remote(run, actor, f'grep -c "param has been updated" {F.BOX_DIR}/logs/client-1.log || true\n',
                   timeout=60, check=False).stdout.strip()
    return int(out or 0)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run')
    p.add_argument('--wait-min', type=float, default=30, help='how long to wait for the actors to get parameters')
    a = p.parse_args(argv)
    run = F.Run.load(a.run)
    actors = [b for b in run.live('actor') if b.get('status') == 'running']
    learner = run.boxes.get('learner')
    if not actors or not learner:
        print('no running actor and learner')
        return 1
    actor = actors[0]
    t0 = time.time()
    while (n := updated(run, actor)) == 0 and time.time() - t0 < a.wait_min * 60:
        time.sleep(60)
    print(f'1. {actor["name"]} got the trainer\'s parameters {n} times' if n else
          f'1. FAIL: no parameters on {actor["name"]} after {a.wait_min:g} min')
    cfg = f'{F.BOX_ROOT}/{F.BOX_DIR}/cfg/actor-1.toml'
    out = F.remote(run, actor, PROBE.format(root=F.BOX_ROOT, cfg=cfg, py=F.BOX_PY), timeout=180, check=False)
    replies = [line.split()[1:] for line in out.stdout.splitlines() if line.startswith('REPLY')]
    denied = replies == [['submit_param', 'denied'], ['drain', 'denied']]
    print(f'2. through the tunnel without the token: {replies or out.stderr.strip()[-300:]}')
    refused = F.remote(run, learner, f'grep -c "refused .* no valid token" {F.BOX_DIR}/logs/server.log || true\n',
                       timeout=60, check=False).stdout.strip()
    print(f'3. refusals in the learner\'s server.log: {refused}')
    ok = bool(n) and denied and int(refused or 0) >= 2
    run.event('token_check', passed=ok, updates=n, replies=repr(replies), refused=refused)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
