#!/usr/bin/env python3
"""One box's health as JSON, for `tools/fleet/fleet.py health` (03 §2, §3.7).

Runs on the box from the repo root (stdlib only) and reads workdir/fleet-box/manifest.json, which
`fleet.py start` writes. Reports raw facts; fleet.py judges them:
- every fleet process: running (its tmux session and the process itself), restarts of its
  session (supervise.log), age of its last log line;
- client.py: last submission, games and blocked seconds in the last hour, recent connection errors;
- server.py: replay-queue length, logs drained in the last hour;
- train_online.py: last parameter submission, checkpoint age;
- the tunnel (actors) and the server port (learner): listening on loopback;
- GPU, memory (cgroup and host), disk, CPU busy over one second.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime

BOX = 'workdir/fleet-box'
# prelude.py's logging format (asctime, level, file:line). tqdm writes \r-separated bars into
# the same files, so a record can start mid-line.
TS = re.compile(rb'(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d{3} +[A-Z]+ +[\w.]+:\d+ +')
HOUR = 3600


def tail_lines(path, nbytes=1 << 20):
    try:
        with open(path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - nbytes))
            data = f.read()
    except FileNotFoundError:
        return []
    return [line for line in re.split(rb'[\r\n]+', data) if line]


def log_events(lines):
    """(unix time, message) for every logging record in the lines, oldest first."""
    out = []
    for line in lines:
        ms = list(TS.finditer(line))
        for i, m in enumerate(ms):
            end = ms[i + 1].start() if i + 1 < len(ms) else len(line)
            t = datetime.strptime(m.group(1).decode(), '%Y-%m-%d %H:%M:%S').timestamp()
            out.append((t, line[m.end():end].decode(errors='replace').strip()))
    return out


def last(events, needle):
    for t, msg in reversed(events):
        if needle in msg:
            return t, msg
    return None, None


def supervise_stats(lines, name, now):
    """Restarts of one process from supervise.log lines: `<iso> <name> exit <rc> after <s>s`."""
    exits = []
    for line in lines:
        parts = line.decode(errors='replace').split()
        if len(parts) >= 4 and parts[1] == name and parts[2] == 'exit':
            exits.append((datetime.fromisoformat(parts[0]).timestamp(), int(parts[3])))
    return {
        'restarts': len(exits),
        'restarts_1h': sum(1 for t, _ in exits if now - t < HOUR),
        'last_exit_code': exits[-1][1] if exits else None,
        'last_exit_age_s': round(now - exits[-1][0]) if exits else None,
    }


def listening(port, tcp_table):
    """Whether 127.0.0.1:<port> is in LISTEN state in a /proc/net/tcp table."""
    want = f'0100007F:{port:04X}'
    for line in tcp_table.splitlines()[1:]:
        parts = line.split()
        if len(parts) > 3 and parts[1] == want and parts[3] == '0A':
            return True
    return False


def age(now, t):
    return None if t is None else round(now - t)


def client_facts(events, now):
    t_submit, _ = last(events, 'logs have been submitted')
    games = blocked = 0
    for t, msg in events:
        if now - t >= HOUR:
            continue
        if msg.startswith('trainee rankings: ['):
            games += sum(int(x) for x in msg[len('trainee rankings: ['):].split(']')[0].split())
        elif msg.startswith('blocked '):
            blocked += int(msg.split()[1].rstrip('s'))
    return {
        'last_submit_age_s': age(now, t_submit),
        'games_1h': games,
        'blocked_s_1h': blocked,
        'unreachable_15m': sum(1 for t, msg in events if now - t < 900 and 'unreachable' in msg),
    }


def server_facts(events, now):
    _, msg = last(events, 'total buffer size: ')
    return {
        'queue': int(msg.rsplit(':', 1)[1]) if msg else None,
        'drained_1h': sum(int(msg.rsplit(':', 1)[1]) for t, msg in events
                          if now - t < HOUR and msg.startswith('files transferred to trainer:')),
    }


def trainer_facts(events, now, state_file):
    t_param, _ = last(events, 'param has been submitted')
    try:
        ckpt_age = round(now - os.path.getmtime(state_file))
    except (OSError, TypeError):
        ckpt_age = None
    return {'last_param_age_s': age(now, t_param), 'checkpoint_age_s': ckpt_age}


def tmux_alive(name):
    return subprocess.run(['tmux', 'has-session', '-t', name], capture_output=True).returncode == 0


def processes():
    """(argv, environ) of every process this probe can read."""
    out = []
    for pid in os.listdir('/proc'):
        if not pid.isdigit():
            continue
        try:
            with open(f'/proc/{pid}/cmdline', 'rb') as f:
                argv = f.read().split(b'\0')
            with open(f'/proc/{pid}/environ', 'rb') as f:
                env = f.read().split(b'\0')
        except OSError:
            continue
        out.append(([a.decode(errors='replace') for a in argv], {e.decode(errors='replace') for e in env}))
    return out


def running(procs, script, cfg):
    """Whether `python <script>` runs with MORTAL_CFG=<cfg> (the config is in the environment)."""
    return any(script in argv[1:2] and f'MORTAL_CFG={cfg}' in env for argv, env in procs)


def read(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return None


def system_facts():
    out = {}
    try:
        q = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=20)
        util, used, total = (int(x) for x in q.stdout.split(',')[:3])
        out['gpu'] = {'util': util, 'mem_used_mb': used, 'mem_total_mb': total}
    except Exception as e:  # noqa: BLE001 (a health probe reports, never raises)
        out['gpu'] = {'error': repr(e)}
    mem = dict(line.split(':', 1) for line in (read('/proc/meminfo') or '').splitlines() if ':' in line)
    out['mem_avail_gb'] = round(int(mem.get('MemAvailable', '0 kB').split()[0]) / 2**20, 1)
    cg_max, cg_cur = read('/sys/fs/cgroup/memory.max'), read('/sys/fs/cgroup/memory.current')
    if cg_max and cg_cur and cg_max.strip() != 'max':
        out['cgroup_mem_avail_gb'] = round((int(cg_max) - int(cg_cur)) / 2**30, 1)
    out['disk_free_gb'] = round(shutil.disk_usage('.').free / 2**30, 1)
    out['load1'] = round(os.getloadavg()[0], 1)

    def usage():
        stat = read('/sys/fs/cgroup/cpu.stat') or ''
        m = re.search(r'^usage_usec (\d+)$', stat, re.M)
        return int(m.group(1)) if m else None
    u0, t0 = usage(), time.monotonic()
    if u0 is not None:
        time.sleep(1)
        out['cpu_busy'] = round((usage() - u0) / 1e6 / (time.monotonic() - t0), 1)
    return out


def main():
    now = time.time()
    try:
        with open(f'{BOX}/manifest.json') as f:
            manifest = json.load(f)
    except FileNotFoundError:
        print(json.dumps({'time': now, 'started': False, **system_facts()}))
        return
    sup = tail_lines(f'{BOX}/logs/supervise.log')
    table = processes()
    procs = {}
    for name, spec in manifest['procs'].items():
        events = log_events(tail_lines(spec['log']))
        p = {'alive': tmux_alive(spec['session']) and running(table, spec['script'], spec['cfg']),
             **supervise_stats(sup, spec['session'], now),
             'last_log_age_s': age(now, events[-1][0] if events else None)}
        if spec['kind'] == 'client':
            p.update(client_facts(events, now))
        elif spec['kind'] == 'server':
            p.update(server_facts(events, now))
        elif spec['kind'] == 'trainer':
            p.update(trainer_facts(events, now, manifest.get('state_file')))
        lines = tail_lines(spec['log'], 4096)
        p['last_line'] = lines[-1].decode(errors='replace')[-200:] if lines else None
        procs[name] = p
    tcp = read('/proc/net/tcp') or ''
    out = {'time': now, 'started': True, 'manifest_started': manifest['started'], 'role': manifest['role'],
           'procs': procs}
    if manifest['role'] == 'actor':
        out['tunnel'] = {'alive': tmux_alive('tunnel'), 'listening': listening(manifest['port'], tcp)}
    else:
        out['server_listening'] = listening(manifest['port'], tcp)
    out.update(system_facts())
    print(json.dumps(out))


if __name__ == '__main__':
    sys.exit(main())
