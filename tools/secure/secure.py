#!/usr/bin/env python3
"""Secure-tier boxes for SL work on Tenhou data (D23; docs/design/secure-data-path-draft.md).

Hyperstack VMs in CANADA-1, driven from the workstation. The bucket credentials, the Hyperstack API
key and the bundle key stay here: a box gets the bundle through presigned URLs, and the key reaches
its gpg through SSH stdin and a pipe, never a file.

    keypair          make ~/.ssh/hyperstack_ed25519 and import it into CANADA-1 (once)
    up RUN           rent one VM (--flavor, default n3-L40x1) with SSH open to this workstation's
                     IPv4 only; start the watchdog (--hours) and the box's own shutdown an hour later
    setup RUN        sync the repo (a clean commit) and the frozen inputs, then
                     tools/secure/bootstrap_vm.sh: Python 3.12 + torch 2.6 cu124, tools/fleet/setup_box.sh
    ship RUN         unpack a bundle (--bundle, default the newest) into the box's data directory
                     (tools/secure/unpack.py) and write its file indexes
    exec RUN CMD     run CMD on the box, in the repo, as the login user (output here)
    pull RUN         copy the box's workdir/secure-box (outputs only, never data) back
    wipe RUN         delete the box's data directory
    down RUN         pull, wipe, destroy the VM and wait until it is gone; --after N waits first (the
                     watchdog). Safe to repeat
    status           every run: VM state, hours, cost so far
    ssh RUN          print the ssh command

State: workdir/secure/runs/RUN/ (state.json, known_hosts, logs, box/). Every box that is rented,
shipped data, wiped or destroyed is in workdir/secure/ledger.jsonl (append-only): D23's audit trail.
"""
import argparse
import fcntl
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hyperstack  # noqa: E402  (the API key and base URL)

REPO = Path(__file__).resolve().parents[2]
SECURE = REPO / 'workdir' / 'secure'
RUNS = SECURE / 'runs'
LEDGER = SECURE / 'ledger.jsonl'
BUNDLES = SECURE / 'bundle'
BUNDLE_KEY = Path.home() / '.config' / 'mortal-secure' / 'bundle.key'
S3_PY = Path.home() / '.venvs' / 's3' / 'bin' / 'python'
SSH_KEY = Path.home() / '.ssh' / 'hyperstack_ed25519'
KEYPAIR = 'mortal-secure'
REGION = 'CANADA-1'
ENVIRONMENT = 'default-CANADA-1'
IMAGE = 'Ubuntu Server 22.04 LTS R550 CUDA 12.4'
USER = 'ubuntu'
BOX_ROOT = '/workspace/Mortal-Policy'
BOX_OUT = 'workdir/secure-box'  # on the box, under BOX_ROOT: outputs only; `pull` copies it back
BOX_PY = '/venv/main/bin/python'
SHUTDOWN_GRACE_MIN = 60  # the box powers itself off this long after the watchdog's deadline


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def log(run, msg):
    line = f'{datetime.now(timezone.utc):%H:%M:%S} [{run}] {msg}'
    print(line, flush=True)
    d = RUNS / run
    d.mkdir(parents=True, exist_ok=True)
    with open(d / 'secure.log', 'a') as f:
        f.write(line + '\n')


def ledger(run, event, **kw):
    SECURE.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, 'a') as f:
        f.write(json.dumps({'time': now(), 'run': run, 'event': event, **kw}) + '\n')


def api(method, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(hyperstack.API + path, data=data, method=method,
                                 headers={'api_key': hyperstack.api_key(), 'Accept': 'application/json',
                                          'Content-Type': 'application/json', 'User-Agent': 'mortal-policy-secure/1'})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f'{method} {path}: HTTP {e.code}: {e.read().decode(errors="replace")[:400]}') from None


def vm(vm_id):
    """The VM's fields, or None once it is gone."""
    try:
        return api('GET', f'/core/virtual-machines/{vm_id}').get('instance')
    except RuntimeError as e:
        if 'HTTP 404' in str(e):
            return None
        raise


def state_path(run):
    return RUNS / run / 'state.json'


def load(run):
    p = state_path(run)
    if not p.exists():
        sys.exit(f'no run {run!r} ({p})')
    return json.loads(p.read_text())


def save(st):
    p = state_path(st['run'])
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(st, indent=1) + '\n')
    os.replace(tmp, p)


@contextmanager
def locked(run):
    (RUNS / run).mkdir(parents=True, exist_ok=True)
    with open(RUNS / run / 'lock', 'w') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


def ssh_args(st):
    return ['ssh', '-i', str(SSH_KEY), '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=accept-new',
            '-o', f'UserKnownHostsFile={RUNS / st["run"] / "known_hosts"}', '-o', 'ConnectTimeout=20',
            '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=4', '-o', 'LogLevel=ERROR',
            f'{USER}@{st["ip"]}']


def box(st, cmd, check=True, input=None, capture=False, timeout=None):
    """Run a shell command on the box; stream its output here unless captured."""
    r = subprocess.run(ssh_args(st) + [cmd], input=input, text=True, timeout=timeout,
                       stdout=subprocess.PIPE if capture else None, stderr=subprocess.STDOUT if capture else None)
    if check and r.returncode != 0:
        raise RuntimeError(f'box command failed ({r.returncode}): {cmd[:120]}'
                           + (f'\n{r.stdout[-2000:]}' if capture and r.stdout else ''))
    return r


def rsync(st, *args):
    e = shlex.join(ssh_args(st)[:-1])
    subprocess.run(['rsync', '-az', '--partial', '-e', e, *args], check=True)


def flavor_price(flavor):
    """$/h for a flavor: its GPU's price-book value × GPU count, plus the public IP."""
    for group in api('GET', '/core/flavors').get('data') or []:
        for f in group.get('flavors') or []:
            if f.get('name') == flavor:
                prices = {p['name']: float(p['value']) for p in api('GET', '/pricebook')}
                return prices[f['gpu']] * f['gpu_count'] + prices.get('PublicIP', 0.0), f
    sys.exit(f'no flavor {flavor!r} (tools/secure/hyperstack.py flavors)')


def public_ipv4():
    req = urllib.request.Request('https://api.ipify.org', headers={'User-Agent': 'mortal-policy-secure/1'})
    with urllib.request.urlopen(req, timeout=20) as r:
        ip = r.read().decode().strip()
    if ip.count('.') != 3:
        sys.exit(f'not an IPv4 address: {ip!r}')
    return ip


def cmd_keypair(a):
    if not SSH_KEY.exists():
        subprocess.run(['ssh-keygen', '-q', '-t', 'ed25519', '-N', '', '-C', 'mortal-secure', '-f', str(SSH_KEY)],
                       check=True)
    have = [k for k in api('GET', '/core/keypairs').get('keypairs') or [] if k.get('name') == KEYPAIR]
    if have:
        print(f'key pair {KEYPAIR} exists (id {have[0]["id"]})')
        return
    r = api('POST', '/core/keypairs', {'name': KEYPAIR, 'environment_name': ENVIRONMENT,
                                       'public_key': SSH_KEY.with_suffix('.pub').read_text().strip()})
    print(f'imported {SSH_KEY}.pub as {KEYPAIR} (id {(r.get("keypair") or {}).get("id")})')


def cmd_up(a):
    if state_path(a.run).exists() and not load(a.run).get('destroyed'):
        sys.exit(f'run {a.run} already has a VM; `down` it first')
    price, flavor = flavor_price(a.flavor)
    ip = public_ipv4()
    print(f'{a.flavor}: {flavor["gpu"]} ×{flavor["gpu_count"]}, {flavor["cpu"]} vCPU, {flavor["ram"]} GB RAM, '
          f'${price:.3f}/h; SSH from {ip}/32; watchdog at {a.hours} h (≈ ${price * a.hours:.2f})')
    if not flavor.get('stock_available'):
        sys.exit(f'{a.flavor} is out of stock')
    if not a.yes:
        sys.exit('a rental: add --yes')
    payload = {
        'name': f'mp-secure-{a.run}'[:50], 'environment_name': ENVIRONMENT, 'image_name': IMAGE,
        'flavor_name': a.flavor, 'key_name': KEYPAIR, 'count': 1, 'assign_floating_ip': True,
        'enable_port_randomization': False,  # keep SSH on port 22 of the floating IP
        'security_rules': [{'direction': 'ingress', 'ethertype': 'IPv4', 'protocol': 'tcp',
                            'remote_ip_prefix': f'{ip}/32', 'port_range_min': 22, 'port_range_max': 22}],
    }
    r = api('POST', '/core/virtual-machines', payload)
    inst = (r.get('instances') or [None])[0]
    if not inst:
        sys.exit(f'no VM in the response: {json.dumps(r)[:400]}')
    st = {'run': a.run, 'vm_id': inst['id'], 'name': payload['name'], 'flavor': a.flavor, 'gpu': flavor['gpu'],
          'price_per_h': price, 'created': now(), 'ssh_from': ip, 'deadline_h': a.hours}
    save(st)
    ledger(a.run, 'rented', provider='hyperstack', region=REGION, vm_id=inst['id'], flavor=a.flavor)
    log(a.run, f'rented VM {inst["id"]} ({a.flavor}, ${price:.3f}/h)')

    # the watchdog first, so even a failure below can't leave the VM billing
    wd = subprocess.Popen([sys.executable, __file__, 'down', a.run, '--after', str(int(a.hours * 3600))],
                          stdout=open(RUNS / a.run / 'watchdog.log', 'a'), stderr=subprocess.STDOUT,
                          stdin=subprocess.DEVNULL, start_new_session=True)
    st['watchdog_pid'] = wd.pid
    save(st)
    log(a.run, f'watchdog {wd.pid}: down in {a.hours} h')

    t0 = time.time()
    while True:
        v = vm(st['vm_id'])
        if v is None:
            sys.exit('the VM disappeared while starting')
        if v.get('status') == 'ERROR':
            sys.exit(f'the VM failed to start: {json.dumps(v)[:300]}; run `down {a.run}`')
        if v.get('status') == 'ACTIVE' and v.get('floating_ip') and v.get('floating_ip_status') in (None, 'ATTACHED'):
            break
        if time.time() - t0 > 1800:
            sys.exit(f'not ACTIVE after 30 min ({v.get("status")}); run `down {a.run}`')
        time.sleep(15)
    st['ip'] = v['floating_ip']
    save(st)
    log(a.run, f'ACTIVE at {st["ip"]} after {time.time() - t0:,.0f} s')
    for _ in range(40):
        if box(st, 'true', check=False, capture=True).returncode == 0:
            break
        time.sleep(15)
    else:
        sys.exit(f'no SSH after 10 min; run `down {a.run}`')
    # a last stop of the box's own: power off an hour after the watchdog's deadline
    box(st, f'sudo shutdown -h +{int(a.hours * 60) + SHUTDOWN_GRACE_MIN} "mortal-secure deadline" 2>/dev/null')
    log(a.run, f'ready: {shlex.join(ssh_args(st))}')


def cmd_setup(a):
    st = load(a.run)
    dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=REPO,
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        sys.exit(f'commit before syncing:\n{dirty}')
    box(st, f'sudo apt-get install -y -qq rsync >/dev/null 2>&1 || (sudo apt-get update -qq && sudo apt-get install -y -qq rsync); '
            f'sudo install -d -o {USER} -g {USER} /workspace /venv')
    rsync(st, "--filter=:- .gitignore", '--exclude=.claude/', '--exclude=.pytest_cache/', f'{REPO}/',
          f'{USER}@{st["ip"]}:{BOX_ROOT}/')
    frozen = [p for _, p in (line.split() for line in (REPO / 'tools/fleet/frozen.sha256').read_text().splitlines()
                             if line.strip()) if (REPO / p).exists()]
    subprocess.run(['rsync', '-azR', '-e', shlex.join(ssh_args(st)[:-1]), *frozen, f'{USER}@{st["ip"]}:{BOX_ROOT}/'],
                   cwd=REPO, check=True)
    log(a.run, f'synced {subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()} '
               f'and {len(frozen)} frozen inputs')
    box(st, f'sudo bash {BOX_ROOT}/tools/secure/bootstrap_vm.sh root')
    t0 = time.time()
    box(st, f'cd {BOX_ROOT} && bash tools/secure/bootstrap_vm.sh user {"--skip-tests" if a.skip_tests else ""}')
    log(a.run, f'set up in {time.time() - t0:,.0f} s')


def data_dir(st):
    if st.get('data_dir'):
        return st['data_dir']
    # the flavor's ephemeral disk if it has one (L40: 725 GB at /ephemeral), else the root disk
    d = box(st, 'if mountpoint -q /ephemeral; then echo /ephemeral/mj; else echo /data/mj; fi',
            capture=True).stdout.strip()
    st['data_dir'] = d
    save(st)
    return d


def cmd_ship(a):
    st = load(a.run)
    bdir = Path(a.bundle) if a.bundle else max((p for p in BUNDLES.iterdir() if p.is_dir() and not p.name.startswith('staging-')),
                                              key=lambda p: (p / 'manifest.json').stat().st_mtime)
    manifest = json.loads((bdir / 'manifest.json').read_text())
    bundle = manifest['bundle_id'][:16]
    with locked(a.run):
        d = data_dir(st)
        box(st, f'sudo install -d -o {USER} -g {USER} {d}')
        urls = subprocess.run([str(S3_PY), str(REPO / 'tools/secure/bucket.py'), 'urls', str(bdir)],
                              check=True, capture_output=True, text=True).stdout
        key = BUNDLE_KEY.read_text().strip()
        log(a.run, f'shipping bundle {bundle} ({len(manifest["shards"])} objects) to {d}')
        ledger(a.run, 'shipping', vm_id=st['vm_id'], bundle=bundle, data_dir=d)
        t0 = time.time()
        box(st, f'python3 {BOX_ROOT}/tools/secure/unpack.py {d} --jobs {a.jobs}', input=key + '\n' + urls)
        # the loader's file indexes, with this box's paths
        box(st, f'cd {BOX_ROOT} && {BOX_PY} - <<\'EOF\'\n'
                f'import torch\nd = {d!r}\n'
                "for name, splits in (('train', ('train_E', 'train_O')), ('heldout_game', ('heldout_game',))):\n"
                "    files = sorted({f'{d}/mjai/{l.strip()}' for s in splits for l in open(f'{d}/splits/{s}.txt') if l.strip()})\n"
                "    torch.save({'file_list': files}, f'{d}/index-{name}.pth')\n"
                "    print(name, len(files))\n"
                f"open(f'{{d}}/BUNDLE', 'w').write({bundle!r} + '\\n')\n"
                'EOF')
        st['shipped'] = {'bundle': bundle, 'time': now(), 'seconds': round(time.time() - t0)}
        save(st)
        ledger(a.run, 'shipped', vm_id=st['vm_id'], bundle=bundle, data_dir=d, seconds=round(time.time() - t0))
        log(a.run, f'shipped in {time.time() - t0:,.0f} s')


def cmd_exec(a):
    st = load(a.run)
    r = box(st, f'cd {BOX_ROOT} && export PATH=$HOME/.cargo/bin:$HOME/.local/bin:$PATH && {a.command}', check=False)
    sys.exit(r.returncode)


def pull(st):
    dst = RUNS / st['run'] / 'box'
    dst.mkdir(parents=True, exist_ok=True)
    box(st, f'mkdir -p {BOX_ROOT}/{BOX_OUT}')
    rsync(st, f'{USER}@{st["ip"]}:{BOX_ROOT}/{BOX_OUT}/', f'{dst}/')


def cmd_pull(a):
    st = load(a.run)
    pull(st)
    log(a.run, f'pulled {BOX_OUT} to {RUNS / a.run / "box"}')


def wipe(st):
    d = st.get('data_dir')
    if not d:
        return False
    box(st, f'sudo rm -rf {shlex.quote(d)} && test ! -e {shlex.quote(d)}')
    ledger(st['run'], 'wiped', vm_id=st['vm_id'], data_dir=d)
    return True


def cmd_wipe(a):
    st = load(a.run)
    with locked(a.run):
        if wipe(st):
            st['wiped'] = now()
            save(st)
            log(a.run, f'wiped {st["data_dir"]}')
        else:
            log(a.run, 'nothing shipped: nothing to wipe')


def cmd_down(a):
    if a.after:
        time.sleep(a.after)
    st = load(a.run)
    with locked(a.run):
        st = load(a.run)
        if st.get('destroyed'):
            log(a.run, 'already destroyed')
            return
        v = vm(st['vm_id'])
        if v is not None and st.get('ip') and v.get('status') == 'ACTIVE':
            for step, fn in (('pull', pull), ('wipe', wipe)):
                try:
                    fn(st)
                    log(a.run, f'{step}: ok')
                except Exception as e:  # destroying matters more than either
                    log(a.run, f'{step} failed, destroying anyway: {e}')
        if v is not None:
            api('DELETE', f'/core/virtual-machines/{st["vm_id"]}')
            t0 = time.time()
            while (v := vm(st['vm_id'])) is not None and time.time() - t0 < 900:
                if v.get('status') == 'DELETED':
                    break
                time.sleep(10)
            if v is not None and v.get('status') != 'DELETED':
                ledger(a.run, 'destroy_unconfirmed', vm_id=st['vm_id'], status=v.get('status'))
                sys.exit(f'VM {st["vm_id"]} still {v.get("status")} 15 min after DELETE: check the console')
        st['destroyed'] = now()
        save(st)
        ledger(a.run, 'destroyed', vm_id=st['vm_id'])
        hours = (time.time() - datetime.fromisoformat(st['created']).timestamp()) / 3600
        log(a.run, f'VM {st["vm_id"]} destroyed after {hours:.2f} h (≈ ${hours * st["price_per_h"]:.2f})')
    wd = st.get('watchdog_pid')
    if wd and wd != os.getpid() and not a.after:
        try:
            os.kill(wd, 15)
        except ProcessLookupError:
            pass


def cmd_status(a):
    runs = sorted(p.parent.name for p in RUNS.glob('*/state.json')) if RUNS.exists() else []
    for run in runs:
        st = load(run)
        end = datetime.fromisoformat(st['destroyed']).timestamp() if st.get('destroyed') else time.time()
        hours = (end - datetime.fromisoformat(st['created']).timestamp()) / 3600
        live = 'destroyed' if st.get('destroyed') else ((vm(st['vm_id']) or {}).get('status') or 'gone')
        print(f'{run}: VM {st["vm_id"]} {st["flavor"]} {live}, {hours:.2f} h, ≈ ${hours * st["price_per_h"]:.2f}'
              f'{", shipped " + st["shipped"]["bundle"] if st.get("shipped") else ""}'
              f'{", wiped" if st.get("wiped") else ""}')
    others = [v for v in api('GET', '/core/virtual-machines').get('instances') or []
              if v.get('id') not in {load(r)['vm_id'] for r in runs if not load(r).get('destroyed')}]
    for v in others:
        print(f'NOT OURS OR UNTRACKED: VM {v.get("id")} {v.get("name")} {v.get("status")}')


def cmd_ssh(a):
    print(shlex.join(ssh_args(load(a.run))))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('keypair')
    s = sub.add_parser('up')
    s.add_argument('run')
    s.add_argument('--flavor', default='n3-L40x1')
    s.add_argument('--hours', type=float, required=True, help="the watchdog's deadline")
    s.add_argument('--yes', action='store_true', help='rent (without it: show the quote)')
    s = sub.add_parser('setup')
    s.add_argument('run')
    s.add_argument('--skip-tests', action='store_true')
    s = sub.add_parser('ship')
    s.add_argument('run')
    s.add_argument('--bundle', help='a bundle directory (default: the newest in workdir/secure/bundle)')
    s.add_argument('--jobs', type=int, default=4)
    s = sub.add_parser('exec')
    s.add_argument('run')
    s.add_argument('command')
    for name in ('pull', 'wipe', 'ssh'):
        sub.add_parser(name).add_argument('run')
    s = sub.add_parser('down')
    s.add_argument('run')
    s.add_argument('--after', type=int, default=0, help=argparse.SUPPRESS)
    sub.add_parser('status')
    a = ap.parse_args(argv)
    globals()['cmd_' + a.cmd](a)


if __name__ == '__main__':
    main()
