"""tools/fleet (03 §2, D48): the workstation side's pure logic, the box-side health probe's
parsers, and the box scripts' syntax. The tunnel restrictions run against a real sshd in
test_fleet_tunnel.py; the full loop only on rented boxes (tools/fleet/README.md)."""
import importlib.util
import os
import subprocess
import tomllib
from datetime import datetime

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLEET = os.path.join(ROOT, 'tools', 'fleet')
P0 = os.path.join(ROOT, 'experiments', 'configs', 'p0.toml')


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(FLEET, f'{name}.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fleet = load('fleet')
health = load('box_health')


def test_box_config_rewrites_root_and_splits_actor_dirs():
    text, cfg = fleet.load_config(P0)
    learner = tomllib.loads(fleet.box_config(text))
    assert fleet.WS_ROOT not in fleet.box_config(text)
    assert learner['control']['state_file'] == cfg['control']['state_file'].replace(fleet.WS_ROOT, fleet.BOX_ROOT)
    actor = tomllib.loads(fleet.box_config(text, actor=2))
    assert actor['train_play']['default']['log_dir'] == learner['train_play']['default']['log_dir'] + '-2'
    actor['train_play']['default']['log_dir'] = learner['train_play']['default']['log_dir']
    assert actor == learner


def test_load_config_refuses_a_public_server(tmp_path):
    text, _ = fleet.load_config(P0)
    bad = tmp_path / 'bad.toml'
    bad.write_text(text.replace("host = '127.0.0.1'", "host = '0.0.0.0'"))
    with pytest.raises(fleet.FleetError, match='127.0.0.1'):
        fleet.load_config(bad)


def test_inputs_are_the_frozen_checkpoints_plus_the_learner_init():
    _, cfg = fleet.load_config(P0)
    actor = fleet.inputs(cfg, 'actor')
    assert actor == ['workdir/grp-v2/grp-v2-frozen.pth', 'workdir/pilot/qs-a/pilot-qs-a-final.pth',
                     'workdir/pilot/qs-b/pilot-qs-b-final.pth']  # every frozen input, whichever pilot
    init = 'workdir/pilot/qs-a/pilot-qs-a-final.pth'
    assert fleet.inputs(cfg, 'learner', init) == actor + [init]


def test_offers_rank_by_dollars_per_hanchan():
    offers = [
        {'id': 1, 'machine_id': 10, 'cpu_name': 'AMD EPYC 9684X 96-Core Processor', 'dph_total': 0.46},
        {'id': 2, 'machine_id': 11, 'cpu_name': 'AMD Ryzen 9 7950X 16-Core Processor', 'dph_total': 0.60},
        {'id': 3, 'machine_id': 12, 'cpu_name': 'AMD EPYC 7V13 64-Core Processor', 'dph_total': 0.30},
        {'id': 4, 'machine_id': 13, 'cpu_name': 'AMD EPYC 7742 64-Core Processor', 'dph_total': 0.20},
        {'id': 5, 'machine_id': 14, 'cpu_name': 'Intel Xeon Gold 6338', 'dph_total': 0.10},
    ]
    assert [o['id'] for o in fleet.rank_offers(offers)] == [3, 1, 2]
    assert [o['id'] for o in fleet.rank_offers(offers, allow_zen2=True)] == [4, 3, 1, 2]
    assert [o['id'] for o in fleet.rank_offers(offers, exclude={12})] == [1, 2]
    assert fleet.rank_offers(offers)[1]['usd_per_m'] == pytest.approx(0.46 / 50_000 * 1e6)
    custom = [{'id': 6, 'machine_id': 15, 'cpu_name': 'AMD EPYC 7B13 64-Core Processor', 'dph_total': 0.4},
              {'id': 7, 'machine_id': 16, 'cpu_name': 'AMD EPYC 7K62 48-Core Processor', 'dph_total': 0.4},
              {'id': 8, 'machine_id': 17, 'cpu_name': 'AMD Ryzen 9 7945HX with Radeon Graphics', 'dph_total': 0.6}]
    assert [o['id'] for o in fleet.rank_offers(custom)] == [6, 8]  # Zen 3 custom, Zen 4 mobile
    assert [o['est_rate'] for o in fleet.rank_offers(custom, allow_zen2=True)] == [40_000, 50_000, 31_000]  # $10, $12, $12.9 per M


def test_authorized_line_restricts_the_key():
    line = fleet.authorized_line('ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIGx0 fleet@actor', 5000, 'fleet-r-actor1-7')
    opts, kind, blob, comment = line.split(' ')
    assert opts == 'restrict,port-forwarding,permitopen="127.0.0.1:5000",command="/bin/false"'
    assert (kind, blob, comment) == ('ssh-ed25519', 'AAAAC3NzaC1lZDI1NTE5AAAAIGx0', 'fleet-r-actor1-7')
    for bad in ('ssh-rsa AAAAB3Nza x', 'ssh-ed25519 AAAA;rm x', 'ssh-ed25519 AAAA"x'):
        with pytest.raises(fleet.FleetError):
            fleet.authorized_line(bad, 5000, 'c')


def test_run_names_are_safe_in_labels_and_sed():
    for bad in ('P0', 'a b', 'a/b', "a'b", ''):
        with pytest.raises(fleet.FleetError):
            fleet.Run(bad)


def probe(role, **over):
    now = 1_000_000.0
    procs = ({'server': {'alive': True, 'restarts_1h': 0, 'queue': 100, 'drained_1h': 9000},
              'trainer': {'alive': True, 'restarts_1h': 0, 'last_param_age_s': 60, 'checkpoint_age_s': 90}}
             if role == 'learner' else
             {f'client-{k}': {'alive': True, 'restarts_1h': 0, 'last_submit_age_s': 120, 'games_1h': 28_000,
                              'blocked_s_1h': 36, 'unreachable_15m': 0} for k in (1, 2)})
    p = {'time': now, 'started': True, 'manifest_started': now - 7200, 'role': role, 'procs': procs,
         'gpu': {'util': 20}, 'mem_avail_gb': 50.0, 'disk_free_gb': 30.0}
    p.update({'tunnel': {'alive': True, 'listening': True}} if role == 'actor' else {'server_listening': True})
    p.update(over)
    return p


def test_assess_verdicts():
    assert fleet.assess('learner', 'running', probe('learner'), capacity=1600) == ('ok', [])
    assert fleet.assess('actor', '?', probe('actor')) == ('ok', [])  # Vast unknown, box fine
    assert fleet.assess('actor', 'exited', probe('actor'))[0] == 'lost'
    assert fleet.assess('actor', None, None) == ('lost', ['instance gone'])
    assert fleet.assess('actor', 'running', None)[0] == 'unreachable'
    assert fleet.assess('actor', 'running', {'time': 0, 'started': False})[0] == 'fail'

    p = probe('actor', tunnel={'alive': True, 'listening': False})
    assert fleet.assess('actor', 'running', p) == ('fail', ['tunnel down'])
    p = probe('actor')
    p['procs']['client-1']['alive'] = False
    p['procs']['client-2'].update(last_submit_age_s=4000, restarts_1h=5)
    verdict, problems = fleet.assess('actor', 'running', p)
    assert verdict == 'fail' and problems[0] == 'client-1 not running' and len(problems) == 3
    p = probe('actor')
    p['procs']['client-1']['last_submit_age_s'] = None  # up 2 h, never submitted
    assert fleet.assess('actor', 'running', p)[0] == 'warn'

    p = probe('learner', server_listening=False)
    assert fleet.assess('learner', 'running', p) == ('fail', ['server.py not listening'])
    p = probe('learner')
    p['procs']['server']['queue'] = 1600
    assert fleet.assess('learner', 'running', p, capacity=1600)[0] == 'warn'
    p = probe('learner')
    p['procs']['trainer']['last_param_age_s'] = 1000
    p['procs']['server']['queue'] = 4800
    assert fleet.assess('learner', 'running', p, capacity=1600)[1][0] == 'trainer stalled'
    p['procs']['server']['queue'] = 100  # no data waiting: slow, not stalled
    assert 'trainer stalled' not in fleet.assess('learner', 'running', p, capacity=1600)[1]
    p = probe('learner', disk_free_gb=2.0, cgroup_mem_avail_gb=0.5)
    assert fleet.assess('learner', 'running', p)[1] == ['disk 2.0 GB free', '0.5 GB memory available']


def ts(s):
    return datetime.strptime(s, '%Y-%m-%d %H:%M:%S').timestamp()


def test_health_log_parsing():
    lines = [
        b'2026-10-06 19:14:05,262     INFO    client.py:82   trainee rankings: [191 190 215 204] (2.54, -2.25pt)',
        b'2026-10-06 19:14:05,775     INFO    client.py:105  logs have been submitted',
        b'TRAIN:  25%|##5       | 101/400 [00:00<?, ?batch/s]2026-10-06 19:14:11,898     INFO train_online.py:422  first minibatch',
        b'2026-10-06 19:20:00,000     INFO    client.py:73   blocked 42s waiting for params (samples overflow)',
        b'2026-10-06 19:21:00,000  WARNING    client.py:33   server (127.0.0.1, 5000) unreachable (ConnectionRefusedError()); retrying in ~1s',
    ]
    events = health.log_events(lines)
    assert [m.split()[0] for _, m in events] == ['trainee', 'logs', 'first', 'blocked', 'server']
    assert events[2][0] == ts('2026-10-06 19:14:11')
    facts = health.client_facts(events, ts('2026-10-06 19:22:00'))
    assert facts == {'last_submit_age_s': 7 * 60 + 55 - 0, 'games_1h': 800, 'blocked_s_1h': 42, 'unreachable_15m': 1}
    assert health.client_facts(events, ts('2026-10-06 21:00:00'))['games_1h'] == 0


def test_health_server_and_supervise_parsing():
    events = health.log_events([
        b'2026-10-06 19:14:06,282     INFO    server.py:97   total buffer size: 800',
        b'2026-10-06 19:14:07,932     INFO    server.py:124  files transferred to trainer: 800',
        b'2026-10-06 19:14:07,932     INFO    server.py:125  total buffer size: 0',
    ])
    assert health.server_facts(events, ts('2026-10-06 19:30:00')) == {'queue': 0, 'drained_1h': 800}
    sup = [b'2026-10-06T19:00:00+00:00 learner start', b'2026-10-06T19:10:00+00:00 learner exit 1 after 600s',
           b'2026-10-06T19:10:10+00:00 learner start', b'2026-10-06T19:10:15+00:00 client-1 exit 0 after 5s']
    now = datetime.fromisoformat('2026-10-06T19:20:00+00:00').timestamp()
    assert health.supervise_stats(sup, 'learner', now) == {
        'restarts': 1, 'restarts_1h': 1, 'last_exit_code': 1, 'last_exit_age_s': 600}


def test_health_listening():
    table = ('  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n'
             '   0: 0100007F:1388 00000000:0000 0A 00000000:00000000 00:00000000 00000000     0        0 1 1\n'
             '   1: 00000000:0016 00000000:0000 0A 00000000:00000000 00:00000000 00000000     0        0 2 1\n'
             '   2: 0100007F:1389 0100007F:9C40 01 00000000:00000000 00:00000000 00000000     0        0 3 1\n')
    assert health.listening(5000, table)
    assert not health.listening(22, table)    # listening, but not on loopback
    assert not health.listening(5001, table)  # an established connection, not a listener


def test_box_scripts_parse():
    for name in ('box_tunnel.sh', 'box_supervise.sh', 'box_learner.sh', 'setup_box.sh', 'box_bench.sh'):
        subprocess.run(['bash', '-n', os.path.join(FLEET, name)], check=True)


def test_budget_accounting_and_labels(monkeypatch):
    run = fleet.Run.__new__(fleet.Run)
    now = 1_000_000.0
    run.data = {'boxes': {'learner': {'instance': 1, 'created_ts': now - 7200, 'dph': 0.5}},
                'retired': [{'instance': 2, 'created_ts': now - 7200, 'destroyed_ts': now - 3600, 'dph': 0.4},
                            {'instance': None, 'created_ts': now - 7200, 'dph': 0.4}]}  # never created
    monkeypatch.setattr(fleet.time, 'time', lambda: now)
    assert fleet.box_hours(run) == pytest.approx(3.0)
    assert fleet.cost(run) == pytest.approx(2 * 0.5 + 1 * 0.4)
    assert [fleet.snapshot_label(f) for f in (0.25, 0.5, 1.0)] == ['p25', 'p50', 'full']


def test_snapshot_is_verified_frozen_and_never_overwritten(tmp_path, monkeypatch):
    import hashlib
    import json
    import stat
    monkeypatch.setattr(fleet, 'RUNS', tmp_path)
    run = fleet.Run('t')
    run.dir.mkdir()
    run.data = {'config': 'experiments/configs/p0.toml', 'boxes': {'learner': {
        'name': 'learner', 'role': 'learner', 'instance': 7, 'ip': '192.0.2.1', 'ssh_port': 22,
        'created_ts': 0, 'dph': 0.5}}, 'retired': []}
    run.save()
    ckpt = tmp_path / 'box-copy.pth'
    ckpt.write_bytes(b'checkpoint bytes')
    sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    scripts = []

    def fake_remote(run_, box, script, timeout=300, check=True):
        scripts.append(script)
        return subprocess.CompletedProcess([], 0, f'{sha}\n4200\n', '')

    def fake_rsync(run_, box, args, timeout=1800):
        assert args[0] == f'root@192.0.2.1:{fleet.BOX_ROOT}/{fleet.BOX_DIR}/snapshots/t-p50.pth'
        (tmp_path / args[1]).write_bytes(ckpt.read_bytes())
    monkeypatch.setattr(fleet, 'remote', fake_remote)
    monkeypatch.setattr(fleet, 'rsync', fake_rsync)

    rec = fleet.snapshot(run, 'p50')
    local = tmp_path / 't' / 'snapshots' / 'p50.pth'
    assert rec['sha256'] == sha and rec['steps'] == 4200
    assert not os.stat(local).st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    assert json.loads((tmp_path / 't' / 'snapshots.jsonl').read_text())['label'] == 'p50'
    # the state file the trainer replaces atomically is copied on the box before hashing
    assert f'cp {fleet.BOX_ROOT}/workdir/p0/mortal.pth' in scripts[0]
    subprocess.run(['bash', '-n'], input=scripts[0], text=True, check=True)
    with pytest.raises(fleet.FleetError, match='never overwritten'):
        fleet.snapshot(run, 'p50')

    # a copy that doesn't match the box's hash is refused and removed
    monkeypatch.setattr(fleet, 'rsync', lambda r, b, args, timeout=1800: (tmp_path / args[1]).write_bytes(b'torn'))
    with pytest.raises(fleet.FleetError, match='sha256'):
        fleet.snapshot(run, 'full')
    assert not (tmp_path / 't' / 'snapshots' / 'full.pth').exists()


def test_overrides_change_only_the_named_key():
    text, _ = fleet.load_config(P0)
    out = fleet.box_config(text, overrides={'dataset.num_workers': '8', 'train_play.default.games': '800'})
    a, b = tomllib.loads(fleet.box_config(text)), tomllib.loads(out)
    assert b['dataset']['num_workers'] == 8 and a['dataset']['num_workers'] == 2
    b['dataset']['num_workers'] = 2
    assert a == b
    with pytest.raises(fleet.FleetError, match='no such key'):
        fleet.box_config(text, overrides={'dataset.no_such_key': '1'})


def test_learner_start_never_overwrites_a_checkpoint(tmp_path):
    state, init = tmp_path / 'run' / 'mortal.pth', tmp_path / 'init.pth'
    init.write_text('pilot')
    line = fleet.learner_init_line(str(state), str(init))
    sh = lambda: subprocess.run(['bash', '-c', 'set -euo pipefail\n' + line + '\necho ok'],
                                capture_output=True, text=True)  # noqa: E731
    assert sh().returncode == 0 and state.read_text() == 'pilot'  # absent: copied
    state.write_text('trained')
    assert sh().returncode == 0 and state.read_text() == 'trained'  # present: untouched
    state.unlink()
    init.unlink()
    p = sh()
    assert p.returncode == 1 and 'no init to copy' in p.stderr  # neither: refuse


def test_budget_rebase_excludes_hours_before_it(monkeypatch):
    run = fleet.Run.__new__(fleet.Run)
    now = 1_000_000.0
    run.data = {'boxes': {'learner': {'instance': 1, 'created_ts': now - 10 * 3600, 'dph': 0.5}},
                'retired': [{'instance': 2, 'created_ts': now - 10 * 3600, 'destroyed_ts': now - 8 * 3600, 'dph': 0.5}]}
    monkeypatch.setattr(fleet.time, 'time', lambda: now)
    assert fleet.box_hours(run) == pytest.approx(12.0)
    excluded = sum(h for _, h in fleet.billed(run, until=now - 4 * 3600))
    assert excluded == pytest.approx(6.0 + 2.0)
    run.data['budget_rebase'] = {'box_hours': excluded}
    assert fleet.box_hours(run) == pytest.approx(4.0)
    assert fleet.cost(run) == pytest.approx(6.0)  # cost still counts everything


deadline = load('box_deadline')


def test_box_deadline_reads_the_container_key_and_builds_the_vast_request(tmp_path):
    proc1 = tmp_path / 'environ'
    proc1.write_bytes(b'PATH=/bin\0CONTAINER_ID=55\0CONTAINER_API_KEY=k3y\0')
    assert deadline.container_env({}, str(proc1)) == {'CONTAINER_ID': '55', 'CONTAINER_API_KEY': 'k3y'}
    assert deadline.container_env({'CONTAINER_ID': '7', 'CONTAINER_API_KEY': 'e'}, str(proc1))['CONTAINER_ID'] == '7'
    stop, destroy = deadline.request('stop', 55, 'k3y'), deadline.request('destroy', 55, 'k3y')
    assert (stop.method, stop.full_url, stop.data) == ('PUT', 'https://console.vast.ai/api/v0/instances/55/',
                                                       b'{"state": "stopped"}')
    assert (destroy.method, destroy.data) == ('DELETE', b'{}')
    assert stop.get_header('Authorization') == 'Bearer k3y'
    f = tmp_path / 'deadline.json'
    assert deadline.read_deadline(str(f)) is None
    f.write_text('{"instance": 55, "deadline": 100, "action": "reboot"}')
    with pytest.raises(ValueError):
        deadline.read_deadline(str(f))


def run_deadline(tmp_path, monkeypatch, capsys, spec, replies, env_id='55'):
    """box_deadline.main on a fake clock; returns (exit code, requests, log)."""
    import io
    import json
    f = tmp_path / 'deadline.json'
    f.unlink(missing_ok=True)
    if spec is not None:
        f.write_text(json.dumps(spec))
    monkeypatch.setenv('CONTAINER_ID', env_id)
    monkeypatch.setenv('CONTAINER_API_KEY', 'sekrit-key')
    clock, sent = [1000.0], []

    class Reply(io.BytesIO):
        status = 200

    def urlopen(req, timeout):
        sent.append((req.method, req.full_url))
        r = replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return Reply(json.dumps(r).encode())

    def sleep(s):
        if len(sent) >= 3:
            raise KeyboardInterrupt  # a stop retried twice: enough
        clock[0] += s
    try:
        code = deadline.main(str(f), urlopen=urlopen, now=lambda: clock[0], sleep=sleep)
    except KeyboardInterrupt:
        code = None
    out = capsys.readouterr().out
    assert 'sekrit-key' not in out
    return code, sent, out


def test_box_deadline_acts_only_at_the_deadline(tmp_path, monkeypatch, capsys):
    import urllib.error
    code, sent, out = run_deadline(tmp_path, monkeypatch, capsys,
                                   {'instance': 55, 'deadline': 5000, 'action': 'destroy'},
                                   [urllib.error.URLError('down'), {'success': True}])
    assert code == 0 and sent == [('DELETE', 'https://console.vast.ai/api/v0/instances/55/')] * 2
    assert 'armed: destroy instance 55' in out and 'failed: URLError' in out
    # a stop keeps asking (the box goes away when it takes effect)
    code, sent, _ = run_deadline(tmp_path, monkeypatch, capsys,
                                 {'instance': 55, 'deadline': 5000, 'action': 'stop'}, [{'success': True}] * 3)
    assert code is None and [m for m, _ in sent] == ['PUT'] * 3
    # disarmed: no file, or a file naming another instance
    assert run_deadline(tmp_path, monkeypatch, capsys, None, [])[:2] == (0, [])
    assert run_deadline(tmp_path, monkeypatch, capsys, {'instance': 56, 'deadline': 0, 'action': 'destroy'},
                        [])[:2] == (2, [])


def test_box_deadline_spec_follows_the_hard_deadline():
    run = fleet.Run.__new__(fleet.Run)
    run.data = {'watchdog': None}
    assert fleet.deadline_spec(run, {'instance': 1, 'role': 'actor'}) is None
    # runs started before hard_deadline was recorded fall back to the watchdog's
    run.data = {'watchdog': {'pid': 1, 'deadline': '2026-10-16T10:01:20+00:00'}}
    hard = datetime.fromisoformat('2026-10-16T10:01:20+00:00').timestamp()
    assert fleet.deadline_spec(run, {'instance': 9, 'role': 'learner'}) == {
        'instance': 9, 'deadline': hard + 3600, 'action': 'stop'}
    run.data = {'watchdog': None, 'hard_deadline': '2026-10-16T10:01:20+00:00'}
    assert fleet.deadline_spec(run, {'instance': 8, 'role': 'actor'})['action'] == 'destroy'


def test_down_keeps_an_unpulled_learner_that_is_not_running(tmp_path, monkeypatch):
    monkeypatch.setattr(fleet, 'RUNS', tmp_path)
    run = fleet.Run('t')
    run.dir.mkdir()

    def fresh():
        run.data = {'boxes': {
            'learner': {'name': 'learner', 'role': 'learner', 'instance': 1, 'status': 'running'},
            'actor1': {'name': 'actor1', 'role': 'actor', 'instance': 2, 'status': 'running'}}, 'retired': []}
        return list(run.boxes.values())
    for vast_status, kept in (('stopped', True), ('exited', True), ('running', False), (None, False)):
        bs = fresh()
        monkeypatch.setattr(fleet, 'instances', lambda: {1: {'actual_status': vast_status}} if vast_status else {})
        msgs = fleet.keep_unpulled(run, bs)
        assert ([b['name'] for b in bs] == ['actor1']) is kept and bool(msgs) is kept
        # out of the run's boxes, so no watch (of any version) replaces it
        assert ('learner' in run.boxes) is not kept
        assert [b['instance'] for b in run.data.get('kept', [])] == ([1] if kept else [])
    # Vast can't say: keep rather than destroy what may hold the only copy
    bs = fresh()
    monkeypatch.setattr(fleet, 'instances', lambda: None)
    assert fleet.keep_unpulled(run, bs) and run.data['kept'][0]['status'] == 'kept'
    # still billed (storage), and never swept
    run.data['kept'][0].update(created_ts=1, dph=0.5)
    assert any(b['instance'] == 1 for b, _ in fleet.billed(run))
    destroyed = []
    monkeypatch.setattr(fleet, 'instances', lambda: {1: {'label': 'mp-t-learner'}, 3: {'label': 'mp-t-actor9'}})
    monkeypatch.setattr(fleet, 'destroy_instance', lambda iid: destroyed.append(iid) or True)
    fleet.sweep(run)
    assert destroyed == [3]


def test_watchdog_waits_on_the_wall_clock(monkeypatch):
    """A suspended workstation: the wall clock jumps past the deadline during one short sleep."""
    clock, slept = [0.0], []

    def sleep(s):
        slept.append(s)
        clock[0] += s + (50_000 if len(slept) == 2 else 0)

    class Fired(Exception):
        pass

    def fire(name):
        raise Fired(clock[0])
    monkeypatch.setattr(fleet.time, 'time', lambda: clock[0])
    monkeypatch.setattr(fleet.time, 'sleep', sleep)
    monkeypatch.setattr(fleet.Run, 'load', staticmethod(fire))
    a = fleet.argparse.Namespace(after=100_000, run='t', no_pull=False, name=None, kept=False)
    with pytest.raises(Fired) as e:
        fleet.cmd_down(a)
    assert max(slept) <= 60 and e.value.args[0] >= 100_000 and sum(slept) < 100_000


def test_value_runs_start_the_value_trainer():
    _, p0 = fleet.load_config(P0)
    _, v = fleet.load_config(os.path.join(ROOT, 'experiments', 'configs', 'v-a.toml'))
    assert fleet.trainer_script(p0) == 'train_online.py'
    assert fleet.trainer_script(v) == 'train_value.py'
