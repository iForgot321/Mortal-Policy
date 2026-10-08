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
    assert actor == ['workdir/grp-v2/grp-v2-frozen.pth', 'workdir/pilot/qs-a/pilot-qs-a-final.pth']
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
