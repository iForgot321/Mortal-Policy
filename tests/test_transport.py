"""Behaviour sidecars travel with their logs through the server (D46): submit
→ buffer → drain, under the same names, without counting as logs."""
import os
import sys
import types
from threading import Lock

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def server(tmp_path):
    os.environ.setdefault('MORTAL_CFG', os.path.join(ROOT, 'experiments', 'configs', 'pilot-qs-a.toml'))
    sys.path.insert(0, os.path.join(ROOT, 'mortal'))
    # prelude reconfigures stdin, which pytest has replaced; logging setup is all it adds
    sys.modules.setdefault('prelude', types.ModuleType('prelude'))
    import server
    dirs = {k: tmp_path / k for k in ('buffer', 'drain', 'buffer_behaviour', 'drain_behaviour')}
    for d in dirs.values():
        d.mkdir()
    server.S = server.State(
        buffer_dir=str(dirs['buffer']), drain_dir=str(dirs['drain']),
        behaviour_buffer_dir=str(dirs['buffer_behaviour']), behaviour_drain_dir=str(dirs['drain_behaviour']),
        capacity=4, force_sequential=True, dir_lock=Lock(), param_lock=Lock(),
        buffer_size=0, submission_id=0, buffer_versions={},
        mortal_param=None, dqn_param=None, param_version=7, idle_param_version=0,
    )
    return server


def call(server, method, *args):
    replies = []
    handler = server.Handler.__new__(server.Handler)  # no socket: capture replies
    handler.send_msg = lambda msg, packed=False: replies.append(msg)
    getattr(handler, method)(*args)
    return replies


def submit(server, names, version, with_behaviour=True):
    msg = {'type': 'submit_replay', 'param_version': version,
           'logs': {n: f'log {n}'.encode() for n in names}}
    if with_behaviour:
        msg['behaviour'] = {n: f'bp {n}'.encode() for n in names}
    call(server, 'handle_submit_replay', msg)


def test_submit_and_drain_keep_sidecars_paired(server):
    submit(server, ['a.json.gz', 'b.json.gz'], version=5)
    submit(server, ['a.json.gz', 'c.json.gz'], version=6)  # same name from another actor
    assert server.S.buffer_size == 4

    (rsp,) = call(server, 'handle_drain')
    assert rsp['count'] == 4
    logs = sorted(os.listdir(rsp['drain_dir']))
    assert logs == ['0_a.json.gz', '0_b.json.gz', '1_a.json.gz', '1_c.json.gz']
    assert sorted(os.listdir(rsp['behaviour_drain_dir'])) == logs
    for name in logs:
        with open(os.path.join(rsp['behaviour_drain_dir'], name), 'rb') as f:
            assert f.read() == f'bp {name.split("_", 1)[1]}'.encode()
    assert rsp['param_versions'] == {'0_a.json.gz': 5, '0_b.json.gz': 5, '1_a.json.gz': 6, '1_c.json.gz': 6}
    assert rsp['param_version'] == 7
    assert os.listdir(server.S.buffer_dir) == os.listdir(server.S.behaviour_buffer_dir) == []

    # the next drain replaces the previous one in both directories
    submit(server, ['d.json.gz', 'e.json.gz', 'f.json.gz', 'g.json.gz'], version=8)
    (rsp,) = call(server, 'handle_drain')
    assert sorted(os.listdir(rsp['drain_dir'])) == sorted(os.listdir(rsp['behaviour_drain_dir']))
    assert len(os.listdir(rsp['drain_dir'])) == 4
    assert set(rsp['param_versions']) == set(os.listdir(rsp['drain_dir']))


def test_sequential_drain_waits_for_capacity(server):
    submit(server, ['a.json.gz'], version=1)
    (rsp,) = call(server, 'handle_drain')
    assert rsp['count'] == 0 and rsp['param_versions'] == {}
    assert os.listdir(server.S.behaviour_buffer_dir) == ['0_a.json.gz']


def test_client_without_sidecars_still_works(server):
    submit(server, ['a.json.gz', 'b.json.gz', 'c.json.gz', 'd.json.gz'], version=2, with_behaviour=False)
    (rsp,) = call(server, 'handle_drain')
    assert rsp['count'] == 4
    assert os.listdir(rsp['behaviour_drain_dir']) == []


class Exploit:
    """Unpickling this would run code: what a hostile actor host could send (D48)."""
    def __reduce__(self):
        return (os.system, ('echo pwned',))


def test_recv_msg_refuses_code_and_passes_messages(server):
    import pickle
    import socket
    import torch
    from common import recv_msg, send_msg
    msg = {'type': 'submit_replay', 'param_version': 3, 'logs': {'a.json.gz': b'\x1f\x8b'},
           'behaviour': {}, 'state': {'w': torch.ones(2)}, 'is_idle': True}
    a, b = socket.socketpair()
    with a, b:
        send_msg(a, msg)
        got = recv_msg(b)
        assert got['logs'] == msg['logs'] and torch.equal(got['state']['w'], msg['state']['w'])
        send_msg(a, {'type': 'get_param', 'x': Exploit()})
        with pytest.raises(pickle.UnpicklingError):
            recv_msg(b)


def test_server_binds_loopback_only(server):
    server.check_bind_host('127.0.0.1')
    server.check_bind_host('localhost')
    with pytest.raises(SystemExit):
        server.check_bind_host('0.0.0.0')
    server.check_bind_host('0.0.0.0', allow_non_loopback=True)
