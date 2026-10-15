"""tools/fleet/evalbox.py: the ledger merge (box → workstation, append-only), the engine files a job
needs, and the lab lease. The box side runs only on a rented box (tools/fleet/README.md)."""
import importlib.util
import os
import sys

import pytest

from lab import registry

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location('evalbox', os.path.join(ROOT, 'tools', 'fleet', 'evalbox.py'))
evalbox = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evalbox)
spec = importlib.util.spec_from_file_location('labq', os.path.join(ROOT, 'tools', 'fleet', 'labq.py'))
labq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(labq)


def write(d, name, text):
    d.mkdir(exist_ok=True)
    (d / name).write_text(text)


def test_merge_appends_box_rows(tmp_path):
    local, box = tmp_path / 'local', tmp_path / 'box'
    write(local, 'runs.jsonl', '{"a": 1}\n')
    write(local, 'nonces.jsonl', '{"n": 1}\n')
    write(box, 'runs.jsonl', '{"a": 1}\n{"a": 2}\n{"a": 3}\n')
    write(box, 'nonces.jsonl', '{"n": 1}\n')
    assert evalbox.merge_ledgers(local, box) == {'runs.jsonl': 2, 'nonces.jsonl': 0}
    assert (local / 'runs.jsonl').read_text() == (box / 'runs.jsonl').read_text()
    assert evalbox.merge_ledgers(local, box) == {'runs.jsonl': 0, 'nonces.jsonl': 0}


def test_merge_refuses_a_fork_and_writes_nothing(tmp_path):
    local, box = tmp_path / 'local', tmp_path / 'box'
    write(local, 'runs.jsonl', '{"a": 1}\n')
    write(local, 'nonces.jsonl', '{"n": 1}\n{"n": "local"}\n')
    write(box, 'runs.jsonl', '{"a": 1}\n{"a": 2}\n')
    write(box, 'nonces.jsonl', '{"n": 1}\n{"n": "box"}\n')
    with pytest.raises(evalbox.F.FleetError, match='forked'):
        evalbox.merge_ledgers(local, box)
    assert (local / 'runs.jsonl').read_text() == '{"a": 1}\n'


def test_engine_files():
    pilot = 'workdir/pilot/qs-a/pilot-qs-a-final.pth'
    if not os.path.exists(os.path.join(ROOT, pilot)):
        pytest.skip('needs the frozen pilot')
    args = ['measure', '--challenger', pilot + '?eps=0.1', '--champion', 'builtin:shanten-greedy', '--sets', '10']
    assert evalbox.engine_files(args) == [pilot]
    with pytest.raises(evalbox.F.FleetError, match='repo-relative'):
        evalbox.engine_files(['measure', '--challenger', os.path.join(ROOT, pilot)])


def test_lease_blocks_only_the_real_ledger(tmp_path, monkeypatch):
    real = str(tmp_path / 'nonces.jsonl')
    monkeypatch.setattr(registry, 'NONCES', real)
    monkeypatch.setattr(registry, 'LEASE', str(tmp_path / 'lab-lease.json'))
    registry.allocate_nonces('B', 10, 'r1', 'x', path=real)
    (tmp_path / 'lab-lease.json').write_text('{}\n')
    with pytest.raises(RuntimeError, match='eval box'):
        registry.allocate_nonces('B', 10, 'r2', 'x', path=real)
    registry.allocate_nonces('B', 10, 'r3', 'x', path=str(tmp_path / 'other.jsonl'))


def test_a_job_that_cannot_start_fails_without_killing_the_runner(tmp_path, monkeypatch):
    """A missing interpreter once killed the runner and stalled the queue for hours (2026-10-15)."""
    monkeypatch.setattr(labq, 'Q', str(tmp_path))
    job = {'id': 'j1', 'args': ['--x'], 'module': 'deploy.riichidev_parity', 'python': '/no/such/python'}
    st = labq.run_job(job)
    assert st['status'] == 'failed' and st['exit'] is None and 'did not start' in st['note']
    assert labq.state('j1') == st
    ok = labq.run_job({'id': 'j2', 'args': ['-c', 'pass'], 'module': 'json.tool', 'python': sys.executable})
    assert ok['status'] in ('done', 'failed') and labq.state('j2')['exit'] == ok['exit']
