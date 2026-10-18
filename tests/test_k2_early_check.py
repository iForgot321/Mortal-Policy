"""The G0 retry's early check (D59): the pre-registered verdict, and the watcher's hook that runs it
and downs the run on 'stop' (`experiments/k2_early_check.py`, `tools/fleet/fleet.py`)."""
import importlib.util
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'experiments'))

import k2_early_check as ec  # noqa: E402


def load_fleet():
    spec = importlib.util.spec_from_file_location('fleet', os.path.join(ROOT, 'tools', 'fleet', 'fleet.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fleet = load_fleet()


def test_verdict_thresholds():
    assert ec.verdict(0.02, 0.95, 0.0, 0.006) == ('continue', [])
    # P0/P1 at p25 (rl-diagnosis.md): both drift measures fire
    assert ec.verdict(0.35, 0.81) == ('stop', ['rkl', 'agreement'])
    assert ec.verdict(0.12, 0.95, 0.0, 0.006) == ('stop', ['rkl'])
    assert ec.verdict(0.02, 0.89, 0.0, 0.006) == ('stop', ['agreement'])
    assert ec.verdict(0.02, 0.95, -0.02, 0.006) == ('stop', ['du'])  # CI [-0.032, -0.008]
    assert ec.verdict(0.02, 0.95, -0.01, 0.006) == ('continue', [])  # CI reaches above 0


def test_verdict_boundaries_are_strict():
    assert ec.verdict(ec.MAX_RKL, ec.MIN_AGREEMENT, -ec.Z * 0.005, 0.005) == ('continue', [])
    assert ec.verdict(0.02, 0.95) == ('continue', [])  # measure 3 not played


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(fleet, 'RUNS', tmp_path)
    r = fleet.Run('t')
    r.dir.mkdir()
    r.data = {'run': 't', 'boxes': {}, 'retired': [], 'bad_machines': []}
    r.save()
    return r


def events(run):
    p = run.dir / 'events.jsonl'
    return [json.loads(line)['kind'] for line in p.read_text().splitlines()] if p.exists() else []


def test_early_check_waits_for_its_snapshot_then_launches_once(run):
    launched = []
    launch = lambda argv, log: launched.append(argv) or os.getpid()  # noqa: E731
    assert fleet.early_check(run, 'py check.py --init I', 'p25', launch) is None
    assert launched == []  # no p25 snapshot yet
    run.data['snapshots'] = {'p25': {'path': 'workdir/fleet/runs/t/snapshots/p25.pth'}}
    assert fleet.early_check(run, 'py check.py --init I', 'p25', launch) is None
    assert launched[0][:4] == ['py', 'check.py', '--init', 'I']
    assert launched[0][4:6] == ['--ckpt', str(fleet.REPO / 'workdir/fleet/runs/t/snapshots/p25.pth')]
    assert launched[0][6:] == ['--out', str(run.dir / 'early-check-p25.json')]
    # still running (our own pid is alive): nothing more
    assert fleet.early_check(run, 'py check.py --init I', 'p25', launch) is None
    assert len(launched) == 1 and events(run) == ['early_check_started']
    assert fleet.Run.load('t').data['early_check']['pid'] == os.getpid()


@pytest.mark.parametrize('verdict', ['stop', 'continue'])
def test_early_check_returns_the_verdict_once(run, verdict):
    run.data['snapshots'] = {'p25': {'path': 'x.pth'}}
    fleet.early_check(run, 'check', 'p25', lambda argv, log: os.getpid())
    (run.dir / 'early-check-p25.json').write_text(json.dumps({
        'verdict': verdict, 'fired': ['rkl'] if verdict == 'stop' else [],
        'drift': {'rkl': 0.3, 'agreement': 0.8}, 'argmax': None}))
    assert fleet.early_check(run, 'check', 'p25') == verdict
    assert fleet.early_check(run, 'check', 'p25') is None  # handled
    assert events(run) == ['early_check_started', 'early_check']
    assert fleet.Run.load('t').data['early_check']['verdict'] == verdict


def test_a_check_that_died_is_reported_once_and_never_stops(run):
    run.data['snapshots'] = {'p25': {'path': 'x.pth'}}
    dead = subprocess.Popen(['true'])
    dead.wait()
    fleet.early_check(run, 'check', 'p25', lambda argv, log: dead.pid)
    assert fleet.early_check(run, 'check', 'p25') is None
    assert fleet.early_check(run, 'check', 'p25') is None
    assert events(run) == ['early_check_started', 'early_check_failed']


def test_anchor_alert_crosses_both_ways(run):
    (run.dir / 'learner-run').mkdir()
    log = run.dir / 'learner-run' / 'anchor.jsonl'
    now = 1_000_000.0

    def write(kls, start):
        with open(log, 'a') as f:
            for i, kl in enumerate(kls):
                f.write(json.dumps({'time': start + 600 * i, 'kl': kl}) + '\n')

    assert fleet.anchor_alert(run, 0.10, now=now) is None  # no log yet
    write([0.5, 0.5], now - 1200)
    assert fleet.anchor_alert(run, 0.10, now=now) is None  # fewer than 3 entries
    write([0.2], now)
    log.write_text(log.read_text() + '{"time": 1, "kl"')  # a line torn by the pull
    assert fleet.anchor_alert(run, 0.10, now=now) == pytest.approx(0.4)
    assert fleet.anchor_alert(run, 0.10, now=now) == pytest.approx(0.4)  # one event per crossing
    assert events(run) == ['anchor_alert']
    log.write_text('')
    write([0.01, 0.01, 0.01], now + 3 * 3600)
    assert fleet.anchor_alert(run, 0.10, now=now + 4 * 3600) == pytest.approx(0.01)
    assert events(run) == ['anchor_alert', 'anchor_alert_cleared']
    assert fleet.anchor_alert(run, None, now=now) is None  # off
