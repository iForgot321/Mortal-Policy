"""The lab's free-space guard (lab/diskguard.py)."""
import argparse

import pytest

from lab import __main__ as cli
from lab import diskguard, registry


@pytest.fixture
def disk(monkeypatch):
    """A fake machine: free GB per watched path, set by the test."""
    free = {'/lab': 500.0, '/host': 500.0}
    monkeypatch.setattr(diskguard, 'watched', lambda: list(free))
    monkeypatch.setattr(diskguard, 'free_gb', lambda path: free[path])
    monkeypatch.delenv('LAB_MIN_FREE_GB', raising=False)
    return free


def test_log_estimate_matches_the_measured_runs():
    # 3.3 GB per 20K-set measure and 17-18 GB per 100K-set run, four neural seats
    assert 3.3 <= diskguard.log_gb(20_000) <= 4.0
    assert 17.0 <= diskguard.log_gb(100_000) <= 19.0
    assert diskguard.log_gb(100_000, logs=False) == 0


def test_start_needs_the_logs_plus_the_reserve(disk):
    need = diskguard.log_gb(20_000) + diskguard.DEFAULT_MIN_FREE_GB
    disk['/host'] = need + 0.1
    diskguard.check_start(20_000)
    disk['/host'] = need - 0.1  # the host drive alone is short
    with pytest.raises(SystemExit, match='/host has'):
        diskguard.check_start(20_000)
    diskguard.check_start(20_000, logs=False)  # without logs only the reserve counts
    disk['/host'] = 2.2  # the state of 2026-11-10
    with pytest.raises(SystemExit, match='disk guard'):
        diskguard.check_start(20_000, logs=False)


def test_the_reserve_is_set_by_the_environment(disk, monkeypatch):
    disk['/host'] = 2.2
    monkeypatch.setenv('LAB_MIN_FREE_GB', '0')
    diskguard.check_start(100_000)
    assert diskguard.wait_for_space() == 0
    monkeypatch.setenv('LAB_MIN_FREE_GB', '1')
    diskguard.check_start(5_000)
    with pytest.raises(SystemExit):
        diskguard.check_start(20_000)


def test_the_host_drive_is_watched_only_under_wsl(monkeypatch, tmp_path):
    monkeypatch.setattr(diskguard, 'RUNS_DIR', str(tmp_path / 'not' / 'yet'))
    monkeypatch.delenv('LAB_HOST_DRIVE', raising=False)
    monkeypatch.setattr(diskguard, '_is_wsl', lambda: False)
    assert diskguard.watched() == [str(tmp_path)]  # the nearest existing parent
    host = tmp_path / 'c'
    host.mkdir()
    monkeypatch.setenv('LAB_HOST_DRIVE', str(host))
    assert diskguard.watched() == [str(tmp_path), str(host)]
    monkeypatch.setenv('LAB_HOST_DRIVE', '')
    assert diskguard.watched() == [str(tmp_path)]
    monkeypatch.setenv('LAB_HOST_DRIVE', str(tmp_path / 'missing'))
    assert diskguard.watched() == [str(tmp_path)]
    assert diskguard.free_gb(str(tmp_path)) > 0


def test_a_running_run_waits_until_space_is_back(disk, monkeypatch):
    naps = []

    def nap(seconds):
        naps.append(seconds)
        if len(naps) == 3:
            disk['/host'] = 6.0  # someone freed space

    monkeypatch.setattr(diskguard.time, 'sleep', nap)
    assert diskguard.wait_for_space() == 0 and not naps  # plenty free: no wait
    disk['/host'] = 4.9  # under half the 10 GB reserve
    assert diskguard.wait_for_space(poll=60) == 180
    assert naps == [60, 60, 60]


def test_a_refused_run_reserves_no_nonces(disk, monkeypatch):
    def no_nonces(*a, **k):
        raise AssertionError('nonces were reserved before the disk check')

    monkeypatch.setattr(registry, 'allocate_nonces', no_nonces)
    disk['/host'] = 2.2
    args = argparse.Namespace(sets=20_000, no_logs=False, test='T-std', sigma_set=None, family='B')
    with pytest.raises(SystemExit, match='disk guard'):
        cli._fixed(args, 'measure', 'A', 'test')
    with pytest.raises(SystemExit, match='disk guard'):
        cli.cmd_run(args)
