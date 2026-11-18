"""D68's check at each stage boundary of D31's curves (`experiments/size-curves.md`): the
pre-registered limits, the stop check through the lab, and the watcher's hook that runs it after
each re-anchor and downs the run on 'stop' (`experiments/curves_stage_check.py`,
`tools/fleet/fleet.py`)."""
import importlib.util
import json
import os
import subprocess
import sys
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'experiments'))

import curves_stage_check as sc  # noqa: E402


def load_fleet():
    spec = importlib.util.spec_from_file_location('fleet', os.path.join(ROOT, 'tools', 'fleet', 'fleet.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fleet = load_fleet()
H0 = 0.5166  # b12c256's entropy on its own states


def test_the_limits():
    assert sc.verdict(0.05, 0.93, 0.42, H0) == ('continue', [])
    assert sc.verdict(0.16, 0.93, 0.42, H0) == ('stop', ['rkl'])
    assert sc.verdict(0.05, 0.87, 0.42, H0) == ('stop', ['agreement'])
    assert sc.verdict(0.05, 0.93, 0.36, H0) == ('stop', ['entropy'])  # below 0.7 × 0.5166 = 0.3616
    assert sc.verdict(0.05, 0.93, 0.42, H0, ci=[-0.012, -0.001]) == ('stop', ['du'])
    assert sc.verdict(0.05, 0.93, 0.42, H0, ci=[-0.012, 0.0005]) == ('continue', [])  # reaches above 0
    # strict at every limit
    assert sc.verdict(sc.MAX_RKL, sc.MIN_AGREEMENT, sc.MIN_ENTROPY_FRACTION * H0, H0, ci=[-0.01, 0.0]) == ('continue', [])


def args(tmp_path, **kw):
    for name in ('snap.pth', 'anchor.pth', 'student.pth'):
        (tmp_path / name).write_bytes(name.encode())
    a = types.SimpleNamespace(run='curves-b12-s1', start=str(tmp_path / 'student.pth'), states=str(tmp_path / 'states'),
                              ckpt=str(tmp_path / 'snap.pth'), anchor=str(tmp_path / 'anchor.pth'),
                              anchor_label='p25', label='p50', out=str(tmp_path / 'out.json'), sets=sc.SETS)
    vars(a).update(kw)
    return a


def drift(rkl=0.05, agreement=0.93, entropy=0.42):
    def fn(states, start, ckpts):
        assert ckpts == [os.path.abspath(ckpts[0])] or len(ckpts) == 1
        return H0, [{'decisions': 15635, 'rkl': rkl, 'rkl_median': rkl / 2, 'kl': rkl, 'agreement': agreement,
                     'entropy': entropy}]
    return fn


def row(desc, mean, ci, run_id='20261120-010101-measure'):
    return {'run_id': run_id, 'test': 'measure', 'verdict': 'measured', 'description': desc, 'nonce_start': 130000,
            'result': {'sets': 10000, 'mean': mean, 'sd': 0.43, 'ci': ci}}


def test_a_drift_limit_stops_without_playing_the_measure(tmp_path):
    def never(argv, desc):
        raise AssertionError('the stop check must not be played')
    rec = sc.check(args(tmp_path), drift_fn=drift(rkl=0.2, entropy=0.3), measure_fn=never)
    assert rec['verdict'] == 'stop' and rec['fired'] == ['rkl', 'entropy'] and rec['measure'] is None
    assert rec['drift']['start_entropy'] == H0 and rec['ckpt_sha256'] == sc.sha256(str(tmp_path / 'snap.pth'))


def test_the_stop_check_plays_the_snapshot_against_its_anchor(tmp_path):
    seen = {}

    def measure(argv, desc):
        seen['argv'], seen['desc'] = argv, desc
        return row(desc, -0.006, [-0.0102, -0.0018])
    rec = sc.check(args(tmp_path), drift_fn=drift(), measure_fn=measure)
    assert seen['desc'] == 'curves boundary: curves-b12-s1 p50 at 3x p25'
    a = seen['argv']
    assert a[1:4] == ['-m', 'lab', 'measure']
    flag = lambda k: a[a.index(k) + 1]  # noqa: E731
    assert flag('--challenger') == str(tmp_path / 'snap.pth') and flag('--champion') == str(tmp_path / 'anchor.pth')
    assert (flag('--sets'), flag('--family'), flag('--block')) == ('10000', 'A', 'curves-boundary')
    assert '--no-logs' not in a  # a stop rule keeps its games
    assert rec['verdict'] == 'stop' and rec['fired'] == ['du'] and rec['measure']['ci'] == [-0.0102, -0.0018]
    rec = sc.check(args(tmp_path), drift_fn=drift(), measure_fn=lambda argv, desc: row(desc, 0.004, [0.0, 0.008]))
    assert rec['verdict'] == 'continue' and rec['measure']['du'] == 0.004


def test_play_measure_reads_a_finished_row_and_waits_for_the_lease(tmp_path):
    registry = tmp_path / 'runs.jsonl'
    lease, lock = tmp_path / 'lease.json', tmp_path / 'lab' / 'stage-check.lock'
    desc = 'curves boundary: r p25 at 3x student'
    calls = []

    def fake_run(argv, cwd, env):
        calls.append(argv)
        with open(registry, 'a') as f:
            f.write(json.dumps(row(desc, 0.001, [-0.003, 0.005])) + '\n')
        return subprocess.CompletedProcess(argv, 0)
    kw = dict(lease=str(lease), lock=str(lock), registry=str(registry), wait_s=0)
    got = sc.play_measure(['py', '-m', 'lab'], desc, run=fake_run, **kw)
    assert got['result']['mean'] == 0.001 and len(calls) == 1
    got = sc.play_measure(['py', '-m', 'lab'], desc, run=fake_run, **kw)
    assert len(calls) == 1  # a relaunch reads the row, it doesn't play again

    lease.write_text('{}')
    with pytest.raises(RuntimeError, match='held the lab registry'):
        sc.play_measure(['py'], 'another', run=fake_run, lease_wait_s=0, **kw)
    lease.unlink()
    with pytest.raises(RuntimeError, match='without a measured row'):
        sc.play_measure(['py'], 'another', run=lambda argv, cwd, env: subprocess.CompletedProcess(argv, 1), **kw)


def test_a_check_that_cannot_be_made_writes_an_error_verdict(tmp_path):
    out = tmp_path / 'out.json'
    rc = sc.main(['--run', 'r', '--start', str(tmp_path / 'missing.pth'), '--states', str(tmp_path),
                  '--ckpt', str(tmp_path / 'missing.pth'), '--anchor', str(tmp_path / 'missing.pth'),
                  '--anchor-label', 'student', '--label', 'p25', '--out', str(out)])
    rec = json.loads(out.read_text())
    assert rc == 1 and rec['verdict'] == 'error' and 'missing.pth' in rec['error']


# ---------------------------------------------------------------- the watcher's hook

@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(fleet, 'RUNS', tmp_path)
    r = fleet.Run('t')
    r.dir.mkdir()
    r.data = {'run': 't', 'config': 'experiments/configs/curves-b12-s1.toml', 'boxes': {}, 'retired': [],
              'bad_machines': [], 'snapshots': {'p25': {'path': 'runs/t/snapshots/p25.pth'},
                                                'p50': {'path': 'runs/t/snapshots/p50.pth'}},
              'anchors': [{'label': 'p25', 'previous': None}]}
    r.save()
    return r


def events(run):
    p = run.dir / 'events.jsonl'
    return [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []


def sleeper(marker):
    """A live process whose command line holds `marker`, as a running check's holds its --out.
    Returned once it has exec'd: until then /proc shows the parent's command line, and on a busy
    box (every rented box runs this suite at setup) the check would read pytest's (2026-11-18)."""
    import time
    p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)', marker])
    for _ in range(500):
        try:
            with open(f'/proc/{p.pid}/cmdline', 'rb') as f:
                if marker.encode() in f.read():
                    return p
        except OSError:
            pass
        time.sleep(0.01)
    p.kill()
    raise RuntimeError('the sleeper never exec\'d')


def test_a_check_starts_at_each_re_anchor_against_the_stages_anchor(run):
    launched, procs = [], []

    def launch(argv, log):
        launched.append(argv)
        procs.append(sleeper(argv[argv.index('--out') + 1]))
        return procs[-1].pid
    try:
        assert fleet.stage_check(run, None, launch) is None and not launched  # off without --stage-check
        cmd = 'py experiments/curves_stage_check.py --run t --start S --states D'
        assert fleet.stage_check(run, cmd, launch) is None
        assert fleet.stage_check(run, cmd, launch) is None  # running: not launched again
        assert len(launched) == 1
        a = launched[0]
        flag = lambda k: a[a.index(k) + 1]  # noqa: E731
        assert a[:2] == ['py', 'experiments/curves_stage_check.py']
        assert flag('--ckpt') == str(fleet.REPO / 'runs/t/snapshots/p25.pth')
        # the first stage's anchor is the student, the config's opponent
        assert flag('--anchor').endswith('students-s1/ema-b12c256.pth') and flag('--anchor-label') == 'student'
        assert flag('--out') == str(run.dir / 'stage-check-p25.json')
        run.data['anchors'].append({'label': 'p50', 'previous': 'p25'})
        fleet.stage_check(run, cmd, launch)
        b = launched[1]
        assert b[b.index('--anchor') + 1] == str(fleet.REPO / 'runs/t/snapshots/p25.pth')
        assert b[b.index('--anchor-label') + 1] == 'p25' and b[b.index('--label') + 1] == 'p50'
        assert [e['kind'] for e in events(run)] == ['stage_check_started'] * 2
    finally:
        for p in procs:
            p.kill()


def test_a_stop_verdict_is_returned_once(run):
    procs = []

    def launch(argv, log):
        procs.append(sleeper(argv[argv.index('--out') + 1]))
        return procs[-1].pid
    try:
        fleet.stage_check(run, 'check', launch)
        (run.dir / 'stage-check-p25.json').write_text(json.dumps({
            'verdict': 'continue', 'fired': [], 'drift': {'rkl': 0.03, 'agreement': 0.94, 'entropy': 0.45},
            'measure': {'du': 0.004, 'ci': [0.0, 0.008]}}))
        assert fleet.stage_check(run, 'check', launch) is None
        run.data['anchors'].append({'label': 'p50', 'previous': 'p25'})
        fleet.stage_check(run, 'check', launch)
        (run.dir / 'stage-check-p50.json').write_text(json.dumps({
            'verdict': 'stop', 'fired': ['du'], 'drift': {'rkl': 0.05, 'agreement': 0.93, 'entropy': 0.44},
            'measure': {'du': -0.006, 'ci': [-0.010, -0.002]}}))
        assert fleet.stage_check(run, 'check', launch) == 'stop'
        assert fleet.stage_check(run, 'check', launch) is None  # read once
        got = [e for e in events(run) if e['kind'] == 'stage_check']
        assert [(e['label'], e['verdict'], e['fired']) for e in got] == [('p25', 'continue', '-'), ('p50', 'stop', 'du')]
        assert fleet.Run.load('t').data['stage_checks']['p50']['verdict'] == 'stop'
    finally:
        for p in procs:
            p.kill()


def test_a_check_that_died_is_relaunched_once_then_reported(run):
    def dead(argv, log):
        p = subprocess.Popen(['true'])
        p.wait()
        return p.pid
    fleet.stage_check(run, 'check', dead)
    fleet.stage_check(run, 'check', dead)  # died without a verdict: launched again
    assert fleet.stage_check(run, 'check', dead) is None  # died twice: reported, never a stop
    assert fleet.stage_check(run, 'check', dead) is None
    assert [e['kind'] for e in events(run)] == ['stage_check_started', 'stage_check_relaunched', 'stage_check_failed']
    # an 'error' verdict (the check couldn't be made) is reported and doesn't stop either
    run.data['anchors'].append({'label': 'p50', 'previous': 'p25'})
    fleet.stage_check(run, 'check', dead)
    (run.dir / 'stage-check-p50.json').write_text(json.dumps({'verdict': 'error', 'fired': [], 'error': 'lease'}))
    assert fleet.stage_check(run, 'check', dead) is None
    assert events(run)[-1]['kind'] == 'stage_check' and events(run)[-1]['error'] == 'lease'


def test_a_pid_from_before_a_reboot_isnt_taken_for_the_check(tmp_path):
    marker = str(tmp_path / 'stage-check-p25.json')
    p = sleeper(marker)
    try:
        assert fleet.check_running(p.pid, marker)
        assert not fleet.check_running(os.getpid(), marker)  # alive, but some other process
    finally:
        p.kill()
        p.wait()
    assert not fleet.check_running(p.pid, marker)
