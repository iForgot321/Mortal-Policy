"""riichi.dev operations (07 §1): latency classes, the daily summary and the supervisor's restart
policy. No riichienv needed."""
import json
import os
import subprocess

from deploy.riichidev_bot import decision_class, latency_summary
from deploy.riichidev_summary import rank_of, summarize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_latency_classes():
    assert decision_class({'type': 'ankan'}) == decision_class({'type': 'kakan'}) == 'kan'
    assert decision_class({'type': 'none'}) == 'none'
    s = latency_summary({'dahai': [float(i) for i in range(1, 101)], 'none': [5.0]})
    assert s['dahai'] == {'n': 100, 'p50': 51.0, 'p99': 100.0, 'max': 100.0}
    assert s['none']['n'] == 1


def test_rank_ties_go_to_the_earlier_seat():
    assert [rank_of([25000, 25000, 30000, 20000], s) for s in range(4)] == [2, 3, 1, 4]


def test_summarize(tmp_path):
    log = tmp_path / 'g1.jsonl'
    log.write_text('\n'.join(json.dumps(r) for r in [
        {'dir': 'in', 'type': 'request_action'},
        {'dir': 'decision', 'class': 'dahai', 'ms': 12.0},
        {'dir': 'decision', 'class': 'none', 'ms': 3.0}]) + '\n')
    t = 1_760_000_000  # 2025-10-09 UTC
    games = [
        {'start': t, 'end': t + 600, 'seat': 1, 'result': {'scores': [20000, 40000, 30000, 10000]},
         'timeouts': 1, 'failures': [], 'log': str(log)},
        {'start': t + 700, 'end': t + 1300, 'seat': 0, 'result': {'scores': [10000, 40000, 30000, 20000]},
         'timeouts': 0, 'failures': [{'kind': 'ack_rejected'}, {'kind': 'mask_mismatch'}], 'log': 'gone'},
        {'start': t + 1400, 'seat': 2, 'result': None, 'timeouts': 0, 'failures': []},
    ]
    days, latency = summarize(games)
    (d,) = days.values()
    assert d['games'] == 3 and d['ranks'] == [1, 0, 0, 1] and d['unfinished'] == 1
    assert d['timeouts'] == 1 and d['chombo'] == 1 and d['failures'] == 1
    assert abs(d['mean_u'] - (0.9 - 1.3) / 2) < 1e-12
    assert latency['dahai']['n'] == 1 and latency['none']['max'] == 3.0


def supervise(tmp_path, codes):
    """Run the supervisor with a fake bot that exits with `codes` in turn; returns (rc, runs)."""
    count = tmp_path / 'count'
    fake = tmp_path / 'fake.sh'
    fake.write_text('#!/usr/bin/env bash\n'
                    f'n=$(cat {count} 2>/dev/null || echo 0); echo $((n + 1)) > {count}\n'
                    f'codes=({" ".join(map(str, codes))}); exit ${{codes[$n]}}\n')
    fake.chmod(0o755)
    p = subprocess.run([os.path.join(ROOT, 'deploy', 'riichidev_supervise.sh'), '--model', 'x'],
                       env={**os.environ, 'RIICHIDEV_PY': str(fake), 'RIICHIDEV_BACKOFF_S': '0'},
                       capture_output=True, text=True, timeout=60)
    return p.returncode, int(count.read_text())


def test_supervisor_restarts_crashes_only(tmp_path):
    # crashes restart; 0 ends; a failed game (3) ends; 5 crashes within an hour end
    for name, codes, want in (('a', [1, 1, 0], (0, 3)), ('b', [1, 3, 0], (3, 2)), ('c', [1] * 5 + [0], (1, 5))):
        (tmp_path / name).mkdir()
        assert supervise(tmp_path / name, codes) == want
