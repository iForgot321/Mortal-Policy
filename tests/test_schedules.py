"""K2's schedules (05 §5.1, D59): power laws on the run clock (`mortal/schedules.py`)."""
import pytest

import schedules
from schedules import PowerLaw


def test_power_law_values():
    s = PowerLaw(2e-5, p=0.3, t0=1000)
    assert s(0) == pytest.approx(2e-5)
    assert s(1000) == pytest.approx(2e-5 * 2 ** -0.3)
    assert s(9000) == pytest.approx(2e-5 * 10 ** -0.3)
    assert s(-5) == pytest.approx(2e-5)  # before the origin: the start value
    assert PowerLaw(0.2)(10 ** 9) == pytest.approx(0.2)  # p = 0: constant
    assert PowerLaw(0.02, p=-0.5, t0=100)(300) == pytest.approx(0.04)  # p < 0 grows


def test_power_law_bounds():
    s = PowerLaw(0.2, p=1.0, t0=10, lo=0.05, hi=0.15)
    assert s(0) == pytest.approx(0.15)
    assert s(30) == pytest.approx(0.05)
    assert s(10 ** 6) == pytest.approx(0.05)
    for bad in ({'t0': 0}, {'lo': 0.2, 'hi': 0.1}):
        with pytest.raises(ValueError):
            PowerLaw(0.1, **bad)


def test_build_fills_constants_and_shares_t0():
    consts = {'clip_ratio': 0.2, 'entropy_weight': 1e-3}
    s = schedules.build({}, consts)
    assert s['lr'] is None
    assert s['clip_ratio'](10 ** 6) == pytest.approx(0.2)
    assert s['entropy_weight'](10 ** 6) == pytest.approx(1e-3)
    s = schedules.build({'t0': 500, 'lr': {'v0': 2e-5, 'p': 0.3, 'lo': 4e-6},
                         'entropy_weight': {'v0': 1e-3, 'p': 0.5, 't0': 100}}, consts)
    assert s['lr'](500) == pytest.approx(2e-5 * 2 ** -0.3)
    assert s['lr'](10 ** 9) == pytest.approx(4e-6)
    assert s['entropy_weight'](300) == pytest.approx(1e-3 * 4 ** -0.5)  # its own t0
    assert s['clip_ratio'](10 ** 6) == pytest.approx(0.2)  # left out: constant
    with pytest.raises(ValueError):
        schedules.build({'learning_rate': {'v0': 1}}, consts)


def test_run_clock_origin():
    assert schedules.origin(None, 0) == 0
    # a run initialized from a B0 snapshot: the clock starts at the snapshot's steps
    assert schedules.origin({'steps': 244_800}, 244_800) == 244_800
    # a restarted trainer keeps the saved origin
    assert schedules.origin({'steps': 300_000, 'schedule_origin': 244_800}, 300_000) == 244_800
