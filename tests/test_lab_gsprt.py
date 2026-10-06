import numpy as np
import pytest

from lab import calibrate as cal
from lab import gsprt, load_config, utility

U = utility()
SCHED = gsprt.Schedule(min_sets=5000, look_every=500, floor_frac=0.25, sigma_set=0.42)
T_STD = gsprt.TestSpec('T-std', 0.0, 0.01, 100_000, -2.944, 2.944)
T_SMOKE = gsprt.TestSpec('T-smoke', -0.02, 0.0, 20_000, -2.944, 2.944)


def test_config_matches_plan():
    cfg = load_config()
    assert cfg['utility'] == [0.90, 0.50, -0.10, -1.30]
    specs = gsprt.specs_from_config(cfg)
    assert (specs['T-std'].mu0, specs['T-std'].mu1, specs['T-std'].cap) == (0.0, 0.01, 100_000)
    assert (specs['T-NI'].mu0, specs['T-NI'].mu1, specs['T-NI'].cap) == (-0.01, 0.0, 60_000)
    assert (specs['T-smoke'].mu0, specs['T-fine'].mu1) == (-0.02, 0.005)
    sched = gsprt.schedule_from_config(cfg)
    assert (sched.min_sets, sched.look_every, sched.floor_frac) == (5000, 500, 0.25)


def test_llr_matches_closed_form():
    # LLR = N/(2 s²)·[(x̄−μ0)² − (x̄−μ1)²]; at x̄ = μ1 it is N δ² / (2 s²)
    n, s2, d = 10_000, 0.42 ** 2, 0.01
    assert gsprt.llr(n, d, s2, 0.0, d) == pytest.approx(n * d * d / (2 * s2))
    assert gsprt.llr(n, d / 2, s2, 0.0, d) == pytest.approx(0.0)
    assert gsprt.llr(n, 0.0, s2, 0.0, d) == pytest.approx(-n * d * d / (2 * s2))


def test_looks_end_at_cap():
    looks = SCHED.looks(20_000)
    assert looks[0] == 5000 and looks[-1] == 20_000
    assert (np.diff(looks) == 500).all()


def test_zero_variance_gives_no_verdict():
    status, value = gsprt.evaluate(10_000, 0.0, 0.0, T_STD, SCHED)
    assert status == gsprt.CONTINUE and value == 0.0
    status, _ = gsprt.evaluate(100_000, 0.0, 0.0, T_STD, SCHED)
    assert status == gsprt.INCONCLUSIVE


def test_variance_floor_applies():
    # constant-ish stream with tiny variance: the floor must dominate
    n, mean, tiny = 10_000, 0.01, 1e-6
    s1 = n * mean
    s2 = n * (mean ** 2) + (n - 1) * tiny
    _, value = gsprt.evaluate(n, s1, s2, T_STD, SCHED)
    assert value == pytest.approx(gsprt.llr(n, mean, SCHED.var_floor, 0.0, 0.01), rel=1e-6)


def test_no_verdict_before_min_sets():
    status, _ = gsprt.evaluate(4999, 4999 * 0.5, 4999 * 0.5, T_STD, SCHED)
    assert status == gsprt.CONTINUE


def _stream(n, mean, seed):
    rng = np.random.default_rng(seed)
    base = cal.independent_ranks(U)
    idx = rng.choice(len(base.values), size=n, p=base.probs)
    return base.values[idx] + mean


@pytest.mark.parametrize('chunk', [137, 500, 4096])
def test_sequential_is_chunking_invariant(chunk):
    x = _stream(100_000, 0.02, seed=1)
    ref = gsprt.SequentialTest(T_STD, SCHED)
    ref.add(x)
    t = gsprt.SequentialTest(T_STD, SCHED)
    for i in range(0, len(x), chunk):
        if t.add(x[i:i + chunk]) != gsprt.CONTINUE:
            break
    assert (t.status, t.n_decided) == (ref.status, ref.n_decided)
    assert t.llr == pytest.approx(ref.llr)
    assert t.status == gsprt.PASS


def test_live_test_agrees_with_simulator():
    # the calibration simulator must implement the same stopping rule
    for seed, mean in [(2, 0.0), (3, 0.01), (4, 0.005), (5, -0.02)]:
        x = _stream(T_STD.cap, mean, seed)
        live = gsprt.SequentialTest(T_STD, SCHED)
        live.add(x)
        looks = SCHED.looks(T_STD.cap)
        c1 = np.cumsum(x)[looks - 1]
        c2 = np.cumsum(np.square(x))[looks - 1]
        paths = np.array([[gsprt.evaluate(n, a, b, T_STD, SCHED)[1] for n, a, b in zip(looks, c1, c2)]])
        r = cal.score_paths(looks, paths, np.zeros_like(paths, dtype=bool), T_STD.lower, T_STD.upper)
        sim_status = gsprt.PASS if r['pass'] else gsprt.FAIL if r['fail'] else gsprt.INCONCLUSIVE
        assert sim_status == live.status
        assert r['mean_sets'] == (live.n_decided or T_STD.cap)


def test_set_scores_groups_rotations():
    seeds = np.repeat([7, 8], 4)
    seats = np.tile(np.arange(4), 2)
    ranks = np.array([0, 1, 2, 3, 0, 0, 0, 0])
    s, x = gsprt.set_scores(seeds, seats, ranks, U)
    assert s.tolist() == [7, 8]
    assert x[0] == 0.0  # exact, not approximate
    assert x[1] == pytest.approx(0.9)
    with pytest.raises(ValueError):
        gsprt.set_scores(seeds[:7], seats[:7], ranks[:7], U)
    with pytest.raises(ValueError):
        gsprt.set_scores(seeds, np.roll(seats, 1), ranks, U)


def test_bootstrap_ci_close_to_normal():
    x = _stream(50_000, 0.0, seed=6)
    lo, hi = gsprt.bootstrap_mean_ci(x, resamples=4000)
    half = 1.96 * x.std(ddof=1) / np.sqrt(len(x))
    assert lo == pytest.approx(x.mean() - half, abs=half * 0.1)
    assert hi == pytest.approx(x.mean() + half, abs=half * 0.1)


def test_confirm_rules():
    x = _stream(50_000, 0.05, seed=7)
    assert gsprt.confirm(x, 'superiority', resamples=2000)['passed']
    y = _stream(50_000, -0.003, seed=8)
    assert not gsprt.confirm(y, 'superiority', resamples=2000)['passed']
    assert gsprt.confirm(y, 'non-inferiority', delta=0.02, resamples=2000)['passed']
