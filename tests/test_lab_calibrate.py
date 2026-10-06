import numpy as np
import pytest

from lab import calibrate as cal
from lab import gsprt, registry, utility

U = utility()


def test_independent_ranks_distribution():
    d = cal.independent_ranks(U)
    assert d.probs.sum() == pytest.approx(1.0)
    assert len(d.values) == 35  # multisets of 4 ranks
    assert d.mean == pytest.approx(0.0, abs=1e-12)
    # per-game SD of u under a uniform split is ≈ 0.831 (02 §2); set SD is half
    game_sd = np.sqrt(np.mean(np.square(U)))
    assert d.sd == pytest.approx(game_sd / 2)


def test_variants_preserve_mean_and_sd():
    base = cal.independent_ranks(U).shifted(0.01)
    for v in cal.variants(base):
        assert v.probs.sum() == pytest.approx(1.0)
        assert v.mean == pytest.approx(base.mean, abs=1e-12), v.name
        assert v.sd == pytest.approx(base.sd, rel=1e-9), v.name


def test_empirical_compresses_continuous_scores():
    x = np.random.default_rng(0).normal(0.003, 0.3, 50_000)
    d = cal.empirical(x)
    assert len(d.values) == cal.MAX_ATOMS
    assert d.mean == pytest.approx(x.mean(), abs=1e-12)


def test_simulated_error_rates_are_sane():
    sched = gsprt.Schedule(min_sets=5000, look_every=500, floor_frac=0.25, sigma_set=0.4155)
    spec = gsprt.TestSpec('T-smoke', -0.02, 0.0, 20_000, -2.944, 2.944)
    base = cal.independent_ranks(U)
    h0 = cal.simulate_paths(base.shifted(spec.mu0), spec, sched, n_tests=2000, seed=1)
    h1 = cal.simulate_paths(base.shifted(spec.mu1), spec, sched, n_tests=2000, seed=2)
    r0 = cal.score_paths(*h0, spec.lower, spec.upper)
    r1 = cal.score_paths(*h1, spec.lower, spec.upper)
    assert r0['pass'] + r0['fail'] + r0['inconclusive'] == pytest.approx(1.0)
    # Wald bounds with α = β = 0.05: both error rates near or below 5%
    assert r0['pass'] < 0.08
    assert 1 - r1['pass'] < 0.08
    # wider boundaries can only reduce false passes
    assert cal.score_paths(*h0, -4.0, 4.0)['pass'] <= r0['pass']


def test_nonce_ledger(tmp_path):
    path = str(tmp_path / 'nonces.jsonl')
    a, key_b = registry.allocate_nonces('B', 100, 'r1', 'x', path=path)
    b, _ = registry.allocate_nonces('B', 50, 'r2', 'x', path=path)
    assert b == a + 100
    c, key_c = registry.allocate_nonces('C', 10, 'r3', 'x', path=path)
    assert key_c != key_b and c >= 2_000_000_000
    blk, _ = registry.allocate_nonces('A', 2000, 'r4', 'track', block='track-1', path=path)
    again, _ = registry.allocate_nonces('A', 2000, 'r5', 'track', block='track-1', path=path)
    assert again == blk
    nxt, _ = registry.allocate_nonces('A', 10, 'r6', 'x', path=path)
    assert nxt == blk + 2000
    with pytest.raises(ValueError):
        registry.allocate_nonces('C', 10, 'r7', 'x', block='reuse', path=path)
