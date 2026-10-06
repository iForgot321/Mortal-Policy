"""Ordered transition extraction (05 §3.3) unit tests (05 §8: determinism, done
alignment, no cross-seat targets, fixed units)."""
import glob
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'mortal'))

import trajectory as tj  # noqa: E402

AT_KYOKU = np.array([0, 0, 0, 1, 1, 3, 3, 3])   # this seat had no decision in kyoku 2
R = np.array([10.0, -5.0, 7.0, 2.0, -1.0])       # per-kyoku GRP-Δ, kyoku 4 after the last decision


def test_k_horizon_done_and_rewards():
    t = tj.build(AT_KYOKU, R, 'K')
    assert t['done'].tolist() == [False, False, True, False, True, False, False, True]
    assert t['reward'].tolist() == [0, 0, 10, 0, -5, 0, 0, 2]
    assert t['next_idx'].tolist() == [1, 2, -1, 4, -1, 6, 7, -1]
    # each decision's Monte-Carlo return is its own kyoku's reward
    ret = tj.monte_carlo_returns(t['reward'], t['done'])
    assert ret.tolist() == [10, 10, 10, -5, -5, 2, 2, 2]


def test_h_horizon_telescopes():
    t = tj.build(AT_KYOKU, R, 'H')
    assert t['done'].tolist() == [False] * 7 + [True]
    # every kyoku's reward is paid exactly once, including kyoku 2 (no
    # decision) and kyoku 4 (after the seat's last decision)
    assert t['reward'].sum() == pytest.approx(R.sum())
    ret = tj.monte_carlo_returns(t['reward'], t['done'])
    assert ret[0] == pytest.approx(R.sum())
    assert ret[3] == pytest.approx(R[1:].sum())


def test_gae_limits():
    t = tj.build(AT_KYOKU, R, 'H')
    v = np.random.default_rng(0).normal(size=len(AT_KYOKU))
    adv, ret = tj.gae(t['reward'], t['done'], v, gamma=1.0, lam=1.0)
    mc = tj.monte_carlo_returns(t['reward'], t['done'])
    assert np.allclose(ret, mc)                    # λ = 1: λ-return = Monte-Carlo return
    adv0, _ = tj.gae(t['reward'], t['done'], v, gamma=1.0, lam=0.0)
    nxt = np.append(v[1:], 0.0) * ~t['done']
    assert np.allclose(adv0, t['reward'] + nxt - v)  # λ = 0: one-step TD error


def test_bit_identical_recomputation():
    v = np.linspace(-1, 1, len(AT_KYOKU))
    a = tj.gae(**{k: tj.build(AT_KYOKU, R, 'K')[k] for k in ('reward', 'done')}, values=v)
    b = tj.gae(**{k: tj.build(AT_KYOKU, R, 'K')[k] for k in ('reward', 'done')}, values=v)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        tj.build([0, 1, 0], R, 'K')                # not ordered: crosses seats or games
    with pytest.raises(ValueError):
        tj.build([0, 9], R, 'K')                   # no reward for kyoku 9
    with pytest.raises(ValueError):
        tj.build(AT_KYOKU, R, 'X')


FILES = sorted(glob.glob(os.path.expanduser('~/mj-data/mjai/2009/04/*.json.gz')))[:5]


@pytest.mark.skipif(not FILES, reason='needs converted data in ~/mj-data')
def test_done_aligns_with_loader_kyoku_ends():
    from libriichi.dataset import GameplayLoader
    for game in GameplayLoader(version=4, oracle=False, featurize=False).load_gz_log_files(FILES):
        for gp in game:
            at_kyoku = gp.take_at_kyoku()
            loader_dones = gp.take_dones()
            n_kyoku = gp.take_grp().take_feature().shape[0]
            t = tj.build(at_kyoku, np.zeros(n_kyoku), 'K')
            # the loader's own per-kyoku done flags agree with the K horizon
            assert t['done'].tolist() == list(loader_dones)
