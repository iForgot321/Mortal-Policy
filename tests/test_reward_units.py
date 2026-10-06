"""Fixed reward units (05 §3.2, D37) and GRP shaping telescoping (01 §1)."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'mortal'))

torch = pytest.importorskip('torch')
PTS = [90.0, 50.0, -10.0, -130.0]


@pytest.fixture(scope='module')
def calc():
    from model import GRP
    from reward_calculator import RewardCalculator
    torch.manual_seed(0)
    grp = GRP(hidden_size=64, num_layers=2).double()  # untrained: telescoping holds for any potential
    shared = {'count': type('V', (), {'value': 0})(), 'mean': type('V', (), {'value': 0.0})(),
              'M2': type('V', (), {'value': 0.0})(), 'lock': __import__('threading').Lock()}
    return RewardCalculator(grp, PTS, shared_stats=shared, standardize=False)


def feature(n_kyoku, seed):
    rng = np.random.default_rng(seed)
    f = np.zeros((n_kyoku, 7))
    f[:, 0] = np.arange(n_kyoku)
    f[:, 3:] = rng.dirichlet(np.ones(4), n_kyoku) * 10  # scores / 10000, summing to 100k
    return f


def test_raw_units_are_deterministic_and_unstandardized(calc):
    f, ranks = feature(9, 1), [2, 0, 3, 1]
    a = calc.calc_delta_pt(0, f, ranks)
    b = calc.calc_delta_pt(0, f, ranks)
    assert np.array_equal(a, b)
    assert calc.shared_stats['count'].value == 0  # no running statistics touched


def test_rewards_telescope_to_final_utility(calc):
    f, ranks = feature(11, 2), [1, 3, 0, 2]
    for player in range(4):
        r = calc.calc_delta_pt(player, f, ranks)
        prob0 = calc.calc_rank_prob(player, f, ranks)[0].numpy()
        assert r.sum() == pytest.approx(PTS[ranks[player]] - prob0 @ np.array(PTS), abs=1e-9)
