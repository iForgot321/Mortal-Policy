"""K1 critic (05 §4): privileged inputs are the opponents' hands only, oracle
inputs never reach the policy, GAE runs per trajectory before shuffling, and
the diagnostics behave at their limits."""
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

import critic as cr
import rl_dataset as rd
import trajectory as tj
from libriichi.consts import obs_shape, oracle_obs_shape
from libriichi.dataset import GameplayLoader
from model import GRP, Brain

PTS = [90, 50, -10, -130]
GRP_NETWORK = {'hidden_size': 64, 'num_layers': 2}
CPU = torch.device('cpu')


@pytest.fixture(scope='module')
def chunk(arena_run, tmp_path_factory):
    """All 8 trainee trajectories in one chunk, as the trainer's DataLoader
    (batch_size=None) delivers them."""
    root, logs, _, _ = arena_run
    torch.manual_seed(1)
    grp_file = tmp_path_factory.mktemp('grp') / 'grp.pth'
    torch.save({'model': GRP(**GRP_NETWORK).state_dict()}, grp_file)
    ds = rd.OrderedDatasetsIter(version=4, file_list=list(logs), behaviour_dir=str(root / 'bp'), pts=PTS,
                                grp_state_file=str(grp_file), grp_network=GRP_NETWORK, targets='gae',
                                horizon='H', file_batch_size=len(logs))
    (chunk,) = list(DataLoader(ds, batch_size=None))
    return chunk


def small_critic(seed=0):
    torch.manual_seed(seed)
    return cr.Critic(conv_channels=32, num_blocks=1).eval()


def test_policy_input_has_no_oracle_channels():
    brain = Brain(version=4, conv_channels=32, num_blocks=1, Norm='GN')
    assert brain.encoder.net[0].in_channels == obs_shape(4)[0] == 1012
    critic = cr.Critic(conv_channels=32, num_blocks=1)
    assert critic.encoder.net[0].in_channels == 1012 + 51
    assert oracle_obs_shape(4)[0] == 217  # rows 51+ (the wall order) are left out


def test_privileged_rows_are_opponents_hands(arena_run):
    _, logs, _, _ = arena_run
    plain = GameplayLoader(version=4, oracle=False, player_names=['trainee']).load_gz_log_files(logs[:2])
    oracle = GameplayLoader(version=4, oracle=True, trust_seed=True, player_names=['trainee']).load_gz_log_files(logs[:2])
    for (p,), (o,) in zip(plain, oracle):
        # the oracle loader featurizes the policy's observation identically
        assert all(np.array_equal(a, b) for a, b in zip(p.take_obs(), o.take_obs()))
        for inv in o.take_invisible_obs():
            priv = cr.privileged(inv)
            assert priv.shape == (51, 34)
            for opp in range(3):
                # tehai rows: one row per copy of a tile, so the row sum counts the
                # concealed tiles: 13 or 14 less 3 per meld
                tiles = priv[17 * opp:17 * opp + 4].sum()
                assert tiles % 3 in (1, 2) and 1 <= tiles <= 14
                shanten = priv[17 * opp + 7:17 * opp + 14].sum(1)
                assert (shanten == 34).sum() == 1  # exactly one shanten row is set


def test_chunk_carries_privileged_channels(chunk):
    assert len(chunk) == 8
    for t in chunk:
        n = len(t['action'])
        assert t['obs'].shape == (n, 1012, 34) and t['priv'].shape == (n, 51, 34)
        assert t['done'][-1] and not t['done'][:-1].any()  # horizon H: the game end


def test_gae_targets_are_deterministic_and_per_trajectory(chunk):
    critic = small_critic()
    a = cr.gae_targets(chunk, critic, device=CPU, lam=0.95)
    b = cr.gae_targets(chunk, critic, device=CPU, lam=0.95)
    for k in ('value', 'advantage', 'returns', 'mc_return'):
        assert torch.equal(a[k], b[k]), k
    # a trajectory's targets don't depend on its neighbours' rewards (same chunk,
    # so the critic sees the same batches and gives bit-identical values)
    shifted = [dict(t) for t in chunk]
    shifted[0]['reward'] = shifted[0]['reward'] + 5.0
    c = cr.gae_targets(shifted, critic, device=CPU, lam=0.95)
    n0 = len(chunk[0]['action'])
    assert torch.equal(c['value'], a['value'])
    assert not torch.equal(c['advantage'][:n0], a['advantage'][:n0])
    for k in ('advantage', 'returns', 'mc_return'):
        assert torch.equal(c[k][n0:], a[k][n0:]), k
    assert (a['group'][:n0, 0] == 0).all() and (a['group'][n0:, 0] >= 1).all()
    # λ = 1: the regression target is the observed return
    one = cr.gae_targets(chunk, critic, device=CPU, lam=1.0)
    assert torch.allclose(one['returns'], one['mc_return'], atol=1e-5)
    assert torch.allclose(one['advantage'], one['mc_return'] - one['value'], atol=1e-5)


def test_zero_critic_gives_monte_carlo_advantages(chunk):
    critic = small_critic()
    torch.nn.init.zeros_(critic.head.weight)
    torch.nn.init.zeros_(critic.head.bias)
    flat = cr.gae_targets(chunk, critic, device=CPU, lam=1.0)
    assert (flat['value'] == 0).all()
    # V = 0, γ = λ = 1: GAE is the Monte-Carlo return
    assert torch.allclose(flat['advantage'], flat['mc_return'], atol=1e-6)
    # V = 0, λ < 1: the λ-discounted return
    flat = cr.gae_targets(chunk, critic, device=CPU, lam=0.9)
    start = 0
    for t in chunk:
        n = len(t['action'])
        want = tj.monte_carlo_returns(np.asarray(t['reward']), np.asarray(t['done']), gamma=0.9)
        assert np.allclose(flat['advantage'][start:start + n].numpy(), want, atol=1e-6)
        start += n


def test_shuffled_minibatches_use_each_decision_once():
    sizes = [700, 300, 600]
    offsets = np.cumsum([0] + sizes)
    flats = [{'id': torch.arange(o, o + n), 'x': torch.arange(o, o + n) * 2} for o, n in zip(offsets, sizes)]
    batches = list(rd.shuffled_minibatches(iter(flats), 256, ('id', 'x')))
    assert len(batches) == sum(sizes) // 256
    ids = torch.cat([b['id'] for b in batches])
    assert len(ids.unique()) == len(ids) == 6 * 256
    assert all(torch.equal(b['x'], b['id'] * 2) for b in batches)  # columns stay aligned


def test_diagnostics_limits():
    rng = np.random.default_rng(0)
    ret = rng.normal(size=1000)
    assert cr.explained_variance(ret, ret) == pytest.approx(1)
    assert cr.explained_variance(np.full(1000, 0.3), ret) == pytest.approx(0)
    assert cr.calibration(ret + rng.normal(scale=0.1, size=1000), ret)[2]
    assert not cr.calibration(-ret, ret)[2]
    group = np.stack((np.zeros(1000), np.repeat(np.arange(10), 100)), axis=1)
    per_kyoku = np.repeat(rng.normal(size=10), 100)
    assert cr.advantage_variance_ratio(per_kyoku, group) == pytest.approx(0, abs=1e-12)  # P0: one advantage per kyoku
    assert cr.advantage_variance_ratio(per_kyoku + ret, group) > 0.1


def test_last_kyoku_marks_each_trajectorys_final_kyoku(chunk):
    critic = cr.Critic(conv_channels=32, num_blocks=1).eval()
    flat = cr.gae_targets(chunk, critic, device=CPU, lam=0.95)
    want = np.concatenate([np.asarray(t['at_kyoku']) == np.asarray(t['at_kyoku'])[-1] for t in chunk])
    assert flat['last_kyoku'].dtype == torch.bool
    assert np.array_equal(flat['last_kyoku'].numpy(), want)
    assert 0 < want.sum() < len(want)
