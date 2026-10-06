"""Stored behaviour probabilities (05 §3.4, D46): the engine's sampling
distribution, and the arena sidecar joined to loader entries (05 §8: the
first-minibatch ratio is 1 against the stored behaviour probability)."""
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'mortal'))

import trajectory as tj  # noqa: E402
from engine import MortalEngine  # noqa: E402
from libriichi.dataset import GameplayLoader  # noqa: E402

CPU = torch.device('cpu')


@pytest.mark.parametrize('eps', [0.0, 0.3, 1.0])
def test_sampler_matches_behaviour_dist(eps):
    eng = MortalEngine(torch.nn.Identity(), torch.nn.Identity(), is_oracle=False, version=4,
                       device=CPU, explore_rate=eps)
    p = torch.zeros(46)
    p[[3, 17, 41, 45]] = torch.tensor([0.5, 0.3, 0.15, 0.05])
    # the second row is not normalized, as reduced-precision softmax output may not be
    q = torch.stack([p, p * 1.002])
    dist = eng.behaviour_dist(q)
    assert torch.allclose(dist.sum(-1), torch.ones(2), atol=1e-5)
    assert dist[0, 3] == pytest.approx(1 - eps + eps * 0.5)

    n = 100_000
    torch.manual_seed(0)
    actions, is_greedy, b = eng.sample(q[0].expand(n, -1))
    assert torch.equal(b, dist[0, actions])
    if eps == 0:
        assert is_greedy.all() and (actions == 3).all()
    freq = torch.bincount(actions, minlength=46).double() / n
    sd = (dist[0].double() * (1 - dist[0].double()) / n).sqrt()
    assert ((freq - dist[0].double()).abs() <= 5 * sd + 1e-12).all()


def sidecar_of(root, log):
    return tj.load_behaviour(root / 'bp' / os.path.basename(log))


def test_every_entry_joins(arena_run):
    root, logs, _, _ = arena_run
    loader = GameplayLoader(version=4, oracle=False, featurize=False)
    passes = unapplied = 0
    for log in logs:
        sidecar = sidecar_of(root, log)
        unapplied += (~sidecar['applied']).sum()
        for game in loader.load_gz_log_files([log])[0]:
            actions = np.asarray(game.take_actions())
            prob = tj.join_behaviour(game.take_event_idx(), actions, game.take_player_id(), sidecar)
            # NaN only for the agari guard or voided reactions, both rare
            assert np.isnan(prob).mean() < 0.01
            ok = ~np.isnan(prob)
            assert ((prob[ok] > 0) & (prob[ok] <= 1)).all()
            passes += (ok & (actions == 45)).sum()
    # passes and overruled claims are the cases the mjai log cannot carry
    assert passes > 0
    assert unapplied > 0


def test_first_minibatch_ratio_is_one(arena_run):
    """The same weights reproduce the stored probability of every sampled
    action from the loader's own features (ε = 1, so b = π)."""
    root, logs, brain, policy = arena_run
    loader = GameplayLoader(version=4, oracle=False, player_names=['trainee'])
    checked = 0
    for log in logs[:2]:
        sidecar = sidecar_of(root, log)
        (game,) = loader.load_gz_log_files([log])[0]
        obs = torch.as_tensor(np.stack(game.take_obs()))
        masks = torch.as_tensor(np.stack(game.take_masks()))
        actions = np.asarray(game.take_actions())
        prob = tj.join_behaviour(game.take_event_idx(), actions, game.take_player_id(), sidecar)
        with torch.inference_mode():
            pi = policy(brain(obs), masks)
        pi_a = pi[torch.arange(len(actions)), torch.as_tensor(actions)].numpy()
        ok = ~np.isnan(prob)
        np.testing.assert_allclose(pi_a[ok] / prob[ok], 1, rtol=1e-4)
        checked += ok.sum()
    assert checked > 100


def test_greedy_seats_store_one(arena_run):
    root, logs, _, _ = arena_run
    loader = GameplayLoader(version=4, oracle=False, featurize=False, player_names=['champion'])
    for log in logs:
        sidecar = sidecar_of(root, log)
        for game in loader.load_gz_log_files([log])[0]:
            prob = tj.join_behaviour(game.take_event_idx(), game.take_actions(), game.take_player_id(), sidecar)
            assert (prob[~np.isnan(prob)] == 1).all()


def test_join_rejects_disagreement():
    sidecar = {
        'event_idx': np.array([5, 5, 9, 12]),
        'actor': np.array([0, 1, 0, 0]),
        'kan_select': np.array([False, False, False, False]),
        'label': np.array([45, 38, 7, 41]),
        'prob': np.array([0.6, 0.4, np.nan, 0.2], dtype=np.float32),
        'applied': np.array([False, True, True, False]),
    }
    # seat 0: a pass the log can show, a guarded discard, and an unapplied
    # pon the log reads as a pass
    prob = tj.join_behaviour([5, 9, 12], [45, 7, 45], 0, sidecar)
    assert prob[0] == pytest.approx(0.6) and np.isnan(prob[1:]).all()
    with pytest.raises(ValueError, match='no behaviour record'):
        tj.join_behaviour([5, 9, 12, 13], [45, 7, 45, 3], 0, sidecar)
    with pytest.raises(ValueError, match='implies label'):
        tj.join_behaviour([5, 9, 12], [45, 8, 45], 0, sidecar)
    with pytest.raises(ValueError, match='without a loader entry'):
        tj.join_behaviour([5], [45], 0, sidecar)
