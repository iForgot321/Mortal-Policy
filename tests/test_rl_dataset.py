"""Ordered RL dataset (05 §3.3, §8): deterministic targets, `done` alignment,
no cross-seat targets, fixed reward units, and a behaviour probability (or a
flag) for every entry."""
import os
import shutil

import numpy as np
import pytest
import torch

import rl_dataset as rd
import trajectory as tj
from libriichi.dataset import GameplayLoader
from model import GRP

PTS = [90, 50, -10, -130]
GRP_NETWORK = {'hidden_size': 64, 'num_layers': 2}


@pytest.fixture(scope='module')
def grp_file(tmp_path_factory):
    torch.manual_seed(1)  # untrained: these are plumbing tests
    f = tmp_path_factory.mktemp('grp') / 'grp.pth'
    torch.save({'model': GRP(**GRP_NETWORK).state_dict()}, f)
    return str(f)


def dataset(arena_run, grp_file, **kw):
    root, logs, _, _ = arena_run
    kw.setdefault('file_list', list(logs))
    ds = rd.OrderedDatasetsIter(version=4, behaviour_dir=str(root / 'bp'), pts=PTS,
                                grp_state_file=grp_file, grp_network=GRP_NETWORK, **kw)
    ds.prepare()
    return ds


def trajectories(arena_run, grp_file, **kw):
    ds = dataset(arena_run, grp_file, **kw)
    return list(ds.trajectories(ds.file_list)), ds


def test_targets_are_bit_identical(arena_run, grp_file):
    for horizon in tj.HORIZONS:
        a, _ = trajectories(arena_run, grp_file, horizon=horizon)
        b, _ = trajectories(arena_run, grp_file, horizon=horizon)
        assert len(a) == len(b) == 8
        for x, y in zip(a, b):
            for k in ('reward', 'done', 'next_idx', 'behaviour_prob', 'action', 'kyoku_rewards'):
                assert np.array_equal(x[k], y[k], equal_nan=k == 'behaviour_prob'), k
            assert all(np.array_equal(o, p) for o, p in zip(x['obs'], y['obs']))


def test_done_aligns_with_kyoku_and_game_ends(arena_run, grp_file):
    _, logs, _, _ = arena_run
    light = GameplayLoader(version=4, oracle=False, featurize=False, player_names=['trainee'])
    loader_dones = [g.take_dones() for f in light.load_gz_log_files(logs) for g in f]
    k, _ = trajectories(arena_run, grp_file, horizon='K')
    h, _ = trajectories(arena_run, grp_file, horizon='H')
    for tk, th, dones in zip(k, h, loader_dones):
        assert tk['done'].tolist() == list(dones)       # K: every kyoku end
        assert th['done'].tolist() == [False] * (len(th['done']) - 1) + [True]  # H: the game end
        assert th['reward'].sum() == pytest.approx(th['kyoku_rewards'].sum())


def test_no_target_crosses_seats(arena_run, grp_file):
    """Every seat's trajectory is unchanged when the other seats' sidecar
    records are scrambled, and links only to its own decisions."""
    root, logs, _, _ = arena_run
    all_seats, ds = trajectories(arena_run, grp_file, player_names=(), file_list=logs[:2])
    assert len(all_seats) == 8
    for i, traj in enumerate(all_seats):
        log = logs[i // 4]
        sidecar = tj.load_behaviour(rd.sidecar_path(log, str(root / 'bp')))
        others = sidecar['actor'] != traj['player_id']
        sidecar['prob'][others] = np.random.default_rng(i).random(others.sum())
        sidecar['label'][others] = 0
        (game,) = [g for g in ds.loader.load_gz_log_files([log])[0] if g.take_player_id() == traj['player_id']]
        again = rd.seat_trajectory(game, sidecar, ds.reward_calc, 'K')
        assert np.array_equal(again['behaviour_prob'], traj['behaviour_prob'], equal_nan=True)
        assert np.array_equal(again['reward'], traj['reward'])
        n = len(traj['action'])
        nxt = traj['next_idx']
        assert ((nxt == -1) == traj['done']).all() and (nxt[~traj['done']] == np.arange(n)[~traj['done']] + 1).all()


def test_rewards_in_fixed_u_units(arena_run, grp_file):
    _, logs, _, _ = arena_run
    trajs, ds = trajectories(arena_run, grp_file, player_names=(), file_list=logs[:1])
    calc = ds.reward_calc
    assert calc.shared_stats['count'].value == 0   # no running statistics touched
    grp = GameplayLoader(version=4, oracle=False, featurize=False).load_gz_log_files(logs[:1])[0][0].take_grp()
    feature, ranks = grp.take_feature(), grp.take_rank_by_player()
    for traj in trajs:
        seat = traj['player_id']
        raw = calc.calc_delta_pt(seat, feature, ranks)
        assert np.array_equal(traj['kyoku_rewards'], raw / rd.PTS_PER_U)
        prior = calc.calc_rank_prob(seat, feature, ranks)[0].numpy() @ np.array(PTS, dtype=np.float64)
        # telescopes to u(final) − E_GRP[u] at the start, in u units
        assert traj['kyoku_rewards'].sum() == pytest.approx((PTS[ranks[seat]] - prior) / 100)
    assert calc.shared_stats['count'].value == 0


def test_every_entry_has_b_or_flag(arena_run, grp_file):
    trajs, _ = trajectories(arena_run, grp_file)
    b = np.concatenate([t['behaviour_prob'] for t in trajs])
    ok = ~np.isnan(b)
    assert ((b[ok] > 0) & (b[ok] <= 1)).all()
    assert ok.mean() > 0.99  # NaN only for the agari guard or voided reactions


def test_iteration_yields_shuffled_decisions(arena_run, grp_file):
    root, logs, _, _ = arena_run
    trajs, _ = trajectories(arena_run, grp_file)
    versions = {os.path.basename(f): 3 for f in logs}
    ds = rd.OrderedDatasetsIter(version=4, file_list=list(logs), behaviour_dir=str(root / 'bp'), pts=PTS,
                                grp_state_file=grp_file, grp_network=GRP_NETWORK, file_batch_size=4,
                                obs_fp16=True, versions=versions)
    entries = list(ds)
    assert len(entries) == sum(len(t['action']) for t in trajs)
    obs, action, mask, adv, b, version = entries[0]
    assert obs.dtype == np.float16 and obs.shape == (1012, 34)
    assert mask[action] and version == 3
    # P0 under K: each decision's advantage is its own kyoku's reward
    want = sorted(np.float32(t['kyoku_rewards'][t['at_kyoku'][i]]) for t in trajs for i in range(len(t['action'])))
    assert sorted(e[3] for e in entries) == want


def test_logs_without_sidecar_are_skipped(arena_run, grp_file, tmp_path):
    root, logs, _, _ = arena_run
    bp = tmp_path / 'bp'
    shutil.copytree(root / 'bp', bp)
    os.remove(rd.sidecar_path(logs[0], str(bp)))
    with_bp, without = rd.split_by_sidecar(logs, str(bp))
    assert without == [logs[0]] and with_bp == logs[1:]
    ds = rd.OrderedDatasetsIter(version=4, file_list=list(logs[:2]), behaviour_dir=str(bp), pts=PTS,
                                grp_state_file=grp_file, grp_network=GRP_NETWORK)
    ds.prepare()
    assert len(list(ds.trajectories(ds.file_list))) == 1
