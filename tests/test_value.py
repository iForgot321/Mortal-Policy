"""Arm V's offline data path and checkpoint loading (experiments/v.md)."""
import glob
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MJ = os.path.expanduser('~/mj-data')
FILES = sorted(glob.glob(os.path.join(MJ, 'mjai', '2009', '03', '*.json.gz')))[:3]
U = np.array([0.9, 0.5, -0.1, -1.3])

torch = pytest.importorskip('torch')


@pytest.fixture(scope='module')
def dataloader():
    os.environ.setdefault('MORTAL_CFG', os.path.join(ROOT, 'experiments', 'configs', 'cql-qs-a.toml'))
    sys.path.insert(0, os.path.join(ROOT, 'mortal'))
    import dataloader
    return dataloader


def test_next_ranks_stable_ties(dataloader):
    # two kyoku starts and the final scores; seats 1 and 2 tie after kyoku 0
    feature = np.zeros((2, 7))
    feature[0, 3:] = [2.5, 2.5, 2.5, 2.5]
    feature[1, 3:] = [3.3, 2.1, 2.1, 2.5]
    final = [20000, 40000, 10000, 30000]
    ranks = [dataloader.next_ranks(feature, final, p) for p in range(4)]
    assert [r[0] for r in ranks] == [0, 1, 2, 3]  # all tied: seat order
    assert [r[1] for r in ranks] == [0, 2, 3, 1]  # the tie between seats 1 and 2 goes to seat 1
    assert [r[2] for r in ranks] == [2, 0, 3, 1]


@pytest.mark.skipif(len(FILES) < 3, reason='needs converted data in ~/mj-data')
def test_value_entries(dataloader):
    from libriichi.dataset import GameplayLoader
    ds = dataloader.FileDatasetsIter(version=4, file_list=list(FILES), pts=[90, 50, -10, -130],
                                     file_batch_size=len(FILES), player_names=[],
                                     compute_advantage=False, obs_fp16=True, value_targets=True)
    it = ds.build_iter()  # loads the GRP and the fixed-unit reward calculator
    entries = list(it)
    light = GameplayLoader(version=4, oracle=False, featurize=False).load_gz_log_files(FILES)
    assert len(entries) == sum(len(g.take_actions()) for f in light for g in f)
    assert all(len(e) == 5 for e in entries)
    q = np.array([e[3] for e in entries])
    rank = np.array([e[4] for e in entries])
    # u units: a kyoku's GRP-Δu can't exceed the utility's range
    assert np.abs(q).max() <= U.max() - U.min()
    assert np.abs(q).max() > 0.01  # not standardized away, not all zero
    assert set(rank) <= {0, 1, 2, 3}
    assert ds.reward_calc.shared_stats['count'].value == 0


@pytest.mark.skipif(len(FILES) < 3, reason='needs converted data in ~/mj-data')
def test_kyoku_targets_telescope(dataloader):
    """Per seat, the kyoku targets sum to u(final rank) − E[u] at the start (the GRP
    shaping telescopes), and the rank after the last kyoku is the final rank."""
    from libriichi.dataset import GameplayLoader
    ds = dataloader.FileDatasetsIter(version=4, file_list=[], pts=[90, 50, -10, -130],
                                     compute_advantage=False, value_targets=True)
    next(ds.build_iter(), None)
    for game in GameplayLoader(version=4, oracle=False, featurize=False).load_gz_log_files(FILES[:1])[0]:
        pid = game.take_player_id()
        grp = game.take_grp()
        feature, rbp = grp.take_feature(), grp.take_rank_by_player()
        r = ds.reward_calc.calc_delta_pt(pid, feature, rbp) / dataloader.PTS_PER_U
        p0 = ds.reward_calc.calc_rank_prob(pid, feature, rbp)[0].numpy()
        assert r.sum() == pytest.approx(U[rbp[pid]] - p0 @ U, abs=1e-9)
        assert dataloader.next_ranks(feature, grp.take_final_scores(), pid)[-1] == rbp[pid]


def test_load_engine_value_checkpoint(tmp_path):
    sys.path.insert(0, ROOT)
    from lab.arena import load_engine
    from model import Brain, DQN
    torch.manual_seed(0)
    brain = Brain(version=4, conv_channels=32, num_blocks=1, Norm='GN').eval()
    dqn = DQN(version=4).eval()
    ckpt = tmp_path / 'v.pth'
    torch.save({'mortal': brain.state_dict(), 'current_dqn': dqn.state_dict(),
                'config': {'control': {'version': 4}, 'resnet': {'conv_channels': 32, 'num_blocks': 1}}}, ckpt)
    engine = load_engine(str(ckpt), device='cpu', enable_amp=False)
    obs = np.random.default_rng(0).random((3, 1012, 34), dtype=np.float32)
    masks = np.zeros((3, 46), dtype=bool)
    masks[:, [0, 5, 37]] = True
    actions, q_out, *_ = engine.react_batch(list(obs), list(masks), None)
    with torch.inference_mode():
        q = dqn(brain(torch.as_tensor(obs)), torch.as_tensor(masks))
    assert actions == q.argmax(-1).tolist()
    assert all(a in (0, 5, 37) for a in actions)
