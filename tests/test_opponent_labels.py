"""GameplayLoader(opponent_labels=True): the SL teacher's tenpai and waits targets (04 §5,
docs/design/teacher-trainer-draft.md). They must equal the oracle encoder's rows for the same
opponents, follow the m↔p augmentation, and leave the observation untouched."""
import glob
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MJ = os.path.expanduser('~/mj-data')
FILES = sorted(glob.glob(os.path.join(MJ, 'mjai', '2019', '03', '*.json.gz')))[:4]
pytestmark = pytest.mark.skipif(len(FILES) < 4, reason='needs converted data in ~/mj-data')
ROWS_PER_OPPONENT = 17  # v2-v5 oracle rows: tehai 4, akas 3, shanten one-hot 7, scalar 1, waits 1, furiten 1


@pytest.fixture(scope='module')
def loader():
    sys.path.insert(0, os.path.join(ROOT, 'mortal'))
    from libriichi.dataset import GameplayLoader
    return GameplayLoader


def games(GameplayLoader, **kw):
    return [g for f in FILES for g in GameplayLoader(version=5, **kw).load_gz_log_files([f])[0]]


def test_labels_equal_the_oracle_rows(loader):
    n = 0
    for g in games(loader, oracle=True, opponent_labels=True):
        labels, inv = g.take_opponent_labels(), g.take_invisible_obs()
        assert len(labels) == len(inv) == len(g.take_actions())
        for lab, o in zip(labels, inv):
            assert lab.shape == (3, 35) and lab.dtype == np.bool_
            for j in range(3):
                b = j * ROWS_PER_OPPONENT
                assert lab[j, 0] == bool(o[b + 7].all())  # shanten 0 (tenpai) one-hot row
                assert (lab[j, 1:] == (o[b + 15] > 0.5)).all()  # the waits row
                assert lab[j, 0] or not lab[j, 1:].any()  # waits only when tenpai
            n += 1
    assert n > 1000


def test_labels_follow_the_augmentation(loader):
    plain = games(loader, oracle=False, opponent_labels=True)
    aug = games(loader, oracle=False, opponent_labels=True, augmented=True)
    swap = np.r_[9:18, 0:9, 18:34]  # manzu <-> pinzu (Tile::augment), honours and souzu fixed
    assert len(plain) == len(aug)
    for p, a in zip(plain, aug):
        lp, la = p.take_opponent_labels(), a.take_opponent_labels()
        assert len(lp) == len(la)
        for x, y in zip(lp, la):
            assert (x[:, 0] == y[:, 0]).all()
            assert (x[:, 1:][:, swap] == y[:, 1:]).all()


def test_observations_unchanged(loader):
    off = games(loader, oracle=False)
    on = games(loader, oracle=False, opponent_labels=True)
    for a, b in zip(off, on):
        for x, y in zip(a.take_obs(), b.take_obs()):
            assert np.array_equal(x, y)
        assert a.take_actions() == b.take_actions()
        assert not a.take_opponent_labels()
