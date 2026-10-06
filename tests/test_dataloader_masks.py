"""Q6/Q8 masks in FileDatasetsIter drop exactly the masked decisions (04a §4, §6.4)."""
import glob
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MJ = os.path.expanduser('~/mj-data')
FILES = sorted(glob.glob(os.path.join(MJ, 'mjai', '2009', '03', '*.json.gz')))[:3]
pytestmark = pytest.mark.skipif(len(FILES) < 3, reason='needs converted data in ~/mj-data')


@pytest.fixture(scope='module')
def loader_mod():
    os.environ.setdefault('MORTAL_CFG', os.path.join(ROOT, 'experiments', 'configs', 'pilot-qs-a.toml'))
    sys.path.insert(0, os.path.join(ROOT, 'mortal'))
    import dataloader
    from libriichi.dataset import GameplayLoader
    return dataloader, GameplayLoader


def entries(dataloader, masks):
    ds = dataloader.FileDatasetsIter(version=4, file_list=list(FILES), pts=[90, 50, -10, -130],
                                     file_batch_size=len(FILES), player_names=[], masks=masks,
                                     compute_advantage=False, obs_fp16=True)
    from libriichi.dataset import GameplayLoader
    ds.loader = GameplayLoader(version=4, oracle=False, player_names=[], excludes=None, augmented=False)
    ds.buffer = []
    ds.populate_buffer(list(FILES))
    return ds.buffer


def test_masks_drop_exactly(loader_mod):
    dataloader, GameplayLoader = loader_mod
    light = GameplayLoader(version=4, oracle=False, featurize=False).load_gz_log_files(FILES)
    lid0 = os.path.basename(FILES[0]).split('.')[0]
    lid1 = os.path.basename(FILES[1]).split('.')[0]
    per_seat0 = {g.take_player_id(): g.take_at_kyoku() for g in light[0]}
    per_seat1 = {g.take_player_id(): len(g.take_actions()) for g in light[1]}
    total = sum(len(v) for v in per_seat0.values()) + sum(per_seat1.values()) + \
        sum(len(g.take_actions()) for g in light[2])

    base = entries(dataloader, {})
    assert len(base) == total
    assert base[0][0].dtype.name == 'float16'

    # game 0: mask kyoku 1 for all seats; game 1: mask seat 2
    masks = {lid0: (frozenset({1}), frozenset()), lid1: (frozenset(), frozenset({2}))}
    in_k1 = sum(sum(1 for k in ks if k == 1) for ks in per_seat0.values())
    assert in_k1 > 0
    masked = entries(dataloader, masks)
    assert len(masked) == total - in_k1 - per_seat1[2]


def test_load_masks_roundtrip(loader_mod, tmp_path):
    dataloader, _ = loader_mod
    p = tmp_path / 'masks.jsonl'
    p.write_text('{"log_id": "x", "kyoku": [0, 3], "seats": []}\n{"log_id": "y", "kyoku": [], "seats": [1]}\n')
    m = dataloader.load_masks(str(p))
    assert m == {'x': (frozenset({0, 3}), frozenset()), 'y': (frozenset(), frozenset({1}))}
