"""tools/fleet/stage_init.py: a run's init for a new stage, made from another run's snapshot
(`experiments/anchor-release.md`)."""
import importlib.util
import os
import stat

import pytest
import torch

import schedules
from checkpoint import load_checkpoint
from kl_anchor import KLAnchor
from model import Brain, CategoricalPolicy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location('stage_init', os.path.join(ROOT, 'tools', 'fleet', 'stage_init.py'))
stage_init = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage_init)

CFG = {'control': {'version': 4}, 'resnet': {'conv_channels': 32, 'num_blocks': 1}}


def policy_file(path, seed):
    torch.manual_seed(seed)
    brain = Brain(version=4, conv_channels=32, num_blocks=1, Norm='GN')
    torch.save({'config': CFG, 'mortal': brain.state_dict(), 'policy_net': CategoricalPolicy().state_dict()}, path)
    return str(path)


@pytest.fixture
def snapshot(tmp_path):
    """A trainer snapshot as train_online.py saves it, 3,000 steps into a run that began at step
    1,000 and anchors to `old.pth`."""
    old = policy_file(tmp_path / 'old.pth', 0)
    state = load_checkpoint(old)
    state.update({'steps': 4000, 'schedule_origin': 1000, 'optimizer': {'lr': 1e-5},
                  'kl_anchor': {'mode': 'adaptive', 'state_file': old, 'adaptive_coef': 0.0759375,
                                'kl_sum': 0.25, 'kl_n': 13, 'reached': True, 'windows': 9}})
    path = tmp_path / 'snap.pth'
    torch.save(state, path)
    return str(path), old


def anchor(state_file, default):
    return KLAnchor({'coef': 0.1, 'state_file': state_file}, default_state_file=default,
                    device=torch.device('cpu'), version=4)


def test_the_clock_restarts_and_the_anchor_keeps_its_coefficient(snapshot, tmp_path):
    src, old = snapshot
    out = str(tmp_path / 'init-a.pth')
    _, dropped = stage_init.write(src, out)
    assert dropped == ['schedule_origin']
    before, after = load_checkpoint(src), load_checkpoint(out)
    # the run clock starts where the new run does
    assert schedules.origin(before, before['steps']) == 1000
    assert schedules.origin(after, after['steps']) == after['steps'] == 4000
    # the same anchor, at the coefficient the old run left it
    a = anchor(None, old)
    assert a.coef(0) == 0.1
    a.load_state_dict(after['kl_anchor'])
    assert a.coef(0) == 0.0759375 and a.reached and a.windows == 9
    # nothing else changed
    assert set(after) == set(before) - {'schedule_origin'}
    assert all(torch.equal(v, after['mortal'][k]) for k, v in before['mortal'].items())
    assert after['optimizer'] == before['optimizer']


def test_a_new_anchor_starts_from_the_config(snapshot, tmp_path):
    src, old = snapshot
    new = policy_file(tmp_path / 'new.pth', 1)
    # as it is, the snapshot refuses another anchor
    with pytest.raises(ValueError, match='the checkpoint anchors to'):
        anchor(new, old).load_state_dict(load_checkpoint(src)['kl_anchor'])
    out = str(tmp_path / 'init-b.pth')
    _, dropped = stage_init.write(src, out, new_anchor=True)
    assert dropped == ['schedule_origin', 'kl_anchor']
    after = load_checkpoint(out)
    # train_online.py loads the anchor's state only if the checkpoint has one
    assert 'kl_anchor' not in after and 'schedule_origin' not in after
    assert after['steps'] == 4000


def test_the_init_is_read_only_and_never_overwritten(snapshot, tmp_path):
    src, _ = snapshot
    out = str(tmp_path / 'sub' / 'init.pth')
    stage_init.write(src, out)
    assert not os.stat(out).st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    with pytest.raises(FileExistsError):
        stage_init.write(src, out)
    assert stage_init.main([src, str(tmp_path / 'cli.pth'), '--new-anchor']) == 0
    assert 'kl_anchor' not in load_checkpoint(str(tmp_path / 'cli.pth'))
