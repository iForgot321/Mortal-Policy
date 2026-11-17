"""Shared fixtures: a small 1v3 arena run with behaviour sidecars (D46)."""
import os
import sys
from glob import glob

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'mortal'))

_MORTAL_CFG_AT_START = os.environ.get('MORTAL_CFG')


def pytest_collection_finish(session):
    """No test module may set MORTAL_CFG while it is imported. Collection imports every module
    before any fixture runs, so a module-level default (one that names experiments/configs/, i.e.
    workstation paths) becomes every later test's config. Here those paths exist and the suite
    passes; on a rented box it fails setup_box.sh (2026-11-17: tests/test_validate.py, two eval
    boxes). Set it inside a fixture instead, as tests/test_teacher.py does."""
    now = os.environ.get('MORTAL_CFG')
    if now != _MORTAL_CFG_AT_START:
        raise pytest.UsageError(f'MORTAL_CFG was set to {now!r} while collecting the tests: set it '
                                'inside a fixture (workdir/box/configs/ first), not at import')


def make_engine(brain, policy, explore_rate, name):
    import torch
    from engine import MortalEngine
    return MortalEngine(
        brain, policy, is_oracle=False, version=4, device=torch.device('cpu'), enable_amp=False,
        enable_rule_based_agari_guard=True, explore_rate=explore_rate, name=name,
    )


@pytest.fixture(scope='session')
def arena_run(tmp_path_factory):
    """Two duplicate sets with a small random network: a trainee that samples
    (ε = 1) against the same network playing greedily. Returns (root, logs,
    brain, policy); sidecars are in root / 'bp'."""
    import torch
    from libriichi.arena import OneVsThree
    from model import Brain, CategoricalPolicy
    torch.manual_seed(0)
    brain = Brain(version=4, conv_channels=32, num_blocks=1, Norm='GN').eval()
    policy = CategoricalPolicy().eval()
    root = tmp_path_factory.mktemp('behaviour')
    env = OneVsThree(disable_progress_bar=True, log_dir=str(root / 'logs'),
                     behaviour_dir=str(root / 'bp'))
    env.py_vs_py(
        challenger=make_engine(brain, policy, 1.0, 'trainee'),
        champion=make_engine(brain, policy, 0.0, 'champion'),
        seed_start=(0, 0x5eed),
        seed_count=2,
    )
    logs = sorted(glob(str(root / 'logs' / '*.json.gz')))
    assert len(logs) == 8
    return root, logs, brain, policy
