"""mortal/checkpoint.py: checkpoints load as data, never as code. Snapshots come back from rented
boxes and are loaded on the workstation (lab, anchor, baseline), so no code path may unpickle one
without the restriction."""
import os
import pickle
import re
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'mortal'))
from checkpoint import load_checkpoint  # noqa: E402

CODE_DIRS = ('mortal', 'lab', 'search', 'tools', 'experiments', 'deploy', 'scripts')


def test_a_trainer_checkpoint_round_trips(tmp_path):
    state = {
        'mortal': {'w': torch.arange(4.0)}, 'optimizer': {'state': {0: {'step': torch.tensor(3.0)}}, 'param_groups': [{'lr': 1}]},
        'steps': 3, 'timestamp': 1.5, 'cursor': {'0': 4}, 'file_order': 'shuffled',
        'config': {'control': {'version': 5}, 'env': {'pts': [90.0, 50.0, -10.0, -130.0]}},
        # v1-era trainers saved numpy scalars here
        'best_perf': {'avg_rank': np.float64(2.5), 'avg_pt': np.float32(1.5), 'games': np.int64(7), 'ok': np.bool_(True)},
    }
    torch.save(state, tmp_path / 'ckpt.pth')
    got = load_checkpoint(tmp_path / 'ckpt.pth')
    assert torch.equal(got['mortal']['w'], state['mortal']['w']) and got['config'] == state['config']
    assert got['best_perf'] == state['best_perf'] and got['cursor'] == {'0': 4}


class CallsAFunction:
    """What a hostile checkpoint does: name any importable function for the unpickler to call.
    os.getpid stands in for os.system."""
    def __reduce__(self):
        return (os.getpid, ())


def test_a_checkpoint_that_calls_a_function_is_refused(tmp_path):
    torch.save({'mortal': {'w': torch.zeros(1)}, 'config': CallsAFunction()}, tmp_path / 'bad.pth')
    with pytest.raises(pickle.UnpicklingError):
        load_checkpoint(tmp_path / 'bad.pth')


@pytest.mark.parametrize('rel', [
    'workdir/pilot/qs-a/pilot-qs-a-final.pth',       # train_offline_phase1.py
    'workdir/phase1/phase1-v0-final.pth',            # v1-era, numpy scalars in best_perf
    'workdir/fleet/runs/k2b/snapshots/full.pth',     # train_online.py with critic and anchor, written on a Vast box
    'workdir/teacher/heads-quick/final.pth',         # train_teacher.py
])
def test_existing_checkpoints_load(rel):
    path = os.path.join(ROOT, rel)
    if not os.path.exists(path):
        pytest.skip(f'no {rel}')
    state = load_checkpoint(path)
    assert 'mortal' in state and 'config' in state


def test_no_unrestricted_load_in_the_code():
    pattern = re.compile(r'weights_only\s*=\s*False|pickle\.loads?\(')
    hits = []
    for top in CODE_DIRS:
        for dirpath, _, names in os.walk(os.path.join(ROOT, top)):
            for name in names:
                if name.endswith(('.py', '.sh')):
                    path = os.path.join(dirpath, name)
                    with open(path, encoding='utf-8') as f:
                        hits += [f'{os.path.relpath(path, ROOT)}:{n}' for n, line in enumerate(f, 1) if pattern.search(line)]
    assert not hits, f'load checkpoints with checkpoint.load_checkpoint: {hits}'
