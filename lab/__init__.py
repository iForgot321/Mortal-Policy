"""Evaluation lab (docs/reference/02_evaluation_lab.md).

Run from the repo root as `python -m lab ...` (see lab/__main__.py).
"""
import os

import numpy as np
import toml

LAB_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(LAB_DIR)
MORTAL_DIR = os.path.join(ROOT, 'mortal')
REGISTRY_DIR = os.path.join(LAB_DIR, 'registry')
# Game logs and per-set scores are bulky and stay out of git.
RUNS_DIR = os.path.join(ROOT, 'workdir', 'lab')


def load_config(path=os.path.join(LAB_DIR, 'lab.toml')):
    with open(path, encoding='utf-8') as f:
        return toml.load(f)


def load_seeds(path=os.path.join(LAB_DIR, 'seeds.toml')):
    with open(path, encoding='utf-8') as f:
        return toml.load(f)['families']


def utility(cfg=None):
    cfg = cfg or load_config()
    u = np.asarray(cfg['utility'], dtype=np.float64)
    assert u.shape == (4,) and abs(u.sum()) < 1e-12, 'utility must have 4 entries summing to 0'
    return u
