"""`experiments/curves_flips.py`'s analysis on hand-made probabilities."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'experiments'))

import curves_flips as cf  # noqa: E402


def test_stage_changes_back_and_on_and_the_student():
    # four decisions over three moves
    student = np.array([[.6, .3, .1], [.6, .3, .1], [.6, .3, .1], [.6, .3, .1]])
    p50 = np.array([[.52, .48, 0], [.9, .1, 0], [.5, .4, .1], [.7, .2, .1]])
    p75 = np.array([[.40, .60, 0], [.2, .8, 0], [.3, .6, .1], [.7, .2, .1]])
    full = np.array([[.70, .30, 0], [.1, .2, .7], [.2, .7, .1], [.8, .1, .1]])
    r = cf.analyse({'student': student, 'p50': p50, 'p75': p75, 'full': full}, np.array(['discard'] * 4))
    assert r['stage3']['n'] == 3 and r['stage3']['changed'] == pytest.approx(0.75)
    # stage 3's gaps in p50 between the old and the new move: .04, .8, .1
    assert r['stage3']['old_gap_bins'] == pytest.approx({'0-0.05': 1 / 3, '0.05-0.2': 1 / 3, '0.2-0.5': 0, '>=0.5': 1 / 3})
    assert r['stage4']['n'] == 2  # decision 0 (1 -> 0) and decision 1 (1 -> 2)
    assert r['both_stages'] == {'n': 2, 'share_of_stage4': 1.0, 'back_to_p50': 1, 'to_a_third_move': 1}
    assert r['stage4_vs_student'] == {'toward': 1, 'away': 0, 'neither': 1}
    assert r['p50_to_full_changed'] == pytest.approx(0.5)


def test_bins_cover_every_gap_once():
    b = cf.bins(np.array([0.0, 0.049, 0.05, 0.3, 0.5, 1.0]))
    assert sum(b.values()) == pytest.approx(1.0) and b['>=0.5'] == pytest.approx(2 / 6)
