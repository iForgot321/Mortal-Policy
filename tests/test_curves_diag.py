"""Option B's diagnostics on curves batch 1 (`experiments/curves_diag.py`): the triage's rule, the
pairing's guards and the descriptions' prefix."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'experiments'))

import curves_diag as cd  # noqa: E402


def run(fraction, lower):
    return {'entropy_fraction': fraction, 'since_p50': {'ci': [lower, lower + 0.01]}}


def test_the_rule():
    assert cd.rule({'a': run(0.75, 0.001), 'b': run(0.70, 0.0001)})['option'] == 'A'  # 70% itself passes
    assert cd.rule({'a': run(0.673, 0.001), 'b': run(0.672, 0.001)})['option'] == 'C'  # entropy alone
    assert cd.rule({'a': run(0.75, 0.0), 'b': run(0.75, -0.002)})['option'] == 'C'  # a bound at 0 isn't above it
    r = cd.rule({'a': run(0.75, 0.001), 'b': run(0.69, 0.001)})
    assert r['option'] == 'disagree' and r['meets_both_conditions'] == {'a': True, 'b': False}


def row(run_id, champion, nonce_start=134200):
    return {'run_id': run_id, 'champion': {'sha256': champion}, 'nonce_start': nonce_start}


def test_paired_differences_need_the_same_deals_and_opponents(monkeypatch):
    seeds = np.arange(200)
    rng = np.random.default_rng(0)
    base = rng.choice([-1.3, -0.1, 0.5, 0.9], size=200)
    data = {'new': (seeds, base + 0.05), 'base': (seeds, base), 'shuffled': (seeds[::-1], base)}
    monkeypatch.setattr(cd, 'scores', lambda r: data[r['run_id']])
    d = cd.paired(row('new', 'x'), row('base', 'x'))
    assert d['du'] == pytest.approx(0.05) and d['ci'][0] == pytest.approx(0.05) and d['sets'] == 200
    with pytest.raises(ValueError, match='different opponents'):
        cd.paired(row('new', 'x'), row('base', 'y'))
    with pytest.raises(ValueError, match='same deals'):
        cd.paired(row('new', 'x'), row('shuffled', 'x'))
    with pytest.raises(ValueError, match='same deals'):
        cd.paired(row('new', 'x'), row('base', 'x', nonce_start=0))


def test_the_descriptions_are_never_the_readings_or_the_stage_checks():
    descs = [d for d, _, _ in cd.measures()]
    assert len(descs) == len(set(descs)) == 5
    assert all(d.startswith('curves diag: ') for d in descs)
    assert not any(d.startswith(('curves: ', 'curves boundary: ')) for d in descs)
    assert descs[:2] == ['curves diag: curves-b12-s1 full at 3x p50', 'curves diag: curves-b16-s1 full at 3x p50']
    argv = cd.lab_argv('c.pth', 'a.pth', descs[0])
    assert argv[argv.index('--block') + 1] == 'curves-boundary' and '--no-logs' in argv and '--no-amp' not in argv


def test_the_offtable_bands_and_names():
    ref = 0.0294  # half: 0.0147
    assert cd.band([0.016, 0.030], ref) == 'carries'
    assert cd.band([-0.002, 0.014], ref) == 'specific'
    assert cd.band([0.010, 0.024], ref) == 'ambiguous'
    t = cd.offtable()
    assert set(t) == {f'{m}@{x}' for m in ('full', 'student', 'p50') for x in ('nosp', 'bc')}
    assert all(d.startswith('curves diag: ') for d, _, _ in t.values())
    assert len({d for d, _, _ in t.values()}) == 6
    assert t['full@nosp'][2] == t['student@nosp'][2] == t['p50@nosp'][2]  # one table, the same opponents
