"""K2's KL anchor (05 §5.1, D59): the exact reverse KL over the legal mask, the frozen π_SL, and
the coefficient rules (`mortal/kl_anchor.py`)."""
import math
import os

import pytest
import torch

from kl_anchor import KLAnchor
from model import Brain, CategoricalPolicy
from ppo import reverse_kl

A = 46


def masked(logits, mask):
    return logits.masked_fill(~mask, -torch.inf)


@pytest.fixture
def batch():
    g = torch.Generator().manual_seed(0)
    mask = torch.rand(64, A, generator=g) < 0.3
    mask[:, 0] = True
    return torch.randn(64, A, generator=g), torch.randn(64, A, generator=g), mask


def test_reverse_kl_is_exact_over_the_mask(batch):
    x, y, mask = batch
    kl = reverse_kl(masked(x, mask), masked(y, mask), mask)
    for i in range(len(x)):
        p = torch.softmax(x[i][mask[i]], -1)
        q = torch.softmax(y[i][mask[i]], -1)
        assert kl[i] == pytest.approx(float((p * (p.log() - q.log())).sum()), abs=1e-5)
    assert (kl >= -1e-6).all()


def test_reverse_kl_is_zero_against_itself_and_ignores_illegal_actions(batch):
    x, y, mask = batch
    assert reverse_kl(masked(x, mask), masked(x, mask), mask).abs().max() < 1e-6
    # whatever an illegal action's raw logit was, it never enters
    kl = reverse_kl(masked(x, mask), masked(y, mask), mask)
    kl2 = reverse_kl(masked(x + 5 * (~mask), mask), masked(y - 3 * (~mask), mask), mask)
    assert torch.equal(kl, kl2)


def test_reverse_kl_gradient_is_finite_and_pulls_toward_the_anchor(batch):
    x, y, mask = batch
    theta = x.clone().requires_grad_(True)
    reverse_kl(masked(theta, mask), masked(y, mask), mask).mean().backward()
    assert torch.isfinite(theta.grad).all()
    assert (theta.grad[~mask] == 0).all()
    # at the anchor the KL is at its minimum: zero gradient
    theta = y.clone().requires_grad_(True)
    reverse_kl(masked(theta, mask), masked(y, mask), mask).mean().backward()
    assert theta.grad.abs().max() < 1e-6
    # descending it moves the policy to the anchor
    theta = x.clone().requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=0.1)
    first = None
    for _ in range(200):
        opt.zero_grad()
        loss = reverse_kl(masked(theta, mask), masked(y, mask), mask).mean()
        first = first if first is not None else float(loss)
        loss.backward()
        opt.step()
    assert float(loss) < first / 100


@pytest.fixture
def anchor_file(tmp_path):
    torch.manual_seed(3)
    brain = Brain(version=4, conv_channels=32, num_blocks=1, Norm='GN')
    policy = CategoricalPolicy()
    path = tmp_path / 'sl.pth'
    torch.save({'mortal': brain.state_dict(), 'policy_net': policy.state_dict(),
                'config': {'control': {'version': 4}, 'resnet': {'conv_channels': 32, 'num_blocks': 1}}}, path)
    return str(path), brain.eval(), policy.eval()


def make(anchor_file, **cfg):
    return KLAnchor(cfg, default_state_file=anchor_file[0], device=torch.device('cpu'), version=4)


def test_the_anchor_is_the_frozen_checkpoint(anchor_file):
    _, brain, policy = anchor_file
    a = make(anchor_file)
    obs = torch.randn(8, 1012, 34)
    masks = torch.rand(8, A) < 0.5
    masks[:, 0] = True
    with torch.no_grad():
        want = policy.logits(brain(obs), masks)
    assert torch.allclose(a.logits(obs, masks), want, atol=1e-5)
    assert not any(p.requires_grad for p in (*a.brain.parameters(), *a.policy.parameters()))
    assert not a.brain.training and not a.policy.training
    with pytest.raises(ValueError):
        KLAnchor({}, default_state_file=anchor_file[0], device=torch.device('cpu'), version=5)


def test_fixed_coefficient_decays_on_a_power_law(anchor_file):
    a = make(anchor_file, mode='fixed', coef=0.4, coef_decay=0.3, decay_steps=1000)
    assert a.coef(0) == pytest.approx(0.4)
    assert a.coef(1000) == pytest.approx(0.4 * 2 ** -0.3)
    assert a.coef(9000) == pytest.approx(0.4 * 10 ** -0.3)
    for step in range(1, 401):
        a.observe(1.0, step)  # 'fixed' never adapts
    assert a.coef(0) == pytest.approx(0.4)


def test_adaptive_coefficient_steers_toward_the_target(anchor_file):
    a = make(anchor_file, mode='adaptive', coef=0.1, target=0.02, adapt_every=4, tolerance=1.5, factor=2.0,
             coef_min=0.01, coef_max=0.8, hold_windows=10)

    def window(kl, start):
        for step in range(start, start + 4):
            a.observe(torch.tensor(kl), step)

    window(0.0, 1)  # below the band before the KL ever reached it: left alone
    assert a.coef(4) == pytest.approx(0.1)
    window(0.1, 5)  # above 0.03: doubled
    assert a.coef(8) == pytest.approx(0.2)
    window(0.025, 9)  # inside [0.0133, 0.03]: unchanged
    assert a.coef(12) == pytest.approx(0.2)
    window(0.001, 13)  # below, once the KL has reached the band: halved
    assert a.coef(16) == pytest.approx(0.1)
    for start in range(17, 45, 4):
        window(1.0, start)
    assert a.coef(44) == pytest.approx(0.8)  # clamped
    for start in range(45, 105, 4):
        window(0.0, start)
    assert a.coef(104) == pytest.approx(0.01)


def test_the_hold_ends_after_hold_windows(anchor_file):
    a = make(anchor_file, mode='adaptive', coef=0.1, target=0.02, adapt_every=1, hold_windows=3)
    for step in range(1, 4):
        a.observe(0.0, step)  # held: the KL hasn't reached the band yet
    assert a.coef(3) == pytest.approx(0.1)
    a.observe(0.0, 4)  # the hold is over: a coefficient that keeps the KL below the band relaxes
    assert a.coef(4) == pytest.approx(0.05)


def test_target_growth_and_state_round_trip(anchor_file):
    a = make(anchor_file, target=0.02, target_growth=0.5, decay_steps=100, adapt_every=10)
    assert a.target(0) == pytest.approx(0.02)
    assert a.target(300) == pytest.approx(0.02 * 2)
    for step in range(1, 16):
        a.observe(torch.tensor(0.5), step)
    state = a.state_dict()
    assert state['kl_n'] == 5 and state['kl_sum'] == pytest.approx(2.5) and state['reached']
    b = make(anchor_file, target=0.02, target_growth=0.5, decay_steps=100, adapt_every=10)
    b.load_state_dict(state)
    assert b.coef(15) == a.coef(15) and b.kl_n == 5 and b.windows == 1
    other = dict(state, state_file='/elsewhere.pth')
    with pytest.raises(ValueError):
        b.load_state_dict(other)


def test_a_box_checkpoint_resumes_on_the_workstation():
    from kl_anchor import same_file
    box = '/workspace/Mortal-Policy/workdir/pilot/qs-a/pilot-qs-a-final.pth'
    assert same_file(box, '/home/me/Mortal-Policy/workdir/pilot/qs-a/pilot-qs-a-final.pth')
    assert not same_file(box, '/home/me/Mortal-Policy/workdir/pilot/qs-b/pilot-qs-b-final.pth')
    assert same_file('/tmp/a.pth', '/tmp/a.pth') and not same_file('/tmp/a.pth', '/tmp/b.pth')


def test_bad_settings_are_refused(anchor_file):
    for cfg in ({'mode': 'both'}, {'tolerance': 1.0}, {'factor': 0.5}, {'coef_min': 0}, {'decay_steps': 0}):
        with pytest.raises(ValueError):
            make(anchor_file, **cfg)
    assert math.isfinite(make(anchor_file).coef(10 ** 9))


# ---------------------------------------------------------------- re-anchoring (D64, D68)

def policy_file(path, seed, version=4):
    torch.manual_seed(seed)
    brain = Brain(version=version, conv_channels=32, num_blocks=1, Norm='GN')
    policy = CategoricalPolicy()
    torch.save({'mortal': brain.state_dict(), 'policy_net': policy.state_dict(), 'steps': 4800,
                'config': {'control': {'version': version}, 'resnet': {'conv_channels': 32, 'num_blocks': 1}}}, path)
    return str(path), brain.eval(), policy.eval()


def request(path, file, sha=None, label='p25'):
    import json
    from kl_anchor import sha256_file
    path.write_text(json.dumps({'label': label, 'file': file, 'sha256': sha or sha256_file(file)}))
    return str(path)


def logits_of(brain, policy, obs, masks):
    with torch.no_grad():
        return policy.logits(brain(obs), masks)


@pytest.fixture
def inputs():
    g = torch.Generator().manual_seed(5)
    obs = torch.randn(8, 1012, 34, generator=g)
    masks = torch.rand(8, A, generator=g) < 0.5
    masks[:, 0] = True
    return obs, masks


def test_a_request_moves_the_anchor_and_carries_the_coefficient(anchor_file, tmp_path, inputs):
    from kl_anchor import request_path
    obs, masks = inputs
    a = make(anchor_file, target=0.02, adapt_every=2, tolerance=1.25, factor=1.5, hold_windows=3)
    for step in range(1, 5):
        a.observe(torch.tensor(0.1), step)  # above the band twice: 0.1 · 1.5²
    a.observe(torch.tensor(0.1), 5)  # half a window, against the old anchor
    coef = a.coef(5)
    assert coef == pytest.approx(0.225) and a.reached and a.windows == 2 and a.kl_n == 1
    snap, brain, policy = policy_file(tmp_path / 'p25.pth', seed=11)
    req = request_path(str(tmp_path / 'run' / 'mortal.pth'))
    assert req == str(tmp_path / 'run' / 'reanchor.json')
    assert a.poll(req, 5, 5) is None  # no request yet
    (tmp_path / 'run').mkdir()
    rec = a.poll(request(tmp_path / 'run' / 'reanchor.json', snap), 4800, 4790)
    assert rec['reanchor'] == 'p25' and rec['stage'] == 2 and rec['steps'] == 4800 and rec['coef'] == coef
    assert a.stage == 2 and a.anchor_sha256 == rec['sha256']
    assert torch.allclose(a.logits(obs, masks), logits_of(brain, policy, obs, masks), atol=1e-5)
    assert not any(p.requires_grad for p in (*a.brain.parameters(), *a.policy.parameters()))
    # the coefficient carries; the old anchor's half window and the hold's progress don't
    assert a.coef(4790) == coef and a.kl_n == 0 and not a.reached and a.windows == 0
    assert a.poll(req, 5200, 5190) is None  # picked up once


def test_the_hold_applies_again_after_a_re_anchor(anchor_file, tmp_path):
    a = make(anchor_file, target=0.02, adapt_every=1, tolerance=1.5, factor=2.0, hold_windows=2)
    for step in range(1, 4):
        a.observe(0.0, step)  # the hold ends after 2 windows: the third halves it
    assert a.coef(3) == pytest.approx(0.05)
    snap, _, _ = policy_file(tmp_path / 'p25.pth', seed=11)
    a.poll(request(tmp_path / 'reanchor.json', snap), 3, 3)
    for step in range(4, 6):
        a.observe(0.0, step)  # held again: the KL to the new anchor starts at ≈ 0
    assert a.coef(5) == pytest.approx(0.05)
    a.observe(0.0, 6)
    assert a.coef(6) == pytest.approx(0.025)


def test_a_bad_request_is_refused_once_and_the_anchor_kept(anchor_file, tmp_path, inputs):
    obs, masks = inputs
    _, brain, policy = anchor_file
    a = make(anchor_file)
    snap, _, _ = policy_file(tmp_path / 'p25.pth', seed=11)
    req = request(tmp_path / 'reanchor.json', snap, sha='0' * 64)
    rec = a.poll(req, 10, 10)
    assert rec['reanchor_failed'] == 'p25' and 'sha256' in rec['error']
    assert a.poll(req, 20, 20) is None  # not retried at every save
    v5, _, _ = policy_file(tmp_path / 'v5.pth', seed=12, version=5)
    assert 'version' in a.poll(request(tmp_path / 'reanchor.json', v5), 30, 30)['error']
    missing = request(tmp_path / 'reanchor.json', snap)
    os.remove(snap)
    assert 'reanchor_failed' in a.poll(missing, 40, 40)
    (tmp_path / 'reanchor.json').write_text('{"label": "p2')  # torn
    assert 'unreadable' in a.poll(str(tmp_path / 'reanchor.json'), 50, 50)['error']
    assert a.stage == 1
    assert torch.allclose(a.logits(obs, masks), logits_of(brain, policy, obs, masks), atol=1e-5)
    # a request naming the run's first anchor changes nothing
    assert a.poll(request(tmp_path / 'reanchor.json', anchor_file[0]), 60, 60) is None and a.stage == 1


def test_a_re_anchored_state_resumes_without_the_anchor_file(anchor_file, tmp_path, inputs):
    """A restarted trainer, or a replaced learner without the snapshot, resumes on the anchor its
    checkpoint names."""
    from checkpoint import load_checkpoint
    obs, masks = inputs
    a = make(anchor_file, adapt_every=4)
    p25, brain25, policy25 = policy_file(tmp_path / 'p25.pth', seed=11)
    a.poll(request(tmp_path / 'reanchor.json', p25), 100, 100)
    p50, brain50, policy50 = policy_file(tmp_path / 'p50.pth', seed=13)
    a.poll(request(tmp_path / 'reanchor.json', p50, label='p50'), 200, 200)
    a.observe(torch.tensor(0.03), 201)
    assert [s['label'] for s in a.stages] == ['p25', 'p50'] and a.stage == 3
    torch.save({'kl_anchor': a.state_dict()}, tmp_path / 'mortal.pth')
    os.remove(p25)
    os.remove(p50)
    b = make(anchor_file, adapt_every=4)
    b.load_state_dict(load_checkpoint(str(tmp_path / 'mortal.pth'))['kl_anchor'])
    assert b.stage == 3 and b.stages == a.stages and b.kl_n == 1 and b.coef(201) == a.coef(201)
    assert torch.allclose(b.logits(obs, masks), logits_of(brain50, policy50, obs, masks), atol=1e-5)
    # the request it was saved under is already done; an earlier stage's is ignored
    assert b.poll(str(tmp_path / 'reanchor.json'), 202, 202) is None
    (tmp_path / 'stale.json').write_text(__import__('json').dumps(
        {'label': 'p25', 'file': p25, 'sha256': a.stages[0]['sha256']}))
    assert b.poll(str(tmp_path / 'stale.json'), 203, 203) is None and b.stage == 3


def test_a_changed_first_anchor_is_refused(anchor_file, tmp_path):
    a = make(anchor_file)
    state = a.state_dict()
    assert 'weights' not in state and state['stages'] == []  # stage 1: the config's file is the anchor
    b = make(anchor_file)
    b.load_state_dict(state)
    old = {k: v for k, v in state.items() if k not in ('sha256', 'stages')}
    b.load_state_dict(old)  # a checkpoint from before re-anchoring existed
    with pytest.raises(ValueError, match='changed'):
        b.load_state_dict(dict(state, sha256='f' * 64))
