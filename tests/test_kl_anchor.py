"""K2's KL anchor (05 §5.1, D59): the exact reverse KL over the legal mask, the frozen π_SL, and
the coefficient rules (`mortal/kl_anchor.py`)."""
import math

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
