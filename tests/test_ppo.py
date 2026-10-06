"""The learner's PPO loss against stored behaviour probabilities (05 §8):
first-minibatch ratio 1, illegal actions at probability exactly 0, and entries
without a behaviour probability kept out of the policy loss."""
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

import rl_dataset as rd
from model import GRP
from ppo import ppo_loss

PTS = [90, 50, -10, -130]
GRP_NETWORK = {'hidden_size': 64, 'num_layers': 2}
HP = {'clip_ratio': 0.2, 'dual_clip': 3.0, 'entropy_weight': 1e-3}


@pytest.fixture(scope='module')
def first_batch(arena_run, tmp_path_factory):
    """The first minibatch the trainer would see, through the real dataset
    (ordered, joined, shuffled) and the default collate."""
    root, logs, brain, policy = arena_run
    torch.manual_seed(1)
    grp_file = tmp_path_factory.mktemp('grp') / 'grp.pth'
    torch.save({'model': GRP(**GRP_NETWORK).state_dict()}, grp_file)
    ds = rd.OrderedDatasetsIter(version=4, file_list=list(logs), behaviour_dir=str(root / 'bp'), pts=PTS,
                                grp_state_file=str(grp_file), grp_network=GRP_NETWORK)
    obs, actions, masks, advantage, b, version = next(iter(DataLoader(ds, batch_size=512)))
    return obs.float(), actions, masks.bool(), advantage, b, brain, policy


def test_first_minibatch_ratio_is_one(first_batch):
    """Learner weights equal to the actor's (ε = 1, CPU FP32): the trainer's
    loss code reproduces every stored probability."""
    obs, actions, masks, advantage, b, brain, policy = first_batch
    with torch.no_grad():
        logits = policy.logits(brain(obs), masks)
    _, stats = ppo_loss(logits, actions, advantage, b, **HP)
    assert stats['excluded_frac'] < 0.02
    assert abs(stats['ratio_min'] - 1) < 1e-4 and abs(stats['ratio_max'] - 1) < 1e-4
    assert stats['clip_frac'] == 0


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_illegal_actions_have_probability_zero(first_batch, dtype):
    obs, actions, masks, advantage, b, brain, policy = first_batch
    with torch.no_grad():
        logits = policy.logits(brain(obs), masks).to(dtype)
    probs = torch.softmax(logits.float(), -1)
    assert (probs[~masks] == 0).all()
    assert torch.allclose(probs.sum(-1), torch.ones(len(probs)), atol=1e-3)
    loss, stats = ppo_loss(logits, actions, advantage, b, **HP)
    assert torch.isfinite(loss) and torch.isfinite(stats['entropy'])


def test_excluded_entries_carry_no_policy_gradient():
    torch.manual_seed(0)
    n, a = 64, 46
    masks = torch.rand(n, a) < 0.5
    masks[:, 45] = True
    actions = torch.multinomial(masks.float(), 1).squeeze(1)
    logits = torch.randn(n, a).masked_fill(~masks, -torch.inf).requires_grad_()
    advantage = torch.randn(n)
    # b = π: every ratio is 1, inside the clip range, so every included entry has a gradient
    b = torch.softmax(logits.detach(), -1)[torch.arange(n), actions]
    b[::4] = torch.nan

    hp = {**HP, 'entropy_weight': 0.0}
    loss, stats = ppo_loss(logits, actions, advantage, b, **hp)
    loss.backward()
    excluded = torch.isnan(b)
    assert (logits.grad[excluded] == 0).all()
    assert (logits.grad[~excluded].abs().sum(-1) > 0).all()
    assert stats['excluded_frac'] == pytest.approx(0.25)

    # their advantages don't enter the normalization either
    advantage2 = advantage.clone()
    advantage2[excluded] = 1e6
    loss2, _ = ppo_loss(logits, actions, advantage2, b, **hp)
    assert loss2.item() == pytest.approx(loss.item())


def test_dual_clip_matches_upstream_formula():
    """Same clipping as upstream's snapshot path, with b in place of the
    snapshot's probability and advantages not normalized."""
    torch.manual_seed(0)
    n = 256
    logits = torch.randn(n, 46)
    actions = torch.randint(0, 46, (n,))
    advantage = torch.randn(n) * 2
    pi = torch.softmax(logits, -1)[torch.arange(n), actions]
    b = (pi * torch.exp(torch.randn(n) * 0.5)).clamp(max=1)
    hp = {**HP, 'entropy_weight': 0.0}
    loss, stats = ppo_loss(logits, actions, advantage, b, advantage_norm=False, **hp)

    ratio = pi / b
    min_loss = torch.min(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage)
    want = -torch.where(advantage < 0, torch.max(min_loss, 3.0 * advantage), min_loss).mean()
    assert loss.item() == pytest.approx(want.item(), rel=1e-5)
    assert stats['clip_frac'] == pytest.approx(((ratio - 1).abs() > 0.2).float().mean().item())
