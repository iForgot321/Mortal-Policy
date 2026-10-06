"""Dual-clip PPO against stored behaviour probabilities (05 §2.4, §3.4; D37, D46).

`train_online.py` with `policy.ratio_reference = 'stored'` calls `ppo_loss` on
every batch; the tests call the same function (05 §8: first-minibatch ratio 1).
"""
import torch
from torch.distributions import Categorical


def ppo_loss(
    logits,
    actions,
    advantage,
    behaviour_prob,
    *,
    clip_ratio,
    dual_clip,
    entropy_weight,
    advantage_norm = True,
):
    """Loss and detached diagnostics for one batch.

    `logits` are the masked raw logits (`CategoricalPolicy.logits`, illegal
    actions at -inf); everything here runs in fp32, whatever autocast produced
    them in. `behaviour_prob` is the stored b(a|s), detached data; NaN keeps an
    entry out of the policy loss (agari guard, voided reaction) but not out of
    the entropy bonus. `advantage_norm` normalizes the detached advantages over
    the batch's included entries.
    """
    logits = logits.float()
    dist = Categorical(logits=logits)
    log_prob = dist.log_prob(actions)
    entropy = dist.entropy()

    included = ~torch.isnan(behaviour_prob)
    n_included = included.sum()
    log_b = torch.where(included, behaviour_prob, 1.0).float().log()
    ratio = (log_prob - log_b).exp()

    adv = advantage.detach().float()
    raw_mean, raw_std = adv[included].mean(), adv[included].std(correction=0)
    if advantage_norm:
        adv = (adv - raw_mean) / (raw_std + 1e-8)
    adv = torch.where(included, adv, 0.0)

    loss1 = ratio * adv
    loss2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
    min_loss = torch.min(loss1, loss2)
    clip_loss = torch.where(adv < 0, torch.max(min_loss, dual_clip * adv), min_loss)
    policy_loss = -torch.where(included, clip_loss, 0.0).sum() / n_included.clamp(min=1)
    loss = policy_loss - entropy_weight * entropy.mean()

    with torch.no_grad():
        r = ratio[included]
        stats = {
            'loss': loss.detach(),
            'policy_loss': policy_loss.detach(),
            'entropy': entropy.mean(),
            'ratio_mean': r.mean(),
            'ratio_p99': torch.quantile(r, 0.99) if len(r) else r.sum(),
            'ratio_min': r.min() if len(r) else r.sum(),
            'ratio_max': r.max() if len(r) else r.sum(),
            'clip_frac': ((r - 1).abs() > clip_ratio).float().mean(),
            'excluded_frac': 1 - included.float().mean(),
            'adv_mean': raw_mean,
            'adv_std': raw_std,
        }
    return loss, stats
