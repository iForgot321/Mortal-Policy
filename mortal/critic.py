"""Critic V(h, i) for discovery package K1 (05 §4).

- Input: the v4 observation h (1012 channels) plus the **opponents' concealed
  hands** from the Invisible oracle encoding: rows 0–50, 3 opponents × 17 rows
  (tehai 4, aka 3, shanten 7 + 1, waits 1, furiten 1; libriichi
  `dataset/invisible.rs`). Rows 51 and above are the wall order (yama,
  rinshan, dora, ura), which 05 §4 excludes. Exact hidden information for
  arena logs needs `GameplayLoader(oracle=True, trust_seed=True)`.
- Network: a separate b8c128 with the policy trunk's v4 layout (GroupNorm,
  Mish, pre-activation) and a scalar head. Its own optimizer.
- Units: u (rl_dataset.py), the same as the rewards and the SL kyoku-Δ head.

Oracle inputs stay on the learner: the critic never reaches the actors or the
policy (CLAUDE.md rule 6). `server.py` only ever carries the policy's weights.
"""
from functools import partial

import numpy as np
import torch
from torch import nn

import trajectory as tj
from libriichi.consts import obs_shape
from model import ResNet

# opponents' hands; rows 51+ are the wall order (05 §4: excluded)
PRIVILEGED_ROWS = 51


def privileged(invisible_obs):
    """The critic's privileged channels of one Invisible encoding (217 × 34)."""
    return invisible_obs[:PRIVILEGED_ROWS]


class Critic(nn.Module):
    def __init__(self, *, conv_channels=128, num_blocks=8, version=4):
        super().__init__()
        self.version = version
        self.encoder = ResNet(
            in_channels = obs_shape(version)[0] + PRIVILEGED_ROWS,
            conv_channels = conv_channels,
            num_blocks = num_blocks,
            norm_builder = partial(nn.GroupNorm, num_channels=conv_channels, num_groups=32, eps=1e-3),
            actv_builder = partial(nn.Mish, inplace=True),
            pre_actv = True,
        )
        self.actv = nn.Mish(inplace=True)
        self.head = nn.Linear(1024, 1)

    def forward(self, obs, priv):
        x = torch.cat((obs, priv), dim=1)
        return self.head(self.actv(self.encoder(x))).squeeze(-1)


def gae_targets(chunk, critic, *, device, gamma=1.0, lam=0.95, enable_amp=False, eval_batch=1024):
    """Critic values (no grad) and GAE targets on whole trajectories, before
    any shuffling (05 §3.3). `chunk` is a list of `OrderedDatasetsIter.compact`
    trajectories as tensors.

    Returns the chunk's decisions concatenated, with `value`, `advantage` and
    `returns` (the λ-return, the critic's regression target; detached) added.
    For the diagnostics: `mc_return`, the observed return (independent of the
    critic), and `group`, each decision's (trajectory, kyoku).
    Nothing crosses trajectories: each is its own GAE recursion.
    """
    keys = ('obs', 'priv', 'action', 'mask', 'behaviour_prob', 'param_version')
    flat = {k: torch.cat([torch.as_tensor(t[k]) for t in chunk]) for k in keys}
    was_training = critic.training
    critic.eval()
    values = []
    with torch.no_grad(), torch.autocast(device.type, enabled=enable_amp):
        for start in range(0, len(flat['obs']), eval_batch):
            obs = flat['obs'][start:start + eval_batch].to(device=device, dtype=torch.float32)
            priv = flat['priv'][start:start + eval_batch].to(device=device, dtype=torch.float32)
            values.append(critic(obs, priv).float().cpu())
    critic.train(was_training)
    values = torch.cat(values).double().numpy()

    advantage, returns, mc_return, group = [], [], [], []
    start = 0
    for i, t in enumerate(chunk):
        n = len(t['action'])
        reward, done = np.asarray(t['reward']), np.asarray(t['done'])
        adv, ret = tj.gae(reward, done, values[start:start + n], gamma=gamma, lam=lam)
        advantage.append(adv)
        returns.append(ret)
        mc_return.append(tj.monte_carlo_returns(reward, done, gamma=gamma))
        group.append(np.stack((np.full(n, i), np.asarray(t['at_kyoku'])), axis=1))
        start += n
    flat['value'] = torch.from_numpy(values).float()
    flat['advantage'] = torch.from_numpy(np.concatenate(advantage)).float()
    flat['returns'] = torch.from_numpy(np.concatenate(returns)).float()
    flat['mc_return'] = torch.from_numpy(np.concatenate(mc_return)).float()
    flat['group'] = torch.from_numpy(np.concatenate(group))
    return flat


def explained_variance(values, returns):
    """1 − Var(returns − V) / Var(returns)."""
    values = np.asarray(values, dtype=np.float64)
    returns = np.asarray(returns, dtype=np.float64)
    var = returns.var()
    return float('nan') if var == 0 else float(1 - (returns - values).var() / var)


def calibration(values, returns, bins=10):
    """Mean prediction and mean target in equal-count bins of the prediction.
    Returns (mean_v, mean_target, monotone)."""
    values = np.asarray(values, dtype=np.float64)
    returns = np.asarray(returns, dtype=np.float64)
    order = np.argsort(values, kind='stable')
    parts = [p for p in np.array_split(order, bins) if len(p)]
    mean_v = np.array([values[p].mean() for p in parts])
    mean_t = np.array([returns[p].mean() for p in parts])
    return mean_v, mean_t, bool(np.all(np.diff(mean_t) >= 0))


def advantage_variance_ratio(advantage, episode_kyoku):
    """Within-kyoku over across-kyoku variance of the advantages (05 §4: should
    rise above ~0 with a critic; it is exactly 0 for P0's per-kyoku returns).

    `episode_kyoku` labels each decision's (trajectory, kyoku) group.
    """
    advantage = np.asarray(advantage, dtype=np.float64)
    _, group = np.unique(np.asarray(episode_kyoku), return_inverse=True, axis=0)
    count = np.bincount(group)
    mean = np.bincount(group, weights=advantage) / count
    within = ((advantage - mean[group]) ** 2).sum() / len(advantage)
    across = (count * (mean - advantage.mean()) ** 2).sum() / len(advantage)
    return float('nan') if across == 0 else float(within / across)
