"""Ordered per-seat trajectories and their targets (05 §3, D37).

Upstream shuffles individual decisions and gives every decision its kyoku's
advantage. Here targets are computed on each seat's ordered decision sequence
first, and only then may the entries be shuffled:

  1. `build`: rewards and done flags under a declared horizon, plus the index
     of the next decision of the same seat;
  2. `gae`: GAE advantages and λ-returns from critic values (detached targets);
  3. shuffle and train.

Horizons (05 §3.1):
  'K' (K-surrogate)  the episode is the kyoku; terminal reward = that kyoku's
                     GRP-Δ. Upstream behaviour and arm P0's objective; a
                     surrogate, not policy-preserving.
  'H' (H-hanchan)    GRP-Δ is paid at each kyoku boundary and returns run to
                     the end of the hanchan (γ = 1); telescopes to
                     u(final) − Φ(start), so it is policy-preserving.

Rewards are in fixed units (raw GRP-Δ in `pts` units, RewardCalculator with
standardize=False); nothing here keeps running statistics.

All inputs are one seat's arrays as GameplayLoader returns them (`at_kyoku`
per decision) plus that seat's per-kyoku rewards; nothing crosses seats.
"""
import numpy as np

HORIZONS = ('K', 'H')


def build(at_kyoku, kyoku_rewards, horizon):
    """Per-decision reward, done and next index for one seat.

    The last decision of each kyoku receives that kyoku's reward; every other
    decision receives 0. `done` marks episode ends under the horizon: the last
    decision of every kyoku for 'K', only the last decision of the game for
    'H'. `next_idx[i]` is i + 1, or -1 where done.
    """
    if horizon not in HORIZONS:
        raise ValueError(f'horizon must be one of {HORIZONS}, got {horizon!r}')
    if isinstance(at_kyoku, (bytes, bytearray)):  # Vec<u8> from libriichi arrives as bytes
        at_kyoku = np.frombuffer(at_kyoku, dtype=np.uint8)
    at_kyoku = np.asarray(at_kyoku, dtype=np.int64)
    kyoku_rewards = np.asarray(kyoku_rewards, dtype=np.float64)
    n = len(at_kyoku)
    if n == 0:
        return {'reward': np.zeros(0), 'done': np.zeros(0, bool), 'next_idx': np.zeros(0, np.int64)}
    if np.any(np.diff(at_kyoku) < 0):
        raise ValueError('at_kyoku must be non-decreasing (one seat, in order)')
    if at_kyoku[-1] >= len(kyoku_rewards):
        raise ValueError('a decision refers to a kyoku without a reward')

    kyoku_end = np.append(at_kyoku[1:] != at_kyoku[:-1], True)
    reward = np.where(kyoku_end, kyoku_rewards[at_kyoku], 0.0)
    if horizon == 'K':
        done = kyoku_end.copy()
    else:
        done = np.zeros(n, dtype=bool)
        done[-1] = True
        # kyoku with no decision of this seat still pay their reward: fold
        # them into the next decision's kyoku boundary so the sum telescopes
        missing = np.setdiff1d(np.arange(at_kyoku[-1] + 1), at_kyoku)
        for k in missing:
            later = np.flatnonzero(kyoku_end & (at_kyoku > k))
            reward[later[0] if len(later) else n - 1] += kyoku_rewards[k]
        # rewards of kyoku after this seat's last decision belong to the end
        reward[-1] += kyoku_rewards[at_kyoku[-1] + 1:].sum()
    next_idx = np.where(done, -1, np.arange(1, n + 1))
    return {'reward': reward, 'done': done, 'next_idx': next_idx}


def gae(reward, done, values, gamma=1.0, lam=0.95, bootstrap=0.0):
    """GAE advantages and λ-returns for one ordered trajectory.

    `values[i]` is the critic's V at decision i (detached); terminal steps
    bootstrap with 0, and a trajectory cut off without `done` at its end
    bootstraps with `bootstrap`. Returns (advantages, returns), with
    returns = advantages + values (the critic's regression target).
    """
    reward = np.asarray(reward, dtype=np.float64)
    done = np.asarray(done, dtype=bool)
    values = np.asarray(values, dtype=np.float64)
    n = len(reward)
    adv = np.zeros(n)
    next_value, next_adv = bootstrap, 0.0
    for i in range(n - 1, -1, -1):
        if done[i]:
            next_value, next_adv = 0.0, 0.0
        delta = reward[i] + gamma * next_value - values[i]
        next_adv = delta + gamma * lam * next_adv
        adv[i] = next_adv
        next_value = values[i]
    return adv, adv + values


def monte_carlo_returns(reward, done, gamma=1.0):
    """Discounted returns to the end of each episode (GAE with λ = 1, V = 0)."""
    adv, _ = gae(reward, done, np.zeros(len(reward)), gamma=gamma, lam=1.0)
    return adv
