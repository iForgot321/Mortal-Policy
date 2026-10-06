"""Ordered RL dataset for arms P0/P1 (05 §2–3, D37, D46).

For each 1v3 log the server drained, and for each trainee seat in it:

  1. load the seat's decisions with GameplayLoader (one seat, in order);
  2. per-kyoku rewards from the frozen GRP, in fixed units (no running
     statistics: RewardCalculator(standardize=False));
  3. `trajectory.build` under the declared horizon, and the stored behaviour
     probability of every decision from the game's sidecar (`join_behaviour`);
  4. targets, then shuffling:
     - `targets = 'mc'` (arm P0): the policy advantage is the Monte-Carlo
       return; under horizon K this is each decision's own kyoku reward,
       upstream's label without standardization. Decisions are shuffled within
       a buffer of `file_batch_size` logs and yielded one by one.
     - `targets = 'gae'` (arm P1, critic K1): GAE needs critic values, and the
       critic lives on the learner GPU. Each `file_batch_size` logs are yielded
       as one chunk of whole trajectories, with the critic's privileged
       channels (`GameplayLoader(oracle=True, trust_seed=True)`); the learner
       computes values and targets (`critic.gae_targets`) and only then
       shuffles (`shuffled_minibatches`).

Units: rewards are GRP-Δu in **u units** (05 §3.2), the GRP-expected change of
u = pts / PTS_PER_U, because `env.pts` holds u × 100 (D2: [90, 50, -10, -130]).
The critic, the SL kyoku-Δ head and the search bootstrap use the same units.
The policy only sees advantages normalized per batch, so the scale doesn't
matter to it.

Behaviour probabilities: NaN means the entry must stay out of ratio terms (the
agari guard replaced the action, or the board voided the reaction). Such
entries are still yielded; the trainer gives them weight 0 in the policy loss
and counts them. A log without a sidecar is never trained on (no guessed b):
`split_by_sidecar` lets the trainer drop and count those up front, and the
dataset skips any that remain. A game whose log and sidecar disagree is
skipped with an error in the log.

SL training keeps using `dataloader.FileDatasetsIter`, which is unchanged;
`dataloader.worker_init_fn` splits this dataset's files between workers too.
"""
import logging
import random
from multiprocessing import Lock, Value
from os import path

import numpy as np
import torch
from torch.utils.data import IterableDataset

import trajectory as tj
from critic import privileged
from libriichi.dataset import GameplayLoader
from model import GRP
from reward_calculator import RewardCalculator

PTS_PER_U = 100
TARGETS = ('mc', 'gae')


def sidecar_path(log, behaviour_dir):
    """The behaviour sidecar of `log`: same file name, in `behaviour_dir`."""
    return path.join(behaviour_dir, path.basename(log))


def split_by_sidecar(file_list, behaviour_dir):
    """(logs with a sidecar, logs without one)."""
    with_bp, without = [], []
    for f in file_list:
        (with_bp if path.exists(sidecar_path(f, behaviour_dir)) else without).append(f)
    return with_bp, without


def fixed_unit_rewards(grp, pts):
    """A RewardCalculator in fixed units. Its statistics are local and must
    stay at count 0 (05 §8: no running statistics in the reward path)."""
    stats = {'count': Value('i', 0), 'mean': Value('d', 0.0), 'M2': Value('d', 0.0), 'lock': Lock()}
    return RewardCalculator(grp, pts, shared_stats=stats, standardize=False)


def seat_trajectory(game, sidecar, reward_calc, horizon, oracle=False):
    """One seat's ordered trajectory from a featurized `Gameplay` (consumed).

    Returns numpy arrays per decision: obs, (invisible_obs,) action, mask,
    reward, done, next_idx, behaviour_prob, at_kyoku, plus `player_id` and the
    per-kyoku `kyoku_rewards` in u units.
    """
    player_id = game.take_player_id()
    actions = np.asarray(game.take_actions(), dtype=np.int64)
    event_idx = game.take_event_idx()
    at_kyoku = np.frombuffer(bytes(game.take_at_kyoku()), dtype=np.uint8)
    grp = game.take_grp()
    kyoku_rewards = reward_calc.calc_delta_pt(player_id, grp.take_feature(), grp.take_rank_by_player()) / PTS_PER_U

    out = tj.build(at_kyoku, kyoku_rewards, horizon)
    out['behaviour_prob'] = tj.join_behaviour(event_idx, actions, player_id, sidecar)
    out['obs'] = game.take_obs()
    if oracle:
        out['invisible_obs'] = game.take_invisible_obs()
    out['action'] = actions
    out['mask'] = game.take_masks()
    out['at_kyoku'] = at_kyoku
    out['player_id'] = player_id
    out['kyoku_rewards'] = kyoku_rewards
    return out


class OrderedDatasetsIter(IterableDataset):
    """`targets = 'mc'`: yields (obs, action, mask, advantage, behaviour_prob,
    param_version) per decision, shuffled within a buffer only after targets
    are attached. `targets = 'gae'`: yields one list of whole trajectories
    (`compact`) per `file_batch_size` logs, unshuffled; use a DataLoader with
    `batch_size=None`.

    `versions` maps a log's file name to the parameter version it was played
    with (the server's drain reply); -1 where unknown.
    """

    def __init__(
        self,
        version,
        file_list,
        behaviour_dir,
        pts,
        grp_state_file,
        grp_network,
        horizon = 'K',
        targets = 'mc',
        player_names = ('trainee',),
        file_batch_size = 20,
        num_epochs = 1,
        obs_fp16 = False,
        versions = None,
    ):
        super().__init__()
        if horizon not in tj.HORIZONS:
            raise ValueError(f'horizon must be one of {tj.HORIZONS}, got {horizon!r}')
        if targets not in TARGETS:
            raise ValueError(f'targets must be one of {TARGETS}, got {targets!r}')
        self.version = version
        self.file_list = file_list
        self.behaviour_dir = behaviour_dir
        self.pts = pts
        self.grp_state_file = grp_state_file
        self.grp_network = grp_network
        self.horizon = horizon
        self.targets = targets
        self.player_names = list(player_names)
        self.file_batch_size = file_batch_size
        self.num_epochs = num_epochs
        # fp16 obs halve worker RAM; identical under AMP (see FileDatasetsIter)
        self.obs_fp16 = obs_fp16
        self.versions = versions or {}
        # the critic's privileged channels need the exact hidden information
        self.oracle = targets == 'gae'
        self.iterator = None

    def prepare(self):
        """Load the GRP and the loader (in the worker process)."""
        grp = GRP(**self.grp_network)
        grp.load_state_dict(torch.load(self.grp_state_file, weights_only=True, map_location='cpu')['model'])
        self.reward_calc = fixed_unit_rewards(grp, self.pts)
        self.loader = GameplayLoader(
            version = self.version,
            oracle = self.oracle,
            trust_seed = self.oracle,
            player_names = self.player_names,
        )

    def build_iter(self):
        self.prepare()
        for _ in range(self.num_epochs):
            random.shuffle(self.file_list)
            for start in range(0, len(self.file_list), self.file_batch_size):
                trajs = self.trajectories(self.file_list[start:start + self.file_batch_size])
                if self.targets == 'gae':
                    chunk = [self.compact(t) for t in trajs]
                    if chunk:
                        yield chunk
                    continue
                buffer = []
                for traj in trajs:
                    buffer.extend(self.decisions(traj))
                random.shuffle(buffer)
                yield from buffer

    def trajectories(self, file_list):
        """(log, trajectory) of every selected seat in `file_list`, in order."""
        file_list = [f for f in file_list if self.has_sidecar(f)]
        try:
            data = self.loader.load_gz_log_files(file_list)
            names = file_list
        except RuntimeError:
            # one bad log aborts the batch: retry file by file
            data, names = [], []
            for filename in file_list:
                try:
                    data.extend(self.loader.load_gz_log_files([filename]))
                    names.append(filename)
                except RuntimeError as e:
                    logging.warning(f'skipping bad log {filename}: {e}')
        for filename, games in zip(names, data):
            sidecar = tj.load_behaviour(sidecar_path(filename, self.behaviour_dir))
            for game in games:
                try:
                    traj = seat_trajectory(game, sidecar, self.reward_calc, self.horizon, self.oracle)
                except ValueError as e:
                    logging.error(f'skipping a seat of {filename}: log and behaviour sidecar disagree: {e}')
                    continue
                if len(traj['action']) == 0:
                    continue
                traj['param_version'] = self.versions.get(path.basename(filename), -1)
                yield traj

    def has_sidecar(self, filename):
        if path.exists(sidecar_path(filename, self.behaviour_dir)):
            return True
        logging.warning(f'skipping {filename}: no behaviour sidecar')
        return False

    def decisions(self, traj):
        advantage = tj.monte_carlo_returns(traj['reward'], traj['done']).astype(np.float32)
        version = traj['param_version']
        for i, obs in enumerate(traj['obs']):
            yield (
                obs.astype(np.float16) if self.obs_fp16 else obs,
                traj['action'][i],
                traj['mask'][i],
                advantage[i],
                traj['behaviour_prob'][i],
                version,
            )

    def compact(self, traj):
        """One whole trajectory as stacked arrays, for the learner's GAE."""
        dtype = np.float16 if self.obs_fp16 else np.float32
        n = len(traj['action'])
        return {
            'obs': np.stack(traj['obs']).astype(dtype),
            'priv': np.stack([privileged(x) for x in traj['invisible_obs']]).astype(dtype),
            'action': traj['action'],
            'mask': np.stack(traj['mask']),
            'reward': traj['reward'],
            'done': traj['done'],
            'behaviour_prob': traj['behaviour_prob'],
            'at_kyoku': traj['at_kyoku'].astype(np.int64),
            'param_version': np.full(n, traj['param_version'], dtype=np.int64),
        }

    def __iter__(self):
        if self.iterator is None:
            self.iterator = self.build_iter()
        return self.iterator


def shuffled_minibatches(flats, batch_size, keys):
    """Minibatches of `batch_size` decisions, shuffled within each flat chunk
    of decisions (whose targets are already attached) plus the remainder
    carried over from the previous chunk. A final remainder smaller than
    `batch_size` is dropped, as upstream does."""
    leftover = None
    for flat in flats:
        if leftover is not None:
            flat = {k: torch.cat((leftover[k], flat[k])) for k in keys}
        n = len(flat[keys[0]])
        perm = torch.randperm(n)
        full = n - n % batch_size
        for start in range(0, full, batch_size):
            idx = perm[start:start + batch_size]
            yield {k: flat[k][idx] for k in keys}
        leftover = {k: flat[k][perm[full:]] for k in keys}
