import os
import random
import logging
import torch
import numpy as np
from multiprocessing import Lock, Value
from torch.utils.data import IterableDataset
from model import GRP
from reward_calculator import RewardCalculator
from libriichi.dataset import GameplayLoader
from config import config

# env.pts holds u × 100 (D2); rl_dataset.PTS_PER_U, kept here to avoid its imports
PTS_PER_U = 100

class FileDatasetsIter(IterableDataset):
    def __init__(
        self,
        version,
        file_list,
        pts,
        shared_stats=None,
        oracle = False,
        file_batch_size = 20, 
        reserve_ratio = 0,
        player_names = None,
        excludes = None,
        num_epochs = 1,
        enable_augmentation = False,
        augmented_first = False,
        masks = None,
        compute_advantage = True,
        obs_fp16 = False,
        value_targets = False,
    ):
        super().__init__()
        self.version = version
        self.file_list = file_list
        self.pts = pts
        self.oracle = oracle
        self.file_batch_size = file_batch_size
        self.reserve_ratio = reserve_ratio
        self.player_names = player_names
        self.excludes = excludes
        self.num_epochs = num_epochs
        self.enable_augmentation = enable_augmentation
        self.augmented_first = augmented_first
        self.iterator = None
        self.shared_stats = shared_stats 
        # {log_id: (masked kyoku indices, masked seats)} from the dataset's
        # masks.jsonl (04a §4: Q6 disconnect windows, Q8 bot seats). Rewards
        # are still computed on the full game; masked decisions are skipped.
        self.masks = masks or {}
        # False for behaviour cloning: no GRP, every advantage is 0
        self.compute_advantage = compute_advantage
        # fp16 obs halve worker RAM; identical under AMP, whose first conv
        # casts its input to fp16 anyway. Leave off for fp32 training.
        self.obs_fp16 = obs_fp16
        # True for arm V (Equim's train.py): each entry ends with the Q target
        # (the kyoku's GRP-Δu in fixed u units, 05 §3.2) and the seat's rank
        # after the kyoku (the next-rank auxiliary) instead of an advantage
        self.value_targets = value_targets

    def build_iter(self):
        if self.compute_advantage or self.value_targets:
            self.grp = GRP(**config['grp']['network'])
            grp_state = torch.load(config['grp']['state_file'], weights_only=True, map_location=torch.device('cpu'))
            self.grp.load_state_dict(grp_state['model'])
            if self.value_targets:
                # fixed units: local statistics that stay at count 0, as in
                # rl_dataset.fixed_unit_rewards (a Manager can't start in a worker)
                stats = {'count': Value('i', 0), 'mean': Value('d', 0.0), 'M2': Value('d', 0.0), 'lock': Lock()}
                self.reward_calc = RewardCalculator(self.grp, self.pts, shared_stats=stats, standardize=False)
            else:
                self.reward_calc = RewardCalculator(
                    self.grp,
                    self.pts,
                    shared_stats=self.shared_stats,
                    standardize=config['env'].get('reward_standardization', True),
                )

        for _ in range(self.num_epochs):
            yield from self.load_files(self.augmented_first)
            if self.enable_augmentation:
                yield from self.load_files(not self.augmented_first)

    def load_files(self, augmented):
        random.shuffle(self.file_list)
        self.loader = GameplayLoader(
            version = self.version,
            oracle = self.oracle,
            player_names = self.player_names,
            excludes = self.excludes,
            augmented = augmented,
        )
        self.buffer = []

        for start_idx in range(0, len(self.file_list), self.file_batch_size):
            old_buffer_size = len(self.buffer)
            self.populate_buffer(self.file_list[start_idx:start_idx + self.file_batch_size])
            buffer_size = len(self.buffer)

            reserved_size = int((buffer_size - old_buffer_size) * self.reserve_ratio)
            if reserved_size > buffer_size:
                continue

            random.shuffle(self.buffer)
            yield from self.buffer[reserved_size:]
            del self.buffer[reserved_size:]
        random.shuffle(self.buffer)
        yield from self.buffer
        self.buffer.clear()

    def populate_buffer(self, file_list):
        try:
            data = self.loader.load_gz_log_files(file_list)
            names = file_list
        except RuntimeError:
            # rare semantically-invalid logs (converter bugs, e.g. impossible
            # aka counts) abort the whole batch; retry file-by-file and drop
            # only the offenders
            data = []
            names = []
            for filename in file_list:
                try:
                    data.extend(self.loader.load_gz_log_files([filename]))
                    names.append(filename)
                except RuntimeError as e:
                    logging.warning(f'skipping bad log {filename}: {e}')
        for filename, file in zip(names, data):
            log_id = os.path.basename(filename).split('.', 1)[0]
            masked_kyoku, masked_seats = self.masks.get(log_id, ((), ()))
            for game in file:
                if game.take_player_id() in masked_seats:
                    continue
                # per move
                obs = game.take_obs()
                if self.oracle:
                    invisible_obs = game.take_invisible_obs()
                actions = game.take_actions()
                masks = game.take_masks()
                at_kyoku = game.take_at_kyoku()

                # per game
                grp = game.take_grp()
                player_id = game.take_player_id()

                game_size = len(obs)

                if self.value_targets:
                    grp_feature = grp.take_feature()
                    rank_by_player = grp.take_rank_by_player()
                    advantage = self.reward_calc.calc_delta_pt(player_id, grp_feature, rank_by_player) / PTS_PER_U
                    assert len(advantage) >= at_kyoku[-1] + 1
                    ranks = next_ranks(grp_feature, grp.take_final_scores(), player_id)
                elif self.compute_advantage:
                    grp_feature = grp.take_feature()
                    rank_by_player = grp.take_rank_by_player()
                    advantage = self.reward_calc.calc_delta_pt(player_id, grp_feature, rank_by_player)
                    assert len(advantage) >= at_kyoku[-1] + 1 
                else:
                    advantage = np.zeros(at_kyoku[-1] + 1 if game_size else 0)

                for i in range(game_size):
                    if at_kyoku[i] in masked_kyoku:
                        continue
                    entry = [
                        obs[i].astype(np.float16) if self.obs_fp16 else obs[i],
                        actions[i],
                        masks[i],
                        advantage[at_kyoku[i]],
                    ]
                    if self.value_targets:
                        entry.append(ranks[at_kyoku[i] + 1])
                    if self.oracle:
                        entry.insert(1, invisible_obs[i])
                    self.buffer.append(entry)

    def __iter__(self):
        if self.iterator is None:
            self.iterator = self.build_iter()
        return self.iterator

def worker_init_fn(*args, **kwargs):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    per_worker = int(np.ceil(len(dataset.file_list) / worker_info.num_workers))
    start = worker_info.id * per_worker
    end = start + per_worker
    dataset.file_list = dataset.file_list[start:end]


def next_ranks(grp_feature, final_scores, player_id):
    """The seat's 0-based rank at the start of each kyoku and at the end of the
    game, as Equim's loader computes it: ranks[k + 1] is the rank after kyoku k.
    Ties go to the lower seat (stable sort), as in libriichi."""
    scores_seq = np.concatenate((grp_feature[:, 3:] * 1e4, [final_scores]))
    rank_by_player_seq = (-scores_seq).argsort(-1, kind='stable').argsort(-1, kind='stable')
    return rank_by_player_seq[:, player_id]


def load_masks(path):
    """Read a dataset masks.jsonl (tools/data/splits.py) into the form
    FileDatasetsIter expects."""
    import json
    out = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            out[row['log_id']] = (frozenset(row['kyoku']), frozenset(row['seats']))
    return out
