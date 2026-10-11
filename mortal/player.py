import torch
import numpy as np
import os
import shutil
import secrets
import logging
from os import path
from model import Brain, DQN,CategoricalPolicy
from engine import MortalEngine, ValueEngine
from libriichi.stat import Stat
from libriichi.arena import OneVsThree
from config import config

def head_kind():
    """The trained network's head: 'policy' (CategoricalPolicy) or 'value'
    (arm V's DQN, train_value.py)."""
    kind = config['control'].get('head', 'policy')
    assert kind in ('policy', 'value'), kind
    return kind

def build_head(kind, version):
    return DQN(version=version) if kind == 'value' else CategoricalPolicy()

def load_frozen(baseline_cfg):
    """The frozen opponent as a greedy engine, from a policy checkpoint
    ('policy_net') or a value one ('current_dqn', arm V)."""
    device = torch.device(baseline_cfg['device'])

    # weights_only=False: trainer-saved checkpoints carry numpy scalars that
    # torch>=2.6 safe-load rejects; we only ever load our own checkpoints
    state = torch.load(baseline_cfg['state_file'], weights_only=False, map_location=torch.device('cpu'))
    cfg = state['config']
    version = cfg['control'].get('version', 1)
    conv_channels = cfg['resnet']['conv_channels']
    num_blocks = cfg['resnet']['num_blocks']
    stable_mortal = Brain(version=version, num_blocks=num_blocks, conv_channels=conv_channels,Norm = "GN").eval()
    stable_mortal.load_state_dict(state['mortal'])
    if 'current_dqn' in state:
        stable_dqn = build_head('value', version).eval()
        stable_dqn.load_state_dict(state['current_dqn'])
    else:
        stable_dqn = build_head('policy', version).eval()
        stable_dqn.load_state_dict(state['policy_net'])
    if baseline_cfg['enable_compile']:
        stable_mortal.compile()
        stable_dqn.compile()

    return MortalEngine(
        stable_mortal,
        stable_dqn,
        is_oracle = False,
        version = version,
        device = device,
        enable_amp = True,
        enable_rule_based_agari_guard = True,
        name = 'baseline',
    )

class TestPlayer:
    def __init__(self):
        self.baseline_engine = load_frozen(config['baseline']['test'])
        self.chal_version = config['control']['version']
        self.log_dir = path.abspath(config['test_play']['log_dir'])

    def test_play(self, seed_count, mortal, dqn, device):
        torch.backends.cudnn.benchmark = False
        engine_chal = MortalEngine(
            mortal,
            dqn,
            is_oracle = False,
            version = self.chal_version,
            device = device,
            enable_amp = True,
            name = 'mortal',
        )

        if path.isdir(self.log_dir):
            shutil.rmtree(self.log_dir)

        env = OneVsThree(
            disable_progress_bar = False,
            log_dir = self.log_dir,
        )
        env.py_vs_py(
            challenger = engine_chal,
            champion = self.baseline_engine,
            seed_start = (10000, 0x2000),
            seed_count = seed_count,
        )

        stat = Stat.from_dir(self.log_dir, 'mortal')
        torch.backends.cudnn.benchmark = config['control']['enable_cudnn_benchmark']
        return stat

class TrainPlayer:
    def __init__(self):
        self.baseline_engine = load_frozen(config['baseline']['train'])

        profile = os.environ.get('TRAIN_PLAY_PROFILE', 'default')
        logging.info(f'using profile {profile}')
        cfg = config['train_play'][profile]
        self.chal_version = config['control']['version']
        self.log_dir = path.abspath(cfg['log_dir'])
        # behaviour sidecars, one per log under the same name (D46)
        self.behaviour_dir = path.abspath(cfg.get('behaviour_dir', self.log_dir + '_behaviour'))
        self.train_key = secrets.randbits(64)
        self.train_seed = 10000
        self.head = head_kind()
        if self.head == 'value':
            # Equim's exploration (ValueEngine)
            self.explore = {k: cfg[k] for k in ('boltzmann_epsilon', 'boltzmann_temp', 'top_p')}
        else:
            self.explore = {'explore_rate': cfg['explore_rate']}
        self.seed_count = cfg['games'] // 4
        self.repeats = cfg['repeats']

        self.repeat_counter = 0

    def train_play(self, mortal, dqn, device):
        torch.backends.cudnn.benchmark = False
        engine_chal = (ValueEngine if self.head == 'value' else MortalEngine)(
            mortal,
            dqn,
            is_oracle = False,
            version = self.chal_version,
            device = device,
            enable_amp = True,
            name = 'trainee',
            **self.explore,
        )

        for d in (self.log_dir, self.behaviour_dir):
            if path.isdir(d):
                shutil.rmtree(d)

        env = OneVsThree(
            disable_progress_bar = False,
            log_dir = self.log_dir,
            behaviour_dir = self.behaviour_dir,
        )
        rankings = env.py_vs_py(
            challenger = engine_chal,
            champion = self.baseline_engine,
            seed_start = (self.train_seed, self.train_key),
            seed_count = self.seed_count,
        )
        self.repeat_counter += 1
        if self.repeat_counter == self.repeats:
            self.train_seed += self.seed_count
            self.repeat_counter = 0

        rankings = np.array(rankings)
        file_list = list(map(lambda p: path.join(self.log_dir, p), os.listdir(self.log_dir)))

        torch.backends.cudnn.benchmark = config['control']['enable_cudnn_benchmark']
        return rankings, file_list
