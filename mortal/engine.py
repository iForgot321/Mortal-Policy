import json
import traceback
import torch
import numpy as np
from torch.distributions import Normal, Categorical
from typing import *

class MortalEngine:
    def __init__(
        self,
        brain,
        dqn,
        is_oracle,
        version,
        device = None,
        stochastic_latent = False,
        enable_amp = False,
        enable_quick_eval = True,
        enable_rule_based_agari_guard = False,
        name = 'NoName',
        explore_rate = 0,
    ):
        self.engine_type = 'mortal'
        self.device = device or torch.device('cpu')
        assert isinstance(self.device, torch.device)
        self.brain = brain.to(self.device).eval()
        self.dqn = dqn.to(self.device).eval()   # 
        self.is_oracle = is_oracle
        self.version = version
        self.stochastic_latent = stochastic_latent

        self.enable_amp = enable_amp
        self.enable_quick_eval = enable_quick_eval
        self.enable_rule_based_agari_guard = enable_rule_based_agari_guard
        self.name = name
        self.explore_rate = explore_rate

    def react_batch(self, obs, masks, invisible_obs):
        try:
            with (
                torch.autocast(self.device.type, enabled=self.enable_amp),
                torch.inference_mode(),
            ):
                return self._react_batch(obs, masks, invisible_obs)
        except Exception as ex:
            raise Exception(f'{ex}\n{traceback.format_exc()}')

    def _react_batch(self, obs, masks, invisible_obs):
        obs = torch.as_tensor(np.stack(obs, axis=0), device=self.device)
        masks = torch.as_tensor(np.stack(masks, axis=0), device=self.device)
        if invisible_obs is not None:
            invisible_obs = torch.as_tensor(np.stack(invisible_obs, axis=0), device=self.device)
        batch_size = obs.shape[0]

        match self.version:
            case 1:
                mu, logsig = self.brain(obs, invisible_obs)
                if self.stochastic_latent:
                    latent = Normal(mu, logsig.exp() + 1e-6).sample()
                else:
                    latent = mu
                q_out = self.dqn(latent, masks)
            case 2 | 3 | 4 | 5:
                phi = self.brain(obs)
                q_out = self.dqn(phi, masks)

        actions, is_greedy, behaviour_probs = self.sample(q_out)
        return actions.tolist(), q_out.tolist(), masks.tolist(), is_greedy.tolist(), behaviour_probs.tolist()

    def sample(self, q_out):
        """Actions drawn from `behaviour_dist(q_out)`, whether each was the
        greedy branch, and the probability of each drawn action (D46)."""
        batch_size = q_out.shape[0]
        if self.explore_rate > 0:
            is_greedy = torch.full((batch_size,), 1-self.explore_rate, device=self.device).bernoulli().to(torch.bool)
            actions = torch.where(is_greedy, q_out.argmax(-1), Categorical(probs=q_out).sample())
        else:
            is_greedy = torch.ones(batch_size, dtype=torch.bool, device=self.device)
            actions = q_out.argmax(-1)
        behaviour_probs = self.behaviour_dist(q_out).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        return actions, is_greedy, behaviour_probs

    def behaviour_dist(self, q_out):
        """The distribution `sample` draws from (05 §3.4, D46):
        (1 - ε)·1[a = argmax] + ε·π(a), with π normalized as Categorical does."""
        q = q_out.float()
        greedy = torch.nn.functional.one_hot(q.argmax(-1), q.shape[-1]).float()
        return (1 - self.explore_rate) * greedy + self.explore_rate * q / q.sum(-1, keepdim=True)

class ValueEngine(MortalEngine):
    """Arm V's actor (Equim's engine, a24399b^:mortal/engine.py). `dqn` is a DQN
    (Q at -inf where illegal): greedy on Q, except that with probability ε the
    action is drawn from softmax(Q/τ) restricted to its top-p nucleus."""
    def __init__(self, *args, boltzmann_epsilon=0, boltzmann_temp=1, top_p=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.boltzmann_epsilon = boltzmann_epsilon
        self.boltzmann_temp = boltzmann_temp
        self.top_p = top_p

    def sample(self, q_out):
        batch_size = q_out.shape[0]
        greedy = q_out.argmax(-1)
        if self.boltzmann_epsilon > 0:
            is_greedy = torch.full((batch_size,), 1-self.boltzmann_epsilon, device=self.device).bernoulli().to(torch.bool)
            sampled = self.nucleus(q_out).multinomial(1).squeeze(-1)
            actions = torch.where(is_greedy, greedy, sampled)
        else:
            is_greedy = torch.ones(batch_size, dtype=torch.bool, device=self.device)
            actions = greedy
        behaviour_probs = self.behaviour_dist(q_out).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        return actions, is_greedy, behaviour_probs

    def nucleus(self, q_out):
        """softmax(Q/τ) on its top-p nucleus, renormalized: the distribution
        Equim's `sample_top_p` draws from (the smallest set of most likely
        actions whose mass reaches p; p ≤ 0 is greedy)."""
        q = q_out.float()
        if self.top_p <= 0:
            return torch.nn.functional.one_hot(q.argmax(-1), q.shape[-1]).float()
        probs = (q / self.boltzmann_temp).softmax(-1)
        if self.top_p >= 1:
            return probs
        probs_sort, probs_idx = probs.sort(-1, descending=True)
        keep_sort = probs_sort.cumsum(-1) - probs_sort <= self.top_p
        keep = torch.zeros_like(keep_sort).scatter(-1, probs_idx, keep_sort)
        probs = probs * keep
        return probs / probs.sum(-1, keepdim=True)

    def behaviour_dist(self, q_out):
        """(1 - ε)·1[a = argmax Q] + ε·nucleus(a): what `sample` draws from."""
        q = q_out.float()
        greedy = torch.nn.functional.one_hot(q.argmax(-1), q.shape[-1]).float()
        return (1 - self.boltzmann_epsilon) * greedy + self.boltzmann_epsilon * self.nucleus(q_out)

class ExampleMjaiLogEngine:
    def __init__(self, name: str):
        self.engine_type = 'mjai-log'
        self.name = name
        self.player_ids = None

    def set_player_ids(self, player_ids: List[int]):
        self.player_ids = player_ids

    def react_batch(self, game_states):
        res = []
        for game_state in game_states:
            game_idx = game_state.game_index
            state = game_state.state
            events_json = game_state.events_json

            events = json.loads(events_json)
            assert events[0]['type'] == 'start_kyoku'

            player_id = self.player_ids[game_idx]
            cans = state.last_cans
            if cans.can_discard:
                tile = state.last_self_tsumo()
                res.append(json.dumps({
                    'type': 'dahai',
                    'actor': player_id,
                    'pai': tile,
                    'tsumogiri': True,
                }))
            else:
                res.append('{"type":"none"}')
        return res

    # They will be executed at specific events. They can be no-op but must be
    # defined.
    def start_game(self, game_idx: int):
        pass
    def end_kyoku(self, game_idx: int):
        pass
    def end_game(self, game_idx: int, scores: List[int]):
        pass
