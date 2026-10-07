"""Observation v5 = v4 with the single-player tables (rows 889-1011) zeroed (SP ablation,
docs/design/sp-ablation-handoff.md). Checked on the encoder, the SL loader, the model and the
arena, where a v5 engine plays against v4 ones."""
import gzip
import json

import numpy as np
import pytest

SP_START = 889


def test_encoder_v5_is_v4_without_sp(arena_run):
    from libriichi.state import PlayerState

    _, logs, _, _ = arena_run
    checked = sp_filled = 0
    for path in logs[:4]:
        with gzip.open(path, 'rt') as f:
            events = [json.loads(line) for line in f]
        states = [PlayerState(i) for i in range(4)]
        for ev in events:
            ev.pop('meta', None)
            line = json.dumps(ev)
            for st in states:
                cans = st.update(line)
                if not cans.can_act:
                    continue
                for at_kan_select in ([False, True] if cans.can_ankan or cans.can_kakan else [False]):
                    obs4, mask4 = st.encode_obs(4, at_kan_select)
                    obs5, mask5 = st.encode_obs(5, at_kan_select)
                    assert obs5.shape == obs4.shape == (1012, 34)
                    assert np.array_equal(obs5[:SP_START], obs4[:SP_START])
                    assert not obs5[SP_START:].any()
                    assert np.array_equal(mask5, mask4)
                    checked += 1
                    sp_filled += bool(obs4[SP_START:].any())
    assert checked > 1000
    assert sp_filled > checked // 4  # not vacuous: v4 fills the rows at most decisions


def test_loader_v5_is_v4_without_sp(arena_run):
    """The SL path (GameplayLoader) featurizes v5 the same way, oracle channels included."""
    from libriichi.dataset import GameplayLoader

    _, logs, _, _ = arena_run
    games = {
        v: GameplayLoader(version=v, oracle=True, trust_seed=True).load_gz_log_files(logs[:2])
        for v in (4, 5)
    }
    n = 0
    for file4, file5 in zip(games[4], games[5]):
        for g4, g5 in zip(file4, file5):
            assert g4.take_actions() == g5.take_actions()
            assert all(np.array_equal(a, b) for a, b in zip(g4.take_masks(), g5.take_masks()))
            assert all(np.array_equal(a, b) for a, b in zip(g4.take_invisible_obs(), g5.take_invisible_obs()))
            for o4, o5 in zip(g4.take_obs(), g5.take_obs()):
                assert np.array_equal(o5[:SP_START], o4[:SP_START]) and not o5[SP_START:].any()
                n += 1
    assert n > 500


def test_brain_v5_builds_v4_modules():
    """Same input shape, so the same seed gives the same initial weights: the matched pair
    differs only in the SP inputs."""
    import torch
    from model import Brain, DQN

    nets = {}
    for v in (4, 5):
        torch.manual_seed(1)
        nets[v] = (Brain(version=v, conv_channels=32, num_blocks=1, Norm='GN').state_dict(),
                   DQN(version=v).state_dict())
    for sd4, sd5 in zip(nets[4], nets[5]):
        assert sd4.keys() == sd5.keys()
        assert all(torch.equal(sd4[k], sd5[k]) for k in sd4)


@pytest.mark.parametrize('chal_version', [5, 4])
def test_arena_mixed_versions(chal_version):
    """Each agent encodes with its own engine's version: the v5 side sees zeroed SP rows,
    the v4 side doesn't."""
    import torch
    from engine import MortalEngine
    from libriichi.arena import OneVsThree
    from model import Brain, CategoricalPolicy

    class Recorder(MortalEngine):
        def react_batch(self, obs, masks, invisible_obs):
            self.sp_filled += sum(bool(o[SP_START:].any()) for o in obs)
            self.seen += len(obs)
            return super().react_batch(obs, masks, invisible_obs)

    def engine(version, name):
        torch.manual_seed(0)
        e = Recorder(Brain(version=version, conv_channels=32, num_blocks=1, Norm='GN').eval(),
                     CategoricalPolicy().eval(), is_oracle=False, version=version,
                     device=torch.device('cpu'), enable_amp=False,
                     enable_rule_based_agari_guard=True, name=name)
        e.sp_filled = e.seen = 0
        return e

    chal = engine(chal_version, 'challenger')
    cham = engine(9 - chal_version, 'champion')
    records = OneVsThree(disable_progress_bar=True).py_vs_py_games(
        challenger=chal, champion=cham, seed_start=(0, 0x5eed), seed_count=1)
    assert len(records) == 4
    assert sorted(r[1] for r in records) == [0, 1, 2, 3]  # the challenger takes each seat once
    v5, v4 = (chal, cham) if chal_version == 5 else (cham, chal)
    assert v5.seen > 0 and v5.sp_filled == 0
    assert v4.seen > 0 and v4.sp_filled > 0
