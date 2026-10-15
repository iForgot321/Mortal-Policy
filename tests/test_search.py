"""Search (06): branch-and-rollout from arena seeds (the pilot's oracle stage) and the GRP
valuation of a rollout's kyoku result."""
import gzip
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from conftest import make_engine  # noqa: E402
from lab.arena import ShantenGreedyEngine  # noqa: E402
from libriichi.search import Beliefs, rollouts, root_observations, self_play  # noqa: E402
from search.value import GRP_STATE_FILE, GrpValue, next_kyoku, start_row  # noqa: E402

KEY = 0x5ea7c4


def kyokus(path):
    """The game's kyokus as lists of mjai events, `start_kyoku` to `end_kyoku`."""
    with gzip.open(path, 'rt') as f:
        events = [json.loads(line) for line in f]
    out = []
    for ev in events:
        if ev['type'] == 'start_kyoku':
            out.append([])
        if out and (not out[-1] or out[-1][-1]['type'] != 'end_kyoku'):
            out[-1].append(ev)
    return events[0]['seed'], out


def plain(ev):
    return {k: v for k, v in ev.items() if k != 'meta'}


def roots_of(seed, kyoku, every=1):
    """Each own-turn discard or riichi: (root, logged action)."""
    lines = [json.dumps(ev) for ev in kyoku]
    out = []
    for i, ev in enumerate(kyoku):
        if ev['type'] in ('dahai', 'reach'):
            out.append(((tuple(seed), lines, i), action_of(ev)))
    return out[::every]


def action_of(ev):
    return 37 if ev['type'] == 'reach' else TILES.index(ev['pai'])


TILES = [f'{n}{s}' for s in 'mps' for n in range(1, 10)] + list('ESWNPFC') + ['5mr', '5pr', '5sr']


@pytest.fixture(scope='module')
def greedy_games(tmp_path_factory):
    d = tmp_path_factory.mktemp('search-sg')
    scores = self_play(ShantenGreedyEngine(), (0, KEY), 3, str(d))
    return [str(d / f'{n}_{KEY}.json.gz') for n in range(3)], scores


def test_self_play_writes_one_log_per_seed(greedy_games):
    paths, scores = greedy_games
    assert all(os.path.exists(p) for p in paths)
    assert all(sum(s) == 100_000 for s in scores)


def test_logged_action_reproduces_every_kyoku(greedy_games):
    """S-2's test (06 §3) through the Python API: from the true latent history, the logged
    action and the same deterministic agent give back the logged kyoku."""
    paths, _ = greedy_games
    roots, jobs, want = [], [], []
    for p in paths:
        seed, kys = kyokus(p)
        for ky in kys:
            for root, action in roots_of(seed, ky, every=7):
                jobs.append((len(roots), action, None))
                roots.append(root)
                want.append([plain(e) for e in ky])
    assert len(jobs) > 50
    outs = rollouts(ShantenGreedyEngine(), roots, jobs, keep_logs=True)
    for out, w in zip(outs, want):
        assert [plain(json.loads(line)) for line in out.log] == w


def test_next_kyoku_follows_the_arena(greedy_games):
    """`next_kyoku` predicts each next `start_kyoku`, and the final scores at the game's end."""
    paths, scores = greedy_games
    for p, final in zip(paths, scores):
        seed, kys = kyokus(p)
        roots = [roots_of(seed, ky)[0] for ky in kys]
        outs = rollouts(ShantenGreedyEngine(), [r for r, _ in roots],
                        [(i, a, None) for i, (_, a) in enumerate(roots)])
        for i, (out, ky) in enumerate(zip(outs, kys)):
            row, end = next_kyoku(out, ky[0]['honba'])
            if i + 1 < len(kys):
                assert end is None and row == start_row(kys[i + 1][0])
            else:
                assert row is None and end == list(final)


def test_root_observations_match_the_decider(greedy_games):
    paths, _ = greedy_games
    seed, kys = kyokus(paths[0])
    pairs = roots_of(seed, kys[0])[:6]
    obs = root_observations([r for r, _ in pairs], 4)
    for ((_, lines, i), action), (seat, o, mask) in zip(pairs, obs):
        assert seat == json.loads(lines[i])['actor']
        assert o.shape == (1012, 34) and mask[action]


def test_redeal_keeps_the_root_and_varies_the_rest(greedy_games):
    paths, _ = greedy_games
    seed, kys = kyokus(paths[0])
    (root, action), = roots_of(seed, kys[0])[5:6]
    i = root[2]
    jobs = [(0, action, None)] + [(0, action, s) for s in range(8)]
    outs = rollouts(ShantenGreedyEngine(), [root], jobs, keep_logs=True)
    logs = [[plain(json.loads(line)) for line in o.log] for o in outs]
    assert all(log[:i] == [plain(e) for e in kys[0][:i]] for log in logs)
    assert len({json.dumps(log) for log in logs[1:]}) > 1


def test_rollouts_refuse_a_non_branch_action(greedy_games):
    paths, _ = greedy_games
    seed, kys = kyokus(paths[0])
    (root, _), = roots_of(seed, kys[0])[:1]
    with pytest.raises(Exception, match='discard or riichi'):
        rollouts(ShantenGreedyEngine(), [root], [(0, 43, None)])


def test_calls_and_kans_replay_and_reproduce(tmp_path):
    """A small random network calls and kans freely. Its kyokus replay through every chi, pon
    and kan, the logged action reproduces the kyoku, and redealt branches play to the end.

    Self-play and rollouts batch the network differently, and a random network has near-ties
    that another CPU's kernels may flip, so reproduction is checked at 95% here (100% on the
    workstation). The deterministic agent above is checked exactly."""
    import torch
    from model import Brain, CategoricalPolicy
    torch.manual_seed(0)
    brain = Brain(version=4, conv_channels=32, num_blocks=1, Norm='GN').eval()
    engine = make_engine(brain, CategoricalPolicy().eval(), 0.0, 'bp')
    self_play(engine, (0, KEY), 3, str(tmp_path))
    roots, jobs, want, types = [], [], [], set()
    for n in range(3):
        seed, kys = kyokus(str(tmp_path / f'{n}_{KEY}.json.gz'))
        for ky in kys:
            types |= {e['type'] for e in ky}
            for root, action in roots_of(seed, ky, every=3):
                jobs += [(len(roots), action, None), (len(roots), action, len(roots))]
                roots.append(root)
                want.append([plain(e) for e in ky])
    assert {'chi', 'pon', 'daiminkan', 'kakan'} <= types
    outs = rollouts(engine, roots, jobs, keep_logs=True)
    same = [[plain(json.loads(line)) for line in out.log] == w for out, w in zip(outs[::2], want)]
    assert sum(same) >= 0.95 * len(same)
    for out in outs[1::2]:
        assert json.loads(out.log[-1])['type'] == 'end_kyoku'
        assert sum(out.scores) + 1000 * out.kyotaku_left == 100_000


def test_beliefs_sample_the_seats_view_and_roll_out(greedy_games):
    """Particles come from the deciding seat's view only: the same particles from the full log
    and from the log with the other hands and draws hidden. Each rolls out to the end."""
    paths, _ = greedy_games
    seed, kys = kyokus(paths[0])
    pairs = roots_of(seed, kys[0])[4:7]

    def hide(lines, seat):
        out = []
        for line in lines:
            ev = json.loads(line)
            if ev['type'] == 'start_kyoku':
                ev['tehais'] = [h if s == seat else ['?'] * 13 for s, h in enumerate(ev['tehais'])]
            elif ev['type'] == 'tsumo' and ev['actor'] != seat:
                ev['pai'] = '?'
            out.append(json.dumps(ev))
        return out

    seats = [json.loads(lines[i])['actor'] for (_, lines, i), _ in pairs]
    full = Beliefs([(lines, i) for (_, lines, i), _ in pairs], 8, seed=1)
    seen = Beliefs([(hide(lines, s), i) for ((_, lines, i), _), s in zip(pairs, seats)], 8, seed=1)
    assert full.seats() == seats == seen.seats()
    assert full.counts() == [(8, 8)] * len(pairs)
    for r in range(len(pairs)):
        assert all(full.tiles(r, p) == seen.tiles(r, p) for p in range(8))
    jobs = [(r, p, a) for r, (_, a) in enumerate(pairs) for p in range(8)]
    for out in full.rollouts(ShantenGreedyEngine(), jobs):
        assert sum(out.scores) + 1000 * out.kyotaku_left == 100_000


@pytest.mark.skipif(not os.path.exists(GRP_STATE_FILE), reason='needs the frozen GRP')
def test_grp_value_is_a_utility():
    """Mid-game values lie inside the utility range; a game-ending result is its exact utility."""
    class Out:
        def __init__(self, kyoku, scores, can_renchan=False, has_hora=True, abortive=False):
            self.kyoku, self.scores, self.kyotaku_left = kyoku, scores, 0
            self.can_renchan, self.has_hora, self.has_abortive_ryukyoku = can_renchan, has_hora, abortive

    value = GrpValue()
    e1 = [[0, 0, 0, 2.5, 2.5, 2.5, 2.5]]
    mid = Out(0, [33000, 23000, 22000, 22000])
    end = Out(7, [10000, 40000, 30000, 20000])
    v = value([(e1, mid, 0, 0), (e1, mid, 0, 1), (e1, end, 0, 0), (e1, end, 0, 1)])
    assert -1.3 < v[1] < v[0] < 0.9
    assert v[2] == pytest.approx(-1.3) and v[3] == pytest.approx(0.9)


def test_hand_examples_and_root_views(greedy_games):
    """The hand model's data: true concealed hands of the other seats at each decision, and a
    root's view from the seat's own events equals the full replay's observation."""
    from libriichi.search import hand_examples, root_views
    paths, _ = greedy_games
    obs, hands, akas, unseen = hand_examples(paths[0])
    assert obs.shape[1:] == (1012, 34) and hands.shape == (len(obs), 3, 34) and akas.shape == (len(obs), 9)
    assert set(hands.sum(-1).ravel()) <= {1, 4, 7, 10, 13}
    assert (hands.sum(1) <= unseen).all()
    seed, kys = kyokus(paths[0])
    pairs = roots_of(seed, kys[0])[:6]
    views = root_views([(lines, i) for (_, lines, i), _ in pairs])
    for (root, _), (seat, o, _, sizes, _), (s2, o2, _) in zip(pairs, views, root_observations([r for r, _ in pairs], 4)):
        assert seat == s2 and (o == o2).all() and set(sizes) <= {1, 4, 7, 10, 13}


def test_hand_model_density_is_exact():
    """A sample's log q equals the training path's log-probability of the same hands, and every
    sample respects the hand sizes and the unseen copies."""
    import numpy as np
    import torch
    from model import Brain
    from search.handmodel import HandModel
    torch.manual_seed(0)
    model = HandModel(Brain(version=4, conv_channels=32, num_blocks=1, Norm='GN'), hidden=32).eval()
    obs = torch.rand(1, 1012, 34)
    unseen = torch.tensor([3, 2, 4, 1, 3] * 6 + [2, 0, 4, 1])
    sizes = torch.tensor([13, 10, 7])
    hands, logq = model.sample(obs, unseen, sizes, 32)
    assert (hands.sum(-1) == sizes).all() and (hands.sum(1) <= unseen).all()
    with torch.no_grad():
        lp = model.log_prob(obs.expand(32, -1, -1), hands, unseen.expand(32, -1)).sum(-1)
    assert np.allclose(lp.numpy(), logq.numpy(), atol=1e-4)


def test_waits_and_shanten():
    """123m 456m 789m 11p 23s waits on 1s and 4s; one tile off, it isn't tenpai."""
    from libriichi.search import shanten, waits
    hand = [0] * 34
    for k in list(range(9)) + [9, 9, 19, 20]:
        hand[k] += 1
    assert shanten([hand]) == [0]
    assert [k for k, w in enumerate(waits([hand])[0]) if w] == [18, 21]
    hand[20] -= 1
    hand[30] += 1
    assert shanten([hand]) == [1] and not any(waits([hand])[0])
