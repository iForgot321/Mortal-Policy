"""The value of a rollout that ends a kyoku: expected utility in u units under
the frozen GRP (05 §3.2, D37), the units every reward, critic and search
bootstrap shares. Comparing two branches of one root, this is the difference of
their K-surrogate rewards for the kyoku.
"""
import os

import numpy as np
import torch

from lab import utility

from . import ROOT

GRP_STATE_FILE = os.path.join(ROOT, 'workdir', 'grp-v2', 'grp-v2-frozen.pth')
HANCHAN = 8


def grp_row(kyoku, honba, kyotaku, scores):
    """A kyoku start as the GRP reads it (libriichi/src/dataset/grp.rs):
    [grand kyoku (E1 = 0), honba, kyotaku, scores / 10000 by seat]."""
    return [kyoku, honba, kyotaku, *(s / 10000 for s in scores)]


def start_row(start_kyoku):
    """`grp_row` of a `start_kyoku` event."""
    ev = start_kyoku
    grand = 'ESW'.index(ev['bakaze']) * 4 + ev['kyoku'] - 1
    return grp_row(grand, ev['honba'], ev['kyotaku'], ev['scores'])


def top_seat(scores):
    """The first seat with the highest score, as the arena breaks ties."""
    return max(range(4), key=lambda i: (scores[i], -i))


def ranks(scores):
    """0-based rank of each seat; ties go to the lower seat (rankings.rs)."""
    order = sorted(range(4), key=lambda i: (-scores[i], i))
    out = [0] * 4
    for rank, seat in enumerate(order):
        out[seat] = rank
    return out


def next_kyoku(outcome, honba, length=HANCHAN):
    """What follows a kyoku, as `Game::poll` plays it (arena/game.rs): the
    next kyoku's `grp_row`, or the final scores (kyotaku to the top) when the
    game ends there. Returns `(row, final_scores)`, one of them None."""
    kyoku, scores, kyotaku = outcome.kyoku, list(outcome.scores), outcome.kyotaku_left
    in_renchan = False
    if any(s < 0 for s in scores):
        return None, final_scores(scores, kyotaku)
    if outcome.has_abortive_ryukyoku:
        honba += 1
    elif not outcome.can_renchan:
        kyoku += 1
        honba = 0 if outcome.has_hora else honba + 1
    else:
        oya = kyoku % 4
        if kyoku >= length - 1 and scores[oya] >= 30000 and top_seat(scores) == oya:
            return None, final_scores(scores, kyotaku)
        in_renchan = True
        honba += 1
    if kyoku >= length + 4 or kyoku >= length and not in_renchan and max(scores) >= 30000:
        return None, final_scores(scores, kyotaku)
    return grp_row(kyoku, honba, kyotaku, scores), None


def final_scores(scores, kyotaku):
    scores = list(scores)
    scores[top_seat(scores)] += kyotaku * 1000
    return scores


class GrpValue:
    """Expected utility of each seat after a kyoku, from the frozen GRP."""

    def __init__(self, state_file=GRP_STATE_FILE, u=None):
        from model import GRP
        self.grp = GRP(hidden_size=64, num_layers=2)
        state = torch.load(state_file, weights_only=True, map_location='cpu')
        self.grp.load_state_dict(state['model'])
        self.grp.eval()
        self.u = torch.as_tensor(utility() if u is None else u, dtype=torch.float64)

    def __call__(self, items):
        """`items`: (history, outcome, honba, seat) per rollout, where
        `history` is the game's `grp_row`s up to and including the kyoku the
        rollout played. Returns the seat's expected utility after it, in u."""
        values = np.empty(len(items))
        seqs, idx = [], []
        for i, (history, outcome, honba, seat) in enumerate(items):
            row, final = next_kyoku(outcome, honba)
            if final is not None:
                values[i] = float(self.u[ranks(final)[seat]])
            else:
                seqs.append(torch.tensor([*history, row], dtype=torch.float64))
                idx.append(i)
        if seqs:
            with torch.inference_mode():
                matrix = self.grp.calc_matrix(self.grp(seqs))  # (N, seat, rank)
            seats = torch.tensor([items[i][3] for i in idx])
            ev = matrix[torch.arange(len(idx)), seats] @ self.u
            values[idx] = ev.numpy()
        return values
