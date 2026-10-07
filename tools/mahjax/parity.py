#!/usr/bin/env python
"""Mahjax rule-parity harness (03 §7, spike item 1).

Replays libriichi-generated mjai logs (the lab's arena logs) through Mahjax's
`red_mahjong` env and compares the two engines at every step:

- the legal-action set of every seat at every decision, projected onto
  Mortal's 46-action space (37 discards, riichi, chi low/mid/high, pon, kan,
  agari, ryukyoku, pass), plus the kan-select tile set. libriichi's side comes
  from `PlayerState.action_mask`, fed the same log;
- agari and exhaustive-draw deltas;
- every kyoku start (round, honba, kyotaku, dealer, scores, dealt hands) and
  the game end (whether the game ends, final scores).

Mahjax deals each wall from the key passed to `env.step`. The harness rebuilds
each kyoku's wall from the log (haipai, live and rinshan draws, dora and ura
markers; unseen slots get the remaining tiles) and passes it in place of that
key. Mahjax asks one seat at a time about a discard, in priority order. Each
question is answered from the log: the logged call or ron, otherwise PASS.

A disagreement that leaves the replay in step (a mask, a delta, a score) is
counted and the replay goes on. One that doesn't (the log's action is illegal
in Mahjax, or one engine ends the kyoku or game and the other doesn't) stops
that game.

Run in the `mahjax` conda env (JAX + mahjax), from the repo root:

    XLA_PYTHON_CLIENT_PREALLOCATE=false ~/miniconda3/envs/mahjax/bin/python \\
        tools/mahjax/parity.py --logs workdir/lab/<run>/logs --n 10000 \\
        --out workdir/mahjax-spike/parity-<name>.json
"""
import argparse
import collections
import glob
import gzip
import json
import os
import sys
import time
from multiprocessing import get_context

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, 'mortal'))

# ---------------------------------------------------------------- tiles

_SUIT = {'m': 0, 'p': 9, 's': 18}
_HONOR = {'E': 27, 'S': 28, 'W': 29, 'N': 30, 'P': 31, 'F': 32, 'C': 33}
_AKA = {'5mr': 34, '5pr': 35, '5sr': 36}
TILE_NAMES = [f'{n}{s}' for s in 'mps' for n in range(1, 10)] + list('ESWNPFC') + ['5mr', '5pr', '5sr']
# 136 tiles over the 37 kinds: three black and one red five per suit
FULL_SET = np.array([4] * 34 + [1, 1, 1])
FULL_SET[[4, 13, 22]] = 3


def tid(name):
    if name in _AKA:
        return _AKA[name]
    if name in _HONOR:
        return _HONOR[name]
    return _SUIT[name[1]] + int(name[0]) - 1


def deaka(t):
    return (4, 13, 22)[t - 34] if t >= 34 else t


# Mortal's action space (libriichi consts::ACTION_SPACE = 46)
M_RIICHI, M_CHI_LOW, M_CHI_MID, M_CHI_HIGH, M_PON, M_KAN, M_AGARI, M_RYUKYOKU, M_PASS = range(37, 46)
M_NAMES = TILE_NAMES + ['riichi', 'chi_low', 'chi_mid', 'chi_high', 'pon', 'kan', 'agari', 'ryukyoku', 'pass']

# Mahjax red_mahjong actions (mahjax/red_mahjong/action.py)
X_TSUMOGIRI, X_RIICHI, X_TSUMO, X_RON, X_PON, X_PON_RED, X_OPEN_KAN = 71, 72, 73, 74, 75, 76, 77
X_CHI_L, X_CHI_L_RED, X_CHI_M, X_CHI_M_RED, X_CHI_R, X_CHI_R_RED = 78, 79, 80, 81, 82, 83
X_PASS, X_KYUUSHU = 84, 85
X_SELFKAN = 37  # 37 + tile type: closed or added kan
X_CALLS = slice(X_RON, X_CHI_R_RED + 1)

ROUND_WIND = {'E': 0, 'S': 1, 'W': 2, 'N': 3}


def project(row, last_draw):
    """A Mahjax mask row (87) onto Mortal's 46 actions, and its kan-select tile set."""
    m = np.zeros(46, bool)
    m[:37] = row[:37]
    if row[X_TSUMOGIRI]:
        m[last_draw] = True
    m[M_RIICHI] = row[X_RIICHI]
    m[M_CHI_LOW] = row[X_CHI_L] | row[X_CHI_L_RED]
    m[M_CHI_MID] = row[X_CHI_M] | row[X_CHI_M_RED]
    m[M_CHI_HIGH] = row[X_CHI_R] | row[X_CHI_R_RED]
    m[M_PON] = row[X_PON] | row[X_PON_RED]
    selfkan = row[X_SELFKAN:X_SELFKAN + 34]
    m[M_KAN] = row[X_OPEN_KAN] | selfkan.any()
    m[M_AGARI] = row[X_TSUMO] | row[X_RON]
    m[M_RYUKYOKU] = row[X_KYUUSHU]
    m[M_PASS] = row[X_CALLS].any()
    kan = frozenset(np.flatnonzero(selfkan).tolist())
    return m, kan


def project_reaction(row, target):
    m, _ = project(row, 0)
    m[:37] = False
    kan = frozenset([deaka(target)]) if row[X_OPEN_KAN] else frozenset()
    return m, kan


# ---------------------------------------------------------------- log side

def read_log(path):
    with gzip.open(path, 'rt') as f:
        events = [json.loads(line) for line in f]
    for ev in events:
        ev.pop('meta', None)
    return events


def build_wall(kyoku_events):
    """Mahjax's 136-slot wall for one kyoku (red_mahjong/env.py layout): hands
    at 84 + 13 * seat, live draws from 83 down, dora/ura indicators at 9/8 and
    kan indicators at 9 - 2k / 8 - 2k, rinshan draws at 10 + k."""
    sk = kyoku_events[0]
    deck = np.full(136, -1, np.int16)

    def put(i, t):
        if deck[i] not in (-1, t):
            raise ValueError(f'wall slot {i} gets {TILE_NAMES[t]} and {TILE_NAMES[deck[i]]}')
        deck[i] = t

    for seat, tehai in enumerate(sk['tehais']):
        for j, p in enumerate(tehai):
            put(84 + 13 * seat + j, tid(p))
    put(9, tid(sk['dora_marker']))
    live, rinshan, n_dora, after_kan = 83, 0, 0, False
    for ev in kyoku_events[1:]:
        kind = ev['type']
        if kind == 'tsumo':
            if after_kan:
                put(10 + rinshan, tid(ev['pai']))
                rinshan += 1
                after_kan = False
            else:
                put(live, tid(ev['pai']))
                live -= 1
        elif kind in ('ankan', 'kakan', 'daiminkan'):
            after_kan = True
        elif kind == 'dora':
            n_dora += 1
            put(9 - 2 * n_dora, tid(ev['dora_marker']))
        elif kind == 'hora':
            for k, u in enumerate(ev.get('ura_markers') or []):
                put(8 - 2 * k, tid(u))
    counts = FULL_SET.copy()
    np.subtract.at(counts, deck[deck >= 0], 1)
    if (counts < 0).any():
        raise ValueError('more copies of a tile than exist')
    deck[deck < 0] = np.repeat(np.arange(37), counts)
    return deck.astype(np.int8)


def libriichi_decisions(events):
    """{(event index, seat): (mask46, kan-select tiles)} for every seat that
    can act after each event, from libriichi's PlayerState; and the set of
    (event index, seat) at which a seat is furiten after a discard or kakan."""
    from libriichi.state import PlayerState

    states = [PlayerState(i) for i in range(4)]
    out, furiten = {}, set()
    for i, ev in enumerate(events):
        line = json.dumps(ev)
        for s, st in enumerate(states):
            cans = st.update(line)
            if ev['type'] in ('dahai', 'kakan') and st.at_furiten:
                furiten.add((i, s))
            if cans.can_act:
                mask = np.array(st.action_mask(False), bool)
                kan = frozenset()
                if cans.can_kan:
                    kan = frozenset(np.flatnonzero(st.action_mask(True)[:34]).tolist())
                out[(i, s)] = (mask, kan)
    return out, furiten


def final_scores(events):
    """libriichi's final scores: last kyoku's start scores + its deltas − the
    riichi sticks it took, with the sticks left on the table to the top
    (arena/game.rs `commit`; ties to the lower seat)."""
    last = max(i for i, ev in enumerate(events) if ev['type'] == 'start_kyoku')
    sk = events[last]
    scores = np.array(sk['scores'], np.int64)
    kyotaku = sk['kyotaku']
    for ev in events[last + 1:]:
        if ev['type'] == 'reach_accepted':
            scores[ev['actor']] -= 1000
            kyotaku += 1
        elif ev['type'] in ('hora', 'ryukyoku') and ev.get('deltas'):
            scores += np.array(ev['deltas'])
            if ev['type'] == 'hora':
                kyotaku = 0
    if kyotaku:
        scores[int(np.argmax(scores))] += 1000 * kyotaku
    return scores


def prepare_game(path):
    """CPU side of one game (runs in a worker pool)."""
    events = read_log(path)
    starts = [i for i, ev in enumerate(events) if ev['type'] == 'start_kyoku']
    ends = starts[1:] + [len(events)]
    walls = [build_wall(events[a:b]) for a, b in zip(starts, ends)]
    return (path, events, walls, *libriichi_decisions(events))


# ---------------------------------------------------------------- replay

class Divergence(Exception):
    """The replay can't continue in step with the log."""


ENV_EVENTS = ('dora', 'reach_accepted')


class Game:
    def __init__(self, path, events, walls, lib, lib_furiten):
        self.path, self.events, self.walls, self.lib = path, events, walls, lib
        self.lib_furiten = lib_furiten
        self.mj = {}                 # (event index, seat) -> (mask46, kan set), Mahjax side
        self.reaction_seen = set()   # trigger events whose reactions Mahjax evaluated
        self.ignored = set()         # libriichi decisions with no Mahjax counterpart by design
        self.consumed = set()
        self.ptr = 1                 # next event to process (0 is start_game)
        self.kyoku = -1              # index of the current kyoku in self.walls
        self.in_kyoku = False
        self.trigger = None          # event that opened the current own-turn decision
        self.react_trigger = None    # dahai/kakan/ankan whose reactions are pending
        self.pending_snapshot = None
        self.pending_hora = None     # (event index, fan, fu) to check against rewards
        self.status = 'running'
        self.divergence = None       # (event index, kind, detail)
        self.notes = []              # non-fatal disagreements: (kind, event index, detail)
        self.steps = 0

    # -- helpers
    def ev(self, i):
        return self.events[i]

    def skip_env(self):
        while self.ptr < len(self.events) and (
                self.events[self.ptr]['type'] in ENV_EVENTS or self.ptr in self.consumed):
            self.ptr += 1

    def note(self, kind, idx, detail):
        self.notes.append((kind, idx, detail))

    def next_wall(self):
        return self.walls[min(self.kyoku + 1, len(self.walls) - 1)]

    # -- one Mahjax state -> one action
    def decide(self, f):
        self.steps += 1
        if self.pending_snapshot is not None:
            self.snapshot_reactions(f)
        if self.pending_hora is not None:
            idx, fan, fu = self.pending_hora
            self.pending_hora = None
            got = (np.asarray(f['rewards']) * 100).round().astype(int).tolist()
            want = self.ev(idx)['deltas']
            if got != want:
                self.note('hora_deltas', idx, {'libriichi': want, 'mahjax': got, 'mahjax_fan_fu': [fan, fu]})
        if f['terminated']:
            return self.game_end(f)
        fresh = int(f['round_step']) == 0
        if fresh and self.in_kyoku:
            self.end_kyoku(f)
        if not fresh and not self.in_kyoku:
            raise Divergence(('mahjax_continues_kyoku', self.ptr, None))
        if fresh:
            self.start_kyoku(f)

        mask = f['mask']
        if mask[X_KYUUSHU] and mask.sum() == 1:
            return self.special_draw()
        if mask[X_PASS]:
            return self.answer_query(f)
        return self.own_turn(f)

    def snapshot_reactions(self, f):
        trig, actor = self.pending_snapshot
        self.pending_snapshot = None
        self.reaction_seen.add(trig)
        if f['terminated'] or int(f['round_step']) == 0 or not f['mask'][X_PASS]:
            return  # nobody could react
        for s in range(4):
            if s != actor and f['rows'][s][X_CALLS].any():
                self.mj[(trig, s)] = project_reaction(f['rows'][s], int(f['target']))

    def start_kyoku(self, f):
        self.skip_env()
        if self.ptr < len(self.events) and self.ev(self.ptr)['type'] == 'end_kyoku':
            self.ptr += 1
        sk = self.ev(self.ptr)
        if sk['type'] != 'start_kyoku':
            raise Divergence(('mahjax_extra_kyoku', self.ptr, sk['type']))
        self.kyoku += 1
        want = {
            'round': ROUND_WIND[sk['bakaze']] * 4 + sk['kyoku'] - 1,
            'honba': sk['honba'], 'kyotaku': sk['kyotaku'], 'dealer': sk['oya'],
            'scores': sk['scores'],
        }
        got = {
            'round': int(f['round']), 'honba': int(f['honba']), 'kyotaku': int(f['kyotaku']),
            'dealer': int(f['dealer']), 'scores': (np.asarray(f['score']) * 100).tolist(),
        }
        for k in want:
            if want[k] != got[k]:
                if k in ('round', 'dealer'):
                    raise Divergence(('kyoku_start_' + k, self.ptr, {'libriichi': want, 'mahjax': got}))
                self.note('kyoku_start_' + k, self.ptr, {'libriichi': want[k], 'mahjax': got[k]})
        # dealt hands (the dealer's includes the first draw, checked at its tsumo)
        first = self.ev(self.ptr + 1)
        for s in range(4):
            hand = np.zeros(37, int)
            np.add.at(hand, [tid(p) for p in sk['tehais'][s]], 1)
            if s == sk['oya'] and first['type'] == 'tsumo':
                hand[tid(first['pai'])] += 1
            if not np.array_equal(hand, np.asarray(f['hwr'][s])):
                raise Divergence(('deal', self.ptr, s))
        self.ptr += 1
        self.in_kyoku = True

    def end_kyoku(self, f):
        """Mahjax has ended the kyoku: the log must have no actions left in it."""
        self.in_kyoku = False
        self.skip_env()
        ev = self.ev(self.ptr)
        if ev['type'] == 'ryukyoku':
            if not f['terminated']:
                got = (np.asarray(f['rewards']) * 100).round().astype(int).tolist()
                if ev.get('deltas') and got != ev['deltas']:
                    self.note('ryukyoku_deltas', self.ptr, {'libriichi': ev['deltas'], 'mahjax': got})
            self.ptr += 1
            self.skip_env()
            ev = self.ev(self.ptr)
        if ev['type'] != 'end_kyoku':
            raise Divergence(('mahjax_ended_kyoku_early', self.ptr, ev['type']))

    def game_end(self, f):
        if self.in_kyoku:
            self.end_kyoku(f)
        self.skip_env()
        if self.ev(self.ptr)['type'] == 'end_kyoku':
            self.ptr += 1
        if self.ev(self.ptr)['type'] != 'end_game':
            raise Divergence(('mahjax_ended_game_early', self.ptr, self.ev(self.ptr)['type']))
        want = final_scores(self.events).tolist()
        got = (np.asarray(f['score']) * 100).tolist()
        if want != got:
            self.note('final_scores', self.ptr, {'libriichi': want, 'mahjax': got})
        self.status = 'done'
        return None

    def special_draw(self):
        """Mahjax's pending abortive draw (four winds/riichi/kans, triple ron)
        takes a KYUUSHU step to deal the next kyoku."""
        self.skip_env()
        ev = self.ev(self.ptr)
        if ev['type'] == 'tsumo' and self.ev(self.ptr + 1)['type'] == 'ryukyoku':
            # libriichi's four-riichi draw comes one step late: it deals the next
            # tile and discards that player's decision (arena/board.rs), where
            # Tenhou and Mahjax end the kyoku when the fourth riichi is accepted
            self.note('four_riichi_after_a_draw', self.ptr, None)
            self.ignored.add(self.ptr)
            self.ptr += 1
            ev = self.ev(self.ptr)
        if ev['type'] != 'ryukyoku':
            rons = [i for i in self.consumed if self.ev(i)['type'] == 'hora' and i > self.react_trigger]
            if len(rons) >= 3:
                # Tenhou (and Mahjax) abort a triple ron; libriichi's arena pays all three.
                # Mahjax's rollback of the first two shows up as the third ron's reward.
                self.notes = [n for n in self.notes if not (n[0] == 'hora_deltas' and n[1] in rons)]
                raise Divergence(('triple_ron', self.react_trigger, None))
            raise Divergence(('mahjax_abortive_draw', self.ptr, ev['type']))
        return X_KYUUSHU

    def own_turn(self, f):
        cp = int(f['cp'])
        last_draw = int(f['last_draw'])
        self.skip_env()
        ev = self.ev(self.ptr)
        if ev['type'] == 'tsumo':
            if ev['actor'] != cp or tid(ev['pai']) != last_draw:
                raise Divergence(('draw', self.ptr, {
                    'libriichi': [ev['actor'], ev['pai']],
                    'mahjax': [cp, TILE_NAMES[last_draw] if last_draw >= 0 else None]}))
            self.trigger = self.ptr
            self.ptr += 1
            self.skip_env()
            ev = self.ev(self.ptr)

        kind = ev['type']
        prev = self.ev(self.ptr - 1)
        if kind == 'ryukyoku' and prev['type'] == 'tsumo' and prev['actor'] == cp:
            action = X_KYUUSHU  # kyuushu kyuuhai
        elif kind in ('ryukyoku', 'end_kyoku'):
            raise Divergence(('mahjax_continues_kyoku', self.ptr, kind))
        elif ev.get('actor') != cp:
            raise Divergence(('turn', self.ptr, {'libriichi': [kind, ev.get('actor')], 'mahjax': cp}))
        elif kind == 'dahai':
            t = tid(ev['pai'])
            same = t == last_draw
            action = X_TSUMOGIRI if same and (ev['tsumogiri'] or f['hwr'][cp][t] == 1) else t
        elif kind == 'reach':
            action = X_RIICHI
        elif kind == 'ankan':
            action = X_SELFKAN + deaka(tid(ev['consumed'][0]))
        elif kind == 'kakan':
            action = X_SELFKAN + deaka(tid(ev['pai']))
        elif kind == 'hora' and ev['target'] == cp:
            action = X_TSUMO
            self.pending_hora = (self.ptr, int(f['fan'][cp][0]), int(f['fu'][cp][0]))
        else:
            raise Divergence(('unexpected_event', self.ptr, kind))
        self.mj[(self.trigger, cp)] = project(f['mask'], last_draw)
        if not f['mask'][action]:
            raise Divergence(('illegal_in_mahjax', self.ptr, {'event': ev, 'action': int(action)}))
        if kind == 'reach':
            self.trigger = self.ptr
        if kind in ('dahai', 'kakan', 'ankan'):
            self.react_trigger = self.ptr
            self.pending_snapshot = (self.ptr, cp)
        self.consumed.add(self.ptr)
        self.ptr += 1
        return action

    def responses(self, trig):
        """The log's reactions to event `trig`: ron actors (event index each),
        the call that took it (if any), and whether a ryukyoku follows."""
        discarder = self.ev(trig)['actor']
        rons, call, ryukyoku = {}, None, False
        j = trig + 1
        while self.ev(j)['type'] in ENV_EVENTS:
            j += 1
        while self.ev(j)['type'] == 'hora' and self.ev(j)['target'] == discarder:
            rons[self.ev(j)['actor']] = j
            j += 1
        if not rons:
            kind = self.ev(j)['type']
            if kind in ('chi', 'pon', 'daiminkan'):
                call = j
            elif kind == 'ryukyoku':
                ryukyoku = True
        return rons, call, ryukyoku

    def answer_query(self, f):
        cp = int(f['cp'])
        trig = self.react_trigger
        rons, call, ryukyoku = self.responses(trig)
        action = X_PASS
        if (ryukyoku and f['mask'][X_RON] and self.ev(trig + 1)['type'] == 'ryukyoku'
                and tiles_left(self.events, trig + 1) > 0 and int(f['n_kan'].sum()) < 4):
            # Tenhou logs a triple ron (三家和) as an abortive ryukyoku right after
            # the discard: every seat that could ron declared it
            action = X_RON
            idx = None
        elif cp in rons:
            action = X_RON
            idx = rons[cp]
            self.pending_hora = (idx, int(f['fan'][cp][0]), int(f['fu'][cp][0]))
        elif call is not None and self.ev(call)['actor'] == cp:
            idx = call
            action = self.call_action(self.ev(call))
            if not f['mask'][action] and f['rows'][cp][action]:
                # the call waits until the claims above it (a ron, this seat's
                # own included) are declined; Mahjax offers it after that
                action = X_PASS
        if not f['mask'][action]:
            raise Divergence(('illegal_in_mahjax', trig, {'seat': cp, 'action': int(action),
                                                          'legal': np.flatnonzero(f['mask']).tolist()}))
        if action != X_PASS and idx is not None:
            self.consumed.add(idx)
            if action != X_RON:
                self.trigger = idx
        return action

    @staticmethod
    def call_action(ev):
        kind = ev['type']
        consumed = [tid(p) for p in ev['consumed']]
        red = any(t >= 34 for t in consumed)
        if kind == 'pon':
            return X_PON_RED if red else X_PON
        if kind == 'daiminkan':
            return X_OPEN_KAN
        called = deaka(tid(ev['pai']))
        lo = min(deaka(t) for t in consumed)
        if called < lo:
            return X_CHI_L_RED if red else X_CHI_L
        if called > max(deaka(t) for t in consumed):
            return X_CHI_R_RED if red else X_CHI_R
        return X_CHI_M_RED if red else X_CHI_M

    # -- after the replay
    def compare_decisions(self):
        """Per-decision comparison of the two engines' legal sets."""
        limit = self.divergence[1] if self.divergence else len(self.events)
        out = []
        keys = set(k for k in self.lib if k[0] < limit and k[0] not in self.ignored)
        keys |= set(k for k in self.mj if k[0] < limit)
        for key in sorted(keys):
            idx, seat = key
            kind = self.ev(idx)['type']
            reaction = kind in ('dahai', 'kakan', 'ankan')
            if reaction and idx not in self.reaction_seen and key not in self.mj:
                continue  # Mahjax never got to evaluate this trigger
            lib = self.lib.get(key)
            mj = self.mj.get(key)
            if lib is None or mj is None:
                which = 'libriichi_only' if mj is None else 'mahjax_only'
                bits = np.flatnonzero((lib or mj)[0]).tolist()
                names = [M_NAMES[b] for b in bits]
                if mj is not None and set(names) <= {'agari', 'pass'} and key in self.lib_furiten:
                    names.append('(libriichi: furiten)')
                out.append((which, kind, idx, seat, names))
                continue
            diff = np.flatnonzero(lib[0] != mj[0]).tolist()
            if diff or lib[1] != mj[1]:
                detail = ['%s%s' % ('-' if lib[0][b] else '+', M_NAMES[b]) for b in diff]
                if diff == [M_AGARI] and mj[0][M_AGARI] and key in self.lib_furiten:
                    detail.append('(libriichi: furiten)')
                if lib[1] != mj[1]:
                    detail.append('kan_select lib=%s mj=%s' % (
                        sorted(TILE_NAMES[t] for t in lib[1]), sorted(TILE_NAMES[t] for t in mj[1])))
                out.append(('mask', kind, idx, seat, detail))
            else:
                out.append(('equal', kind, idx, seat, None))
        return out


# ---------------------------------------------------------------- Mahjax side

def make_jax_fns(rules=None):
    """Batched init/step/extract. `rules` overrides GameConfig fields; it is
    for the negative control only (a changed rule must show up as mismatches)."""
    import jax
    import jax.numpy as jnp
    import mahjax.red_mahjong.env as E
    from mahjax.red_mahjong.hand import Hand
    from mahjax.red_mahjong.state import GameConfig

    def prepare_from_deck(deck, game_config=None):
        """`_prepare_next_round_assets` with the wall given instead of dealt."""
        deck = jnp.asarray(deck, jnp.int8)
        hand_with_red = Hand.make_init_hand(deck)
        hand = jax.vmap(Hand.to_34)(hand_with_red)
        dora = jnp.full(5, -1, jnp.int8).at[0].set(deck[9])
        ura = jnp.full(5, -1, jnp.int8).at[0].set(deck[8])
        can_ron = E.v_can_win(hand, E.TILE_RANGE)
        return deck, dora, ura, hand, hand_with_red, can_ron

    # every round transition in the step path deals through this function
    E._prepare_next_round_assets = prepare_from_deck
    config = GameConfig()
    for key, value in (rules or {}).items():
        config = config.replace(**{key: jnp.asarray(value, getattr(config, key).dtype)})
    env = E.RedMahjong(round_mode='half', order_points=[0, 0, 0, 0], game_config=config)

    def first_state(deck):
        base = E._make_state(
            current_player=jnp.int8(0), dealer=jnp.int8(0),
            init_wind=E._calc_wind(0), seat_wind=E._calc_wind(0),
            round=jnp.int8(0), round_limit=jnp.int8(7),
            order_points=jnp.zeros(4, jnp.int32),
        )
        return E._init_for_next_round_from_prepared(base, prepare_from_deck(deck), env.game_config)

    def extract(s):
        p, r = s.players, s.round_state
        return dict(
            cp=s.current_player, mask=s.legal_action_mask, rows=p.legal_action_mask,
            target=r.target, last_draw=r.last_draw, terminated=s.terminated, rewards=s.rewards,
            score=r.score, round=r.round, honba=r.honba, kyotaku=r.kyotaku, dealer=r.dealer,
            round_step=r.round_step, fan=p.fan, fu=p.fu, hwr=p.hand_with_red, n_kan=p.n_kan,
        )

    init_b = jax.jit(jax.vmap(first_state))
    step_b = jax.jit(jax.vmap(env.step))
    extract_b = jax.jit(jax.vmap(extract))
    return init_b, step_b, extract_b


def replay_batch(games, fns):
    import jax
    import jax.numpy as jnp

    init_b, step_b, extract_b = fns
    B = len(games)
    states = init_b(jnp.asarray(np.stack([g.walls[0] for g in games])))
    while True:
        f = jax.device_get(extract_b(states))
        actions = np.zeros(B, np.int32)
        keys = np.zeros((B, 136), np.int8)
        running = 0
        for b, g in enumerate(games):
            fb = {k: v[b] for k, v in f.items()}
            a = None
            if g.status == 'running':
                try:
                    a = g.decide(fb)
                except Divergence as e:
                    g.status = 'diverged'
                    g.divergence = e.args[0]
                if g.status == 'running':
                    running += 1
            if a is None:
                a = int(np.flatnonzero(fb['mask'])[0])  # keep stepping a finished game harmlessly
            actions[b] = a
            keys[b] = g.next_wall()
        if not running:
            break
        states = step_b(states, jnp.asarray(actions), jnp.asarray(keys))


def tiles_left(events, i):
    """Live-wall tiles left before event i (70 draws per kyoku, rinshan included)."""
    start = max(j for j in range(i) if events[j]['type'] == 'start_kyoku')
    return 70 - sum(e['type'] == 'tsumo' for e in events[start:i])


class Summary:
    """Accumulates per-game results, so a game's events can be freed after its batch."""

    def __init__(self):
        self.games = 0
        self.status = collections.Counter()
        self.kyoku = 0
        self.steps = 0
        self.decisions = collections.Counter()
        self.mismatches = collections.Counter()
        self.notes = collections.Counter()
        self.divergences = collections.Counter()
        self.offered = collections.Counter()
        self.outcomes = collections.Counter()
        self.examples = {k: collections.defaultdict(list) for k in ('decision', 'note', 'divergence')}

    def example(self, kind, key, g, **fields):
        if len(self.examples[kind][key]) < 5:
            self.examples[kind][key].append({'log': os.path.relpath(g.path, REPO), **fields})

    def add(self, g):
        self.games += 1
        self.status[g.status] += 1
        self.kyoku += g.kyoku + 1
        self.steps += g.steps
        for which, kind, idx, seat, detail in g.compare_decisions():
            self.decisions[which] += 1
            if which != 'equal':
                shown = ' '.join(detail) if which == 'mask' else 'offered ' + ','.join(detail)
                key = f'{which} after {kind}: {shown}'
                self.mismatches[key] += 1
                self.example('decision', key, g, event_idx=idx, seat=seat)
        for kind, idx, detail in g.notes:
            self.notes[kind] += 1
            self.example('note', kind, g, event_idx=idx, detail=detail)
        if g.divergence:
            kind, idx, detail = g.divergence
            self.divergences[kind] += 1
            self.example('divergence', kind, g, event_idx=idx, detail=detail)
        self.add_coverage(g)

    def add_coverage(self, g):
        """What the log exercised: libriichi decisions by offered action, and outcomes."""
        for mask, _ in g.lib.values():
            for b in range(M_RIICHI, 46):
                if mask[b]:
                    self.offered[M_NAMES[b]] += 1
        ev, out = g.events, self.outcomes
        for i, e in enumerate(ev):
            kind = e['type']
            if kind == 'hora':
                out['tsumo' if e['actor'] == e['target'] else 'ron'] += 1
                if ev[i + 1]['type'] == 'hora' and ev[i - 1]['type'] != 'hora':
                    out['multi_ron'] += 1
                if e.get('ura_markers'):
                    out['hora_with_ura'] += 1
            elif kind == 'ryukyoku':
                prev = ev[i - 1]['type']
                deltas = e.get('deltas') or [0, 0, 0, 0]
                if prev == 'tsumo':
                    out['ryukyoku_kyuushu'] += 1
                elif prev == 'reach_accepted':
                    out['ryukyoku_four_riichi'] += 1
                elif any(deltas) or tiles_left(ev, i) == 0:
                    out['ryukyoku_exhaustive'] += 1
                else:
                    out['ryukyoku_abortive_other'] += 1  # four winds, four kans
                if any(abs(d) >= 4000 for d in deltas):
                    out['nagashi_mangan'] += 1
            elif kind in ('chi', 'pon', 'daiminkan', 'ankan', 'kakan', 'reach'):
                out[kind] += 1
            elif kind == 'start_kyoku' and e['bakaze'] == 'W' and e['kyoku'] == 1 and e['honba'] == 0:
                out['games_into_west'] += 1
        if (final_scores(ev) < 0).any():
            out['games_ended_by_tobi'] += 1

    def as_dict(self):
        return {
            'games': self.games,
            'status': dict(self.status),
            'kyoku_replayed': self.kyoku,
            'mahjax_steps': self.steps,
            'decisions': dict(self.decisions),
            'decision_mismatches': dict(self.mismatches.most_common()),
            'decision_examples': dict(self.examples['decision']),
            'notes': dict(self.notes),
            'note_examples': dict(self.examples['note']),
            'divergences': dict(self.divergences),
            'divergence_examples': dict(self.examples['divergence']),
            'coverage': {'libriichi_decisions_offering': dict(self.offered),
                         'outcomes': dict(sorted(self.outcomes.items()))},
        }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--logs', required=True, nargs='+', help='directories of *.json.gz mjai logs')
    ap.add_argument('--n', type=int, default=10000, help='number of logs (sorted, first N)')
    ap.add_argument('--batch', type=int, default=500, help='games replayed in parallel')
    ap.add_argument('--workers', type=int, default=8, help='CPU workers for log parsing and libriichi')
    ap.add_argument('--exclude', help='file of log ids (file names without .json.gz) to skip')
    ap.add_argument('--out', required=True, help='summary JSON')
    ap.add_argument('--upstream-cond', action='store_true',
                    help='keep lax.cond/switch as is (default: run them as selects, select_patch.py)')
    ap.add_argument('--rule', action='append', default=[], metavar='FIELD=VALUE',
                    help='override a Mahjax GameConfig field (negative control only)')
    args = ap.parse_args()
    rules = {k: int(v) for k, v in (r.split('=', 1) for r in args.rule)}

    if not args.upstream_cond:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import select_patch
        select_patch.apply()
    import jax
    import mahjax

    exclude = set()
    if args.exclude:
        with open(args.exclude) as f:
            exclude = {line.strip() for line in f if line.strip()}
    paths = sorted(p for d in args.logs for p in glob.glob(os.path.join(d, '*.json.gz'))
                   if os.path.basename(p)[:-len('.json.gz')] not in exclude)[:args.n]
    print(f'{len(paths)} logs; jax {jax.__version__} on {jax.devices()[0]}; mahjax {mahjax.__version__}',
          flush=True)
    t0 = time.time()
    fns = make_jax_fns(rules)
    acc = Summary()
    with get_context('forkserver').Pool(args.workers) as pool:
        for i in range(0, len(paths), args.batch):
            batch = [Game(*r) for r in pool.map(prepare_game, paths[i:i + args.batch], chunksize=8)]
            replay_batch(batch, fns)
            for g in batch:
                acc.add(g)
            del batch
            print(f'{acc.games}/{len(paths)} games, {dict(acc.status)}, {time.time() - t0:.0f} s', flush=True)
    summary = acc.as_dict()
    summary.update({
        'wall_seconds': round(time.time() - t0, 1),
        'logs': args.logs, 'exclude': args.exclude, 'n': len(paths), 'rules': rules, 'cond_as_select': not args.upstream_cond, 'jax': jax.__version__, 'mahjax': mahjax.__version__,
        'device': str(jax.devices()[0]),
    })
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: summary[k] for k in ('status', 'decisions', 'notes', 'divergences')}, indent=1))


if __name__ == '__main__':
    main()
