"""riichi.dev (RiichiLab) adapter (07 §1): one engine plays validation or ranked games over the
server's WebSocket.

For every `request_action`, the observation is decoded with riichienv, and `obs.new_events()`
(the mjai events new to this seat) are fed to a libriichi mjai `Bot`. Its reaction is sent only
if it matches one of the request's `possible_actions`, echoing `request_id`. Otherwise a legal
fallback is sent and the game is marked failed.

A shadow libriichi `PlayerState`, fed the same events, gives libriichi's action mask at every
request. It is compared with the mask built from `possible_actions` (07 §1, parity). Any
mismatch fails the game, and the runner stops queueing more games (fail closed).

Usage, from the repo root, in the riichidev venv:
    RIICHIDEV_TOKEN=... python -m deploy.riichidev_bot --model PATH --mode validate
The bot token never goes in the repo, a log or a community box (07 §1).
"""
import argparse
import asyncio
import json
import os
import sys
import time

from . import ROOT, RUNS_DIR

URLS = {
    'validate': 'wss://game.riichi.dev/ws/validate',
    'ranked': 'wss://game.riichi.dev/ws/ranked',
}
ACTION_SPACE = 46
TILES = [f'{n}{s}' for s in 'mps' for n in range(1, 10)] + ['E', 'S', 'W', 'N', 'P', 'F', 'C']
AKA = {'5mr': 34, '5pr': 35, '5sr': 36}
TILE_INDEX = {t: i for i, t in enumerate(TILES)} | AKA
KANS = ('daiminkan', 'ankan', 'kakan')
GAME_EVENTS = ('start_game', 'start_kyoku', 'tsumo', 'dahai', 'chi', 'pon', 'reach',
               'reach_accepted', 'dora', 'hora', 'ryukyoku', 'end_kyoku', 'end_game') + KANS


def deaka(pai):
    return pai[:-1] if pai in AKA else pai


def masks_from_legal(legal):
    """libriichi's action mask and kan-select mask (`PlayerState.action_mask(False/True)`, 46
    each) for a list of mjai legal moves."""
    mask = [False] * ACTION_SPACE
    kan = [False] * ACTION_SPACE
    for a in legal:
        t = a['type']
        if t == 'dahai':
            mask[TILE_INDEX[a['pai']]] = True
        elif t == 'reach':
            mask[37] = True
        elif t == 'chi':
            p = TILE_INDEX[deaka(a['pai'])]
            c = sorted(TILE_INDEX[deaka(x)] for x in a['consumed'])
            # low: the called tile is the lowest of the run (libriichi set_can_chi_from_tile)
            shape = {(p + 1, p + 2): 38, (p - 1, p + 1): 39, (p - 2, p - 1): 40}.get(tuple(c))
            if shape is None:
                raise ValueError(f'chi with no shape: {a}')
            mask[shape] = True
        elif t == 'pon':
            mask[41] = True
        elif t in KANS:
            mask[42] = True
            tile = a['consumed'][0] if t == 'ankan' else a['pai']
            kan[TILE_INDEX[deaka(tile)]] = True
        elif t == 'hora':
            mask[43] = True
        elif t == 'ryukyoku':
            mask[44] = True
        elif t == 'none':
            mask[45] = True
        else:
            raise ValueError(f'unknown legal move: {a}')
    return mask, kan


def same_move(reaction, legal):
    """Whether an mjai reaction is the legal move `legal`. Fields only one side carries (actor,
    tsumogiri, meta, a hora's pai) don't count."""
    if reaction['type'] != legal['type']:
        return False
    for k in ('pai', 'target'):
        if k in reaction and k in legal and reaction[k] != legal[k]:
            return False
    if 'consumed' in reaction and 'consumed' in legal:
        return sorted(reaction['consumed']) == sorted(legal['consumed'])
    return True


def fallback(legal, drawn=None):
    """The server's own timeout default (pass, else tsumogiri), else the first legal move."""
    for a in legal:
        if a['type'] == 'none':
            return a
    for a in legal:
        if a['type'] == 'dahai' and a['pai'] == drawn:
            return a
    return legal[0]


class Seat:
    """One game from one seat: riichienv observations in, mjai responses out."""

    def __init__(self, engine, seat):
        from libriichi.mjai import Bot
        from libriichi.state import PlayerState
        self.seat = seat
        self.bot = Bot(engine, seat)
        self.shadow = PlayerState(seat)
        self.drawn = None
        # what went wrong in this game; empty means clean
        self.failures = []

    def fail(self, kind, **detail):
        self.failures.append({'kind': kind, **detail})

    def decide(self, observation, legal, request_id=None, events=None):
        """The response to one `request_action`, and a record of the decision for the game log.

        The mjai events new to this seat come from the observation's `new_events()`, unless
        `events` (mjai JSON lines) gives them."""
        record = {'request_id': request_id, 'legal': legal}
        try:
            reaction = self._react(observation, legal, record, events)
        except Exception as ex:  # never drop a game: a disconnect forfeits the rest of it
            self.fail('exception', request_id=request_id, error=repr(ex))
            reaction = None
        if reaction is not None:
            matches = [a for a in legal if same_move(reaction, a)]
            if not matches:
                self.fail('illegal_reaction', request_id=request_id, reaction=reaction)
        if reaction is None or not matches:
            response = dict(fallback(legal, self.drawn))
            record['fallback'] = True
        else:
            response = dict(matches[0])
            if response['type'] == 'dahai' and 'tsumogiri' in reaction:
                response['tsumogiri'] = reaction['tsumogiri']
        if response['type'] != 'none':
            response.setdefault('actor', self.seat)
        if request_id is not None:
            response['request_id'] = request_id
        record['response'] = response
        return response, record

    def _react(self, observation, legal, record, events):
        if events is None:
            from riichienv import Observation
            events = Observation.deserialize_from_base64(observation).new_events()
        lines = list(events)
        record['events'] = lines
        reaction = None
        for i, line in enumerate(lines):
            out = self.bot.react(line)
            self.shadow.update(line)
            ev = json.loads(line)
            if ev['type'] == 'tsumo' and ev.get('actor') == self.seat:
                self.drawn = ev['pai']
            if out is not None and i < len(lines) - 1:
                # libriichi could act where the server didn't ask
                self.fail('unrequested_action', request_id=record['request_id'], event=line, reaction=out)
            if i == len(lines) - 1:
                reaction = out
        mask, kan = masks_from_legal(legal)
        ours = list(self.shadow.action_mask(False))
        if ours != mask:
            self.fail('mask_mismatch', request_id=record['request_id'],
                      libriichi=[i for i, m in enumerate(ours) if m],
                      riichienv=[i for i, m in enumerate(mask) if m])
        elif mask[42]:
            ours_kan = list(self.shadow.action_mask(True))
            if ours_kan != kan:
                self.fail('kan_mask_mismatch', request_id=record['request_id'],
                          libriichi=[i for i, m in enumerate(ours_kan) if m],
                          riichienv=[i for i, m in enumerate(kan) if m])
        if reaction is None:
            self.fail('no_reaction', request_id=record['request_id'])
            return None
        reaction = json.loads(reaction)
        record['reaction'] = reaction
        reaction = {k: v for k, v in reaction.items() if k != 'meta'}
        return reaction


def load_engine(model, device, amp):
    """The lab's engine (`lab.arena.load_engine`), so the bot plays what the lab evaluated."""
    if not amp:
        # true FP32, as the lab's deterministic A/A (D43): Ampere runs cuDNN convs in TF32
        import torch
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    sys.path.insert(0, ROOT)
    from lab.arena import load_engine as lab_load_engine
    return lab_load_engine(model, device=device, name='riichidev', enable_amp=amp,
                           enable_rule_based_agari_guard=True)


def event_key(ev):
    ev = json.loads(ev) if isinstance(ev, str) else ev
    return [ev['type'], ev.get('actor')]


async def play_game(url, token, engine, log, source='observation'):
    """One game on one connection. Returns a summary dict.

    The server sends the game's mjai events over the socket as well as in each observation.
    `source` picks which feeds the bot; the other is compared with it at every request, by
    event type and actor, and any difference fails the game."""
    import websockets
    seat = None
    # game events received on the socket since the last request
    stream = []
    summary = {'url': url, 'start': time.time(), 'requests': 0, 'timeouts': 0, 'max_ms': 0.0,
               'result': None, 'source': source, 'failures': []}

    def write(direction, msg):
        log.write(json.dumps({'t': round(time.time(), 3), 'dir': direction, **msg},
                             ensure_ascii=False, separators=(',', ':')) + '\n')

    async with websockets.connect(url, additional_headers={'Authorization': f'Bearer {token}'},
                                  open_timeout=60, max_size=2**23) as ws:
        async for raw in ws:
            msg = json.loads(raw)
            kind = msg.get('type')
            if kind == 'request_action':
                summary['requests'] += 1
                t0 = time.perf_counter()
                if seat is None:
                    response, record = dict(fallback(msg['possible_actions'])), {}
                    response['request_id'] = msg.get('request_id')
                    summary['failures'].append({'kind': 'request_before_start_game'})
                else:
                    try:
                        from riichienv import Observation
                        observed = Observation.deserialize_from_base64(msg['observation']).new_events()
                    except Exception as ex:
                        observed = None
                        summary['failures'].append({'kind': 'observation_undecodable', 'error': repr(ex)})
                    streamed = [json.dumps(e) for e in stream]
                    response, record = seat.decide(msg['observation'], msg['possible_actions'],
                                                   msg.get('request_id'),
                                                   streamed if source == 'stream' else observed)
                await ws.send(json.dumps(response, separators=(',', ':')))
                ms = (time.perf_counter() - t0) * 1000
                summary['max_ms'] = max(summary['max_ms'], ms)
                write('in', {k: v for k, v in msg.items() if k != 'observation'})
                write('decision', {**record, 'ms': round(ms, 2)})
                if seat is not None and observed is not None:
                    a = [event_key(e) for e in observed]
                    b = [event_key(e) for e in streamed]
                    if a != b:
                        summary['failures'].append({'kind': 'event_source_mismatch',
                                                    'request_id': msg.get('request_id'),
                                                    'observation': a, 'stream': b})
                stream = []
                continue
            write('in', msg)
            if kind in GAME_EVENTS:
                stream.append(msg)
            if kind == 'start_game':
                seat = Seat(engine, msg['id'])
            elif kind == 'action_ack':
                if msg.get('status') == 'defaulted':
                    summary['timeouts'] += 1
                elif msg.get('status') != 'accepted':
                    summary['failures'].append({'kind': 'ack_' + str(msg.get('status')), 'ack': msg})
            elif kind == 'end_game':
                summary['result'] = msg
                if '/validate' not in url:
                    break
            elif kind == 'validation_result':
                summary['validation'] = msg
                break
    if seat is not None:
        summary['seat'] = seat.seat
        summary['failures'] += seat.failures
    summary['end'] = time.time()
    return summary


def read_token(path):
    if path is None:
        token = os.environ.get('RIICHIDEV_TOKEN')
        if not token:
            raise SystemExit('set RIICHIDEV_TOKEN or pass --token-file')
        return token
    path = os.path.realpath(path)
    if path.startswith(os.path.realpath(ROOT) + os.sep):
        raise SystemExit('the token file must live outside the repo')
    with open(path, encoding='utf-8') as f:
        return f.read().strip()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--model', required=True, help='checkpoint (the lab spec of the deployed version)')
    p.add_argument('--mode', choices=sorted(URLS), default='validate')
    p.add_argument('--games', type=int, default=1)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--amp', action='store_true', help='AMP inference (default: true FP32, D43)')
    p.add_argument('--token-file', help='file holding the bot token, outside the repo '
                   '(default: $RIICHIDEV_TOKEN)')
    p.add_argument('--log-dir', default=os.path.join(RUNS_DIR, 'games'))
    p.add_argument('--events', choices=('observation', 'stream'), default='observation',
                   help="which source feeds the bot; the other is checked against it")
    p.add_argument('--keep-going', action='store_true',
                   help='queue more games after a failed one (default: stop)')
    a = p.parse_args(argv)
    if a.mode == 'validate' and a.games != 1:
        raise SystemExit('validation is one game')
    token = read_token(a.token_file)
    engine = load_engine(a.model, a.device, a.amp)
    os.makedirs(a.log_dir, exist_ok=True)
    summaries_path = os.path.join(os.path.dirname(a.log_dir), 'summaries.jsonl')
    for n in range(a.games):
        stamp = time.strftime('%Y%m%d-%H%M%S', time.gmtime())
        path = os.path.join(a.log_dir, f'{stamp}-{a.mode}.jsonl')
        with open(path, 'w', encoding='utf-8') as log:
            summary = asyncio.run(play_game(URLS[a.mode], token, engine, log, a.events))
        summary.update(model=a.model, amp=a.amp, mode=a.mode, log=path)
        with open(summaries_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(summary, separators=(',', ':')) + '\n')
        print(f"game {n + 1}/{a.games}: seat {summary.get('seat')}, scores "
              f"{(summary['result'] or {}).get('scores')}, {summary['requests']} requests, "
              f"max {summary['max_ms']:.0f} ms, {summary['timeouts']} timeouts, "
              f"{len(summary['failures'])} failures"
              + (f", validation {summary['validation']}" if 'validation' in summary else ''),
              flush=True)
        if summary['failures'] and not a.keep_going:
            print('stopping: failures in this game (see the log)', file=sys.stderr)
            return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
