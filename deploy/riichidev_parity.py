"""Parity tests for the riichi.dev adapter (07 §1), on lab game logs.

Each log is replayed through riichienv (`RiichiEnv.apply_event`). Whenever a seat has legal
moves, its observation goes through `Seat.decide` exactly as a server request would: base64
round trip, `possible_actions` from `legal_actions()`. At every request:
1. riichienv's legal moves must equal libriichi's mask (`Seat.decide` checks this);
2. the adapter's observation tensor must equal that of a libriichi `PlayerState` fed the lab log
   directly, masked for that seat as the arena masks it. The adapter's inputs are then exactly
   the lab engine's;
3. the adapter's move must equal the log's, where the log shows that seat's move. Where it
   doesn't, the adapter must have passed, or made a claim that a higher-priority claim
   outranked (counted, not verified).

Check 3 needs logs played in every seat by the engine given with --model, in true FP32 (D43),
e.g. the lab's deterministic A/A. --names limits check 3 to seats with those player names.

Usage, from the repo root, in the riichidev venv:
    python -m deploy.riichidev_parity --model CKPT --logs 'workdir/lab/RUN/logs/*.json.gz' --games 1000
"""
import argparse
import gzip
import json
import os
import sys
import time
from glob import glob

from . import RUNS_DIR
from .riichidev_bot import KANS, Seat, event_key, load_engine, same_move

# events that are a seat's own move
MOVES = ('dahai', 'chi', 'pon', 'daiminkan', 'ankan', 'kakan', 'reach', 'hora', 'ryukyoku')
# claim priority on a discard (or a kakan): a lower claim can be outranked by a higher one
PRIORITY = {'chi': 1, 'pon': 2, 'daiminkan': 2, 'hora': 3}


def masked_for(ev, seat):
    """The event as the arena shows it to `seat`: others' hands and draws hidden."""
    if ev['type'] == 'start_kyoku':
        ev = dict(ev, tehais=[h if i == seat else ['?'] * len(h) for i, h in enumerate(ev['tehais'])])
    elif ev['type'] == 'tsumo' and ev['actor'] != seat:
        ev = dict(ev, pai='?')
    return ev


def replies_to(lines, i):
    """The moves that answer event i: a multi-ron's hora events, else the next event. A dora
    or reach_accepted line can sit between a decision and its move (tsumo > dora > dahai)."""
    out = []
    for ev in lines[i + 1:]:
        if ev['type'] in ('dora', 'reach_accepted'):
            continue
        if out and not (ev['type'] == 'hora' and out[0]['type'] == 'hora'):
            break
        out.append(ev)
    return out


def check_move(response, replies, seat):
    """'match', 'mismatch' or 'outranked' for one response against the log."""
    own = [ev for ev in replies if ev.get('actor') == seat and ev['type'] in MOVES]
    if own:
        ev = own[0]
        if not same_move(response, ev):
            return 'mismatch'
        if ev['type'] == 'dahai' and response.get('tsumogiri', ev['tsumogiri']) != ev['tsumogiri']:
            return 'mismatch'
        return 'match'
    if response['type'] == 'none':
        return 'match'
    first = replies[0] if replies else {}
    if response['type'] == 'ryukyoku' and first.get('type') == 'ryukyoku':
        # nine terminals: the log's ryukyoku carries no actor
        return 'match'
    if first.get('type') == 'ryukyoku' and response['type'] == 'hora':
        # a triple ron aborts (D51) and the log shows no hora
        return 'outranked'
    if PRIORITY.get(first.get('type'), 0) > PRIORITY.get(response['type'], 9):
        return 'outranked'
    return 'mismatch'


def replay(path, engine, version, names=None):
    """Replay one log; returns (counts, problems)."""
    from libriichi.state import PlayerState
    from riichienv import RiichiEnv

    with gzip.open(path, 'rt', encoding='utf-8') as f:
        lines = [json.loads(line) for line in f if line.strip()]
    for ev in lines:
        ev.pop('meta', None)
    player_names = lines[0].get('names', [''] * 4)
    checked = [names is None or player_names[s] in names for s in range(4)]

    env = RiichiEnv(game_mode=2)
    seats = [Seat(engine, s) for s in range(4)]
    native = [PlayerState(s) for s in range(4)]
    counts = dict(requests=0, match=0, mismatch=0, outranked=0, unchecked=0, obs_mismatch=0)
    problems = []
    request_id = 0
    for i, ev in enumerate(lines):
        env.apply_event(ev)
        for s in range(4):
            native[s].update(json.dumps(masked_for(ev, s)))
        # Replay only: apply_event leaves a decision open across the next event, so riichienv
        # would ask twice: a riichi discard's calls again after reach_accepted, and an open
        # kan's rinshan discard again after its dora. Step mode (the server) asks once, at the
        # discard and at the rinshan tsumo, as libriichi's arena does.
        if ev['type'] in ('end_kyoku', 'end_game', 'reach_accepted', 'dora'):
            continue
        # Replay only, too: right after a kan, riichienv offers its declarer a discard before
        # the rinshan tsumo. Others may still be asked (chankan).
        declarer = ev['actor'] if ev['type'] in KANS else None
        # only seats with a decision may be observed: get_observation advances new_events
        probe = env.clone()
        asked = [s for s in range(4) if s != declarer
                 and (o := probe.get_observation(s)) is not None and o.legal_actions()]
        for s in asked:
            obs = env.get_observation(s)
            legal = [json.loads(a.to_mjai()) for a in obs.legal_actions()]
            request_id += 1
            counts['requests'] += 1
            response, record = seats[s].decide(obs.serialize_to_base64(), legal, request_id)
            ours, theirs = seats[s].shadow, native[s]
            same = all(
                (a == b).all()
                for a, b in zip(ours.encode_obs(version, False), theirs.encode_obs(version, False)))
            if not same:
                counts['obs_mismatch'] += 1
                problems.append({'log': path, 'event': i, 'seat': s, 'kind': 'obs_mismatch',
                                 'adapter_events': record.get('events')})
            if not checked[s]:
                counts['unchecked'] += 1
                continue
            verdict = check_move(response, replies_to(lines, i), s)
            counts[verdict] += 1
            if verdict == 'mismatch':
                problems.append({'log': path, 'event': i, 'seat': s, 'kind': 'move_mismatch',
                                 'response': response, 'log_next': replies_to(lines, i),
                                 'reaction': record.get('reaction')})
    for s, seat in enumerate(seats):
        for f in seat.failures:
            problems.append({'log': path, 'seat': s, **f})
    counts['adapter_failures'] = sum(len(seat.failures) for seat in seats)
    return counts, problems


def selfplay(seed, engine, version):
    """One riichienv game in step mode, the server's flow, with the adapter in all four seats.

    `env.step` hands every seat its new events at every step, so its observations can't stand
    in for the server's, which carry all events new to a seat since its previous request
    (`new_events()`, as riichi.dev documents it). A mirror env, fed the game's log by
    `apply_event` and observed only for the seat being asked, scopes them that way. The
    legal moves come from the step-mode env. At every request, the legal moves are checked
    against libriichi's mask (`Seat.decide`), and the observation against a libriichi
    `PlayerState` fed the full log, masked for that seat. Returns (counts, problems, ranks)."""
    from libriichi.state import PlayerState
    from riichienv import RiichiEnv

    env = RiichiEnv(game_mode=2, seed=seed)
    mirror = RiichiEnv(game_mode=2)
    seats = [Seat(engine, s) for s in range(4)]
    native = [PlayerState(s) for s in range(4)]
    fed = [[] for _ in range(4)]
    counts = dict(requests=0, obs_mismatch=0, fallbacks=0)
    problems = []
    cursor = 0
    request_id = 0
    obs = env.get_observations()
    while not env.done():
        log = env.mjai_log
        for ev in log[cursor:]:
            ev = ev if isinstance(ev, dict) else json.loads(ev)
            mirror.apply_event(ev)
            for s in range(4):
                native[s].update(json.dumps(masked_for(ev, s)))
        cursor = len(log)
        actions = {}
        for s, o in obs.items():
            if not o.legal_actions():
                continue
            legal = [json.loads(a.to_mjai()) for a in o.legal_actions()]
            request_id += 1
            counts['requests'] += 1
            observation = mirror.get_observation(s).serialize_to_base64()
            response, record = seats[s].decide(observation, legal, request_id)
            fed[s] += record.get('events', [])
            counts['fallbacks'] += bool(record.get('fallback'))
            same = all(
                (a == b).all()
                for a, b in zip(seats[s].shadow.encode_obs(version, False), native[s].encode_obs(version, False)))
            if not same:
                counts['obs_mismatch'] += 1
                problems.append({'seed': seed, 'seat': s, 'kind': 'obs_mismatch', 'request_id': request_id})
            actions[s] = o.select_action_from_mjai({k: v for k, v in response.items() if k != 'request_id'})
        obs = env.step(actions)
    # every seat must have been fed the game in order, up to its last request (types and actors)
    full = [event_key(e) for e in env.mjai_log]
    for s in range(4):
        got = [event_key(e) for e in fed[s]]
        if got != full[:len(got)]:
            problems.append({'seed': seed, 'seat': s, 'kind': 'event_gap', 'fed': len(got), 'log': len(full)})
    for s, seat in enumerate(seats):
        for f in seat.failures:
            problems.append({'seed': seed, 'seat': s, **f})
    counts['adapter_failures'] = sum(len(seat.failures) for seat in seats)
    return counts, problems, list(env.ranks())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--model', required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--logs', help='glob of lab mjai logs (.json.gz) to replay')
    src.add_argument('--selfplay', type=int, metavar='N',
                     help='instead, play N riichienv games in step mode, the adapter in every seat')
    p.add_argument('--seed-start', type=int, default=0, help='first riichienv seed for --selfplay')
    p.add_argument('--games', type=int, default=1000, help='logs to replay')
    p.add_argument('--names', help='comma-separated player names whose moves are checked (default: all)')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--amp', action='store_true')
    p.add_argument('--out', default=None, help='report path (default: workdir/riichidev/parity-<time>.json)')
    a = p.parse_args(argv)

    engine = load_engine(a.model, a.device, a.amp)
    names = set(a.names.split(',')) if a.names else None
    if a.selfplay:
        jobs = [(seed, lambda seed=seed: selfplay(seed, engine, engine.version)[:2])
                for seed in range(a.seed_start, a.seed_start + a.selfplay)]
    else:
        paths = sorted(glob(a.logs))[:a.games]
        if not paths:
            raise SystemExit(f'no logs match {a.logs}')
        jobs = [(path, lambda path=path: replay(path, engine, engine.version, names)) for path in paths]
    total, problems = {}, []
    t0 = time.time()
    for n, (_, job) in enumerate(jobs, 1):
        counts, probs = job()
        for k, v in counts.items():
            total[k] = total.get(k, 0) + v
        problems += probs
        if n % 50 == 0 or n == len(jobs):
            print(f'{n}/{len(jobs)} games, {time.time() - t0:.0f} s: {total}', flush=True)
    passed = not problems and not total.get('mismatch')
    report = {'model': a.model, 'source': f'selfplay {a.seed_start}+{a.selfplay}' if a.selfplay else a.logs,
              'games': len(jobs), 'amp': a.amp, 'names': sorted(names) if names else None,
              'counts': total, 'passed': passed, 'problems': problems[:200],
              'problem_kinds': {k: sum(p['kind'] == k for p in problems) for k in {p['kind'] for p in problems}},
              'seconds': round(time.time() - t0)}
    out = a.out or os.path.join(RUNS_DIR, time.strftime('parity-%Y%m%d-%H%M%S.json', time.gmtime()))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=1)
    print(('PASS' if passed else 'FAIL') + f': {total} {report["problem_kinds"]}; report {out}')
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
