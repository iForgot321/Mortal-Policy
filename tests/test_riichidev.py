"""The riichi.dev adapter (07 §1): move mapping, parity on lab logs and in step mode, and the
WebSocket runner against a local mock of the validation game. Needs riichienv and websockets
(the riichidev venv, `deploy/__init__.py`); skipped in the plain mortal env."""
import asyncio
import json
from glob import glob

import pytest

pytest.importorskip('riichienv')
pytest.importorskip('websockets')

from conftest import make_engine  # noqa: E402
from deploy.riichidev_bot import fallback, masks_from_legal, play_game, same_move  # noqa: E402
from deploy.riichidev_parity import check_move, masked_for, replay, selfplay  # noqa: E402


def on(mask):
    return [i for i, m in enumerate(mask) if m]


def test_masks_from_legal():
    mask, kan = masks_from_legal([
        {'type': 'dahai', 'pai': '1m'}, {'type': 'dahai', 'pai': 'C'}, {'type': 'dahai', 'pai': '5pr'},
        {'type': 'reach'}, {'type': 'hora'}, {'type': 'ryukyoku'},
        {'type': 'ankan', 'consumed': ['5s', '5s', '5s', '5sr']},
        {'type': 'kakan', 'pai': 'E', 'consumed': ['E', 'E', 'E']},
    ])
    assert on(mask) == [0, 33, 35, 37, 42, 43, 44]
    assert on(kan) == [22, 27]
    calls, kan = masks_from_legal([
        {'type': 'chi', 'pai': '3m', 'consumed': ['4m', '5mr']},   # low
        {'type': 'chi', 'pai': '3m', 'consumed': ['2m', '4m']},    # mid
        {'type': 'chi', 'pai': '3m', 'consumed': ['1m', '2m']},    # high
        {'type': 'pon', 'pai': '3m', 'consumed': ['3m', '3m']},
        {'type': 'daiminkan', 'pai': '3m', 'consumed': ['3m', '3m', '3m']},
        {'type': 'none'},
    ])
    assert on(calls) == [38, 39, 40, 41, 42, 45]
    assert on(kan) == [2]
    with pytest.raises(ValueError):
        masks_from_legal([{'type': 'chi', 'pai': '3m', 'consumed': ['5m', '6m']}])


def test_same_move_and_fallback():
    assert same_move({'type': 'dahai', 'actor': 0, 'pai': '5mr', 'tsumogiri': True}, {'type': 'dahai', 'pai': '5mr'})
    assert not same_move({'type': 'dahai', 'pai': '5m'}, {'type': 'dahai', 'pai': '5mr'})
    assert same_move({'type': 'chi', 'pai': '3m', 'consumed': ['5m', '4m']},
                     {'type': 'chi', 'pai': '3m', 'consumed': ['4m', '5m']})
    assert not same_move({'type': 'chi', 'pai': '3m', 'consumed': ['4m', '5m']},
                         {'type': 'chi', 'pai': '3m', 'consumed': ['4m', '5mr']})
    assert same_move({'type': 'hora', 'actor': 1, 'target': 0}, {'type': 'hora', 'actor': 1, 'target': 0, 'pai': '1s'})
    legal = [{'type': 'dahai', 'pai': '1m'}, {'type': 'dahai', 'pai': '9s'}, {'type': 'reach'}]
    assert fallback(legal, drawn='9s') == {'type': 'dahai', 'pai': '9s'}
    assert fallback(legal + [{'type': 'none'}]) == {'type': 'none'}


def test_check_move():
    hora = {'type': 'hora', 'actor': 2, 'target': 0}
    assert check_move({'type': 'none'}, [hora], 1) == 'match'
    assert check_move({'type': 'pon', 'pai': '1m', 'consumed': ['1m', '1m']}, [hora], 1) == 'outranked'
    assert check_move({'type': 'hora'}, [hora], 1) == 'mismatch'
    assert check_move({'type': 'hora'}, [hora, {'type': 'hora', 'actor': 1, 'target': 0}], 1) == 'match'
    assert check_move({'type': 'ryukyoku', 'actor': 1}, [{'type': 'ryukyoku', 'deltas': [0] * 4}], 1) == 'match'
    assert check_move({'type': 'dahai', 'pai': '1m', 'tsumogiri': True},
                      [{'type': 'dahai', 'actor': 1, 'pai': '1m', 'tsumogiri': False}], 1) == 'mismatch'


def test_replay_lab_logs(arena_run):
    """Inputs identical to the lab engine's at every request; the greedy champion's moves
    identical to the log's (the trainee samples, so its moves aren't checked)."""
    root, logs, brain, policy = arena_run
    engine = make_engine(brain, policy, 0.0, 'adapter')
    total = {}
    for path in sorted(glob(str(root / 'logs' / '*.json.gz'))):
        counts, problems = replay(path, engine, 4, names={'champion'})
        assert problems == [], problems[:3]
        for k, v in counts.items():
            total[k] = total.get(k, 0) + v
    assert total['requests'] > 1000 and total['unchecked'] > 0
    assert total['mismatch'] == 0 and total['obs_mismatch'] == 0 and total['adapter_failures'] == 0
    assert total['match'] > 0.9 * (total['requests'] - total['unchecked'])


def test_selfplay_step_mode(arena_run):
    _, _, brain, policy = arena_run
    counts, problems, ranks = selfplay(11, make_engine(brain, policy, 0.0, 'adapter'), 4)
    assert problems == [], problems[:3]
    assert counts['requests'] > 500 and counts['fallbacks'] == 0
    assert sorted(ranks) == [1, 2, 3, 4]


def test_play_game_mock_validation(arena_run, tmp_path):
    """The runner against a local mock of the validation game: seat 0 over the socket,
    tsumogiri bots elsewhere, events streamed and in the observation."""
    import websockets
    from riichienv import RiichiEnv
    _, _, brain, policy = arena_run
    engine = make_engine(brain, policy, 0.0, 'adapter')
    seen = {'requests': 0, 'rejected': 0}

    async def server(ws):
        seen['auth'] = ws.request.headers.get('Authorization')
        env, mirror = RiichiEnv(game_mode=1, seed=5), RiichiEnv(game_mode=1)
        await ws.send(json.dumps({'type': 'start_game', 'id': 0}))
        cursor, request_id = 0, 0
        obs = env.get_observations()
        while True:
            for ev in env.mjai_log[cursor:]:
                ev = ev if isinstance(ev, dict) else json.loads(ev)
                mirror.apply_event(ev)
                if ev['type'] != 'start_game':
                    await ws.send(json.dumps(masked_for(ev, 0)))
            cursor = len(env.mjai_log)
            if env.done():
                break
            actions = {}
            for s, o in obs.items():
                if not o.legal_actions():
                    continue
                legal = [json.loads(a.to_mjai()) for a in o.legal_actions()]
                if s != 0:
                    actions[s] = o.select_action_from_mjai(fallback(legal))
                    continue
                request_id += 1
                seen['requests'] += 1
                await ws.send(json.dumps({
                    'type': 'request_action', 'request_id': request_id,
                    'time': {'grace_ms': 3000, 'bank_ms': 15000, 'deadline_ms': 18000},
                    'possible_actions': legal,
                    'observation': mirror.get_observation(0).serialize_to_base64()}))
                response = json.loads(await ws.recv())
                ok = response.pop('request_id') == request_id and any(same_move(response, a) for a in legal)
                seen['rejected'] += not ok
                await ws.send(json.dumps({'type': 'action_ack', 'request_id': request_id,
                                          'status': 'accepted' if ok else 'rejected'}))
                actions[0] = o.select_action_from_mjai(response)
            obs = env.step(actions)
        await ws.send(json.dumps({'type': 'validation_result', 'passed': True}))
        await ws.wait_closed()

    async def run():
        async with websockets.serve(server, '127.0.0.1', 0) as srv:
            port = srv.sockets[0].getsockname()[1]
            with open(tmp_path / 'game.jsonl', 'w', encoding='utf-8') as log:
                return await play_game(f'ws://127.0.0.1:{port}/ws/validate', 'test-token', engine, log)

    summary = asyncio.run(run())
    assert seen['auth'] == 'Bearer test-token'
    assert summary['failures'] == [], summary['failures'][:3]
    assert summary['requests'] == seen['requests'] > 20 and seen['rejected'] == 0
    assert summary['validation'] == {'type': 'validation_result', 'passed': True}
    assert summary['result']['type'] == 'end_game' and summary['seat'] == 0
    # the token never reaches the game log
    assert 'test-token' not in (tmp_path / 'game.jsonl').read_text(encoding='utf-8')
