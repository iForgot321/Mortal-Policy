"""PlayerState.action_mask from Python (the Mahjax parity harness relies on it):
equal to encode_obs's mask at every decision of every seat, kan select included."""
import gzip
import json

import numpy as np


def test_action_mask_matches_encode_obs(arena_run):
    from libriichi.state import PlayerState

    _, logs, _, _ = arena_run
    checked = 0
    for path in logs[:2]:
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
                for at_kan_select in ([False, True] if cans.can_kan else [False]):
                    _, mask = st.encode_obs(4, at_kan_select)
                    assert np.array_equal(np.asarray(st.action_mask(at_kan_select)), mask)
                    checked += 1
    assert checked > 500
