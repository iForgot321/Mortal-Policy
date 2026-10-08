"""What riichi.dev does on a triple ron (D12a), from riichienv, whose Rust core runs the RiichiLab
server ("local simulation behavior matches the online server exactly").

Stages one: three closed hands with a dragon triplet (the yaku), all waiting on 2m (3m4m); the
dealer discards 2m; all three declare ron. Prints the outcome under the default rule and each
preset. Verified 2026-10-08 with riichienv 0.4.10: the default (`RiichiEnv(rule=None)`, i.e. Rust
`GameRule::default()` = `default_tenhou()`) aborts it (`ryukyoku`, reason `sanchaho`, no deltas);
`default_mjsoul()` and the Python no-argument `GameRule()` pay all three.

Run it outside the frozen `mortal` env:
    python -m venv /tmp/rienv && /tmp/rienv/bin/pip install riichienv==0.4.10
    /tmp/rienv/bin/python tools/rules/riichienv_sanchaho.py
"""
import json

from riichienv import ActionType, GameRule, RiichiEnv

HANDS = [
    [4, 5, 108, 109, 110, 112, 113, 116, 117, 120, 121, 60, 64, 24],   # dealer, 14 tiles, holds 2m (4)
    [124, 125, 126, 36, 40, 44, 96, 100, 104, 8, 12, 68, 69],          # P x3, 123p, 789s, 3m4m, 9p9p
    [128, 129, 130, 37, 41, 45, 97, 101, 105, 9, 13, 32, 33],          # F x3, 123p, 789s, 3m4m, 9m9m
    [132, 133, 134, 38, 42, 46, 98, 102, 106, 10, 14, 72, 73],         # C x3, 123p, 789s, 3m4m, 1s1s
]


def run(rule):
    env = RiichiEnv(seed=1, rule=rule)
    env.reset(oya=0)
    env.hands = [sorted(h) for h in HANDS]
    env.drawn_tile = 4
    discard = next(a for a in env.get_observations([0])[0].legal_actions()
                   if a.action_type == ActionType.Discard and a.tile == 4)
    obs = env.step({0: discard})
    rons = {}
    for pid, o in obs.items():
        r = [a for a in o.legal_actions() if a.action_type == ActionType.Ron]
        if r:
            rons[pid] = r[0]
    start = len(env.mjai_log)
    env.step(rons)
    after = [e for e in env.mjai_log[start - 1:] if e['type'] in ('dahai', 'hora', 'ryukyoku', 'end_kyoku', 'start_kyoku', 'end_game')]
    return sorted(rons), after, env.scores()


for name, rule in [('RiichiEnv(rule=None)', None), ('GameRule.default_tenhou()', GameRule.default_tenhou()),
                   ('GameRule.default_mjsoul()', GameRule.default_mjsoul()), ('GameRule() (Python no-arg)', GameRule())]:
    rons, events, scores = run(rule)
    print(f'\n== {name}: ron offered to {rons}; scores after {scores}')
    for e in events:
        print('  ', json.dumps({k: v for k, v in e.items() if k in ('type', 'actor', 'target', 'pai', 'deltas', 'reason', 'honba', 'oya', 'kyotaku')}))
