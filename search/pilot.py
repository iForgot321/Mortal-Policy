"""The search pilot (06 §0, D32): riichi vs dama at closed tenpai.

Stages, each writing under one run directory:

1. `selfplay`: hanchans of the blueprint at all four seats (family-A seeds).
2. `decisions`: every own-turn decision where riichi is legal and the policy's
   best non-riichi action is a discard; a fixed-seed sample of them, with the
   policy's probabilities at the root.
3. `oracle`: for each decision, rollouts of both candidates (riichi, and the
   policy's best dama discard) to the end of the kyoku with the blueprint at
   every seat. The opponents' hands and every past draw are the true ones from
   the arena seed; the undealt tiles are dealt again for each rollout, with
   the same deals for both candidates (common random numbers). Each candidate
   is also played once in the true world, which is the evaluation.
4. `report`: the gain of each search rule over the policy, in u, on the true
   worlds.

The oracle stage reads arena seeds: a lab-only diagnostic, never cited as a
strength result and never part of a deployed bot (06 §0, §2).
"""
import gzip
import json
import math
import os
import random
from glob import glob

import numpy as np

RIICHI = 37


def read_game(path):
    """(seed, kyokus): each kyoku as its raw mjai lines, `start_kyoku` to `end_kyoku`."""
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        lines = f.read().splitlines()
    seed = tuple(json.loads(lines[0])['seed'])
    kyokus, cur = [], None
    for line in lines[1:]:
        kind = json.loads(line)['type']
        if kind == 'start_kyoku':
            cur = []
        if cur is not None:
            cur.append(line)
        if kind == 'end_kyoku':
            kyokus.append(cur)
            cur = None
    return seed, kyokus


def grp_history(kyokus, k):
    """GRP rows of the game's kyoku starts up to and including kyoku `k`."""
    from .value import start_row
    return [start_row(json.loads(ky[0])) for ky in kyokus[:k + 1]]


def riichi_roots(lines):
    """`(until, seat)` for each own-turn decision of the kyoku where riichi is
    legal: the seat that just drew can riichi, and `lines[until]` is what it did."""
    from libriichi.state import PlayerState
    states = [PlayerState(s) for s in range(4)]
    out = []
    for i, line in enumerate(lines[:-1]):
        for s, state in enumerate(states):
            cans = state.update(line)
            if cans.can_riichi:
                out.append((i + 1, s))
    return out


# ---- 1. self-play ----------------------------------------------------------

def selfplay(engine, run_dir, nonce_start, key, games, chunk=500):
    from libriichi.search import self_play
    log_dir = os.path.join(run_dir, 'logs')
    done = {os.path.basename(p) for p in glob(os.path.join(log_dir, '*.json.gz'))}
    for start in range(nonce_start, nonce_start + games, chunk):
        n = min(chunk, nonce_start + games - start)
        if all(f'{s}_{key}.json.gz' in done for s in range(start, start + n)):
            continue
        self_play(engine, (start, key), n, log_dir)
        print(f'self-play: {start + n - nonce_start}/{games} games', flush=True)


# ---- 2. decisions ----------------------------------------------------------

def policy_at(engine, roots, batch=512):
    """The engine's output at each root: (seat, q_out over the action space, mask)."""
    from libriichi.search import root_observations
    out = []
    for i in range(0, len(roots), batch):
        obs = root_observations(roots[i:i + batch], engine.version)
        seats = [s for s, _, _ in obs]
        _, q, masks, *_ = engine.react_batch([o for _, o, _ in obs], [m for _, _, m in obs], None)
        out += zip(seats, q, masks)
    return out


def decisions(engine, run_dir, count, sample_seed=0):
    """Scan the self-play logs, keep the class (riichi legal, best non-riichi
    action a discard), sample `count` with a fixed seed and record the policy."""
    paths = sorted(glob(os.path.join(run_dir, 'logs', '*.json.gz')))
    found = []
    for path in paths:
        _, kyokus = read_game(path)
        for k, lines in enumerate(kyokus):
            found += [(os.path.basename(path), k, until, seat) for until, seat in riichi_roots(lines)]
    random.Random(sample_seed).shuffle(found)

    rows, cache, excluded = [], {}, 0
    for i in range(0, len(found), 1024):
        if len(rows) >= count:
            break
        part = found[i:i + 1024]
        roots = []
        for name, k, until, _ in part:
            if name not in cache:
                cache[name] = read_game(os.path.join(run_dir, 'logs', name))
            seed, kyokus = cache[name]
            roots.append((seed, kyokus[k], until))
        for (name, k, until, seat), (s, q, mask) in zip(part, policy_at(engine, roots)):
            assert s == seat and mask[RIICHI], (name, k, until)
            q = np.where(np.asarray(mask, dtype=bool), np.asarray(q, dtype=np.float64), -np.inf)
            best = int(q.argmax())
            q_no_riichi = q.copy()
            q_no_riichi[RIICHI] = -np.inf
            dama = int(q_no_riichi.argmax())
            if dama > 36:
                # winning, a kan or an abortive draw: not a riichi-vs-dama choice
                excluded += 1
                continue
            policy = RIICHI if best == RIICHI else dama
            seed, kyokus = cache[name]
            rows.append({
                'id': len(rows), 'game': name, 'seed': list(seed), 'kyoku': k, 'until': until,
                'seat': seat, 'honba': json.loads(kyokus[k][0])['honba'],
                'dama': dama, 'policy': policy,
                'q_riichi': float(q[RIICHI]), 'q_dama': float(q[dama]),
            })
            if len(rows) >= count:
                break
    with open(os.path.join(run_dir, 'decisions.jsonl'), 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    summary = {'scanned_games': len(paths), 'riichi_legal': len(found), 'sampled': len(rows),
               'excluded_non_discard': excluded, 'sample_seed': sample_seed}
    with open(os.path.join(run_dir, 'decisions-summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    return summary


# ---- 3. oracle search ------------------------------------------------------

def redeal_seed(decision_id, r):
    """The deal of rollout `r` at a decision, shared by both candidates."""
    return decision_id << 16 | r


def oracle(engine, value, run_dir, redeals, chunk=16):
    """Rollouts for each decision in `decisions.jsonl`, appended to
    `oracle.jsonl` (resumable): per candidate, the value of each redealt
    rollout and of the true world, in u."""
    from libriichi.search import rollouts
    with open(os.path.join(run_dir, 'decisions.jsonl'), encoding='utf-8') as f:
        rows = [json.loads(line) for line in f]
    out_path = os.path.join(run_dir, 'oracle.jsonl')
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding='utf-8') as f:
            done = {json.loads(line)['id'] for line in f}
    todo = [r for r in rows if r['id'] not in done]
    cache = {}

    for i in range(0, len(todo), chunk):
        part = todo[i:i + chunk]
        roots, jobs, items = [], [], []
        for j, d in enumerate(part):
            if d['game'] not in cache:
                if len(cache) >= 64:
                    cache.clear()
                cache[d['game']] = read_game(os.path.join(run_dir, 'logs', d['game']))
            seed, kyokus = cache[d['game']]
            roots.append((seed, kyokus[d['kyoku']], d['until']))
            history = grp_history(kyokus, d['kyoku'])
            for action in (RIICHI, d['dama']):
                for r in [None, *range(redeals)]:
                    jobs.append((j, action, None if r is None else redeal_seed(d['id'], r)))
                    items.append(history)
        outs = rollouts(engine, roots, jobs, keep_logs=True)
        values = value([(h, o, part[j]['honba'], part[j]['seat'])
                        for h, o, (j, _, _) in zip(items, outs, jobs)])
        per = 2 * (redeals + 1)
        with open(out_path, 'a', encoding='utf-8') as f:
            for j, d in enumerate(part):
                v = values[j * per:(j + 1) * per]
                riichi, dama = v[:redeals + 1], v[redeals + 1:]
                # the policy's action in the true world replays the self-play game (06 §3)
                true_policy = outs[j * per + (0 if d['policy'] == RIICHI else redeals + 1)]
                f.write(json.dumps({
                    'id': d['id'],
                    'true_riichi': round(float(riichi[0]), 6), 'true_dama': round(float(dama[0]), 6),
                    'riichi': [round(float(x), 6) for x in riichi[1:]],
                    'dama': [round(float(x), 6) for x in dama[1:]],
                    'reproduced': same_kyoku(true_policy.log, roots[j][1]),
                }) + '\n')
        print(f'oracle: {len(done) + i + len(part)}/{len(rows)} decisions', flush=True)


def same_kyoku(log, lines):
    """Whether a rollout's kyoku log equals the logged kyoku, `meta` aside."""
    def plain(line):
        ev = json.loads(line)
        ev.pop('meta', None)
        return ev
    return len(log) == len(lines) and all(plain(a) == plain(b) for a, b in zip(log, lines))


# ---- 3b. belief search ------------------------------------------------------

def belief(engine, value, model, run_dir, particles, chunk=16, seed=0):
    """Rollouts for each decision from particles of the learned belief (D58): hands sampled from
    what the deciding seat has seen, completed into histories, the undealt tiles in random order.
    Both candidates are played on the same particles. Appended to `belief.jsonl` (resumable): per
    candidate, the value of each particle's rollout, in u. Reads no seed."""
    from .handmodel import beliefs
    with open(os.path.join(run_dir, 'decisions.jsonl'), encoding='utf-8') as f:
        rows = [json.loads(line) for line in f]
    out_path = os.path.join(run_dir, 'belief.jsonl')
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding='utf-8') as f:
            done = {json.loads(line)['id'] for line in f}
    todo = [r for r in rows if r['id'] not in done]
    cache = {}

    for i in range(0, len(todo), chunk):
        part = todo[i:i + chunk]
        roots, histories = [], []
        for d in part:
            if d['game'] not in cache:
                if len(cache) >= 64:
                    cache.clear()
                cache[d['game']] = read_game(os.path.join(run_dir, 'logs', d['game']))
            _, kyokus = cache[d['game']]
            roots.append((kyokus[d['kyoku']], d['until']))
            histories.append(grp_history(kyokus, d['kyoku']))
        # twice the draws: a hand that fits no history (a riichi hand short of tenpai) is dropped
        b, _ = beliefs(model, roots, 2 * particles, seed=seed + part[0]['id'])
        counts = [(min(valid, particles), drawn) for valid, drawn in b.counts()]
        jobs, items = [], []
        for j, d in enumerate(part):
            for action in (RIICHI, d['dama']):
                for p in range(counts[j][0]):
                    jobs.append((j, p, action))
                    items.append((histories[j], d['honba'], d['seat']))
        outs = b.rollouts(engine, jobs)
        values = value([(h, o, honba, seat) for (h, honba, seat), o in zip(items, outs)])
        with open(out_path, 'a', encoding='utf-8') as f:
            at = 0
            for j, d in enumerate(part):
                k = counts[j][0]
                riichi, dama = values[at:at + k], values[at + k:at + 2 * k]
                at += 2 * k
                f.write(json.dumps({
                    'id': d['id'], 'drawn': counts[j][1],
                    'riichi': [round(float(x), 6) for x in riichi],
                    'dama': [round(float(x), 6) for x in dama],
                }) + '\n')
        print(f'belief: {len(done) + i + len(part)}/{len(rows)} decisions', flush=True)


def latency(engine, value, model, run_dir, particles, decisions=100, seed=0):
    """06 §0's latency: one belief search at a time, request to answer: sample `2 * particles`
    hands from the seat's view, complete them, roll both candidates out on up to `particles`
    of them and value the rollouts. Returns per-stage seconds and the p50/p99 of the total."""
    import time
    import torch
    from .handmodel import beliefs
    with open(os.path.join(run_dir, 'decisions.jsonl'), encoding='utf-8') as f:
        rows = [json.loads(line) for line in f][:decisions]
    stages = {'beliefs': [], 'rollouts': [], 'value': [], 'total': []}
    for d in rows:
        seed_, kyokus = read_game(os.path.join(run_dir, 'logs', d['game']))
        root = (kyokus[d['kyoku']], d['until'])
        history = grp_history(kyokus, d['kyoku'])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        b, _ = beliefs(model, [root], 2 * particles, seed=seed + d['id'])
        k = min(b.counts()[0][0], particles)
        t1 = time.perf_counter()
        outs = b.rollouts(engine, [(0, p, a) for a in (RIICHI, d['dama']) for p in range(k)])
        t2 = time.perf_counter()
        value([(history, o, d['honba'], d['seat']) for o in outs])
        t3 = time.perf_counter()
        for name, dt in zip(stages, (t1 - t0, t2 - t1, t3 - t2, t3 - t0)):
            stages[name].append(dt)
    out = {name: {'p50': float(np.percentile(v, 50)), 'p99': float(np.percentile(v, 99)),
                  'max': float(np.max(v))} for name, v in stages.items()}
    out['decisions'] = len(rows)
    out['particles'] = particles
    return out


# ---- 4. report -------------------------------------------------------------

def clustered_se(x, groups):
    """Standard error of the mean of `x`, clustered by `groups` (decisions of
    one game share its seed and its policy)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    resid = x - x.mean()
    sums = {}
    for g, e in zip(groups, resid):
        sums[g] = sums.get(g, 0.0) + e
    g = len(sums)
    var = sum(s * s for s in sums.values()) / n**2 * g / max(g - 1, 1)
    return math.sqrt(var)


def choices(d, o, rule):
    """The search's distribution over (riichi, dama) at one decision under
    `rule`: ('plain', z) deviates from the policy when the paired z of the
    estimated gain exceeds z; ('regularized', lam) is π·exp(Q/λ) over the two
    candidates, played by its argmax."""
    riichi, dama = np.asarray(o['riichi']), np.asarray(o['dama'])
    if len(riichi) < 2:
        # no worlds to judge by: keep the policy's choice
        return float(d['policy'] == RIICHI)
    kind, param = rule
    if kind == 'plain':
        policy_riichi = d['policy'] == RIICHI
        # the estimated gain of switching away from the policy's choice
        switch = (dama - riichi) if policy_riichi else (riichi - dama)
        mean = switch.mean()
        se = switch.std(ddof=1) / math.sqrt(len(switch))
        z = mean / se if se > 0 else (math.inf if mean > 0 else 0.0)
        return float(policy_riichi != (z > param))
    if kind == 'regularized':
        p = np.array([d['q_riichi'], d['q_dama']], dtype=np.float64)
        p = p / p.sum()
        logits = np.log(np.maximum(p, 1e-12)) + np.array([riichi.mean(), dama.mean()]) / param
        return 1.0 if logits[0] > logits[1] else 0.0
    raise ValueError(rule)


RULES = (('plain', 0.0), ('plain', 2.0), ('regularized', 0.05), ('regularized', 0.1))
PRIMARY = ('plain', 2.0)  # D58


def report(run_dir, rules=RULES):
    """Gains over the policy on the true worlds, for the oracle and (if run) the belief
    condition, each search rule; and D32's recovery ratio under the primary rule (D58)."""
    with open(os.path.join(run_dir, 'decisions.jsonl'), encoding='utf-8') as f:
        ds = {r['id']: r for r in map(json.loads, f)}
    with open(os.path.join(run_dir, 'oracle.jsonl'), encoding='utf-8') as f:
        oracle = {o['id']: o for o in map(json.loads, f)}
    searched = {'oracle': oracle}
    path = os.path.join(run_dir, 'belief.jsonl')
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            searched['belief'] = {b['id']: b for b in map(json.loads, f)}
    # common decisions only: every condition is scored on the same true worlds
    ids = sorted(set.intersection(*(set(c) for c in searched.values())))
    groups = [ds[i]['game'] for i in ids]
    true_policy = np.array([oracle[i]['true_riichi'] if ds[i]['policy'] == RIICHI else oracle[i]['true_dama']
                            for i in ids])

    out = {'decisions': len(ids), 'games': len(set(groups)), 'primary_rule': f'{PRIMARY[0]}({PRIMARY[1]})',
           'policy_riichi_rate': float(np.mean([ds[i]['policy'] == RIICHI for i in ids])),
           'reproduced': float(np.mean([oracle[i]['reproduced'] for i in ids]))}
    clair = np.array([max(oracle[i]['true_riichi'], oracle[i]['true_dama']) for i in ids]) - true_policy
    out['clairvoyant_gain'] = summarize(clair, groups)
    for name, rows in searched.items():
        out[name] = {}
        for rule in rules:
            p_riichi = np.array([choices(ds[i], rows[i], rule) for i in ids])
            v = np.array([p * oracle[i]['true_riichi'] + (1 - p) * oracle[i]['true_dama']
                          for p, i in zip(p_riichi, ids)])
            dev = np.array([(p > 0.5) != (ds[i]['policy'] == RIICHI) for p, i in zip(p_riichi, ids)])
            out[name][f'{rule[0]}({rule[1]})'] = {**summarize(v - true_policy, groups),
                                                  'deviation_rate': float(dev.mean())}
    if 'belief' in searched:
        key = f'{PRIMARY[0]}({PRIMARY[1]})'
        o, b = out['oracle'][key]['gain_u'], out['belief'][key]['gain_u']
        out['recovery'] = b / o if o else None
    with open(os.path.join(run_dir, 'report.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    return out


def summarize(gain, groups):
    mean = float(np.mean(gain))
    se = clustered_se(gain, groups)
    return {'gain_u': mean, 'se': se, 'z': mean / se if se > 0 else 0.0}
