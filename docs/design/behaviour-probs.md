# Design note: stored behaviour probabilities (05 §3.4, D37)

**Status:** decided 2026-10-05: option A (D46), implemented. Written 2026-10-03.

## The requirement

05 §3.4: "Actors store the behaviour probability of the sampled action. Ratios use it, detached."
05 §3.3: per-seat ordered trajectories carry `behaviour_prob` for every trainee decision.

## The gap

Actors produce mjai logs through `libriichi::arena`. At each poll, `Board` collects one reaction per
player and keeps only the winner (hora > pon/daiminkan > others > none, `arena/board.rs`). Then:

- **Pass decisions** (`Event::None`) are never logged. MortalBatchAgent computes their meta
  (probabilities and mask), but `Board` discards it.
- **Overridden reactions** are never logged either: a chi that loses to another player's pon, or a pon
  that loses to a ron.
- Logged events carry `meta.q_values` (softmax over the legal mask, fp32) and `mask_bits`. The
  behaviour probability of the taken action could be recovered for those, but not for the two cases
  above, and passes are ~10–20% of decisions.

The upstream learner avoids the issue by computing ratios against a periodically refreshed
`Old_mortal` snapshot, which D37 replaces.

## Options

| | Mechanism | Cost | Risk |
|---|---|---|---|
| **A** (recommended) | `Board` records every reaction's `(event_idx, actor, label, behaviour_prob)` into a per-game sidecar in `GameResult`. Actors dump it next to the mjai log (`<log>.bp.json.gz`), and the learner's ordered extraction joins on `(event_idx, actor)`, using `Gameplay.event_idx`, which now exists. | Moderate Rust change in `arena/` plus a loader join | Low: the mjai logs are unchanged, so the training and featurizer paths stay hash-identical |
| B | Learner keeps the last K parameter versions and recomputes π_{v}(a\|s) for each sample with the actor's version | Extra forward per sample; version bookkeeping | Lag beyond K versions is unrecoverable |
| C | Emit non-standard `{"type":"none","actor":k,"meta":…}` events into the mjai log | Small arena change | Breaks mjai compatibility and every log consumer, including the loader's 4-event windows |

Under A, `behaviour_prob` is the probability the actor's policy assigned to the label it emitted,
including exploration: with `explore_rate` ε, b(a) = (1 − ε)·1[a = argmax] + ε·π(a). Decisions altered
after sampling by the rule-based agari guard must be flagged and excluded from ratio terms.

## Implementation (option A)

| Piece | Where |
|---|---|
| Engine reports the probability its sampler gave each sampled action, as a 5th return value of `react_batch` | `MortalEngine.sample` / `behaviour_dist` (`mortal/engine.py`) |
| Agent attaches it to the reaction: 1 for forced moves (quick-eval discard, forced kan tile), `None` when the agari guard replaced the action | `EventExt.behaviour`, never serialized (`libriichi/src/agent/mortal.rs`, `mjai/event.rs`) |
| Board records every reaction, with `applied` | `BoardState::record_reactions` (`arena/board.rs`) |
| Per-game records with whole-log `event_idx` | `GameResult.behaviour`, `BatchGame.record_behaviour` (`arena/game.rs`, `result.rs`) |
| Sidecar: gzipped JSON of columns `event_idx, actor, kan_select, label, prob, applied` | `OneVsThree(behaviour_dir=…)`; `TrainPlayer` writes to `train_play.<profile>.behaviour_dir`, default `log_dir + '_behaviour'` |
| Join | `trajectory.load_behaviour` / `join_behaviour` |

Two departures from the option-A sketch above:
- **Separate directory, same file name**, not `<log>.bp.json.gz` next to the log. `TrainPlayer`, `client.py`, `server.py` and `train_online.py` treat every file in a log directory as a game log.
- **The join key includes `kan_select`.** The loader emits a kan-select entry at the same `event_idx` as its kan.

**Engines that don't report.** An engine that returns the old 4-tuple is still accepted; its network decisions get `prob = None`. `lab/arena.py`'s `HobbledEngine` returns the 4-tuple on purpose, because its random replacements change the sampling distribution.

**Recording is opt-in.** Only `behaviour_dir` turns it on, so lab runs keep neither the records nor their memory.

**Not done yet: transport and the learner, for Phase 1 arms P0/P1.**
- `client.py` doesn't yet submit sidecars, and `server.py` doesn't buffer or drain them.
- `train_online.py` still computes ratios against `Old_mortal`. The stored probabilities replace it when the ordered-trajectory loader is wired into training (05 §2).

## Tests

All in `tests/test_behaviour.py`:
- **Ratio:** the first-minibatch ratio against the stored behaviour probability is 1 within 1e-4. Same weights, loader features, ε = 1, CPU FP32.
- **Join:** every loader entry finds a record with the same label, and every applied record finds an entry.
- **Distribution:** the behaviour distribution sums to 1 within 1e-5, and the sampler's empirical frequencies match it for ε ∈ {0, 0.3, 1}.
- **Greedy seats** store 1.
- **Disagreement:** the join rejects each kind of mismatch between log and sidecar.

`libriichi` also tests that `behaviour` never reaches the serialized mjai event.

**Stress check (2026-10-05, not committed):** 1,000 games of pilot A (ε = 1) vs pilot A (greedy, agari guard on).
- All 633,347 loader entries joined.
- Coverage: 66 multi-rons, 2 four-riichi draws, 870 kans (each with its kan-select record), 2,882 unapplied records.
- The 12 agari-guard replacements were exactly the 12 entries without a probability.
