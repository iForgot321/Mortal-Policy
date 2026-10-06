# Design note: stored behaviour probabilities (05 §3.4, D37)

**Status:** open — needs a decision before arms P0/P1 (Phase 1). Written 2026-10-03.

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

## Tests (to add with the implementation)

- First-minibatch ratio is exactly 1.0 against the stored behaviour probability (05 §8 unit test).
- Every trainee entry from the loader finds exactly one sidecar record, and vice versa.
- Behaviour probs over a decision's legal actions sum to 1 within 1e-5.
