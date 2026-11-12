# Draft: before T1 (D64), the critic check and the anchor-release pilot, for the user's review

**Status: draft, 2026-11-10.** D64 says T1 and D31's curves don't start on K2's fixed anchor, and names
two things that come first. This is their design. Nothing here has run. Each part becomes a
pre-registration (an `experiments/` note, fixed before any result) once the user has chosen below;
the pilot is a rental and needs a yes.

## The evidence behind D64

| What | Number | Source |
|---|---|---|
| `k2b`'s anchor coefficient | 0.0759 from the run's first tenth, never moved | the learner's TensorBoard |
| Reverse KL to pilot A / argmax agreement over the run | 0.025 → 0.019 / 94.1% → 95.3% | same |
| The run's second half (`full` vs `p50`, ≈ $130) | +0.0025 [+0.0004, +0.0046] | `experiments/k2-retry.md` |
| 47.8 h more at the same anchor (the soak, $125) | +0.0016 [−0.0050, +0.0082] | `experiments/fleet-soak.md` |
| Critic explained variance, all decisions / last kyoku | 1.2–1.8% / 6–10% | the learner's TensorBoard |
| Critic loss, start → end | 0.0417 → 0.0424 | same |
| The SL kyoku-Δ head, public information only | ≈ 10% of a kyoku's Δ (Huber 0.0287 vs ≈ 0.032 for a constant) | the teachers' validation |

The policy moved to where the PPO push and the anchor's pull balance, and stayed. The critic, which
sees the opponents' hands, predicts a kyoku's outcome about as well as a head that doesn't.

## 1. The critic check (workstation, $0)

**Done 2026-11-12** (`experiments/critic-ceiling.md`): K1 explains 13.7% of the kyoku's GRP-Δu, above every
supervised arm (8.4–9.0%). By the reading the critic isn't the lever, so the pilot below runs with K1 as
it is. The design that was run:

**Question.** How much of a kyoku's Δ can a critic explain with the opponents' hands, how much
without, and how far below that is K1's online critic?

**Data (exists).** The K1 learning-rate check's games (`experiments/k1-lr-check.md`): 100,000 hanchan
of pilot A's B0 play in `workdir/k1-lr-check/pool`, and 4,000 held-out ones in `fresh` (634,796
trainee decisions). No lab nonces, no registry rows.

**Arms,** all scored on the held-out games:

| Arm | Input | Target | Training |
|---|---|---|---|
| K1 as `k2b` left it | obs + opponents' hands | (no training) | the critic in `k2b`'s `full.pth`, scored as is |
| Public | obs | the current kyoku's GRP-Δu | supervised, b8c128, passes over the pool until held-out EV stops rising |
| Privileged | obs + opponents' hands | the same | the same |
| Privileged, larger | obs + opponents' hands | the same | b16c256, the same |

**Metric.** Explained variance of the current kyoku's Δ on the held-out games: over all decisions, by
third of the kyoku, and in the game's last kyoku (where K1's own figure, 6–10%, is the same target).
Calibration in 10 bins.

**Why the kyoku's Δ and not the H return.** Under H, 86% of the return's variance comes from later
kyoku (`k1_h_ceiling.py`), which no input predicts. The check also reports the mean later-kyoku
return by kyoku index: if that is near a constant, a critic trained on the kyoku's Δ plus that
constant serves H's GAE.

**Reading, fixed before it runs:**
- **Privileged ≥ 2× K1's last-kyoku figure** (≥ 0.20 against ≈ 0.10): the critic is a lever. T1's
  critic is trained this way before any policy step (a supervised B0), and the pilot below uses it.
- **Privileged ≈ public** (within 0.02): the hidden hands add little at this size. The critic stays
  as it is, and sample efficiency moves up the list instead (RL-6 four-seat harvesting, RL-8 reuse).
- **Larger ≥ privileged + 0.03:** T1's critic takes the larger net.

**Cost.** ≈ 6–8 h on the 3060 (one pass of the pool took 1.6 h per critic in the LR check). Nothing
else runs on the workstation meanwhile (RAM). New code: one script, `experiments/critic_ceiling.py`.

## 2. The anchor-release pilot (a rental)

**Question.** Which mechanism lets the policy keep improving past the fixed anchor's equilibrium?

**The baseline exists.** The soak continued `k2b` full at the fixed anchor for 47.8 h and gained
nothing measurable. Both candidates start from the same checkpoint, so "no release" needs no new run.

**Candidates,** each from `k2b` full (v4, pilot A at the table as before), with the soak's LR:

| Candidate | The anchor | Room by the end |
|---|---|---|
| **A. Rising target** | pilot A, target 0.02 → 0.04 nats over the pilot (`target_growth`) | ≈ 0.04 nats from pilot A |
| **B. Re-anchor** | `k2b` full itself, target 0.02 nats, constant | ≈ 0.02 nats from `k2b`, so up to ≈ 0.04 from pilot A |

Both allow about the same distance from pilot A by the end. A gives it gradually around the old
centre; B gives it at once around the new one. B is what a chain of champions would repeat.

**Measures,** each candidate's last checkpoint against `k2b` full, paired on the same deals as in
`experiments/k2b-opponents.md`: 40K sets at a table of pilot A and 40K at a table of pilot B (the
held-out table). 40K sets give a standard error of ≈ 0.0017.

**Pass:** the 95% CI at pilot A's table is above 0, and the estimate at pilot B's table is above 0.
If both pass, the larger gain at pilot B's table is T1's mechanism. If neither passes, D64's
"reopen" applies.

**Size and cost.** Per candidate a learner and 2 actors for 48 h, as `k2b` ran: ≈ $86 at `k2b`'s
$1.80/h, so ≈ $175 for both, plus the measures (workstation, ≈ 28 h, or an eval box at ≈ $5).

**Code before it can run:**
- a stage-initialization tool: a snapshot without its `kl_anchor` state and `schedule_origin`, so a
  run can start from it with a new anchor and a new run clock (`train_online.py` refuses an anchor
  that differs from the checkpoint's). About 20 lines and a test;
- nothing else: `policy.kl_anchor.state_file` and `target_growth` exist.

**Not in this pilot:** who sits at the table. Both candidates keep pilot A there, so the result
compares with `k2b` and the soak. Self-play or a pool is RL-5, after the mechanism is chosen.

## Decisions for the user

1. **Run the critic check now?** Recommended: yes. It costs nothing but the workstation for a night,
   and its reading feeds the pilot.
2. **The pilot's base:** `k2b` full on v4 (recommended, because the soak is its measured baseline), or
   a v5 run, which would need its own fixed-anchor baseline first (≈ $125 more).
3. **The pilot's size:** a learner and 2 actors for 48 h per candidate (≈ $175 for both), or the
   soak's learner and 3 actors (≈ $250).
4. **Candidates:** both A and B (recommended), or one.

## After these

D31's curves (two RL seeds per size, D65) and T1 run on the mechanism and the critic these two
select, from the students the distillation produces (`docs/design/distillation-draft.md`).
