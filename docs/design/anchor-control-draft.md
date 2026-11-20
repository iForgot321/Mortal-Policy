# D69's anchor-control pilot: design draft (2026-11-20)

**Status: draft.** Decided in outline by D69 (the user, 2026-11-20: "Add tests for 1 and 2 to the
plan before T1, feel free to increase the budget accordingly"). It becomes
`experiments/anchor-control.md`, pre-registered, once the open parameters below are set by the
simulation. It runs after the user's yes on a live quote. Exploratory. Nothing has run.

## The question

Which anchor control should T1 run? The candidates are:

- **the curves' recipe** (D68): hard re-anchoring every 72 box-hours, and a ×1.5 multiplier every
  400 steps with a 40K-step hold;
- **C1:** the same re-anchoring with a damped controller;
- **C2:** a soft, moving-average anchor with a weak tie to the student.

## What the curves showed

From the learners' `anchor.jsonl` (`experiments/size-curves.md`, Progress; both sizes alike):

| Stage | LR at its start | Coefficient carried in | KL at +40K steps | What the controller did |
|---|---|---|---|---|
| 1 | 2.0e-5 | 0.11 | ≈ 0.021 | Reached the band in ≈ 20K steps; one raise to 0.165 and back; KL 0.019–0.020 for the stage |
| 2 | 9.4e-6 | 0.11 | 0.008 | Hold ran out below the band; floor (0.01) within 2,400 steps; KL to 0.025 by ≈ +95K; raised to 0.076 / 0.051; settled at 0.0506, KL 0.021–0.022 |
| 3 | 7.7e-6 | 0.0506 | 0.007 | Hold ran out; floor; KL 0.021–0.022 at +110–120K, climbing |

A noisy spring fits the traces: the policy jostled by noisy updates, pulled back in proportion to
its distance from the anchor.

- **Early in a stage the coefficient hardly matters.** The KL climbs at a pace set by the LR and
  the updates' noise.
- **The resting KL scales as LR / coefficient.** At stage 2's LR, 0.11 rests near 0.009, and the
  coefficient that rests at 0.02 is ≈ 0.05; the controller found 0.0506.
- **The time to rest grows as 1 / (LR × coefficient).**
- **The controller acts ≈ 50× faster than the KL answers,** so it ratchets to its limits. It
  overshoots on the way back too, the oscillation Stooke, Achiam & Abbeel (2020) diagnose in
  plain multiplier updates.

## The candidates

Each changes one thing against the curves' recipe. Everything else is the curves': P1 + K2, v5,
`k2.toml`'s schedules on the run clock, the size's curves init, 1 learner + 2 actors × 2
processes, 288 box-hours, snapshots at 72 / 144 / 216 / 288, and the stage checks and drift limits.

### C1, a damped controller (hard re-anchoring kept)

- **The rule,** every 400-step window, after Ziegler et al. (2019), as TRL and TorchRL implement it:

      e = clip(KL / target − 1, −0.2, 0.2)
      coefficient ← coefficient · (1 + e · 400 / H)

- **What it does:** this is an integral controller on log c, with the error capped at ±20%. With
  H ≈ 100K steps, a window changes the coefficient by at most 0.08%, and a stage start at KL 0
  lowers it by ≈ 3% over 40K steps, not to the floor.
- **No hold** is needed.
- **Optional damping terms** (a PID form): a proportional term on e and a derivative term on its
  change. They're added only if the simulation shows the plain rule overshoots.
- **Open, set by the simulation:** H, and whether to add the P and D terms and their gains.

### C2, a soft anchor with a weak tie to the student

- **The anchor:** a moving average (EMA) of the policy's own weights, updated every window:

      anchor ← (1 − β) · anchor + β · policy

  It trails the policy continuously, so there's no reset to KL ≈ 0 and no transient. It follows
  TR-DPO's soft reference update, Magnetic Mirror Descent's magnet, and Elastic Reset's moving
  average.
- **The penalty:** the KL to the anchor at target 0.02, under the curves' controller. With no
  reset, its hold applies only at the run's start.
- **The tie:** a second, fixed penalty c_SL · KL(π ‖ π_student), weak enough to bind only far
  from the student. It's the long leash, as Elastic Reset resets its average to the initial model
  and AlphaStar keeps a KL to its supervised agent.
- **No re-anchors.** The snapshots at 72 / 144 / 216 box-hours are for the stop checks, each
  against the previous snapshot as in the curves.
- **Open, set by the simulation:** β (the anchor's lag) and c_SL.

## The comparator and the reading (to be fixed in the pre-registration)

- **Where:** on the size D31's curves select, from that size's curves init. Both candidates run at
  once, paired like a curves batch.
- **The comparator:** that size's curves runs, two seeds (or three), at the same 288 box-hours.
- **The measures** (on an eval box, 40,000 sets each on family A block `curves`, so each pairs
  with every curves seed's measures): each candidate's `full` and p50 against three copies of the
  student, and its `full` against three copies of BC s1. Six measures.
- **The reading:**
  - D = the candidate's Δu (`full` vs 3× student) − the mean of the curves seeds' Δu;
  - **a candidate passes if D > S**, D68's within-size spread (floored at 0.0043);
  - both pass: the larger D, if the two differ by more than S; otherwise C1, the smaller change;
  - neither passes: T1 runs the curves' recipe.
- **Reported, not deciding:**
  - D's 95% interval, from the curves' seed spread and the lab's noise;
  - the 6-box-day reading;
  - the held-out table;
  - the fraction of windows inside the band, any floor episodes, the drift from the student at
    each snapshot, and throughput.

## Before the rental

1. **Code** (`mortal/kl_anchor.py`, `train_online.py`):
   - C1's controller mode;
   - C2's moving-average anchor, updated on the learner every window and saved with the trainer's
     state, plus the fixed second penalty;
   - tests for each, as for re-anchoring (`tests/test_kl_anchor.py`).
2. **The simulation** ($0, the workstation):
   - fit the noisy-spring model to all four curves runs' recorded traces (stages 1–4 by then);
   - choose C1's H (and gains) and C2's β and c_SL. The aim: at every stage's LR the simulated KL
     enters the band within 10% of a stage and stays in it for ≥ 80% of windows, with no floor
     episode;
   - report the fit and the choice.
3. **Smoke runs** of both on the workstation (`experiments/curves_smoke.py`'s pattern).
4. **Configs and launch:** two configs from the selected size's curves config, launched with the
   curves' scripts (`curves_launch.sh`, generalized to these run names), and their measures.
5. The pre-registration (`experiments/anchor-control.md`) with a live quote, for the user's yes.

## Cost and time

- **Runs:** ≈ 576 box-hours, ≈ $350–440 at batch 1's prices.
- **Final measures:** ≈ $20.
- **The early-research cap** rises to $2.7K (D69).
- **Time:** after the curves' reading (≈ 2026-11-28), ≈ 4 days of runs and ≈ 1 day of measures.
  T1 starts ≈ 5–6 days later.

## Not in this

- **Combining C1 and C2:** if both pass, T1 may combine them only after its own test.
- The rising target (pilot A's mechanism).
- T1's opponents (RL-5).
- A second seed per candidate: the curves' spread stands in for it.
