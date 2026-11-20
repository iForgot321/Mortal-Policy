# D69's anchor-control pilot: design draft (2026-11-20)

**Status: draft.** Decided in outline by D69 (the user, 2026-11-20: "Add tests for 1 and 2 to the
plan before T1, feel free to increase the budget accordingly"). **Amended the same day** after
the simulation (below), by the user: "C2 only; C1 into T1".
- The pilot tests C2 alone.
- C1 goes into T1, without a strength test, if C2 doesn't pass.

It becomes `experiments/anchor-control.md`, pre-registered, once C2's parameters are refitted on
batch 2's traces. It runs after the user's yes on a live quote. Exploratory. Nothing has run.

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

Each changes one thing against the curves' recipe. **Since the amendment only C2 runs in the
pilot.** C1 is kept here as the design T1 adopts if C2 doesn't pass. Everything else is the curves': P1 + K2, v5,
`k2.toml`'s schedules on the run clock, the size's curves init, 1 learner + 2 actors × 2
processes, 288 box-hours, snapshots at 72 / 144 / 216 / 288, and the stage checks and drift limits.

### C1, a damped controller (hard re-anchoring kept): for T1, not run in the pilot

- **The rule,** every 400-step window, after Ziegler et al. (2019), as TRL and TorchRL implement it:

      e = clip(KL / target − 1, −0.2, 0.2)
      coefficient ← coefficient · (1 + e · 400 / H)

- **What it does:** this is an integral controller on log c, with the error capped at ±20%. With
  H ≈ 100K steps, a window changes the coefficient by at most 0.08%, and a stage start at KL 0
  lowers it by ≈ 8% over 40K steps, not to the floor.
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
  with every curves seed's measures): C2's `full` and p50 against three copies of the student,
  and its `full` against three copies of BC s1. Three measures.
- **The reading:**
  - D = C2's Δu (`full` vs 3× student) − the mean of the curves seeds' Δu;
  - **C2 passes if D > S**, D68's within-size spread (floored at 0.0043): then T1 runs C2 as tested;
  - otherwise T1 runs the curves' re-anchoring with C1's controller (horizon 25,000, kp 2.0).
- **Reported, not deciding:**
  - D's 95% interval, from the curves' seed spread and the lab's noise;
  - the 6-box-day reading;
  - the held-out table;
  - the fraction of windows inside the band, any floor episodes, the drift from the student at
    each snapshot, and throughput.

## Before the rental

1. **Code: done** (branch `d69-anchor-control`, `2ccb2f3`; merged into `main` after batch 2 is
   down, since a replacement box gets this checkout's working tree):
   - `mortal/kl_anchor.py`: C1's `'proportional'` mode, and C2's moving average (`ema`), saved
     with the trainer's state;
   - `train_online.py`: the moving-average update, and the tie (`[policy.kl_tie]`);
   - 5 new tests; the suite passes (243).
2. **The simulation** ($0, the workstation):
   - fit the noisy-spring model to all four curves runs' recorded traces (stages 1–4 by then);
   - choose C1's H (and gains) and C2's β and c_SL. The aim: at every stage's LR the simulated KL
     enters the band within 10% of a stage and stays in it for ≥ 80% of windows, with no floor
     episode;
   - report the fit and the choice.
3. **A smoke run** of C2 on the workstation (`experiments/curves_smoke.py`'s pattern), and of C1
   before T1.
4. **Config and launch:** one config from the selected size's curves config, launched with the
   curves' scripts (`curves_launch.sh`, generalized to this run name), and its measures.
5. The pre-registration (`experiments/anchor-control.md`) with a live quote, for the user's yes.

## Simulation (2026-11-20, batch 1's traces; branch `d69-anchor-control`)

`experiments/anchor_control_sim.py` (output `workdir/lab/anchor-control-sim.json`) fits the
noisy-spring model to `curves-b12-s1` and `curves-b16-s1`, stages 1–3 (stage 3 partial).

**The fit:**
- a common pull k = 25.1, and a drive D per stage: 1.34e-6, 4.44e-7, 2.47e-7 per step;
- the record's window-to-window noise is ≈ 1.1% in log;
- the residual is 0.21 in log: one D per stage misses the changes within a stage.
- A first version tied D to the LR by a power law and used the residual as noise. It put the
  drafted controller on its floor in stage 1, where the record never has it, and was replaced.

**Closed loop, the drafted controller against the record:**

| | Simulated | Recorded (b12 / b16) |
|---|---|---|
| Stage 2, floor windows | 126 | 129 / 132 |
| Stage 2, coefficient at the end | 0.0506 | 0.0506 / 0.0506 |
| Stage 2, windows in the band | 94% | 94% / 94% |
| Stage 1, floor windows; end coefficient | 22; 0.17 | 0; 0.11 (the model's weakest stage: the fast start from the student) |

**The record itself: the drafted controller's transients are small.**
- Stage 2 was in the band for 94% of its windows in both runs. It sat below the band for the first
  ≈ 65K steps (6% of the stage), on the floor for 4–5% of windows, and above the band for
  0.1–0.2%.
- Stage 3, still running (≈ 750 windows): 69–73% in the band so far, the floor at 32–48%.

**The candidates** (worst case over stages and 5 noise seeds):

| Setting | In band | Into the band by | Floor windows | KL max |
|---|---|---|---|---|
| Drafted (×1.5, hold 100) | ≥ 93% | ≤ 7% of a stage | 126–328 per stage | 0.0260 |
| **C1: H 25,000, kp 2.0, kd 0** | **≥ 92%** | **≤ 8%** | **0** | **0.0233** |
| C1: H 25,000, kp 0.5 | ≥ 89% | ≤ 11% | 0 | 0.0252 |
| C1: H 100,000 (the draft's example) | ≥ 64–68% | ≤ 32–36% | 0 | 0.0225–0.0234 (too slow for each stage's needs) |
| **C2: ema 5e-4** (lag ≈ 800K steps) | **≥ 96%** | **≤ 4%** (the run's start only) | (the penalty often at its floor; the moving anchor holds the band) | **0.0265** |
| C2: ema 2e-3 or more | 0–96% | — | — | the anchor catches up too fast; the KL falls out of the band |

- **Robustness:**
  - At twice the fitted drive, C1 and C2 both do better.
  - At half the drive, C1 is in the band ≥ 84%, enters by 16% and reaches its floor, but never
    rises above the band (KL max 0.0223). The drafted controller is on the floor for 1,888 windows.
  - So for C1, overshoot, not floor time, is the measure that matters.
- **Proposed parameters:**
  - **C1:** horizon 25,000, kp 2.0, kd 0, p_clip 0.2. The derivative term made no difference.
  - **C2:** ema 5e-4, the curves' controller, and a tie of c_SL ≈ 0.008. That tie's own resting
    KL to the student is ≈ 0.10 at the run's last LR, two-thirds of the 0.15 stop limit.

**What it means for the pilot.** Over a stage, C1 changes little but the transients:
- In the record, the drafted controller lost ≈ 6% of stage 2 to its floor-and-overshoot, and
  never went much above the band.
- C1 removes that transient, but a strength difference from ≈ 6% of each stage is unlikely to
  exceed S, the curves' seed spread, so the pilot probably can't measure C1's effect.
- C2 changes the anchoring itself: no resets, and a trust region that moves continuously. That
  is the substantive question.

The fit is refitted with batch 2's traces before the pre-registration is fixed.

## Cost and time

- **The run:** 288 box-hours, ≈ $175–220 at batch 1's prices.
- **Final measures:** ≈ $10.
- **The early-research cap** rises to $2.5K (D69, amended).
- **Time:** after the curves' reading (≈ 2026-11-28), ≈ 4 days of runs and ≈ 1 day of measures.
  T1 starts ≈ 5–6 days later.

## Not in this

- **Combining C1 and C2:** if both pass, T1 may combine them only after its own test.
- The rising target (pilot A's mechanism).
- T1's opponents (RL-5).
- A second seed per candidate: the curves' spread stands in for it.
