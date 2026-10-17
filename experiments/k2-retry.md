# The G0 retry with K2 (roadmap G0 row, D59): pre-registration

**Status:** being written (2026-10-17). The procedure below is fixed. The arm, the seed and
K2's settings are filled in before launch, and nothing here changes once the run starts.

## Why

P0 and P1 failed G0 (`experiments/p0.md`, `experiments/p1.md`). The diagnosis
(`experiments/rl-diagnosis.md`) found:
- clean credit;
- a weak signal, whose training-game gain is less than lowering the initialization's
  temperature;
- unanchored drift: by p25, 20–25% of argmax decisions changed, at a reverse KL of
  ≈ 0.4 nats/decision.

K2 (`05` §5.1) adds the KL anchor and anneals the update size, LR and entropy together.

## Fixed now (D59)

- **One seed.** Seed 1's `full` gets G0's T-std against its initialization on family B.
  - **Pass:** the retry passes. L2's confirmation on fresh family-C nonces follows before the
    arm is selected.
  - **Fail:** the retry fails. The roadmap's next row applies: the SL incumbent plus inference
    search becomes the main path.
  - **Inconclusive** (the T-std's cap): a second seed only with the user's go-ahead, and only if
    seed 1's point estimate is positive. The arm passes if seed 2 passes.
- **Early check at seed 1's p25 snapshot**, on the workstation, exploratory:

  | # | Measure | Tool | Stop if | P0/P1 at p25 |
  |---|---|---|---|---|
  | 1 | Reverse KL(π_p25 ‖ π_SL), mean per decision | `rl_drift.py` on the diagnosis's states | > 0.10 nats | 0.35–0.38 |
  | 2 | Argmax agreement with π_SL | `rl_drift.py` | < 90% | 78–81% |
  | 3 | Δu at argmax vs the initialization, 4,000 sets on the diagnostic key | `rl_diag_sampling.py --mode argmax` | 95% CI entirely below 0 | p50: −0.022 to −0.033 (family A) |

  - The states are the 200 kept games per pilot (`workdir/diag/credit-a`, `credit-b`), every
    second decision. That's the same set as the diagnosis, so the numbers compare directly.
  - Any one firing stops the run (its fleet goes down) and the result is reported, with no
    verdict recorded. The user decides on a restart inside the cap, or calls the retry failed.
  - Passing the check means only that the run continues.
- **Cap:** ≤ $0.6K from the reserve, the tuning allowance included. One seed is ≈ $250 at
  2026-10 prices.
- **Budget checkpoints:** p25, p50 and full, as before (D49). The same 432 billed box-hours as
  P0/P1, so `full` compares with their results.

## To fill in before launch

- **Arm and seed:** the better arm by G0's point estimates once `p1-a`'s T-std is in. P1 leads
  now (`p1-b` −0.0131 against P0's −0.0268).
- **K2's settings** (`05` §5.1):
  - the anchor's form (reverse KL to the frozen π_SL, exact over the legal mask, on the learner)
    and its coefficient rule (fixed decay or a controller toward a target);
  - the target (0.02 nats/decision at the start) and its decay;
  - the LR, clip ε and entropy schedules.
- **Tuning:** at most 1 box-day of short sweeps (`05` §2.2), written down before they run, with
  what they choose.
- **Monitoring:** the trainer logs the batch's reverse KL to π_SL, the coefficient and the
  argmax agreement, so the early check's first two measures can also be watched live.
