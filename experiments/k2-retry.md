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

## Implementation (2026-10-17)

- **The anchor** (`mortal/kl_anchor.py`, `ppo.reverse_kl`, `[policy.kl_anchor]` in the config):
  - The penalty is c_KL · KL(π_θ ‖ π_SL), exact over the legal mask, averaged over decisions
    with more than one legal action, as `rl_drift.py` measures it.
  - π_SL is a frozen copy on the learner, by default `baseline.train.state_file`.
  - Two coefficient rules:
    - `fixed`: c(t) = c₀ · (1 + t / t₀)^−p, 05 §5.1's power law;
    - `adaptive`: PPO's adaptive-KL rule toward a target, which may grow over the run (the
      anchor's decay). It never lowers the coefficient before the KL first reaches the target
      band.
  - The coefficient's state saves with the checkpoint. TensorBoard gets `anchor/kl`,
    `anchor/agreement`, `anchor/coef` and `anchor/target`.
  - `tests/test_kl_anchor.py` (8 tests): exactness over the mask, illegal logits never
    entering, finite gradients, zero gradient at the anchor, the frozen copy, both rules, and
    the state round trip.
- **Local smoke** (`experiments/configs/k2-smoke.toml`: P1's smoke plus the anchor, the
  controller every 5 steps; 10 min on the workstation from `p1a-b0`):
  - 150 learner steps. The batch KL rose from 0.00005 to 0.020 and agreement fell from 100%
    to 94%. First-minibatch ratios were 1.00004, with no NaNs.
  - **Run 1:** the first version of the controller halved the coefficient from 0.1 to its floor
    (1e-4) while the KL was still near 0, before the policy had moved. Hence the hold: no
    decrease before the KL first reaches the band.
  - **Run 2,** with the hold: 0.1 held the KL near 0.007 (agreement ≈ 97%), below the band for
    good. So the hold ends after `hold_windows` updates (default 5).
  - **Run 3,** with the ending hold: after 5 windows the controller halved each window. The KL
    lags the coefficient, so it went down to the floor before the KL entered the band
    (0.013–0.023, agreement ≈ 95%).
  - **So the default floor is now 0.01** (the 0.02 target needs ≈ 0.03 at this LR). The factor,
    floor and starting coefficient are for the tuning sweep. At full scale the KL reached 0.02
    in ≈ 150 steps, inside the first 400-step window, so the hold rarely matters there.
- **The schedules** (`mortal/schedules.py`, `[policy.schedules]`):
  - The policy LR, clip ε and entropy weight follow value(t) = clip(v₀ · (1 + t/t₀)^−p, lo,
    hi). They share a time scale t₀ unless one sets its own.
  - A name left out keeps its constant. An unscheduled LR stays with
    LinearWarmUpCosineAnnealingLR.
  - TensorBoard gets `schedule/t`, `schedule/lr`, `schedule/clip_ratio` and
    `schedule/entropy_weight`. `tests/test_schedules.py` has 4 tests.
- **Schedule smoke** (`k2-smoke.toml` with t₀ = 50; 5 min, then 3 more from the saved
  checkpoint):
  - The logged LR, clip and entropy values match the formula at each t (e.g. t = 30: LR
    1.58e-5, clip 0.174, entropy 0.00079).
  - The restart resumed at t = 75 from the same origin (step 216,800), with the anchor's
    coefficient restored.
  - **It found a crash:** `toml` parses inline tables (`lr = { v0 = … }`) into a class pickle
    can't save, and the trainer pickles its config into every checkpoint. `mortal/config.py`
    now converts the parsed config to plain dicts (`tests/test_config.py`).
- **The run clock** (both): t counts learner steps since the run's first launch. Its origin is
  saved in the checkpoint (`schedule_origin`), so a restarted trainer keeps it.
  - The anchor's first version used the checkpoint's global `steps`. A run from a B0 snapshot
    starts at ≈ 217K–245K of those, so a decay would have begun well along. It's fixed: the
    anchor runs on the same clock.

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
