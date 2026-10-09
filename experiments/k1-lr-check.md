# K1 critic learning-rate check (D52; pre-registered 2026-10-08, before any data)

- **Question:** the critic's constant learning rate for package K1 (arm P1): **1e-4 or 3e-4**.
  Every other critic setting is fixed by D52 (b8c128, MSE onto λ-returns, γ 1, λ 0.95, AdamW with
  weight decay 0, grad clip 1.0).
- **Budget:** part of P1's tuning allowance (≤ 1 box-day, `05` §2.2). It runs on the workstation,
  costs $0, and takes ≈ 6–7 h.
- **Not a strength claim:** no registry rows, no lab nonces. The games use their own key
  (`b'k1lrchk'`), outside the seed families.
- **Code:** `experiments/k1_lr_check.py`; config `experiments/configs/p1.toml` with
  `critic.warmup = true`.

## Procedure

1. **Data, generated once:** 100,000 hanchan from P1's B0 setup:
   - the policy is pilot A, frozen;
   - the trainee samples (`explore_rate = 1.0`) against three pilot-A baselines with the agari
     guard, as `TrainPlayer` does;
   - behaviour sidecars are on.

   B0 freezes the policy, so its games don't depend on the critic, and one fixed set serves
   both runs.
2. **One run per learning rate.** Each run trains a fresh critic, with the same initialization
   (`control.seed = 1`), through the unmodified trainer (`mortal/train_online.py`, B0, horizon
   H).
   - A replay server stands in for `server.py`. It hands both runs the identical logs in
     identical 800-log drains, in the same order.
   - The trainer logs its critic diagnostics every 100 steps.
3. **Metric: the trainer's own progressive explained variance.** The critic's values on each new
   chunk are compared with that chunk's Monte-Carlo returns *before* the critic trains on it, so
   every point is out-of-sample.

## Decision rule

- **Primary:** the mean explained variance over the windows in the last 20% of each run's steps.
- **Choose 3e-4 only if all three hold:**
  1. its primary metric is ≥ 0.01 above 1e-4's;
  2. its final calibration (10 bins) is monotone;
  3. it is stable: no NaN or inf in the critic loss or explained variance, and its last-20% mean
     is not more than 0.02 below its mean over 40–60% of the run.
- **Otherwise choose 1e-4,** the more conservative setting.
- **Reported, not decisive:** both curves, the critic loss, and calibration over time.
- **Note:** 100K hanchan is a fraction of a full B0 (0.3–0.5M, `05` §4). The check compares
  learning speed and stability early in B0; it doesn't certify convergence.

## Amendment (2026-10-09, before any critic result)

- **First attempt interrupted.** The 100K-game set finished (4.0 h, 0 missing sidecars, `6f80e88`).
  The 1e-4 run was then killed at step 100 of ~31K, when the system ran critically low on memory
  under P1's cloud loader settings (2 workers × 50 logs per file batch). That partial run is kept
  aside as `workdir/k1-lr-check/lr-1e-4.killed-20261008/` and not used.
- **Change:** both runs use **1 loader worker and 16 logs per file batch**, the same for both.
  - These settings only change memory and how the data is batched. The setting under test, and
    every other critic setting, stays as registered.
  - The driver now records peak memory, and stops a run cleanly if available memory falls below
    1.5 GB.
- **Unchanged:** data, drains, seed, metric and decision rule.

## Result

*(filled in after the runs)*
