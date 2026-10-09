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

## Result (2026-10-09)

Both runs saw the same 100,000 games in 125 identical drains (`4bd7936`). Each took 1.6 h with a
peak of 5.6–5.7 GB PSS; no memory-guard stops. Raw output: `workdir/k1-lr-check/` (`report.json`,
each run's `run.json`, `trainer.log` and tensorboard).

| LR | EV, last 20% | EV, 40–60% | EV, final window | Calibration monotone (final) | Critic loss (final) | Finite |
|---|---|---|---|---|---|---|
| **1e-4** | **0.0088** | 0.0059 | 0.0121 | no | 0.0411 | yes |
| 3e-4 | 0.0070 | 0.0038 | 0.0092 | no | 0.0412 | yes |

**Pre-registered decision: 1e-4.** 3e-4 was 0.0018 *lower* over the last 20%, against the 0.01
it needed to win. 3e-4 was stable by the registered test, so the choice doesn't rest on
instability. `experiments/configs/p1.toml` keeps 1e-4.

### Reading (not part of the registered rule)

- **Both critics explain under 1% of the return's variance after 100K hanchan, and neither
  reaches a monotone calibration.** Both curves are still rising (0.004–0.006 at mid-run →
  0.009–0.012 at the end). 100K hanchan is a fifth to a third of a full B0, so B0's stopping rule
  (`05` §4) was not expected to be met yet.
- **Horizon H makes the target mostly unpredictable** (`experiments/k1_h_ceiling.py`, post hoc,
  795K trainee decisions from the last 5,000 games):
  - Var(H return) is 0.480 u²; the current kyoku's reward accounts for 0.066 of it.
  - Even **perfect** knowledge of how the current kyoku ends would explain only **14%** overall:
    11% for decisions in earlier kyoku, and essentially all of it (0.9995) in the game's last
    kyoku (9% of decisions).
  - What a critic can actually reach is lower still, since a kyoku's outcome is partly wall luck,
    which the critic doesn't see.
  - So a low EV is expected under H. 0.009 is consistent with a critic early in training, not with
    a broken pipeline, but it doesn't prove the critic is near its ceiling.
- **Suggested before the real B0:** log EV separately for the last kyoku of the game, where the
  ceiling is ~1. That is the sharpest health check a critic has under H, and the overall EV hides
  it.

### Health check on fresh games (2026-10-09, post hoc)

The trainer now logs `critic/explained_variance_last_kyoku`. `k1_lr_check.py evaluate` scored the
saved critics and an untrained control (same seed) on 4,000 fresh B0 games (nonces from 100,000,
a range the check never used; 634,796 trainee decisions, 9.2% in the last kyoku). It goes through
the trainer's own `gae_targets`.

| Critic | EV, all | EV, last kyoku | EV, earlier kyoku | Calibration monotone (all / last kyoku) |
|---|---|---|---|---|
| untrained (control) | −0.0030 | −0.0104 | −0.0028 | no / no |
| **1e-4** | **+0.0134** | **+0.0634** | +0.0116 | yes / yes |
| 3e-4 | +0.0128 | +0.0614 | +0.0110 | yes / no |

**Reading:**
- **The critic is learning, not broken.** The control sits at ~0. The trained critics are about
  5× better in the last kyoku, as H's structure predicts, and both calibrate monotonically on
  fresh data.
- **Absolute skill is still low:** 6% in the last kyoku, against a ceiling of ~1 for perfect
  knowledge of the hand's outcome. Draw luck caps what is reachable, but a full B0 (0.3–0.5M
  hanchan) should raise it.
- **What to watch in the real B0:** `critic/explained_variance_last_kyoku` should keep rising.
- The 1e-4 vs 3e-4 ordering holds on fresh data, which is consistent with the registered choice.
