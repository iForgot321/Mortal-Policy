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

## The early check, automated (2026-10-18)

- **`experiments/k2_early_check.py --ckpt SNAP --init INIT --states DIR --out FILE`** runs D59's
  three measures and writes the verdict (JSON, atomically, at the end).
  - Measures 1 and 2 come first (≈ 30 s). If either fires, measure 3 isn't played.
  - It reuses `rl_drift.measure` and `rl_diag_sampling.play`, with the diagnosis's states, key
    and nonces.
- **`fleet.py watch --early-check '<that command without --ckpt and --out>'`** launches it
  detached on the workstation once the `p25` snapshot is in (`--early-check-at`). Then:
  - a "stop" verdict downs the run (event `early_check_stop`);
  - "continue" is logged;
  - a check that dies without a verdict is reported once (`early_check_failed`) and never stops
    the run.
- **`fleet.py watch --anchor-alert 0.10`** logs an event when the trainer's batch KL, averaged over
  2 h, crosses 0.10 either way. It's an alert only. The trainer appends `anchor.jsonl` beside its
  state file, and the regular pull brings it back.
- **Tests** (`tests/test_k2_early_check.py`, 7): the verdict's thresholds and strict boundaries;
  the hook waiting for its snapshot, launching once, returning the verdict once, and reporting a
  check that died; the alert's crossings.
- **End to end:** the hook launched the real check on `p1-b`'s p25 snapshot. That gave reverse
  KL 0.3534 and agreement 81.12%, the diagnosis's numbers, so "stop", in 30 s.

## The retry's settings (pre-registered 2026-10-18, before tuning)

`experiments/configs/k2.toml`: P1 seed 1 (pilot A, from the `p1a-b0` snapshot, as `p1-a`) plus:

| K2 part | Setting | Why |
|---|---|---|
| Anchor | adaptive, reverse KL to pilot A, target 0.02 nats/decision, held constant (`target_growth = 0`) | 05 §5.1's start target. A rising target would push the KL toward the early check's limits (0.10 nats; agreement 90%) by p25 by design, so the anchor's decay waits for a later package |
| Anchor controller | start 0.1, ×/÷ 1.5 per 400 steps outside [target/1.5, target×1.5], floor 0.01, ceiling 10, hold 5 updates | The local smokes: 0.1 held the KL near 0.007, and factor 2 overshot |
| LR | 2e-5 · (1 + t/100K)^−0.3, floor 5e-6 | P0/P1's LR at the start. ≈ 6.4e-6 by the end (t ≈ 4.3M steps) |
| Clip ε | 0.2 · (1 + t/100K)^−0.2, floor 0.1 | P0/P1's at the start, 0.1 from t ≈ 3M |
| Entropy weight | 1e-3 · (1 + t/100K)^−0.5, floor 1e-4 | P0/P1's at the start, ≈ 1.5e-4 by the end |

The LR, clip and entropy schedules are fixed here, not tuned: a few hours can't show their long-run
effect. The tuning settles only the anchor's coefficient rule.

## Tuning (05 §2.2's allowance: ≤ 1 box-day; pre-registered 2026-10-18)

- **Question:** at full scale (400-step controller windows, P1's batch sizes), which coefficient
  rule holds the batch KL in the band [0.013, 0.03]?
- **Two runs in parallel**, each the full P1 fleet (learner + 2 actors, 2 processes each), from
  `p1a-b0`, under a 4 h hard deadline:
  - `k2-tune-a` (`k2-tune-a.toml`): the retry's settings above, adaptive;
  - `k2-tune-b` (`k2-tune-b.toml`): the same with 05 §5.1's literal fixed power law,
    c(t) = 0.05 · (1 + t/100K)^−0.3.
- **Cost:** ≤ 24 box-hours, ≈ $13–17 at 2026-10 prices; each watcher has `--max-dollars 10`.
- **Measures**, from each run's `anchor.jsonl` (pulled with the run dir), after its first hour:
  - the fraction of logged batch-KL values inside [0.013, 0.03];
  - their maximum;
  - the coefficient's trace;
  - the learner's drain rate against P1's.
- **Choice rule:**
  - The retry uses the rule with the larger in-band fraction, if it is at least 70%.
  - Within 10 points of each other, adaptive is kept (05 §5.1 states a target).
  - If neither reaches 70%: no launch. Report and adjust the controller.
- **Also, reported only:** the early check (`experiments/k2_early_check.py`) on each run's last
  checkpoint, as a dry run of the p25 check.

## Tuning result (2026-10-18)

Both runs were rented 00:23 UTC. Training started 01:08 (a) and 01:28 (b): actor setup took 40 min
on 256-thread hosts (pytest thread oversubscription, fixed in `setup_box.sh` 675c3db). They
stopped at their 4 h deadlines, 04:23 and 04:25. Cost: $7.20 + $6.92 = **$14.12**.
`experiments/k2_tuning_report.py`:

| After each run's first hour | `k2-tune-a` (adaptive) | `k2-tune-b` (fixed power law) |
|---|---|---|
| Readings / training time | 111 / 3.1 h | 100 / 2.9 h |
| **KL inside [0.013, 0.03]** | **100%** | 95% |
| KL mean / max | 0.0193 / 0.0267 | 0.0262 / 0.0314 |
| Coefficient | 0.1 → 0.01 (floor, 11 min) → 0.114 after one overshoot to 0.0325 at 49 min (inside the first hour); flat at 0.114 since | 0.050 → 0.043 on its power law |
| KL at the end | steady | still rising |
| Agreement with pilot A at the end | 95.0% | 93.6% |
| Learner steps (run clock at the end) | 68,800 | 61,600 |
| Fleet games/h (watcher, last / max) | 61,600 / 76,000 | 68,800 / 71,200 |

- **Choice, by the pre-registered rule:** both reach 70%, and they're within 10 points, so
  **adaptive is kept** (`k2.toml` as written). Beyond the rule, `b`'s KL was still rising as its
  coefficient decayed, which would carry it toward the early check's limits over 6 days.
- **Early-check dry runs** on each run's last checkpoint (`k2_early_check.py`, workstation;
  reported only). Both say "continue":

  | Run | Reverse KL | Agreement | Entropy | Δu at argmax vs pilot A (4,000 sets) |
  |---|---|---|---|---|
  | `k2-tune-a` | 0.0183 | 95.0% | 0.486 | +0.0052 ± 0.0053 |
  | `k2-tune-b` | 0.0324 | 93.2% | 0.455 | +0.0024 ± 0.0054 |

  - Checkpoints after ≈ 3 h (69K and 62K learner steps), so the Δu says nothing about the
    retry's strength.
  - The check's reverse KL on the diagnosis's states (0.018) matches the trainer's batch KL
    (≈ 0.019), so the live alert and the p25 check measure the same thing.
  - Entropy stayed near pilot A's 0.52; P1 had collapsed to ≈ 0.16 by p25.
- **Throughput:** both learners were EPYC 9354s (32 cores). They ran at ≈ 22K steps/h,
  against ≈ 29.6K for P1's average and 33.5K for P0's learner (which has no critic). The vCPU
  quota is ≈ 32 on every host, so core count doesn't explain it; the anchor's extra forward pass
  likely does (see Launch). Not measured: a learner profile would settle it.

## Launch (2026-10-18 11:04 UTC)

- **Approval:** quoted ≈ $250 (stop $275). At launch the live quote was $2.08/h (≈ $300), and the
  user said "Launch now" with the stop at $320. It rented at **$1.93/h**: learner 68591 (EPYC
  7B13), actors 101026 (Ryzen 9 7945HX) and 127613 (Ryzen 9 5950X). That projects ≈ $278 over
  144 h.
- **Commands:**

  ```bash
  F="python3 tools/fleet/fleet.py"
  $F up --run k2 --config experiments/configs/k2.toml --init workdir/fleet/runs/p1a-b0/snapshots/full.pth \
      --actors 2 --procs 2 --set dataset.num_workers=4 --set control.num_threads=8 --hard-hours 156 --verify --yes
  EC="env LD_LIBRARY_PATH=$HOME/miniconda3/envs/mortal/lib $HOME/miniconda3/envs/mortal/bin/python \
      experiments/k2_early_check.py --init workdir/pilot/qs-a/pilot-qs-a-final.pth --states workdir/diag/credit-a"
  setsid -f $F watch --run k2 --replace --allow-zen2 --budget-box-hours 432 --snapshot-at 0.25,0.5 \
      --max-dollars 320 --anchor-alert 0.10 --early-check "$EC" > workdir/fleet/runs/k2/watch.out 2>&1 < /dev/null
  ```

- **Watchdog:** 2026-10-24 23:04 UTC (156 h). `full` comes at 432 box-hours, ≈ 2026-10-24 11:00
  UTC with no losses. The p25 early check comes ≈ 36 h in, ≈ 2026-10-19 23:00 UTC. It runs on
  the workstation, which must stay on and not train meanwhile.
- **Learner speed:** the tuning learners ran ≈ 22K steps/h, against ≈ 29.6K for P1's average
  (`p1-b`: 4.27M steps in 144 h). P0's learner, which has no critic, ran 33.5K. The anchor's
  extra forward pass is the likely cost, so the same box-hours buy ≈ 25% fewer steps than P1
  had.

## Before launch (done)

- **Arm and seed: P1, seed 1** (decided 2026-10-18 05:30 UTC). G0's point estimates: P1
  −0.017 (seeds −0.0209 and −0.0131) against P0 −0.0268. Seed 1 (pilot A, from `p1a-b0`) was
  fixed beforehand. (The tuning ran on P1 before `p1-a`'s verdict, at the user's word. It
  settled only the coefficient rule.)
- **K2's settings** (`05` §5.1):
  - the anchor's form (reverse KL to the frozen π_SL, exact over the legal mask, on the learner)
    and its coefficient rule (fixed decay or a controller toward a target);
  - the target (0.02 nats/decision at the start) and its decay;
  - the LR, clip ε and entropy schedules.
- **Tuning:** at most 1 box-day of short sweeps (`05` §2.2), written down before they run, with
  what they choose.
- **Monitoring:** the trainer logs the batch's reverse KL to π_SL, the coefficient and the
  argmax agreement, so the early check's first two measures can also be watched live.
