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

### First attempt aborted (2026-10-18 11:04–12:39 UTC, $2.30)

- **What happened:** the learner set up fine, at 11:18. All 6 actor attempts, on 6 hosts, failed:
  5 on the code upload (rsync "Broken pipe"), 1 host never ran. Re-provisioning actor1 alone
  worked (13 min upload, 8 min setup with the thread cap), but the ssh that started its clients
  dropped, and the watcher's pulls failed too.
- **Cause: the workstation's internet link.** It had 60% ping loss to 1.1.1.1 with 0% to the router,
  and uploads at ≈ 210 KB/s. The boxes were fine.
- **Why downed:** nothing could train without actors. And with every box looking unreachable,
  `watch --replace` would have destroyed healthy boxes and rented replacements that couldn't
  set up either. The learner had no drains, so its checkpoint was still the initialization:
  nothing was lost.
- **Fix:** `fleet.py watch` now checks its own link (`link_ok`: ping loss to 1.1.1.1 ≤ 20%) before
  judging any box unreachable, and logs `link_down` instead (`tests/test_fleet.py`).
- **Relaunch:** under a new run name (`k2` is taken), with the same config and commands, once the
  link is healthy again. It's within the user's approval for this retry.
  - At 17:10 UTC the live quote was $2.21/h (≈ $318, right at the $320 stop), and the link was still
    at 50–90% loss. The user chose to wait.
  - A detached script (`auto_launch_k2b.sh` in the session scratchpad, log `auto_launch.log`)
    launches run `k2b` with the Launch section's commands once both hold:
    - 20 straight minutes of ≤ 5% ping loss;
    - a live quote of ≤ $2.05/h (≈ $295).
  - It re-checks the price every 15 min and gives up after 48 h.
  - The fleet's rsync now resumes dropped transfers (`--partial-dir`, 12 tries for box syncs).

### Relaunch: run `k2b` (2026-10-18 17:59 UTC)

- **Launched by the waiting script:** the link had held 0% loss since ≈ 17:22, and the quote was
  **$1.80/h** (≈ $259 over 144 h). Learner 52976 (EPYC 9354), actors 13450 and 68591 (EPYC 7B13).
- **Bring-up took 9 min** (setup ≈ 7 min per box with the thread cap; 13450 had taken 40 min before
  the fix). Both actors' clients were running at 18:08.
- **Watchdog:** 2026-10-25 05:59 UTC.
  - **p25 early check** (108 box-hours) ≈ 2026-10-20 06:00 UTC.
  - **full** (432 box-hours) ≈ 2026-10-24 18:00 UTC with no losses.
- The commands are those of the Launch section with `--run k2b`, and the watcher flags are the
  same (`--max-dollars 320`).

## Early check at p25 (2026-10-20): continue

- **p25 snapshot:** 06:02 UTC on 2026-10-20, step 1,048,400, 108.16 box-hours, sha256
  `286cf185ba093274…`. The watcher launched the check on the workstation at once; it took 59 min.
  `workdir/fleet/runs/k2b/early-check-p25.json`.

  | # | Measure | `k2b` p25 | Stop if | P0/P1 at p25 |
  |---|---|---|---|---|
  | 1 | Reverse KL(π_p25 ‖ π_SL), mean per decision (15,215 decisions) | **0.0253** (median 0.0082) | > 0.10 | 0.35–0.38 |
  | 2 | Argmax agreement with π_SL | **94.2%** | < 90% | 78–81% |
  | 3 | Δu at argmax vs pilot A, 4,000 sets, diagnostic key | **+0.0120 ± 0.0055** (≈ [+0.001, +0.023]), avg rank 2.484 | CI entirely below 0 | p50: −0.022 to −0.033 (family A) |

- **Verdict: continue** (nothing fired). The policy's entropy on these states is 0.47; P1's
  collapsed to 0.11.
- The anchor holds the drift at K2's target, an order of magnitude below the unanchored runs.
  This is also the first online checkpoint whose Δu against its initialization is above 0, where
  every P0/P1 checkpoint measured was below it.
- **Not a verdict** (pre-registered above): exploratory, 4,000 sets on the diagnostic key, argmax
  play. Passing means only that the run continues. G0's T-std on family B, `full` vs pilot A,
  decides (≈ 2026-10-24 18:00 UTC).

## Doubling estimate (D49): pre-registered 2026-10-21, before `full` exists

The user approved it on 2026-10-21 (`docs/design/observation-review-d49.md`, option 1). The p50
snapshot exists (2026-10-21 18:00 UTC, step 1,964,400, sha256 `7d5937a792f0ca3e…`); `full` does
not yet.

- **Measurement:** after G0's T-std, on the workstation (≈ 15 h on the 3060):

  ```bash
  python -m lab measure --sets 100000 --family B --arm P1 --package K2 --horizon H \
      --challenger workdir/fleet/runs/k2b/snapshots/full.pth \
      --champion workdir/fleet/runs/k2b/snapshots/p50.pth \
      --description "k2b full vs p50 (doubling estimate, D49)"
  ```

  Fixed N on fresh family-B nonces, AMP on, Δu with a bootstrap 95% CI: the same design and size
  as P0's (`experiments/p0.md`), so the two compare directly. 100K sets give a half-width of
  ≈ 0.002 u.
- **It runs whatever G0's verdict is,** so the choice to measure doesn't depend on the verdict.
  The T-std goes first; this measure doesn't enter D59's verdict.
- **Reading,** P0's thresholds, against v5's break-even of 0.002–0.004 u per doubling
  (`experiments/sp-ablation.md` §7):
  - CI lower bound > 0.004: a doubling clearly pays for v5's extra games. The G0 review weighs a
    switch.
  - CI upper bound < 0.002: it doesn't pay for them. v4 stays.
  - Otherwise grey: no switch on this evidence alone; the user decides at the review with the
    memo's other factors. (P0's rule sent grey to D31's actor-size curves, which were never run.)
  - If the retry fails G0, the number is reported but doesn't decide v4 vs v5: the main path is
    then the SL incumbent plus search, where the question is SL strength and rollout speed (the
    memo).
- **Caveat stated before the result:** K2's anchor target is constant (`target_growth = 0`), so
  the policy stays within ≈ 0.02 nats of π_SL. This measures what a doubling buys under that
  anchor; later doublings may buy less as the policy nears it. Both checkpoints sit near π_SL by
  construction, so unlike P0's, it shouldn't include recovery from drift.
- **Not measured (the user, 2026-10-21):** p50 vs p25, the doubling before.
- **Started by itself** (the user, 2026-10-21): `experiments/k2b_g0_chain.sh` runs detached on the
  workstation since 2026-10-22 03:10 UTC (log `workdir/fleet/runs/k2b/g0-chain.log`). Once the
  watcher's `full` row is in `snapshots.jsonl` and the file's sha256 matches it, the chain runs G0's
  T-std (`full` vs pilot A, the command in the handoff's §1), then this measure. It skips a step
  whose registry row exists for these checkpoints. `resume-supervision.sh` restarts it after a
  reboot; an interrupted step starts again on fresh nonces.

## G0 result (2026-10-24): the retry passes

- **The run:** `k2b` reached its 432 billed box-hours at 18:04 UTC on 2026-10-24: `full` at step
  3,880,000 (sha256 `f5a22c5c16d3d51e…`). Boxes destroyed at 18:04:33; **$259.09** in all (approved
  ≈ $300, stop $320).
- **The T-std**, started by `k2b_g0_chain.sh` at 18:05:58 on the workstation:

  | Test | Challenger vs champion | Verdict | Sets | Δu | ≈ 95% CI | Avg rank | Run id |
  |---|---|---|---|---|---|---|---|
  | T-std (family B) | `k2b` full vs pilot A | **pass** | 7,000 | **+0.0114** | [+0.003, +0.019] | 2.483 | `20261024-110559-T-std` |

  - The GSPRT crossed its upper bound at its fifth look (one every 500 sets from 5,000): LLR +3.88
    against +2.944 (5,000 sets: +1.32; 6,000: +1.13; 6,500: +2.90). Nonces [1,000,920,000,
    +100,000); 18:05–19:20 UTC.
  - The CI is the normal approximation, mean ± 1.96 · sd/√n with sd 0.340; it isn't part of the
    test, and a sequential mean that stopped on crossing is biased upward.
  - Rank histogram over 28,000 games: 7,147 / 7,061 / 6,905 / 6,887.
- **Reading by D59:** **pass. The G0 retry passes**, the first online arm-seed to beat its
  initialization (P0 −0.0268, `p1-a` −0.0209, `p1-b` −0.0131). Consistent with the p25 early check
  (+0.012 ± 0.0055 on the diagnostic key).
- **Next, by the roadmap (`00` G0 row, D59):** L2's confirmation, a T-confirm on fresh family-C
  nonces, before the arm is selected; the fleet soak (`03` §3.7) before T1. Both need the user:
  the confirmation's N is fixed in advance, and the soak is a rental. The doubling estimate (above)
  runs first; it started at 19:20 UTC (`20261024-122034-measure`).

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
