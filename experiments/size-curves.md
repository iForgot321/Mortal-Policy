# D31's actor-size curves: the pre-registration (fixed 2026-11-18)

**Status: fixed** (2026-11-18). The design is D68 (the user: "Update the plan based on the default
recommendation"), with the controller's hold the user chose ("Go with option 1"). The user approved
the rental at the live quote, ≈ $745–750 for the whole chain ("Whole chain"). Nothing here is
changed after the first run starts. Exploratory.

## The question

Which qualified student should RL train from here on: **b12c256 or b16c256?** Both play level with
BC s1 (`experiments/students-s1.md`): the T-NIs passed at +0.0047 and −0.0021, and the golden set
within 0.5 pp everywhere. The qualification shows they reproduce the teacher, not how far RL can take
each (D31). A bigger net may learn more; it also costs more per game, so the same dollars buy it fewer
games. D31 measures which effect wins.

## The runs

| Run | Starting student (EMA weights) | `control.seed` | Batch |
|---|---|---|---|
| `curves-b12-s1` | b12c256, `workdir/secure/runs/students-s1/box/students-s1/ema-b12c256.pth` (`8f8031f4…`) | 1 | 1 |
| `curves-b16-s1` | b16c256, `…/ema-b16c256.pth` (`8aedb80b…`) | 1 | 1 |
| `curves-b12-s2` | b12c256 | 2 | 2 |
| `curves-b16-s2` | b16c256 | 2 | 2 |
| (`curves-b12-s3`, `curves-b16-s3`) | only if the reading is close (below) | 3 | 3 |

Each batch runs two fleets at once. Pairing across sizes keeps size from being confounded with the
period or the hosts.

**Shared by every run** (`experiments/configs/curves-*.toml`, from `k2-v5.toml`; v5):

| Part | Setting |
|---|---|
| Fleet | 1 learner + 2 actors × 2 processes; learner settings as the soak (4 loader workers, 4 trainer threads) |
| Budget | 288 box-hours (12 box-days, ≈ 4 days); snapshots at 72, 144 (the 6-box-day reading), 216 and 288 box-hours |
| Engine | P1 + K2: PPO with stored behaviour probabilities, horizon H, GRP-Δu in fixed units, utility [90, 50, −10, −130], AMP |
| Critic | K1, b8c128. One B0 warm-up per student (policy frozen, until its 10-bin calibration is monotone and its explained variance plateaus), shared by that student's seeds |
| Opponents | 3 frozen copies of the run's starting student |
| Schedules | `k2.toml`'s, on the run clock, not restarted at a re-anchor: LR 2e-5 · (1 + t/100K)^−0.3 (floor 5e-6), clip 0.2 · (1 + t/100K)^−0.2 (floor 0.1), entropy 1e-3 · (1 + t/100K)^−0.5 (floor 1e-4) |
| Anchor | Stage 1: the starting student, target 0.02 nats. Every 72 box-hours (24 h) the current weights become the anchor: 4 stages |
| KL controller | Adaptive, start 0.11 (`k2v5`'s settled coefficient on v5), tolerance 1.25, factor 1.5, floor 0.01, ceiling 10, every 400 steps; the coefficient carries across re-anchors. **Hold 100 windows:** at the start and after each re-anchor it isn't cut before the KL first reaches 0.016 or 40K steps pass (below) |

## Checks while a run trains

At each of the first three stage boundaries, on the workstation (no lab nonces are shared with the
measures below; family A, block `curves-boundary`):

1. **The stop check:** 10,000 sets of the new snapshot against 3× its anchor (the previous stage's
   snapshot). **If the 95% CI is entirely below 0, the run stops** and the result goes to the user.
   The re-anchor isn't held for it: training continues on the new anchor while the check plays.
2. **Monitoring with stop limits** (`experiments/rl_drift.py` on the starting student's own states):
   the run stops if its reverse KL to the starting student exceeds **0.15 nats**, its argmax
   agreement falls below **88%**, or its entropy falls below **70%** of the starting student's.
   For scale, after one re-anchor in the pilot: KL 0.045–0.050, agreement 93%, entropy 76–82%.

**How they run** (`experiments/curves_stage_check.py`, started by `fleet.py watch --stage-check`):
- **One check per re-anchor request.** The request records the stage's anchor: the student for
  p25, then the previous snapshot. The watcher starts the check on the workstation, detached. On a
  `stop` verdict it downs the run; that's a teardown, already authorized.
- **Drift first** (≈ 1 min): the limits on the student's states, 15,635 decisions for b12c256 and
  15,331 for b16c256. The students' entropies are 0.5166 and 0.5177, so the entropy limit sits
  near 0.362.
- **Then the stop check**, if no drift limit fired: a lab `measure`, `--arm P1 --package K2
  --horizon H`, AMP and game logs on (a stop rule keeps its games). Its description is
  `curves boundary: <run> <label> at 3x <anchor label>`.
  - A batch's two runs reach their boundaries together, so the measures take turns under a
    workstation lock.
  - A measure already in the registry is read, not replayed, so a check relaunched after a
    reboot costs nothing.
- **Time:** the bare arena plays b12c256 at ≈ 48K and b16c256 at ≈ 42K hanchan/h on the 3060
  (`tools/fleet/bench_1v3.py`, 2026-11-18). So each 10K-set check plays in about an hour, plus the
  lab's overhead, and both of a boundary's checks fit well inside the 24 h stage.
- **Failures:**
  - A check that dies without a verdict is relaunched once, then reported.
  - A check that can't be made writes an `error` verdict.
  - Neither stops the run.
- **The lab's lease:** while an eval box holds the registry, the workstation's checks wait for it
  (an error after 48 h). So the final measures play only after batch 2 has passed its last
  boundary: in practice, all twelve on one eval box after batch 2's `full`. The reading needs all
  four runs anyway.

## The measures (after the last run's `full`; one eval box)

Family A, block `curves`, AMP, 40,000 sets each, fixed N:

| Measure | For |
|---|---|
| Each run's p50 against 3× its starting student | the 6-box-day reading |
| Each run's `full` against 3× its starting student | the 12-box-day reading |
| Each run's `full` against 3× BC s1 (EMA) | the held-out table (reported) |

**Δu per dollar** for a reading is its Δu over the run's cost at the planning price
($13.2 per box-day), so host prices don't enter the comparison; billed costs are reported beside it.

## Reading, fixed now (D31, D65, D68)

With each size's figure the mean of its seeds' Δu per dollar at 12 box-days, L = b16c256's figure −
b12c256's, and S the larger of the two within-size ranges (floored at 0.0043 Δu over a run's
cost, the lab's noise for a difference of two runs):

1. **L > S:** b16c256. **Otherwise b12c256** (the cheaper net wins ties).
2. **Crossover:** if b16c256 trails at 6 box-days but leads at 12, b16c256, under the same condition.
3. **Close (0.5 S < L < 1.5 S, on either reading):** a third seed per size (batch 3), then rules 1–2
   on three seeds. The two runs (≈ $340) pass the early-research cap, so they need the user's yes at
   that time; if declined, rules 1–2 decide on two seeds.

**Reported, not deciding:** a 95% interval for L from all seeds and the lab's measurement noise; each
run's curve (p25 to `full`), whether each re-anchor's early gain repeats, the held-out table's
carry-over, actor and learner throughput per size, and the stop checks' and limits' values.

## Before the first rental

Where each item stands (2026-11-18; nothing rented):

1. **In-run re-anchoring: done.**
   - `fleet.py reanchor --label pXX`, which `watch --reanchor` runs at each budget snapshot before
     `full`, writes `reanchor.json` beside the trainer's state file. It names the learner's own
     copy of the snapshot, after checking that copy against the snapshot's sha256.
   - The trainer checks for it at start and at every save (`mortal/kl_anchor.py`,
     `train_online.py`). A new request becomes the anchor once the file's sha256 matches. The
     coefficient carries over, and the start-of-run hold applies again (the KL starts again from
     ≈ 0). The KL logged so far in the window, measured against the old anchor, is dropped.
   - **A restart can't lose the anchor:** its weights are saved in the trainer's state. A
     replacement learner also gets the snapshot and the request pushed, in case the last pulled
     checkpoint predates the pickup.
   - Each pickup or refusal is logged to `reanchor.jsonl`. The watcher reads it after each pull
     and logs `reanchored`, `reanchor_failed`, or `reanchor_pending` if nothing has come back
     30 min after the request.
   - Tests: 5 in `tests/test_kl_anchor.py`, 4 in `tests/test_fleet.py`. The second runs the
     box-side check script against real files.
   - **The smoke passed** (the workstation, 2026-11-18, `experiments/curves_smoke.py` on
     `experiments/configs/curves-smoke.toml`: b12c256 from its EMA weights, saves every 5 steps):

     | Check | Seen |
     |---|---|
     | p25 asked for at step 20 | picked up at the next save (step 25), coefficient 0.11 carried; KL to the anchor 0.0070 → 0.0009 |
     | Trainer restarted with p25's file moved away | "kl anchor stage 2" from its own checkpoint; KL 0.0056 before, 0.0061 after (against p25, not the student) |
     | A request with a wrong sha256 | refused, logged with the file's hash; training went on |
     | p50 asked for | stage 3 at step 75, coefficient 0.0733 carried (one cut once the hold had ended) |
     | A `submit_param` without the token | `{'status': 'denied'}` (item 2); the trainer's own requests went through |

2. **`server.py`: done in code; the box check is left.**
   - Tunnel connections reach `server.py` from 127.0.0.1, like the trainer's, so the peer's
     address can't tell them apart. Instead, at each start `server.py` writes a fresh token
     beside its buffers, owner-only. `submit_param` and `drain` must carry it (`mortal/common.py`)
     or get `denied`, and `submit_param` now waits for the server's `ok`.
   - The actors' two requests need no token, and actors never have the file. Pulls leave it on
     the box.
   - Tests: 2 in `tests/test_transport.py`. `tests/test_value.py`'s server + trainer loop runs
     through it, and so did the smoke.
   - The check on a box comes with the B0s: their learner is the first box to run it.
3. **`fleet.py`: done.**
   - `fleet.py up --max-dph` and `evalbox.py up --max-dph` set a per-box limit in the run's
     state. Every offer the run picks stays within it: the first, a provisioning retry's, a
     replacement's and a pinned machine's.
   - Without the flag there is no limit, as before.
   - Test: `test_every_box_of_a_run_stays_within_its_price_limit`.
4. **The two B0s: waiting for the rental's yes.**
   - A student's file holds weights and config only, so its B0 starts from
     `stage_init.py --from-policy` (weights, network, step 0, a fresh AMP scaler).
   - Each B0's `full`, **without its run clock** (`stage_init.py`), is then its size's init.
     Otherwise the B0's clock carries over: `k2v5`'s schedules began at t = 226,800 (LR ≈ 1.4e-5,
     clip 0.158), and the table above starts them at t = 0.
5. **v5 states for the monitoring: done.**
   - Each student's 200 kept games against three copies of itself
     (`experiments/rl_credit_check.py --work workdir/diag/credit-b12c256`, `credit-b16c256`,
     nonces 500,000 + 50 sets on the diagnostic key).
   - On them, `experiments/rl_drift.py` reads 15,635 and 15,331 decisions with a choice. The
     students' own entropies are 0.5166 and 0.5177, so the entropy stop (70%) is at ≈ 0.362 for
     both.
   - The credit check's alignment holds for both: the kyoku's reward against the trainee's point
     change, r = +0.847 and +0.849 at offset 0 (≤ 0.017 at ±1).
6. **The full test suite as a box runs it: passes.** `env -u MORTAL_CFG pytest tests/`: 219
   passed, 5 skipped (14 new). No Rust changed.
7. **The stage-boundary checks: done** (above, "How they run").
   - Tests: 9 in `tests/test_curves_stage_check.py`.
   - A dry run on the smoke's snapshots read drift against the student (reverse KL 0.0166,
     agreement 94.9%), with the lab measure stubbed: a real one would create the
     `curves-boundary` block, whose first use fixes its size.
8. **The scripts: done** (below, "Launch"), with tests of the reading rule and of the quote
   logic against a stand-in `fleet.py`. The B0 inits exist: `workdir/curves/b0-init-b12c256.pth`
   (`cd65fca1…`) and `b0-init-b16c256.pth` (`8abf7aaa…`).
   - The configs are written: `experiments/configs/curves-{b12,b16}-s{1,2,3}.toml` (`k2-v5.toml`
     with the table's settings, the hold included). The two students are now frozen inputs that
     every box checks (`tools/fleet/frozen.sha256`).

### The controller right after each re-anchor (decided 2026-11-18)

The KL to a new anchor starts at ≈ 0, below the band's floor (target / 1.25 = 0.016).

**With `k2.toml`'s hold (5 windows of 400 steps), the coefficient falls to its 0.01 floor ≈ 4,400
steps (≈ 10–15 min) into every stage.** After the hold, each window cuts it by 1.5 before the KL can
respond. Every run so far did this at its start:

| Run | Start → floor | Then |
|---|---|---|
| `k2b` | 0.1 → 0.01 by t = 4.8K | KL over 0.03 near t = 16.8K, coefficient back to 0.0759 by t = 18.4K |
| `k2v5` | 0.1 → 0.01 within 4.4K steps of its start | back to 0.114 by 29K steps in, KL then ≈ 0.016 |
| `release-b` | 0.1 → 0.01 by t = 4.8K | KL to 0.03 by t = 180K, up to 0.0506, then KL 0.0155–0.018: over-tightened |

So under the drafted hold, D68's start of 0.11 and its carry across re-anchors change nothing. Each
stage replays the pilot's floor, overshoot and over-tightening, which is the pattern D68 meant to
avoid.

At the coefficient `k2v5` settled on (0.114), its KL relaxed from 0.028 to 0.016 with a time
constant of ≈ 8K steps. The climb back from 0 should take a few times that.

**Decided: `hold_windows = 100`** (the user, 2026-11-18: "Go with option 1", of three: this, `k2.toml`'s
5 windows, or a controller acting every 4,000 steps). The coefficient may still rise, but it isn't
cut before the KL first reaches 0.016 or 40K steps have passed (≈ 1.5–2 h); then the controller
acts every 400 steps as in every run so far. The cost: the first part of each stage may run
tighter than the target if 0.11 holds the KL below the band. The code resets the hold at each
re-anchor; the six configs carry the value.

## Launch

One detached chain runs everything, each step gated on the one before
(`experiments/curves_chain.sh`):

```bash
setsid -f bash experiments/curves_chain.sh >> workdir/fleet/curves-chain.log 2>&1 < /dev/null
```

1. **Both B0s at once** (`curves_b0.sh b12|b16`): 12 box-hours each, then the token's box check.
2. **The B0 rule and the inits** (`curves_b0.sh SIZE --finish`): a failed rule stops the chain.
3. **Batch 1, then batch 2 on the user's go** (`curves_launch.sh RUN`; the user, 2026-11-22: batch 2
   doesn't start by itself; the chain ends after batch 1 unless it's run with `--batch2`), each
   batch's second run starting 3 min after
   the first: each run runs `watch --budget-box-hours 288 --snapshot-at 0.25,0.5,0.75 --reanchor
   --stage-check …`. A run that ends without `full` stops the chain for the user.
4. **The final measures and the reading** (`curves_measure_chain.sh`, `curves_reading.py` →
   `workdir/lab/curves-reading.json`). Batch 3 isn't started by the chain.

**Limits** (`experiments/curves_common.sh`; the user, 2026-11-18: "the budget is flexible, so if
availability is limited feel free to use a more expensive box"):
- **$1.00/h per box**, held for every box a run rents, retries and replacements included.
- **$2.80/h per fleet** on the live quote, re-quoted every 15 min for up to 6 h.
- **Dollar stop:** each watcher stops its run at 1.25 × the quote over the run, ≈ $220 per run and
  ≈ $11 per B0 at today's quote.

**After a workstation restart:** `bash experiments/curves_launch.sh RUN --resume` (or
`curves_b0.sh SIZE --resume`) for each live run, then relaunch the chain. It skips finished steps
and waits on live runs. Once a batch-2 run is live or done, the relaunch goes on through batch 2
without `--batch2`.

**Batch 2's go** (after batch 1's `full`, on the user's word):

```bash
setsid -f bash experiments/curves_chain.sh --batch2 >> workdir/fleet/curves-chain.log 2>&1 < /dev/null
```

## Cost

| Part | Estimate | At the live quote (2026-11-18) |
|---|---|---|
| Four fleets, 1,152 box-hours | ≈ $630–690 | ≈ $705 (six boxes at once ≈ $3.68/h, ≈ $0.61 per box-hour) |
| Two B0s | ≈ $20 | ≈ $15–20 (one fleet ≈ $1.51/h) |
| Final measures, 12 × 40K sets on an eval box | ≈ $25 | ≈ $25 |
| **Total** | **≈ $680–735** | **≈ $745–750**, inside the ≈ $785 left of the early-research cap |
| If close: two more runs and their measures | + ≈ $340 | |

Only 7 allowlisted machines were offered within $1.00/h when quoted, so prices may move up while
the runs rent; the dollar stops hold each run near its own quote.

Time: ≈ half a day for the B0s, then two 4-day batches; ≈ 9 days with the measures, ≈ 13 if a third
batch runs.

## Not in this

T1's opponents (a fixed table, the latest champion, or a pool of past champions, RL-5) are weighed
after the curves. No search, no mixed tables, no family-C confirmation.

## Progress (2026-11-18)

- **The chain launched at 10:04 UTC** (the user's yes, "Whole chain"; log `workdir/fleet/curves-chain.log`).
- **Most B0 actor setups failed**, on a racy test of mine: `setup_box.sh` runs the suite on every
  box. It read a new process's command line before the process had exec'd (fixed in `2267735`).
  10 of the 11 actor setups that ran failed; both learners passed.
  - The B0s came up with 1 and 0 actors. The missing actors were provisioned by hand between
    10:53 and 11:11, and the machines the failures had marked bad were cleared.
  - The hour each B0 spent short of actors is left out of its 12-box-hour budget: 2.66 and 2.63
    box-hours (`fleet.py exclude`). The cost still counts it.
- **`server.py`'s token, checked on boxes: passes on both fleets.** Through an actor's tunnel, a
  `submit_param` and a `drain` without the token were both denied, the learner logged both
  refusals, and the actors were getting the trainer's parameters.
- **Hosts:**
  - `curves-b12-b0`: learner Ryzen 9 5950X, actors EPYC 7713 and 7742, $1.53/h;
  - `curves-b16-b0`: learner EPYC 7713, actors Threadripper PRO 5995WX and EPYC 7B12, $1.30/h.
- **Next:** each B0's `full` ≈ 15:15 UTC, then the B0 rule and batch 1, by the chain.

## Progress (2026-11-19): batch 1 past its first boundary

- **The B0s passed their rule** (2026-11-18 15:19 UTC; `experiments/b0_check.py`):
  - b12c256: 595 windows, explained variance by third 0.0077 / 0.0110 / 0.0122, final window
    monotone;
  - b16c256: 266 windows, 0.0056 / 0.0089 / 0.0099, final window monotone.
  - $7.59 and $6.53. The inits: `init-b12c256.pth` (`0d656f38…`), `init-b16c256.pth` (`c0111cca…`).
- **Batch 1 rented at 15:20 UTC:** `curves-b12-s1` at $1.74/h (quote $1.93), `curves-b16-s1` at
  $2.30/h (quote $2.30). Dollar stops $232 and $276.
- **p25 (72 box-hours, 2026-11-19 ≈ 15:25 UTC):**
  - the snapshots were re-anchored within 1 and 6 min (b12 at step 1,395,600, b16 at
    1,151,600), coefficient 0.11 carried;
  - **both stage checks: continue.**

  | Run | Reverse KL to the student | Agreement | Entropy (of the student's) | p25 vs 3× student, 10K sets |
  |---|---|---|---|---|
  | `curves-b12-s1` | 0.0176 | 95.2% | 0.490 (95%) | +0.0061 [−0.0002, +0.0125] (`20261119-082213-measure`) |
  | `curves-b16-s1` | 0.0175 | 94.7% | 0.487 (94%) | +0.0061 [−0.0005, +0.0125] (`20261119-091301-measure`) |

- **The controller:**
  - **Stage 1 behaved as designed.** From 0, the KL reached the band within ≈ 20K steps at
    0.11, with one step up to 0.165 and back. It then held at 0.019–0.020 for the rest of the
    stage.
  - **Stage 2 didn't.** After the re-anchor the KL climbed more slowly, at the lower LR (run
    clock t ≈ 1.2–1.4M). The 40K-step hold ran out with the KL at ≈ 0.008, below the band, so the
    coefficient fell to its 0.01 floor within 2,400 steps. The KL then overshot to 0.025 by
    ≈ 95K steps, and the coefficient is climbing back: 0.076 in b12, 0.051 in b16, 120K steps in.
    That's the pattern the hold was meant to prevent, and stages 3 and 4 will likely repeat it.
  - The pre-registration is fixed, so it stays. It's reported here, and with each run's curve in
    the reading.
- **Hosts:**
  - `curves-b16-s1` lost actor1 at 14:24 UTC; the watcher replaced it by 14:44.
  - `curves-b12-s1`'s actor2 (machine 15136) slowed from 81.6K to 33.6K games/h over 90 min,
    with 0% GPU at the probe. The health check still called it ok, so it was replaced by hand
    (18:09–18:17 UTC) and its machine marked bad for the run.
- **Two operational fixes:**
  - The stage checks' registry rows had made the tree dirty, which `fleet.py provision` and `up`
    refused. They now leave `lab/registry/` out of the check, as `evalbox.py` does (`f52ecf5`).
    Batch 2's `up` would otherwise have failed after six hours of quotes.
  - The rows themselves: `f7d40c2`.
- **Spend so far:** B0s $14.12, batch 1 $108 at 80 box-hours. Batch 1 projects to ≈ $370–390.

## Progress (2026-11-20): a workstation restart, and p50 late

- **The workstation restarted at 07:26 UTC** (00:26 local), killing the chain, both watchers and
  both watchdogs. The six boxes trained on unsupervised for ≈ 9 h.
  - Found at 16:36 UTC: watchers and watchdogs resumed (`curves_launch.sh RUN --resume`, same
    stops and deadlines), and the chain relaunched; it found batch 1 live and waits on it.
- **p50 came ≈ 4 box-hours late:** at 147.87 box-hours (b12, step 2,560,800) and 147.76 (b16,
  step 2,237,200), against 144.
  - Stage 2 ran ≈ 76 box-hours, so stage 3 will run ≈ 68 (p75 stays at 216).
  - The reading counts each snapshot's actual box-hours (`curves_reading.py`), so p50's
    6-box-day reading is at ≈ 6.16.
  - Both re-anchored to p50, and both stage checks (p50 against 3× p25) started 16:38 UTC.
- **p50's stage checks: both continue** (p50 against 3× p25, 10K sets):

  | Run | p50 vs 3× p25 | Reverse KL to the student (mean, median) | Agreement |
  |---|---|---|---|
  | `curves-b12-s1` | +0.0104 [+0.0039, +0.0169] | 0.0430, 0.0138 | 92.8% |
  | `curves-b16-s1` | +0.0063 [−0.0004, +0.0130] | 0.0428, 0.0143 | 92.8% |

- **Stage 3's controller repeated stage 2's pattern.** At +36K steps the KL was 0.0073 (b12)
  and 0.0071 (b16), so the 40K-step hold ran out below the band. The coefficient went from
  0.0506 to its floor, and the KL climbed on at ≈ 0.0002 per 1K steps: 0.022 / 0.021 at
  +110–120K (19:20 UTC).
- **p75 (216 box-hours, 2026-11-21 ≈ 15:20 UTC):** both re-anchored; **both stage checks:
  continue.** Stage 3 gained little:

  | Run | p75 vs 3× p50 | Reverse KL to the student (mean, median) | Agreement | Entropy (of the student's) |
  |---|---|---|---|---|
  | `curves-b12-s1` | +0.0017 [−0.0050, +0.0082] (`20261121-082052-measure`) | 0.0719, 0.0251 | 91.3% | 0.387 (75%) |
  | `curves-b16-s1` | +0.0006 [−0.0061, +0.0071] (`20261121-091324-measure`) | 0.0685, 0.0249 | 91.3% | 0.383 (74%) |

  - **The entropy is nearing its 70% stop limit:** 95% at p25, 74–75% at p75. Batch 1's last
    stage has no check (its `full` goes to the final measures).
  - Batch 2's p75 check could fire on it, and then the chain stops for the user. That outcome is
    as pre-registered.

## Progress (2026-11-22): batch 1's `full`, and option B's diagnostics

- **Batch 1 reached `full`** at 288 box-hours and its watchers downed it: `curves-b12-s1` at
  15:22 UTC ($152.54, step 4,703,600), `curves-b16-s1` at 15:28 ($210.05, step 3,941,200).
  - The chain stopped at its new batch-2 gate (the user, 2026-11-22: batch 2 only with
    `curves_chain.sh --batch2`). Nothing is rented.

### Exploratory diagnostics, outside the pre-registration

Option B of the 2026-11-22 review's triage (the user): free workstation checks on both `full`s
before batch 2 is decided.
- **Nothing here is part of D68's reading.** `curves_reading.py` reads `curves: …` rows and the
  `curves` block, and neither is touched. These rows are `curves diag: …`.
- **Method** (`experiments/curves_diag.py`; `workdir/lab/curves-diag/report.json`):
  - **Drift:** `rl_drift.py` on the students' states. It reproduces p75's stage-check numbers
    exactly.
  - **Measures:** 10K sets each on block `curves-boundary`, the stage checks' deals, with AMP on
    and no logs.
  - **Paired differences:** per set against the stage-check rows, with the lab's set-level
    bootstrap.

| | `curves-b12-s1` | `curves-b16-s1` |
|---|---|---|
| Drift at `full`: reverse KL, agreement | 0.094, 90.4% | 0.090, 90.5% |
| Entropy at `full` (of the student's) | 0.3475 (**67.3%**) | 0.3478 (**67.2%**) |
| Gain since p50: `full` vs 3× p50 | **+0.0085 [+0.0015, +0.0155]** (`20261122-115844`) | +0.0051 [−0.0018, +0.0120] (`20261122-124849`) |
| Stage 4's gain: `full`@p50 − p75@p50, paired | +0.0068 [−0.0015, +0.0152] | +0.0045 [−0.0038, +0.0128] |
| Total gain: `full` vs 3× student | +0.0294 [+0.0222, +0.0364] (`20261122-143910`) | +0.0244 [+0.0171, +0.0316] (`20261122-152757`) |
| Gain after p25: `full`@student − p25@student, paired | +0.0233 [+0.0142, +0.0323] | +0.0183 [+0.0094, +0.0275] |

**Head-to-head:** b16 `full` vs 3× b12 `full` is +0.0018 [−0.0054, +0.0088]
(`20261122-134657`), with positive meaning b16 is stronger. The sizes can't be told apart at 10K
sets.

- **Entropy crossed the stage checks' 70% limit during stage 4,** from 74–75% at p75 to 67% at
  `full`. Batch 1's last stage has no check, so nothing fired. Reverse KL and agreement stay
  inside their limits.
- **Stage 4 gained more than stage 3** (+0.0068 / +0.0045 against +0.0017 / +0.0006), but
  neither stage's gain excludes 0 on its own. The gain since p50 excludes 0 for b12 only.
- **Pairing with the stage checks barely narrows the intervals.** The per-set correlation is
  0.23 for stage 4 and 0.11–0.13 for the gain after p25.
- **The gains are measured at the training table.** 3× student is the runs' training opponent
  (`baseline.train`), so these numbers say nothing about other tables.

**The triage's rule** (the user's, not acted on here): option A, batch 2 as pre-registered,
needs entropy at `full` ≥ 70% and a gain since p50 with a lower bound above 0, for both runs.
**Neither run meets both**: both fail on entropy, and b16 also on its bound. So the rule gives
**option C**, with both runs agreeing: don't run batch 2 as is, and enter any amendment to D68
in the decision log first. The call is the user's.

### Stage-to-stage drift (exploratory, 2026-11-22)

`rl_drift.py` with each earlier snapshot as the reference, on the students' states (outputs in
`workdir/lab/curves-diag/stage-drift-*.txt`). A flip is a decision whose top action changed,
among those with more than one legal action.

| | b12: p50 → p75 | b12: p75 → `full` | b12: p50 → `full` | b16: p50 → p75 | b16: p75 → `full` | b16: p50 → `full` |
|---|---|---|---|---|---|---|
| Flips | 4.9% | 4.5% | 6.4% | 5.1% | 4.6% | 6.5% |
| KL(new ‖ old), mean / median | 0.025 / 0.006 | 0.022 / 0.004 | 0.045 / 0.013 | 0.023 / 0.006 | 0.021 / 0.004 | 0.042 / 0.011 |
| Entropy, old → new | 0.438 → 0.387 | 0.387 → 0.348 | | 0.432 → 0.383 | 0.383 → 0.348 | |

- **Stage 4 moved as much as stage 3:** about the same flip rate (4.5% against 4.9–5.1%) and
  the same sharpening (entropy −9 to −10% against −11 to −12%). Its larger lab gain (+0.0068 /
  +0.0045 against +0.0017 / +0.0006) didn't come from more movement, and neither stage's gain
  excludes 0 on its own.
- **The late stages partly churn.** p50 → `full` flipped 6.4–6.5%, well under the two stages'
  sum (9.4–9.7%). So between a third and two-thirds of stage 4's flips fell on decisions that
  stage 3 had already changed; the overlap is at least half the 3.0–3.2 pp gap and at most all of
  it. Some of those went back to p50's action, and the rest moved to a third one.
- **The flips are discards:** about 93% of stage 4's (≈ 650 of ≈ 700 per run). Discards are
  ≈ 78% of these decisions, so flips are concentrated in them. Riichi, call and pass decisions
  flipped 1–6% each, a few to about 20 decisions per kind.
- **The lab plays greedy** (`lab/arena.py`), so the lab gains come from the flips alone.
  Sharpening without a flip matters for training's sampling and the entropy limit only.
- **Not measured:** how much the flips matter. There is no value weighting, and the states are
  the students' own decisions, not those the later snapshots reach.

**Per decision** (`experiments/curves_flips.py`, true FP32; `workdir/lab/curves-diag/flips.json`;
2026-11-23). The rates match the table above exactly, so the changes aren't numerics.

| | b12: stage 3 | b12: stage 4 | b16: stage 3 | b16: stage 4 |
|---|---|---|---|---|
| Decisions changed | 766 (4.9%) | 702 (4.5%) | 780 (5.1%) | 701 (4.6%) |
| The earlier snapshot's gap between the two moves (median) | 0.105 | 0.111 | 0.111 | 0.110 |
| … under 0.05 / 0.05–0.2 / 0.2–0.5 / over 0.5 | 27 / 46 / 23 / 3% | 25 / 49 / 22 / 4% | 27 / 45 / 24 / 5% | 27 / 43 / 26 / 4% |
| The later snapshot's gap for its new move (median) | 0.126 | 0.137 | 0.123 | 0.122 |

- **Moderately close calls, rarely reversals.** Over all decisions, the top move's lead is 0.85–0.91
  (median), and only 11% of decisions are within 0.2. Yet ≈ 72–74% of the changes come from that
  11%, and only 3–5% of them reverse a move preferred by over 0.5. These aren't coin flips: the
  median gap is ≈ 0.11 before the change and ≈ 0.12–0.14 after it.
- **The late stages oscillate.** 248 (b12) and 257 (b16) decisions changed in both stages: 35–37%
  of stage 4's changes, against 4.5% of decisions overall. Of those, 217 and 229 (87–89%) went
  back to p50's move. So about a third of stage 4's changes undid stage 3's.
- **The net movement is small.** Stage 4 restored the student's move on 259 / 266 decisions and
  left it on 392 / 389. The net, ≈ 0.8% of decisions, is the drop in agreement with the student;
  the gross change is 4.5%.

### Off-table transfer (exploratory; fixed 2026-11-23, before any of it plays)

**The question:** does b12's gain at the training table hold at tables the runs never trained
against? The gain is `full` vs 3× student, +0.0294 [+0.0222, +0.0364].

- **The tables:**
  - first, 3× No-SP A, an independent v5 BC student (b8c192) outside the teachers' lineage;
  - second, only if needed, 3× BC s1 (EMA). It is one of the student's own teachers, so an
    exploit of the student could partly carry to it.
- **The measures:** family A, block `curves-offtable` (new), 20,000 sets each, AMP on, no logs
  (`experiments/curves_diag.py play-offtable`). Every measure in the block is on the same deals.
- **D** = `full` − student at the same table, paired per set. Its share is D ÷ 0.0294.
- **The bands** (`curves_diag.band`), against half the training table's gain (0.0147):
  - **carries:** D's 95% interval is entirely above 0.0147;
  - **specific to the student:** it is entirely below;
  - **ambiguous:** otherwise.
- **The order:**
  1. `full` and the student at 3× No-SP A.
  2. If the gain *carries*: p50 at 3× No-SP A, to ask whether the late stages' gain carries too.
     If *ambiguous*: `full` and the student at 3× BC s1. If *specific*: stop and report.
- **Caveats:**
  - The reference +0.0294 has its own ±0.007.
  - A table's strength can change Δu's scale whether or not the policy overfits.
  - `full` at 3× BC s1 is one of the pre-registered final measures. It is reported, not deciding,
    and is played on the `curves` block. Measuring it here, on other deals, is a peek, not a
    change to the reading.

**Results** (2026-11-23; `workdir/lab/curves-diag/offtable.json`). Everything is at 3× No-SP A,
20K sets per measure, on the same deals:

| | Δu [95% CI] | Row |
|---|---|---|
| b12 `full` | +0.0496 [+0.0447, +0.0546] | `20261123-022327` |
| b12 p50 | +0.0449 [+0.0401, +0.0498] | `20261123-051222` |
| b12's starting student | +0.0270 [+0.0222, +0.0317] | `20261123-034926` |
| **D = `full` − student**, paired | **+0.0227 [+0.0164, +0.0290]**, 77% of +0.0294 | |
| p50 − student, paired | +0.0180 [+0.0117, +0.0241] | |
| `full` − p50, paired | +0.0047 [−0.0015, +0.0107] | |

- **The gain carries** by the band fixed above, though narrowly: D's lower bound +0.0164 is
  just above 0.0147. Most of RL's gain over the student isn't specific to the student, at least
  at this table. So single-opponent overfitting looks milder than the review's C1 feared.
- **Most of it was in place at p50:** p50 has 79% of `full`'s off-table gain. The second half
  of the run added +0.0047, which doesn't exclude 0. That matches the training table, where the
  late stages also gained little and noisily (stage 3 +0.0017, stage 4 +0.0068, each
  inconclusive).
- **The comparison isn't exact.** At the training table, the gain since p50 was measured at
  3× p50 (+0.0085), not at 3× student, so the two late-stage numbers come from different
  tables.
- **By the order fixed above, this ends the test.** BC s1 was the step for an ambiguous result.
  It remains open as a second, stronger table: a few hours per measure on the 3060.
