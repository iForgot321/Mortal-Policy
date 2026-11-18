# D31's actor-size curves: the pre-registration (draft, 2026-11-18)

**Status: draft.** The design is D68 (the user, 2026-11-18: "Update the plan based on the default
recommendation"). It becomes fixed when the user approves the rental with a live quote, and nothing
here is changed after the first run starts. Exploratory. Nothing has run.

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
| KL controller | Adaptive, start 0.11 (`k2v5`'s settled coefficient on v5), tolerance 1.25, factor 1.5, floor 0.01, ceiling 10; the coefficient carries across re-anchors |

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

## The measures (after each run's `full`; an eval box)

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
7. **Not built yet:**
   - the stage-boundary checks: the 10K-set stop check and the drift limits, started on the
     workstation by the watcher at each re-anchor, and downing the run when one fires;
   - the launch, B0 and measure scripts.
   - The configs are written: `experiments/configs/curves-{b12,b16}-s{1,2,3}.toml` (`k2-v5.toml`
     with the table's settings). The two students are now frozen inputs that every box checks
     (`tools/fleet/frozen.sha256`).

### Open: the controller right after each re-anchor

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

**Proposed: `hold_windows = 100`.** The coefficient may still rise, but it isn't cut before the KL
first reaches 0.016 or 40K steps have passed (≈ 1.5–2 h). The cost: the first part of each stage
may run tighter than the target if 0.11 holds the KL below the band. This is a config value; the
code resets the hold at each re-anchor either way. The user decides before the pre-registration is
fixed.

## Cost

| Part | Estimate |
|---|---|
| Four fleets, 1,152 box-hours | ≈ $630–690 |
| Two B0s | ≈ $20 |
| Final measures, 12 × 40K sets on an eval box | ≈ $25 |
| **Total** | **≈ $680–735**, inside the ≈ $785 left of the early-research cap |
| If close: two more runs and their measures | + ≈ $340 |

Time: ≈ half a day for the B0s, then two 4-day batches; ≈ 9 days with the measures, ≈ 13 if a third
batch runs.

## Not in this

T1's opponents (a fixed table, the latest champion, or a pool of past champions, RL-5) are weighed
after the curves. No search, no mixed tables, no family-C confirmation.
