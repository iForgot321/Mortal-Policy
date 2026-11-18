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

1. **In-run re-anchoring:** at each snapshot boundary the learner reloads its anchor from that
   snapshot and keeps the controller's coefficient (`mortal/kl_anchor.py`, `train_online.py`, a
   `fleet.py` command). Tests, and a smoke run on the workstation with short stages.
2. **`server.py`:** `submit_param` and `drain` answered only for the learner's own trainer, not any
   tunnel peer (the 2026-11-09 audit's finding 9). Tests, and a check on one box (≈ $1).
3. **`fleet.py`:** provisioning retries that respect the run's price limit (2026-11-17: a retry
   took a $1.04/h host against a $0.75/h limit).
4. **The two B0s** (≈ 12 box-hours each, ≈ $20): their own short rental, in the same approval.
5. **v5 states for the monitoring:** `experiments/rl_credit_check.py --work` for each student, on
   the workstation.
6. The full test suite as a box runs it (`env -u MORTAL_CFG`).

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
