# Fleet soak (03 §3.7, D39, D60): preparation

The soak runs the leading engine, P1 + K2 (`experiments/k2-retry.md`), for ≥ 48 h at T1 scale
(learner + 3 actors + eval) before T1. Two of its criteria looked likely to fail as configured
(2026-10-24):

- **Policy lag:** `k2b` ran at p95 ≈ 8 versions (median 6.7; its learner's TensorBoard, 143 h after
  the first hour), against "p95 ≤ 2". D60 (the user, 2026-10-24) makes lag non-blocking while
  learning stays stable: PPO ratio p99 ≤ 2 in ≥ 95% of intervals, no upward trend in lag, no
  `anchor_alert`. `k2b` met all three (ratio p99 median 1.15, max 1.41; no alert).
- **Learner capacity:** `k2b`'s learner trained 3,624 samples/s (25,483 steps/h × 512), about what
  its two actor boxes produced (≈ 84K games/h × ≈ 155 samples per game). Three actors at ≈ 42K
  games/h each need ≈ 5.4K samples/s.

## Learner capacity check: pre-registered 2026-10-24, before any timing

The user approved it on 2026-10-24 ("experiment 2").

- **What is timed:** `tools/fleet/bench_learner.py --arm P1 --package K2`, `k2b`'s learner path:
  horizon H, the K1 critic (GAE on the learner), PPO with stored behaviour probabilities, and K2's
  KL anchor (π_SL's forward, the reverse KL, the adaptive coefficient), AMP, and torch.compile of
  the policy and critic as `k2.toml` sets it. The drain is 3,200 games of pilot A made on the box
  on the benchmark key (no nonces), the size P1's learner check used. Samples are decisions loaded.
- **Passes, on one box, in order** (`tools/fleet/box_bench.sh <id> <label> learner-k2`):
  1. trainer threads 4, 8 and 16 at 4 loader workers. 4 workers and 8 threads are `k2b`'s settings:
     that pass reproduces `k2b` (expected ≈ 3.6K samples/s on a Zen 4 EPYC like `k2b`'s 9354);
  2. 4 vs 6 loader workers at the fastest thread count;
  3. π_SL compiled too (`--anchor-compile`) at the fastest settings. The trainer leaves it eager;
  4. a py-spy profile of the benchmark at the fastest settings, to name what binds.
- **Only throughput settings:** loader workers and trainer threads don't change what is learned
  from a drain (P1's learner check). Compiling π_SL changes only the order of floating-point
  operations, as the compile of the networks `k2b` already compiles does.
- **Rule:** the fastest `full` samples/s wins; a setting within 5% of the fastest loses to the
  simpler one (fewer workers or threads; π_SL eager), P0's and P1's rule. P1's memory ceiling
  applies: a setting is eligible only if its `full` pass peaks below 32 GB.
- **Target: ≥ 6.0K samples/s,** the ≈ 5.4K three actors need plus 10%.
- **Box:** one RTX 4090 with a Zen 4 EPYC ≥ 32 vCPU and ≥ 64 GB, the fleet's offer query, of
  `k2b`'s learner class (EPYC 9354) if one is offered. ≈ 1.5 h with setup, hard limit 3 h. If it
  misses the target, one more box of another CPU class (≈ $1). Cap $5 in all. Destroyed when the
  results are copied back.
- **Reading:**
  - The chosen settings reach 6.0K: the soak runs with them at learner + 3 actors (written into
    the soak's plan before it runs).
  - They don't: the profile names the bottleneck. The options go to the user: optimize the named
    step (`03` §6), a faster learner CPU, or 2 actors per learner (a change to `03` §3.7's T1
    scale, so an `08` entry first).

### Result (2026-10-25): the target is missed; the main process's chunk concatenation binds

- **Box:** RTX 4090 + EPYC 9474F (Zen 4, 48 cores at a 30.7-CPU quota, 108 GB), Vast instance
  57625912, $0.517/h; no 9354 was offered. The first attempt (57623785) failed setup's test suite:
  `box_bench.sh` synced only pilot A and the GRP, not every frozen checkpoint (fixed in `6b8f196`).
  Both boxes ≈ $0.50 in all. Results: `workdir/fleet/20261025-vast-4090-9474f-learner-k2b/box/`.
- **Drain:** 3,200 games, 514,558 decisions (226 s to generate).

  | Pass | Full, samples/s | Data only | Peak memory, full |
  |---|---|---|---|
  | 4 workers, 4 threads | 4,582 | 11,418 | 24.3 GB |
  | 4 workers, 8 threads (`k2b`'s settings) | 4,683 | 11,541 | 24.9 GB |
  | 4 workers, 16 threads | 4,722 (repeat 4,559) | 11,212 | 24.3 GB |
  | 6 workers, 16 threads | 2,990 | 11,599 | 32.9 GB (over 32: ineligible) |
  | 4 workers, 16 threads, π_SL compiled | 4,149 | 10,720 | 23.5 GB |

  GPU only (training steps on batches already on the GPU): 17.7–21.7K samples/s.
- **By the rule:** the thread counts are within 5% of each other (the repeat varied 3.5%), so the
  fewest wins: **4 workers, 4 threads, π_SL eager, ≈ 4.6K samples/s. The 6.0K target is missed by
  ≈ 23%.** On this faster CPU the benchmark runs ≈ 30% above `k2b`'s live learner (3.6K on an EPYC
  9354), so a live learner would fall further short.
- **The second CPU class wasn't run:** every other offer that day was older (Zen 2/3) or a
  lower-clocked Zen 4 (9334, 9V74), and no desktop Ryzen 7950X/9950X was listed, so none could be
  faster per core than the 9474F.
- **What binds** (py-spy, 50 Hz, 351 s on the main thread; per full pass, from the warm-up and timed
  passes):

  | Main-thread work | Share |
  |---|---|
  | `critic.py:70`: `torch.cat` of each chunk's trajectories (obs and priv: ≈ 35 GB per pass) | ≈ 40% |
  | training steps (forward, backward, AdamW) | ≈ 30% |
  | `critic.py:76`: the critic's values for GAE, a GPU sync every 1,024 decisions | ≈ 20% |
  | `shuffled_minibatches`: concatenation with the remainder and the gathers | ≈ 9% |

  The loader workers deliver 11.5K samples/s on their own, and the GPU trains ≈ 20K/s: the learner
  is bound by memory copies in its main process, as P1's check found before its thread fix.
- **Options (the user's):**
  1. **Optimize the named step** (`03` §6): concatenate in the loader workers, which have ≈ 2.5×
     headroom, and overlap the critic's evaluation with training. The training tensors stay the
     same. Removing ≈ 40% of the main thread's work would put the learner near 7–8K samples/s if
     nothing else binds (an estimate). Code and tests, then a ≈ $0.50 re-benchmark.
  2. **2 actors per learner at T1** (`k2b`'s shape, which met the queue and blocking criteria): a
     change to `03` §3.7's T1 scale, so an `08` entry first.
  3. A faster learner CPU: none offered.

### Optimization (2026-10-25, the user's option 1): the target is met, ≈ 7.5K samples/s

Two rounds on the learner's main process, both throughput only: the learner gets the same tensors,
checked by identity tests in `tests/test_rl_dataset.py` (packed chunks against the plain
concatenation, `gae_targets` on both, and the new minibatch gathers against the old concatenating
version, on the CPU and the GPU). The tests ran on the boxes, as the user asked: 149 and then 153
passed.

1. **`fa61520`:** the loader worker concatenates each gae chunk (`rl_dataset.pack_chunk`, enabled in
   `train_online.py` for the critic path); `gae_targets` syncs the device once per chunk; the
   minibatch shuffle gathers from the remainder and the chunk in place.
2. **`4c30a83`:** `gae_targets` copies each chunk's obs and priv to the GPU once and returns them
   there, so minibatches are gathered on the GPU; mixed gathers use integer indices. Round 1's
   profile had the CPU gathers at ≈ 34% of the main thread, nearly every minibatch holding some
   remainder rows.

Same box class (EPYC 9474F, 30.7-CPU quota), same benchmark (`box_bench.sh … learner-k2`):

| Pass | Before | Round 1 | **Round 2** |
|---|---|---|---|
| 4 workers, 4 threads | 4,582 | 5,503 | **7,514** (repeat 7,607) |
| 4 workers, 8 threads | 4,683 | 5,570 | 7,417 |
| 4 workers, 16 threads | 4,722 (4,559) | 5,757 (5,231) | 7,431 |
| π_SL compiled | 4,149 | 5,171 | 7,530 |
| 6 workers | 2,990 (32.9 GB) | 4,269 (36.8 GB) | 6,384 (36.0 GB): ineligible |
| Peak memory, full (4 workers) | 24.3 GB | 28.5 GB | 25.8 GB |
| GPU only | 19.6K | 19.0K | 20.3K |

- **By the rule:** every 4-worker setting is within 5% of the fastest, so the simplest wins: **4 loader
  workers, 4 trainer threads, π_SL eager: ≈ 7.5K samples/s, 25% above the 6.0K target**, +63% on the
  start. The soak runs with these (`--set dataset.num_workers=4 --set control.num_threads=4`) at
  learner + 3 actors.
- **Not yet known:** how a live learner compares. `k2b`'s ran ≈ 3.6K on an EPYC 9354 against the old
  code's ≈ 4.7K here, and the CPU differs, so the soak itself measures it.
- **Cost:** five box runs (one failed setup on the missing CQL checkpoints, one lost to an sshd restart,
  both fixed in `box_bench.sh`), ≈ $1.40 in all against the $5 cap.

## The soak: pre-registered 2026-10-26, before it runs

The user approved it on 2026-10-26 ("plan and rent the fleet soak"), with the learner on the EPYC
9V74 (≈ $140 in all).

- **Run `soak-a`:** P1 + K2 as registered (`experiments/configs/k2.toml`), continuing from `k2b`'s
  `full` (π_SL stays pilot A), with the learner check's settings (`dataset.num_workers=4`,
  `control.num_threads=4`) and its two optimizations (`fa61520`, `4c30a83`). Learner: machine 12764,
  EPYC 9V74 (Zen 4, Norway, $0.823/h; no 9474F had a long enough contract). 3 actors × 2 processes,
  the fleet's cheapest allowlisted boxes. T1 scale (`03` §3.7).
- **Duration:** `--budget-box-hours 192` (4 boxes × 48 h): p25 at ≈ 12 h, p50 ≈ 24 h, `full` ≈ 48 h,
  then down. Hard deadline 52 h. `--max-dollars 130` for the fleet (≈ $2.37/h, ≈ $118 projected).
- **Eval box:** `evalbox.py` up when p25 lands. It runs exploratory tracking measures, each snapshot
  (p25, then p50) against `k2b` full, 10K sets on a new family-A block `soak`, and goes down with the
  fleet. It doesn't enter the criteria. It comes up after the workstation's search pilot has drawn its
  nonces, because the eval box holds the lab registry.
- **Criteria** (`experiments/fleet_soak_check.py --mode final`, at the end; windows end at the last
  health record):
  1. the replay queue's least-squares line over the last 24 h rises < 25% of capacity (1,600);
  2. policy lag (D60): per-interval mean lag p95 ≤ 2 over the last 24 h, or stable: ratio p99 ≤ 2 in
     ≥ 95% of intervals, the mean lag's line rises < 1 version over the 24 h, no `anchor_alert`;
  3. games drained ≥ 90% of games produced, and actors blocked < 10%, over the soak after its first hour;
  4. every box loss recovered within 30 min, none failed.

  Dry run on `k2b`'s records (2 actors): pass, with the lag passing as stable.
- **Stop rule at p25** (≈ 12 h; `--mode early` as the watcher's `--early-check`, over the last 6 h;
  only clear failures stop): actors blocked > 25%, the learner draining < 75% of production, the ratio
  p99 > 2 in > 20% of intervals, an `anchor_alert`, or a box loss not recovered within 60 min.
  A stop downs the fleet and the result goes to the user.
- **Also reported:** the live learner's samples/s (steps/h × 512) against the benchmark's 7.5K, games/h,
  and the lag figures.
- **Reading:** pass → the soak gate for T1 is met (`03` §3.7). Fail → the dates move, never the test
  sizes (D39), and the failing criterion names the fix.
