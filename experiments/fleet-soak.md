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
