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
