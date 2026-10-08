# SP ablation: v4 without the single-player tables (Phase 0, D12 reopen condition)

- **Hand-off:** `docs/design/sp-ablation-handoff.md`. **Question:** is a student trained without
  v4's single-player (SP) channels non-inferior to one trained with them? And what is dropping
  them worth in speed?
- **Standard:** a recipe claim, so matched pairs (`02` §6.1, D41). One failed pair keeps SP; two
  passes are *preliminary* evidence that SP can go. Confirmation (fresh family-C nonces, reserved
  panel configurations) was not run: `lab confirm` covers the family-C fixed-N part, the reserved
  panel configurations are not implemented.
- **Nothing is adopted here.** Changing the observation is a frozen-item decision (CLAUDE.md
  rule 2) and needs a decision-log entry first; that is the user's call.

## Summary

| Question | Answer |
|---|---|
| v5 built and safe? | Yes. v5 = v4 with the 123 SP rows zeroed; v4 bit-identical (golden hash), v5 = v4 minus SP at every decision tested (§2) |
| Speed | Large and certain: encoding 340× (3.4 vs 1,163 µs per decision), SL loader 90× per thread, SL pipeline 4.1×, SL training 2.9× (8.6 h vs 25 h), arena 2.4× (2.8× per decision), P0 learner 2.3× (§3) |
| Where v4's cost goes | 3-shanten decisions: ~17% of decisions, ~83% of SP time (§3.1) |
| BC fit | Identical by the end of training in both pairs (gap closes by ~100K steps) (§4) |
| T-NI (δ = 0.01) | **Both pairs pass** (§5) |
| Effect size (fixed N, 20K sets per pair) | Pair 1 −0.0060 [−0.0107, −0.0014]; pair 2 +0.0012 [−0.0035, +0.0059]; pooled −0.0024 [−0.0056, +0.0009] (§5.1) |
| Pre-registered verdict | **Inconclusive**: the pairs disagree (z = 2.16). Without the consistency check: grey zone, keep v4. **v5 is not adopted** |
| D12 reopen condition (non-inferior JAX-portable obs) | **Not met** |

Any cost of dropping SP is small (pooled ~0.25 pt per hanchan, plausibly 0 to ~0.6), but the two
training runs disagree by about as much as the effect, so this design can't resolve it.

## 1. Prior evidence (M0, timeboxed)

- Upstream added SP in obs v4 (Equim, 2023-10-31, `2c662b3`, "all: implement v4"; 55 files,
  +3.6K/−1.8K lines) together with other changes: the new encoder layout, the new DQN head and
  whatever training changes "4.0" shipped with. The only public number is the release result,
  "4.0" vs "3.1": average rank 2.482 and +1.228 pt over 1M 1v3 duplicate games
  ([announcement](https://x.com/equim_chan/status/1719382780298277329),
  [strength page](https://mortal.ekyu.moe/perf/strength.html)). That bundles everything, so it says
  nothing about SP alone.
- No SP ablation, v3-vs-v4 observation comparison or design rationale was found in the commit,
  the docs, the mjai-reviewer FAQ or the GitHub discussions (#61 asks what changed between 3.1
  and 4.0 and has no answer).
- nekobean's mahjong-cpp (the calculator's origin) documents what it computes (per-discard
  expected score, win and tenpai probability by turn, required tiles) under a single-player
  model: no opponents, random draws from the unseen tiles. It makes no claim about play strength.
- **So the question is open.** Upstream itself never isolated it.

## 2. Observation v5 (M0–M1)

- **Definition:** v5 = v4 with rows 889–1011 (the 123 SP rows) zero and `single_player_tables()`
  never called. Same shape (1012 × 34), so `Brain(version=5)` builds v4's modules and the same
  seed gives the same initial weights (tested). Zero inputs give their first-layer weights no
  gradient, so it is functionally a removal. When `single_player_tables()` errs in v4 (an agari hand,
  or no tsumo left), the first two SP rows hold the minimum tsumo value instead; v5 zeroes those
  too.
- **Code:** `libriichi/src/consts.rs` (`MAX_VERSION = 5`, shapes), `state/obs_repr.rs` (v4's arms
  take 5; the SP block is skipped for v5), the oracle-obs arms in `dataset/invisible.rs` and
  `arena/board.rs` (oracle shape = v4's), `mortal/model.py` and `mortal/engine.py` (5 is treated
  as 4). Commit `d6b42df`.
- **v4 unchanged:** `tools/obs/hash_obs.py` hashes `encode_obs(4, ·)` (obs and mask) at every
  decision of every seat, kan select included, over 36 logs: 20 pilot A-vs-B games, 8 hobbled-pilot
  games and 8 Tenhou houou games (2024-03), 22,794 decisions. Golden hash before the change
  (`5673d6e`): `b9817059a17a615c…`. Reproduced twice before the change and again after it.
- **v5 = v4 minus SP, checked at every decision:**
  - same 36 logs, `hash_obs.py --check-v5`: all 22,794 decisions, kan select included;
  - Rust: `state::test::validate` now checks it (and equal masks) at every decision of every state
    test, plus `obs_v5_is_v4_without_sp` on a tenpai hand where v4's SP rows are filled at both a
    discard and a call decision; debug build too (the `debug_assert`s run);
  - pytest `tests/test_obs_v5.py`: encoder on arena games (v4's SP rows were filled at most
    decisions, so the check isn't vacuous), the SL loader (`GameplayLoader`, oracle channels
    included), initial weights, and a v5-vs-v4 arena game in both directions (each side encodes
    with its own engine's version).
- **Pre-commit checks:** `cargo fmt` clean; clippy at the 27-error baseline, none on touched
  lines; `cargo test` passes; `pytest tests/` 64 pass (5 new).

## 3. Speed (M2)

Workstation (i5-12600KF, 16 threads; RTX 3060), nothing else running, 2026-10-06, at
`d6b42df` plus the two benchmark-script changes committed with this note. Raw results:
`workdir/sp-ablation/m2/`.

| Measure | v4 | v5 | v5 / v4 |
|---|---|---|---|
| `encode_obs`, µs per decision, mean (median / p99 / max) — `bench_encode.py`, 200 pilot A-vs-B games, 130,710 decisions, 1 thread | 1,163 (106 / 13,908 / 53,160) | 3.4 (3.0 / 9.4 / 160) | 1/340 |
| Encoding CPU-s per hanchan | 0.76 | 0.002 | |
| (v3 for reference) | | 3.4 (3.0 / 9.6 / 150) | |
| SL loader, samples/s, 1 thread — `bench_featurize.py`, 256 Tenhou logs from quick500k, 171K samples | 830 | 74,514 | 90× |
| SL loader, samples/s, 16 threads | 6,044 | 185,375 | 31× |
| SL pipeline as trained (`FileDatasetsIter` + DataLoader, 2 workers, BC, fp16, masks) | 4,632 | 19,218 | 4.1× |
| Arena, hanchan/h, steady median — `bench_1v3.py`, 500 sets per call, AMP, eval mode, 15 min | 22,707 (3 chunks) | 54,413 (7 chunks) | 2.4× |
| Network decisions per game (both sides) | 603 | 711 | |
| Network decisions per hour | 13.7M | 38.7M | 2.8× |
| CPUs busy (whole system) | 10.3 | 2.0 | |
| GPU utilization | 22% | 48% | |

- **The arena v5 run** used `workdir/sp-ablation/pilot-a-as-v5.pth`, a copy of pilot A whose
  config says `version = 5` (sha256 `eaa43ac3…`; read-only). It was trained with SP inputs and
  now sees zeros, so it plays differently: games are ~18% longer in decisions, which is why the
  per-decision ratio (2.8×) is the cleaner number. It is a timing device, not a v5 student.
- **Where the time goes now.** With v4, the rayon encoder kept ~10 CPUs busy and the GPU at 22%.
  With v5 the encoder nearly vanishes: ~2 CPUs (the arena's main thread plus Python inference)
  and the GPU at 48%. The single-threaded arena loop, already the bottleneck on the cloud boxes
  (`fleet-host-benchmarks.md`), is now almost the only cost, so several actor processes per box
  should scale nearly linearly until the GPU saturates. Not measured here.
- **SL:** the pipeline gains 4×, not 90×: past the encoder, Python-side work (shuffle buffer,
  fp16 copies, batching) limits it at ~19K samples/s with 2 workers. v5 training should then be
  GPU-bound on the 3060; §4 checks that. For `03` §4.2 (vCPU × samples/s ≥ 80%
  of the GPU's rate) the CPU requirement for teachers mostly disappears with v5.
- The v4 numbers reproduce the earlier ones (loader 825 / 5.7K, pipeline 4.2K; 3.6K samples/s in
  the pilots' actual training).

**P0 learner** (`tools/fleet/bench_learner.py`, 2026-10-07 23:31, alone on the box): one 800-game
drain (pilot A both sides, trainee sampling, sidecars; 127,546 trainee samples) trained the way
`train_online.py` does (MC targets, 2 workers, `file_batch_size` 50, batch 512, AMP, compile),
timed on a second pass. Raw: `workdir/sp-ablation/learner.json`.

| Mode | v4 | v5 | v5 / v4 |
|---|---|---|---|
| Data pipeline only | 4,267 samples/s | 15,676 | 3.7× |
| **Full learner** | **3,604** | **8,424** | **2.3×** |
| GPU step only (shapes equal) | 15,117 | | |

v4's learner is featurization-bound (full ≈ data only; ~24% of the GPU's rate), and matches the
cloud box's ~3,750 samples/s. v5 lifts it 2.3×; the data and GPU stages then barely overlap.
Since v5 speeds the actors (2.4–2.8×) and the learner (2.3×) about equally, the actor:learner cost
ratio would stay near its v4 value (~3:1), which is the case where Mahjax's end-to-end edge
over libriichi would be largest (~1.8–2×). That is moot while v5 is not adopted.

### 3.1 Where v4's SP cost goes, and what cheaper SP variants would save

`tools/obs/sp_cost.py` (2026-10-07 17:25, alone on the box, 1 thread): `encode_obs(4, False)` timed
at every decision, split by decision type, **real-time** shanten (what the calculator's threshold
uses; `PlayerState.shanten` is not refreshed by a draw) and draws left. Raw:
`workdir/sp-ablation/sp-cost.json`. The two corpora agree to within a few percent:

| | 200 pilot A-vs-B games (130,710 decisions) | 100 Tenhou houou games (66,053) |
|---|---|---|
| Mean / median / p99 / max per decision | 1,145 / 106 / 13,663 / 56,002 µs | 1,159 / 102 / 14,138 / 51,503 µs |
| Share of time in the slowest 1% / 5% / 10% of decisions | 16% / 49% / 71% | 16% / 49% / 71% |

| Real-time shanten | Share of decisions | Mean | **Share of time** |
|---|---|---|---|
| tenpai (0) or agari (−1) | 18% | ~13 µs | 0.2% |
| 1 | 29% | 62 µs | 1.5% |
| 2 | 30–31% | 530–555 µs | 14–15% |
| **3** | **17–18%** | **5.5–5.6 ms** | **83–84%** |
| 4+ (cheap path: required tiles only) | 5–6% | 43 µs | 0.2% |

- **By decision type:** discards (not after own riichi) are 74–75% of decisions and 88% of the
  time; call/pass decisions 21% and 12% (mean ~0.65 ms); discards after the player's own riichi
  5% and ~0%.
- **By draws left:** the first part of the hand (13–17 draws left) is 37% of decisions and 66–68%
  of the time.

Estimated saving per variant (share of v4's SP time; from these timings, not from a modified
encoder; same on both corpora within 1.5 points):

| Variant | Saving | Mean per decision (v4 = 1,145 µs) | Portable to JAX? |
|---|---|---|---|
| Shanten threshold 3 → 2 | 82–84% | ~200 µs (5–6× cheaper) | No |
| No SP at call/pass decisions | 12% | ~1,010 µs | No |
| Work cap at the p95 decision | 19–20% | ~925 µs | No |
| Work cap at the p99 decision | 4% | ~1,100 µs | No |
| Required tiles only (cheap path everywhere) | 97% | ~35 µs | Probably |
| No SP (v5) | 99.7% | 3.4 µs | Yes |

- The cost is the 3-shanten search: one decision in six, mostly early in the hand, takes 5.5 ms
  on average and accounts for five-sixths of the time. Lowering the threshold to 2 removes most of
  it but leaves v4 ~60× above v5; only "required tiles only" gets within ~10× of v5.
- Skipping call/pass decisions or capping the work barely helps: the cost is spread over many
  3-shanten discards, not a few outliers.

## 4. Training runs (M3, M6)

Configs `experiments/configs/sp-nosp-{a,b}.toml`: the pilot configs with `version = 5` and their
own paths, nothing else changed. Run by `experiments/run_sp_ablation_chain.sh` (log:
`workdir/sp-ablation/chain.log`, per-minute RAM: `nosp-*/ram.log`).

**Matched-pair check.** The trainer and data loader are the code the pilots ran
(`train_offline_phase1.py` and `dataloader.py` last modified 2026-10-03 02:26, before pilot A
started at 10:31; v4's encoder hash unchanged). Startup matches (seed, 3,531,008 + 274,222 params,
500,000 files), and like both pilots the run exhausted its data at exactly step 157,869
(156,000 + 1,869), so it saw the same number of samples.

| Run | Time | Steps/s | Samples/s | Peak RAM | Final checkpoint |
|---|---|---|---|---|---|
| Pilot A (v4, seed 1) | 24.8 h | ~1.75 | ~3.6K, featurization-bound | ~8.6 GB (2026-10-03; method not recorded) | step 156,000, `15121432…` |
| Pilot B (v4, seed 2) | 25.4 h | ~1.76 | ~3.6K | — | step 156,000, `b267c026…` |
| **No-SP A** (v5, seed 1), 2026-10-06 22:05 → 10-07 06:39 | **8.6 h** | 5.1 | 10.5K, mostly GPU-bound (GPU ~80%, trainer ~2.7 CPUs) | 13.3 GB (system used, 1-min samples; steady 12.5–13.3) | step 156,000, `498aabc8…` (read-only) |
| **No-SP B** (v5, seed 2), 2026-10-07 08:07 → 16:40 | **8.6 h** | 5.1 | 10.5K | 13.1 GB | step 156,000, `3180f2f0…` (read-only) |

- **RAM:** 13.3 GB system-wide (MemTotal − MemAvailable, including ~1.4 GB of shared memory for
  batches) is above the pilots' recorded 8.6 GB, but steady from the first minute, with no memory
  pressure. Each loader worker holds ~4 GB. Whether the pilots really used less or were measured
  differently is unknown. A stall watchdog ran alongside; nothing tripped.
- **BC loss** (nats per decision, 2K-step means; diagnostic only, CLAUDE.md rule 1). One epoch,
  so every batch is unseen when its loss is taken:

  | Step | Pilot A | Pilot B | No-SP A | No-SP A − A | B − A |
  |---|---|---|---|---|---|
  | 2K | 0.830 | 0.826 | 0.892 | +0.062 | −0.004 |
  | 10K | 0.567 | 0.566 | 0.575 | +0.008 | −0.001 |
  | 20K | 0.541 | 0.543 | 0.549 | +0.007 | +0.002 |
  | 40K | 0.525 | 0.526 | 0.531 | +0.006 | +0.001 |
  | 80K | 0.510 | 0.509 | 0.511 | +0.001 | −0.001 |
  | 120K | 0.495 | 0.497 | 0.495 | −0.000 | +0.002 |
  | 156K | 0.4888 | 0.4887 | 0.4880 | −0.001 | −0.000 |

  The SP inputs speed up early learning (a 0.06-nat gap at 2K steps, ~0.007 at 10–40K), but the
  gap closes by ~100K steps (~200M samples): over steps ≥ 100K it averages −0.0005 (B − A: 0.0000).
  **Pair 2 repeats it:** no-SP B − pilot B is +0.066 at 2K, +0.008 at 10K, +0.006 at 40K, +0.003
  at 80K, +0.000 at 120K and −0.0001 at 156K (final 0.4886 vs 0.4887); mean over steps ≥ 100K
  +0.0003. Like the others, no-SP B exhausted its data at step 157,869.

## 5. Verdicts (M4, M6)

**Smoke** (mixed versions through `lab.arena.play_sets`, benchmark key, 16 sets): runs; each side
encodes with its own engine's version (also covered by `tests/test_obs_v5.py`).

**Pair 1: PASS.** `20261007-064005-T-NI`, no-SP A (v5, seed 1) vs pilot A (v4, seed 1), family B,
AMP, exploratory, calibrated boundaries. Pass at 10,000 sets (40K games, 06:40 → 08:07):
mean Δu = −0.0007, set SD 0.342, LLR +3.66 (boundary +2.944). For scale only (a sequential
estimate, so this interval is not a valid CI): ±1.96·SD/√n ≈ ±0.0067. The running mean stayed
within ±0.01 throughout (+0.008 at 2,000 sets, −0.003 at 8,500).

**Style panel** (challenger seat; `panel.json` in the run dir), against pilot A as challenger in
the A/A′ run (vs 3 × pilot B, 80K games): no metric moved beyond noise.

| Metric | Pilot A | No-SP A |
|---|---|---|
| Avg rank | 2.498 | 2.500 |
| Win / deal-in % | 21.77 / 13.07 | 21.73 / 13.14 |
| Call / riichi % | 28.29 / 20.00 | 28.45 / 20.10 |
| Dama share of wins % | 12.54 | 12.49 |
| Avg win / avg deal-in | 6,355 / −5,200 | 6,386 / −5,223 |
| Mangan+ / honitsu share of wins % | 32.13 / 4.26 | 32.37 / 4.25 |
| First discard honour / 2–8 % | 62.33 / 4.77 | 62.37 / 4.71 |
| Shanten-backward, quiet / facing riichi % | 0.704 / 13.05 | 0.705 / 13.06 |
| 4th as oorasu leader % | 0.66 | 0.71 |

(Deal-in SE ≈ 0.05 points over 420K kyoku; the opponents differ slightly: 3 × pilot B there,
3 × pilot A here.)

**Pair 2: PASS.** `20261007-164116-T-NI`, no-SP B (v5, seed 2) vs pilot B (v4, seed 2), family B,
AMP, exploratory. Pass at the 5,000-set minimum (20K games, 16:41 → 17:24): mean Δu = +0.0044,
set SD 0.328, LLR +4.37. Smoke beforehand: 16 sets ran.

**Two matched pairs pass T-NI: preliminary evidence (`02` §6.1) that the SP channels can be
dropped at this scale** (b8c192 BC quick students, 500K games, one epoch). Not confirmed: no
family-C nonces or reserved panel configurations. The T-NI margin is 0.01 u, so this does not
rule out a smaller cost; the fixed-N effect-size runs (§5.1) measure it.

Style, no-SP B (20K games) next to pilot A's A/A′ panel and no-SP A: mostly within noise, a few
small moves (deal-in 13.33% vs 13.07%, riichi 19.73% vs 20.00%, first-discard honour 63.2% vs
62.3%, shanten-backward quiet 0.65% vs 0.70%, facing riichi 12.5% vs 13.1%). No pilot-B panel as
challenger exists, so seed-to-seed style variation can't be separated from the SP effect here; the
effect-size runs give an 80K-game panel per pair.

### 5.1 Effect size (fixed N)

Agreed with the user 2026-10-07: if both pairs pass, measure each pair at a fixed 20K sets on
fresh family-B nonces (`python -m lab measure --sets 20000 --family B`), pool the two pairs'
set scores, and apply the decision line: **adopt v5 if the pooled 95% lower bound is above
−0.005; keep v4 (and try a cheaper SP variant) if it falls between −0.01 and −0.005.** Sequential
T-NI means are biased by optional stopping, so they don't enter the estimate.

**Amended with the user 2026-10-07 ~22:00, after pair 1's result and before pair 2's existed: a
consistency check.** Each pair's effect is the recipe effect plus run-to-run variation plus
evaluation noise; pooling assumes the first two are shared. So, with d = m₂ − m₁ and
SE_d = √(SE₁² + SE₂²):
- **|d / SE_d| > 1.96 → inconclusive** at the recipe level (the pairs disagree beyond measurement
  noise; more pairs needed), whatever the pooled number says;
- otherwise apply the line above to the pooled mean (the average of the two pair means; 95% CI
  by a stratified bootstrap): adopt v5 above −0.005, keep v4 in (−0.01, −0.005], keep v4 at or
  below −0.01.

A random-effects interval was rejected: with two pairs its t-multiplier is 12.7, so it says
nothing. With two pairs the check only catches large disagreements; "consistent" doesn't prove
equal effects. Code: `experiments/sp_ablation_pool.py <pair-1 run> <pair-2 run>`. Given pair 1
(−0.0060, SE 0.0024), pair 2 is consistent if its mean is in about [−0.0125, +0.0005], and the
pooled line can then no longer reach "adopt".

**Pair 1:** `20261007-173414-measure`, no-SP A vs pilot A, 20,000 sets (80K games, fresh family-B
nonces, 17:34 → 20:30): **mean Δu = −0.0060, 95% bootstrap CI [−0.0107, −0.0014]**, set SD 0.336;
avg rank 2.508, −0.59 display pt per game. The interval excludes 0: at checkpoint level, no-SP A
is measurably weaker than pilot A, by about 0.6 pt per hanchan. That is consistent with its T-NI
pass (−0.0007 over 10K other sets; the two differ by ~1.3 SE), which only ruled out −0.01.

Style (80K games, against pilot A's A/A′ panel, whose opponents were pilot B): win −0.11 points
(SE ≈ 0.045), deal-in +0.07 with a slightly larger average deal-in (−5,228 vs −5,200), riichi
+0.11, **dama share of wins 12.19% vs 12.54% (−0.34, ≈ 4 SE)**, tobi 7.34% vs 7.11%. The dama
shift fits the SP calculator's riichi-vs-dama EV comparison going missing, but one run against a
mismatched reference can't establish that.

**Pair 2:** `20261007-203040-measure`, no-SP B vs pilot B, 20,000 sets (20:30 → 23:28):
**mean Δu = +0.0012, 95% CI [−0.0035, +0.0059]**, set SD 0.334, avg rank 2.499.

**Decision rule** (`experiments/sp_ablation_pool.py`, output `workdir/sp-ablation/pooled.json`):

| | Mean Δu | SE | 95% CI |
|---|---|---|---|
| Pair 1 | −0.0060 | 0.0024 | [−0.0107, −0.0014] |
| Pair 2 | +0.0012 | 0.0024 | [−0.0035, +0.0059] |
| Pooled | −0.0024 | 0.0017 | [−0.0056, +0.0009] (stratified bootstrap) |

- Consistency: d = +0.0072, SE_d = 0.0033, **z = 2.16 → inconsistent** (just past 1.96).
- **Verdict (pre-registered): INCONCLUSIVE at the recipe level.** Without the check the pooled
  lower bound (−0.0056) would have fallen in the grey zone (keep v4). **Either way, v5 is not
  adopted.**
- Style: pair 1's dama shift does not replicate. No-SP B's 80K-game panel has dama 12.76% (up),
  calls 28.7%, deal-in 13.4%, riichi 19.7%, honitsu 4.7%, lower shanten-backward rates (0.66% /
  12.5%). With pilot A's panel as the only reference, seed-to-seed style can't be separated from
  an SP effect.

## 6. What it means, and recommendation

**Run-to-run variation is about as large as the effect.** The pairs differ by 0.0072 against
evaluation noise of 0.0033. Read naively, each trained network carries "luck" of roughly ±0.003 u
(one observation; the A/A′ difference, +0.0015 ± 0.0023, fits it). Then each matched pair has
~0.005 of noise, and resolving an effect of this size to ±0.005 takes ~4 pairs, to ±0.0025 more
than 10. On this workstation an extra pair costs ~37 h (a 25 h v4 pilot, an 8.6 h v5 student, a
3 h measurement). **Lesson for the lab:** recipe claims at T-fine resolution (0.005) need more
than the two pairs `02` §6.1 counts as preliminary evidence; the A/A′ (one pair) understated how
much training runs vary.

**Recommendation (for the user):**
1. **Keep v4 for Phase 1.** That is the pre-registered outcome; the plan already uses v4, so
   nothing changes. Record the result under D12 (reopen condition not met).
2. **Don't buy more SL pairs.** Two more would most likely still leave an effect of ~0.002–0.003
   unresolved.
3. **Revisit SP only if throughput binds at G0**, and then in RL at equal box-days (the v4/v5
   twin design discussed 2026-10-07): an SL comparison can't credit v5's extra games, an RL one at
   equal cost does. The first variant to try then is **shanten threshold 1** (≈ 98% of v5's
   saving, keeps the tenpai/1-shanten EV tables).
4. **Or adopt v5 as a deliberate trade** (a possible cost of up to ~0.5 pt per hanchan for ~3×
   cheaper SL and actors). That is a legitimate call, but it needs a decision-log entry stating
   that the strength evidence was inconclusive.

**Not done / open:** the decision above (user); obs v6 (889 rows) only matters if v5 is adopted;
the lab's git "dirty" flag is set on every run (the nonce ledger is appended before the git state
is read), a cosmetic fix offered, not made.

## Artifacts

`workdir/sp-ablation/`: `golden-v4.json`, `m2/` (speed), `sp-cost.json`, `learner.json`,
`learner-drain/`, `nosp-{a,b}/` (checkpoints `nosp-*-final.pth`, read-only; `train.log`,
`ram.log`, `tb/`), `pilot-a-as-v5.pth` (timing copy), `chain.log`, `tni-*.log`, `measure-*.log`,
`pooled.json`. Lab runs: `workdir/lab/20261007-{064005,164116}-T-NI`,
`workdir/lab/20261007-{173414,203040}-measure`.
