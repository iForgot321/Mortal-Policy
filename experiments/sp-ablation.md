# SP ablation: v4 without the single-player tables (Phase 0, D12 reopen condition)

- **Hand-off:** `docs/design/sp-ablation-handoff.md`. **Question:** is a student trained without
  v4's single-player (SP) channels non-inferior to one trained with them? And what is dropping
  them worth in speed?
- **Standard:** a recipe claim, so matched pairs (`02` §6.1, D41). One failed pair keeps SP; two
  passes are *preliminary* evidence that SP can go. Confirmation (fresh family-C nonces, reserved
  panel configurations) is not implemented in the lab CLI and was not run.
- **Nothing is adopted here.** Changing the observation is a frozen-item decision (CLAUDE.md
  rule 2) and needs a decision-log entry first; that is the user's call.

## Summary

_Filled in at the end (M7)._

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

**Pair 2:** no-SP B (seed 2) training since 2026-10-07 08:07; T-NI vs pilot B follows. _Pending._

## 6. Recommendation

_Pending._
