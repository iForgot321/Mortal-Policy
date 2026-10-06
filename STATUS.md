# STATUS

Running log of where the project is against `docs/reference/00_ROADMAP.md`. Newest first.
Update at the end of every session: what changed, verified vs only attempted, next step.

## Current position

**Phase 0 (Foundations). L0 met 2026-10-05.** The lab is calibrated and trustworthy. Phase 0 exit still needs the fleet throughput numbers (03), which need cloud boxes. The behaviour-probability design is decided (D46) and implemented on the actor side.
Everything runs on the workstation (RTX 3060 12 GB, 15 GB RAM), plus rented Vast.ai boxes from 2026-10-06. Cloud spend: $4.71 of $14.3K (benchmark boxes).

| Phase 0 deliverable | State |
|---|---|
| Lab: GSPRT, calibration simulator, seed families, nonce ledger, registry, CLI | **Done** (code + tests); boundaries **uncalibrated** until A/A′ data exists |
| Lab: deterministic A/A | **Pass, exact**: rule-based, and neural in true FP32 (D43) |
| Lab: shanten-greedy floor bot; hobbled patch; style panel + per-seat table; population-panel definitions | **Done** (mixed-table arena for P-MIX/P-STY/P-RES not yet implemented) |
| Lab: A/A′ (σ_set, recipe noise) | **Done** 2026-10-05: σ_set = 0.328 (37.7% duplicate benefit), recipe noise Δu = +0.0015 [−0.0031, +0.0061], per-seat table flat |
| Lab: §4.1 calibration | **Done** (D44): nominal ±2.944 kept for all test types; worst error 4.77% (T-fine, thin margin); `lab.toml` calibrated = true |
| Lab: hobbled T-std (must fail) + A/A′ T-NI (must not fail) → **L0** | **L0 MET 2026-10-05**: hobbled fails at 5K sets (Δu −0.396, LLR −132.8); A vs B T-NI passes at 6K sets (Δu +0.001, LLR +3.28) |
| Data D1–D8 (snapshot → splits → scoring → golden set → card, determinism) | **Done, frozen (D45):** dataset-v1 `7c6a9e60…` |
| GRP-v2 on `train` | **Done, frozen**: `workdir/grp-v2/grp-v2-frozen.pth` (`db28beac…`), val acc 22.8% |
| Pilot quick student A (b8c192 BC, seed 1) | **Done** 2026-10-04 11:19 (24.8 h): step 156,000, BC loss 0.489, frozen `workdir/pilot/qs-a/pilot-qs-a-final.pth` |
| Pilot quick student B (seed 2) | **Done** 2026-10-05 12:43: step 156,000, frozen `workdir/pilot/qs-b/pilot-qs-b-final.pth` (`b267c026…`) |
| Interfaces (05 §3, D37) | **Done with tests** (2026-10-06): raw-logit head, fixed reward units, ordered trajectories + GAE, stored behaviour probs end to end (sidecars → server → trainer). Arms **P0/P1 are wired and smoke-tested** locally: `experiments/configs/p0.toml`, K1 critic + B0 warm-up mode (`docs/design/rl-wiring-handoff.md`) |
| Fleet (03): VPN, provisioning, benchmarks, Mahjax spike | Box workflow **done** (`tools/fleet/setup_box.sh`, `bench_1v3.py`, `box_bench.sh`). **Hosts benchmarked** 2026-10-06 (eval mode, RTX 4090): Zen 4 EPYC 9684X 49.7K hanchan/h ($9.3/M), Ryzen 7950X 51.4K ($13.9/M), Zen 3 EPYC 7V13 40.0K ($11.6/M, contended host), Zen 2 EPYC 7742 30.9K ($15.8/M); P0 actor 31.7K/h, learner ~3.75K samples/s. **Bottleneck:** libriichi's arena loop on the main thread (~89% of main-thread samples; inference ~2%). P0 at 2,000-game batches +28% actor rate but lag mean ~2, max 4. **Featurization** (workstation): loader 825/s on 1 thread → 5.7K/s on 16; SL pipeline 4.2K/s (`experiments/fleet-host-benchmarks.md`). Open: inference matrix, CPU-only test, VPN, Mahjax spike |

## 2026-10-03 — session 1 of v2

See the git diff / commits for code. Highlights:

**Lab (`lab/`, `tests/test_lab_*.py`)**
- `python -m lab {aa,run,measure,confirm,calibrate,table}`. GSPRT per 02 §4 (5K-set minimum, looks every 500,
  variance floor, caps, zero-variance rule); fixed-N T-confirm with exact multinomial bootstrap; §4.1 calibration
  simulator with adversarial variants (sparse, skew ±, heavy); family keys in `lab/seeds.toml`; nonce ledger.
- libriichi: `OneVsThree.py_vs_py_games` (per-game records), `ShantenGreedy` agent (engine type `shanten-greedy`).
- Deterministic A/A, shanten-greedy, 500 sets: **pass, exactly 0**, after summing utilities in integer hundredths.
- Deterministic A/A, b40c192 (v1-era `phase1-v0-final.pth`), 500 sets: AMP **21 divergent sets**; FP32 (TF32 on)
  **2 divergent sets**, all set scores still exactly 0. Every divergence is an argmax flip at a near-tie (top-2 gap
  ≤ 0.004), i.e. batch-composition-dependent numerics, not bookkeeping. FP32 without TF32: running.
  Details: `experiments/lab-harness.md`.

**Data (`tools/data/`, data in `~/mj-data/`)**
- 2,512,433 Tenhou houou games 2009-02 → 2026-01 (the HF README's ~3.16M is wrong; 2024 alone is 184,424).
- 2009 audit reproduced exactly: 2 placeholder games (Q2), 8 score breaks, Q6 = 0.57% of kyoku, 3.5% BYE games.
  All 8 breaks are one converter quirk (no 4th `reach_accepted` in four-riichi draws) → code `Q4s`, kept.
- libriichi: `GameplayLoader(featurize=False)` / `validate_gz_log_files`: replays all seats through the loader's own
  code path without featurizing (~3K games/s) and errors on any label outside its mask (Q5). `PlayerState.action_mask`
  equals `encode_obs`'s mask (asserted at every decision in the state tests; parity on real games). Entries now carry
  `event_idx`.

**Training path (`mortal/`)**
- `CategoricalPolicy.logits` (masked raw logits, D37); the SL loss uses them in fp32.
- `train_offline_phase1.py`: `control.seed`, `policy.objective = 'bc'|'awr'`, `dataset.masks_file` (Q6/Q8),
  `dataset.obs_fp16`. `train_grp.py`: `optim.lr_steps` + `control.max_steps`.

**Session 1, part 2 (10:00–10:45 after a 7-hour stall)**
- The full index crashed at 02:40 in its final concat (2026 has no DB rows → Arrow `null` columns); my
  waiter only watched for the success line, so nothing resumed the session for ~7 h. Fixed
  (`--combine-only`, type promotion); lesson recorded in Claude's memory: run the job itself in the
  background, or wait on its PID.
- Quarantine over 2.51M games: Q5 3 games (2 newly found illegal-action games), Q3 10 (`00e9` games in
  2025), Q4 7 unexplained (dropped), Q4s ~2.7K (kept), Q6 0.6–1.5% of kyoku, Q8 1,747 bot seats (2025).
- Coverage: held-out window OK (102,054 games, 7 complete months; no D14b). Source DBs cover train fully
  and end mid-Oct 2025.
- Splits frozen (leak tests pass); train = 1,805,376 games, 1.19B decisions. Scoring reproduces 04a §6.1
  (σ_b 0.0246, k ≈ 1,140). Golden set: 164 players, 120K decisions (20K × 6 categories).

**Resolved with the user (2026-10-05)**
- D23 encryption of `~/mj-data` on the workstation: not a concern (user).
- Commits go directly to `main` (this is a fork).

## 2026-10-05 — behaviour probabilities (D46)

The user chose option A. Recorded as D46 and implemented; the design note now describes what was built.
- **Engine:** `MortalEngine.sample` returns the probability that its own sampler gave each sampled action.
- **Arena:** the arena keeps a record for every reaction, including passes and overruled claims. Records carry an `applied` flag, and the agari guard nulls the probability. `OneVsThree(behaviour_dir=…)` writes one sidecar per game, under the log's file name. `TrainPlayer` writes them to `log_dir + '_behaviour'`.
- **Learner:** `trajectory.join_behaviour` matches loader entries on (event_idx, seat, kan_select).
- **Mjai logs are unchanged:** `EventExt.behaviour` is never serialized, and a libriichi test checks this.

**Verified:**
- `cargo test` passes, clippy is still at the baseline 27 errors, and `pytest tests/` passes 35 tests (7 new, ~11 s).
- First-minibatch ratio is 1 within 1e-4 (ε = 1, FP32).
- 1,000-game stress check, pilot A vs pilot A: all 633K entries joined, covering multi-rons, four-riichi draws and kans. The 12 agari-guard replacements were exactly the 12 entries without a probability.
- The lab's hobbled engine still runs.

**Not done:** the actor→server→trainer transport (`client.py`/`server.py`), and switching `train_online.py`'s ratio from `Old_mortal` to the stored probabilities. Both belong to the P0/P1 wiring.

## 2026-10-06 — RL wiring for arms P0/P1 (hand-off M1–M6)

Followed `docs/design/rl-wiring-handoff.md`. Nothing is committed yet.

**Built**
- **M1 transport:** `client.py` submits each log's sidecar. `server.py` buffers and drains sidecars under the log's name, in `buffer_dir + '_behaviour'` / `drain_dir + '_behaviour'`; `buffer_size` still counts logs only. The drain reply also carries each log's parameter version and the server's current version (`common.drain_msg`). Clients without sidecars still work.
- **M2 data:** `mortal/rl_dataset.py`:
  - per trainee seat: GRP rewards in fixed **u units** (pts / 100; written down in the module docstring and `p0.toml`), `tj.build`, `tj.join_behaviour`;
  - P0 (`targets = 'mc'`): Monte-Carlo returns, shuffled only afterwards;
  - P1 (`targets = 'gae'`): whole-trajectory chunks with the critic's privileged channels;
  - logs without a sidecar are skipped and counted; a seat whose log and sidecar disagree is skipped with an error.
  - `FileDatasetsIter` (the SL path) is unchanged.
- **M3 trainer:** `mortal/ppo.py` holds the dual-clip loss on masked raw logits in fp32. In `train_online.py`, new config keys whose defaults keep upstream: `policy.ratio_reference = 'stored'`, `dataset.ordered`, `policy.horizon`, `policy.advantage_norm`. Entries with a NaN `b` get weight 0. Tensorboard `ppo/*` logs ratio mean/p99/min/max, clip fraction, entropy, excluded fraction, policy lag (mean, max) and the first-minibatch ratio of each drain.
- **Configs:** `experiments/configs/p0.toml` (05 §2.4; init = a *copy* of pilot A), plus `p0-smoke.toml` and `p1-smoke.toml`.
- **M5 critic:** `mortal/critic.py`, a separate b8c128 on 1012 + 51 channels (opponents' hands; Invisible rows 0–50; wall order excluded). GAE runs on the learner per chunk before shuffling. MSE to λ-returns, with its own AdamW and LR schedule, saved in the checkpoint. `critic.warmup = true` is **B0**: the policy is frozen and only the critic steps. Diagnostics (`critic/*`): explained variance and 10-bin calibration against Monte-Carlo returns, measured before the critic trains on that data; the within/across-kyoku advantage variance ratio.
- **M6:** `python -m lab … --arm/--package/--horizon` records them in registry rows (02 §9). Existing rows are untouched.
- Tests: the arena fixture moved to `tests/conftest.py`. New: `test_transport.py`, `test_rl_dataset.py`, `test_ppo.py`, `test_critic.py`.

**Verified**
- `pytest tests/`: 58 pass (23 new), covering the 05 §8 items for these paths:
  - first-minibatch ratio 1 within 1e-4 through the real dataset, collate and `ppo_loss` (CPU FP32);
  - illegal-action probability exactly 0 (fp32/fp16/bf16 logits);
  - bit-identical targets on recomputation;
  - `done` matches the loader's kyoku ends (K) and the game end (H);
  - no cross-seat targets (other seats' sidecar records scrambled → identical trajectory);
  - reward statistics count stays 0;
  - oracle-loader obs identical to the plain loader's;
  - policy input exactly 1012 channels.
- `cargo test` passes, clippy is at the baseline, `cargo fmt` is clean (no Rust changes).
- **P0 smoke** (M4; `p0-smoke.toml`, 32 games per version, sequential): 3 versions trained in 116 s, no errors. 32/32 drained logs had same-named sidecars. First-minibatch ratio means 0.99993 / 1.00004 / 1.00005 (p99 ≈ 1.002, AMP). Excluded 0%. Policy lag 0 at drain, 1 after the mid-epoch submit. Peak RAM 5.7 GB.
- **B0 → P1 smoke** (`p1-smoke.toml`, horizon H): B0 ran 2 versions (25 critic steps), and the policy stayed **bit-identical** to pilot A. P1 then restored the critic and ran 2 versions: ratio means 1.00002 / 1.00003, excluded 0%, all `critic/*` scalars logged, advantage variance ratio ≈ 0.1 (exactly 0 under P0). Peak RAM 6.8 GB.
- Explained variance ≈ 0 at this scale is expected and says nothing about K1. These were plumbing runs: no registry rows, no strength claims. Run dirs: `workdir/p0-smoke/`, `workdir/p1-smoke/`.

**Not done / open**
- **P1 horizons:** how the two seeds split between K and H (user decision; handoff §5).
- **K1 critic settings:** LR schedule, λ, loss and grad clip are placeholders in `p1-smoke.toml`. They need to be written down before P1 runs (05 §5.1). P1 has no non-smoke config yet.
- B0 for real (0.3–0.5M hanchan ≈ 14–23 h here) and the arm runs: not started; they need the user's go-ahead.
- Trainer gotcha (unchanged): pilot A restores `steps = 156,000` with a fresh optimizer, because the pilot's config was offline.

## 2026-10-06 — first cloud box (fleet trial)

**Done**
- `01` §10 prices re-checked (complete Vast listing; ≥ 32-vCPU provider table; secure-tier options). The `00` planning price ($0.55) is unchanged; today's ≥ 32-vCPU medians are 4090 $0.50, 5090 $0.67.
- `tools/fleet/`: `setup_box.sh` (packages, Python/torch/GPU checks, Rust 1.95.0, pinned packages, libriichi build, frozen-input hashes, box configs, tests), `requirements-box.txt`, `frozen.sha256`, `bench_1v3.py` (eval-mode hanchan/h on a benchmark-only seed key).
- Workflow chosen with the user: Vast.ai, the `vastai/pytorch:2.6.0-cuda-12.4.1-py312-22.04` image, **no published ports** beyond SSH, code by rsync from a clean commit, Claude drives the box over SSH (`~/.ssh/vast_ed25519`), no credentials on boxes. The GitHub fork stays at upstream; nothing pushed.
- Trial box (Vast contract 54461788; RTX 4090, EPYC 9684X, 32 vCPU, $0.464/h): setup in ~3 min, all tests pass, P0 smoke passes the M4 checks, 49.7K hanchan/h eval mode, 31.7K/h P0 actor, learner ~3.75K samples/s. Details and gotchas: `experiments/fleet-host-benchmarks.md`.

**Verified vs attempted:** everything above ran on the box and the numbers are from its logs (`workdir/fleet/20261006-vast-4090-epyc9684x/`). Not measured: 5090 boxes (torch 2.6 has no Blackwell kernels), featurization rate, multi-box networking/VPN.

**Lesson:** the box idled ~7 h after the trial ($3.78 total vs ~$0.40 of work). Destroy right after copying results back.

**Later the same day**
- Featurization rate measured on the workstation (`tools/fleet/bench_featurize.py`): loader 825 samples/s on 1 thread, 5.7K/s on 16; the SL pipeline gives 3.9K/s (1 worker) and 4.2K/s (2 workers). Reading for 03 §4.2: an H100 can't be fed (needs > 100 threads); a 4090/5090 teacher needs ≥ 48 vCPU *if* the pipeline scales with workers — untested beyond 2 workers (RAM).
- Four CPU classes rented in parallel with `tools/fleet/box_bench.sh` (approved offers; boxes destroyed by the drivers): 7950X 51.4K/h, EPYC 7742 30.9K/h; the 5950X and EPYC 7713 never finished pulling the image (25-min limit) and were destroyed unmeasured, so the py-spy profile and the 2,000-game P0 test didn't run. $0.48.
- Zen 3 retry with the extras (EPYC 7V13, $0.45): 40.0K hanchan/h eval mode on a visibly contended host. py-spy: the main thread spends ~89% in libriichi's arena loop and ~2% in the engine, so the actor bottleneck is Rust-side (03 §6 items 7–8), not Python or the GPU. P0 actor: 22.3K/h at 800-game batches, 28.5K/h at 2,000 (+28%), with policy lag rising to mean ~2, max 4. Driver and destroy worked unattended.
- **Phase 0 exit, as `00` words it** (L0; σ_set, hanchan/h per box type and featurization samples/s logged with commit and config): all four are now logged. Still open among the 03 §10 Phase 0 deliverables: inference matrix, CPU-only verdict, Mahjax spike, VPN/health-check scripts, cost forecast through G1. Whether to declare exit now is the user's call.

## Next steps

Pilots A/B, A/A′, calibration and the L0 checks are all done (see the table).

1. Fleet (03): the inference matrix, CPU-only test, VPN and Mahjax spike; decide whether these gate Phase 0 exit. Optional throughput work: shorten libriichi's per-step main-thread work (the measured actor bottleneck).
2. Phase 1 on the workstation: SL teachers and student toward L1 (`04`); riichi.dev smoke deploy.
3. P0/P1: wiring done (2026-10-06). Next: decide the P1 horizon split and the K1 critic settings; then B0 and the arm runs (cloud, 05 §2.2), with the user's go-ahead.
