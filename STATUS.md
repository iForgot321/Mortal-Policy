# STATUS

Running log of where the project is against `docs/reference/00_ROADMAP.md`. Newest first.
Update at the end of every session: what changed, verified vs only attempted, next step.

## Current position (2026-10-12 13:30 UTC)

**Phase 0 is exited** (D55, the user, 2026-10-11; the bar was met 2026-10-05). **Phase 1 early
research is under way.**

- **Live on the workstation: arm V's quick CQL model B** (seed 2, D54; `experiments/v.md`),
  started 2026-10-12 09:51:48 UTC by `workdir/cql/queue-b.sh`, PID 2502168, log
  `workdir/cql/qs-b/train.log`. It ends ≈ **2026-10-13 11:00 UTC** and is frozen by hand
  (`cql-qs-b-final.pth`, hash, `tools/fleet/frozen.sha256`). It uses the GPU and ≈ 9 GB of RAM,
  so **no lab or arena jobs until it ends**.
- **CQL model A frozen** (2026-10-12 09:51 UTC, step 156,000, sha256 `f8a867da…`, in
  `tools/fleet/frozen.sha256`). Sanity checks pass (data out at step 157,869; losses finite; CQL
  0.547 → 0.495; next rank 0.755 < ln 4). Its exploration gaps barely moved from step 78K.
- **Arm V online ported** (2026-10-11, `experiments/v.md`):
  - `train_value.py` online and `engine.ValueEngine` (Equim's ε/τ/top-p exploration);
  - `control.head = 'value'` in the client, `TrainPlayer` and `fleet.py start`;
  - config v0 `experiments/configs/v-a.toml`.
  Tested on tiny nets on the CPU (sampler, 1v3 actor side, learner loop with `server.py`). The
  full-size check waits for model B. Open items for the user are in `v.md` → "Open before V's
  fleet runs".
  - **D54 (user, 2026-10-11):** seed 2 starts from CQL model B (`v-b.toml`); each seed plays
    against its own frozen CQL model.
  - **Finding:** at v0's exploration (ε 0.005, τ 0.05) the trainee takes about one non-greedy
    action per 90 games, because CQL leaves gaps of ≈ 1.8 u between the best and second-best Q
    (measured on A at step 78K).
  - **Tuning plan written** (`experiments/v-tuning.md`): 4 two-hour runs on 3 boxes (24
    box-hours, ≈ $11–15): Equim's exploration, two stronger settings, and 1e-4 at the default.
    Picked by a 10K-set lab `measure` each against CQL model A, with a 0.01 u margin favouring the
    defaults. **Rental approved by the user 2026-10-11 (≈ $11–15).**
  - **Step 1 done (2026-10-12, `c27ecae`, the pre-registration):** ε measured on frozen A, so
    `e1`/`e1-lr` ε 0.050 at τ 0.5, `e2` ε 0.19 at τ 1.0; the four configs
    `experiments/configs/v-tune-*.toml`. **Runs done 2026-10-12 12:37 UTC** (`experiments/v-tuning.md` → "The runs"): e0,
    e1, e2 valid (e0's learner was lost on Vast and resumed from its 11:16 checkpoint); **e1-lr
    void**: e1's teardown destroyed its boxes at 12:23 (`fleet.py`'s sweep matched labels by
    prefix; fixed with a test). $12.12, 25.5 box-hours, ≈ 1.5 over the allowance, inside the
    approved dollars. **e1-lr rerun approved by the user** (D56: ≈ $3, the sweep ≈ $15.3, 31.5 box-hours; running from ≈ 13:30 UTC). Step-matched
    comparisons (amendments written before any measurement). **Next: 5 measures** after model B
    (e0, e1 final, e1 `t1145`, e2, the rerun e1-lr's match; ≈ 7.5 h). Evaluation on the workstation after model B (≈ 6 h).
- **Box-side deadline built and armed on all 9 boxes** (2026-10-11, user-approved;
  `tools/fleet/README.md`). An hour after each run's hard deadline, actors destroy themselves and
  the learner stops itself (its checkpoints survive).
  - The watchdogs were respawned at the same deadlines on the new code, which waits on the wall
    clock.
  - `down` keeps an unpulled learner that is stopped, out of the reach of `watch`.
  - The three watchers still run the old code: the auto-mode classifier blocked restarting them.
    See the handoff.

- **The eval box is closed** (the user, 2026-10-12): it ran 07:03–08:04 UTC (instance 55513275,
  EPYC 9684X, $0.464/h, ≈ $0.80 with three failed setups). The approval stands; it comes back
  when the user has the riichi.dev token (planned Tuesday 2026-10-13), for the validation game and
  the 100K-game test. `tools/fleet/evalbox.py up --yes` (≈ 10 min); the lab lease is released
  meanwhile, so lab jobs run on the workstation.
  - **Its deterministic A/A on pilot A passed** (2,000 sets, 0 divergent games; FP32).
  - **07's 100K-game illegal-action test is paused:** chunk 0 reached 2,800 of 5,000 games with
    0 failures before the box closed, but no chunk finished, so all 20 run again
    (`experiments/riichidev-adapter.md` → "Before ranked play" item 4).
  - California, riichi.dev TCP connect ≈ 170 ms. Japan had one verified 4090 offer (Zen 2, 27
    cores). Where the bot runs is decided again when the student is ready.
- **Live in the cloud, billing: arm P1, runs `p1-a` (seed 1) and `p1-b` (seed 2),** launched
  2026-10-11 03:16–03:31 UTC at `b3864a7`. 3 boxes each, ≈ $1.39/h per run, ≈ 114K games/h
  trained per run. ≈ $200 each at the end (432 box-hours; safety stop $250). They end by
  themselves ≈ **2026-10-17 03:20 UTC**. Handoff §1b.
- **Live in the cloud, billing: arm P0, run `p0-a`.**
  - 3 Vast boxes, ≈ $1.47/h, ≈ 110K games/h trained; $72 so far (22:42 UTC), ≈ $218 at the end.
  - Supervised by a `fleet.py watch` process **on this workstation**, which must stay on and awake.
  - Ends by itself ≈ **2026-10-16 02:15 UTC**, then the G0 T-std and the doubling `measure` run
    on the workstation.
  - Runbook, commands and checks: **`docs/design/handoff-2026-10-10.md`**. Check it first in any
    new session.
  - **Policy lag** runs at mean ≈ 7, p95 ≈ 9–11 versions, against the soak's p95 ≤ 2. PPO copes
    (ratio p99 ≈ 1.35), so P0 continues. B0 showed the same lag, so P1 will too. Finding and options:
    `experiments/p0.md` → "Policy lag during run p0-a".
- **Done:** the P1 B0 warm-ups. Seed 2 passed; seed 1 passed on its registered 4 box-hour
  extension (`p1a-b0x`, 2026-10-10, $1.99; `experiments/p1.md`). Both P1 seeds can start.
- **Approved 2026-10-11: both P1 runs, ≈ $640 of new spend in all** (P0 ≈ $218 + P1 ≈ $424; past
  the user's $500 line, approved explicitly), after a P1 learner check. The projection at launch
  prices is ≈ $620.
- **P1 learner check done** (`p1-tune`, $0.60; `experiments/p1.md`): 4 loader workers as in B0,
  plus **8 torch threads in the trainer** (new `control.num_threads`). Torch's default of 32
  oversubscribed the box and halved the learner: ≈ 4.8K samples/s ≈ 110K hanchan/h instead of
  2.4K. The benchmark's 'torch.compile hang' was a fork deadlock after in-process generation;
  fixed.
- **riichi.dev adapter built** (`deploy/`, `experiments/riichidev-adapter.md`, 07 §1). Parity at
  the plan's scale passed: a 1,000-game replay of the FP32 A/A (622K requests, inputs and moves
  identical to the lab engine's) and 300 riichienv self-play games (193K requests). Not yet
  connected.
- **riichi.dev decisions (user, 2026-10-12;** `experiments/riichidev-adapter.md` → "Before
  ranked play"**):** ranked play waits for the π_SL student, as 07 says; riichi.dev allows 4 bots
  per account at a time; **the shared eval/deploy 4090 is approved** (≈ $0.55/h, ≈ $13/day,
  ≈ $400/month); pushing `main` to the public fork is approved (AGPL).
- **riichi.dev operations built** (2026-10-12): latency per decision class, the supervisor (restarts
  after crashes only; stops between games on a stop file) and the daily summary.
- **Waiting on the user:** the riichi.dev bot token, stored outside the repo, for the validation
  game.
- **Decisions D48–D55** were made in the last four days (fleet network, obs review at G0, 4090s,
  triple ron, P1 horizon and critic, P1 seeds and B0 budget, V seeds and opponents, Phase 0
  exit). See `08`.
- **Cloud spend:** the three live runs have billed ≈ $126 (22:42 UTC: P0 $72, P1 $27 + $27), on
  top of the earlier one-off boxes (B0 warm-ups, checks, benchmarks); ≈ $4.25/h while all three
  are live, ≈ $620 for the three runs at the end (approved ≈ $640).

### Phase 0 deliverables

| Phase 0 deliverable | State |
|---|---|
| Lab: GSPRT, calibration simulator, seed families, nonce ledger, registry, CLI | **Done** (code + tests), calibrated (D44) |
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
| Fleet (03): VPN, provisioning, benchmarks, Mahjax spike | Box workflow **done** (`tools/fleet/setup_box.sh`, `bench_1v3.py`, `box_bench.sh`). **Hosts benchmarked** 2026-10-06 (eval mode, RTX 4090): Zen 4 EPYC 9684X 49.7K hanchan/h ($9.3/M), Ryzen 7950X 51.4K ($13.9/M), Zen 3 EPYC 7V13 40.0K ($11.6/M, contended host), Zen 2 EPYC 7742 30.9K ($15.8/M); P0 actor 31.7K/h, learner ~3.75K samples/s. **Bottleneck:** libriichi's arena loop on the main thread (~89% of main-thread samples; inference ~2%). P0 at 2,000-game batches +28% actor rate but lag mean ~2, max 4. **Multi-process actors:** 2 processes per box 77.8K/h (1.48×), 3 processes 85.3K/h (1.62×, ~$5.4/M). **Inference matrix (reduced):** actor size up to b16c256 costs ≤ ~6% throughput, b24c512 ~28% (AMP), torch.compile +8–11%. **Featurization** (workstation): loader 825/s on 1 thread → 5.7K/s on 16; SL pipeline 4.2K/s (`experiments/fleet-host-benchmarks.md`). **Mahjax spike done** 2026-10-06: D12 stands (parity fixable, throughput per dollar passes 7.5–12× on a rented 3060, integration ≈ 6–9 weeks; `experiments/mahjax-spike.md`). **Fleet scripts done** 2026-10-08 (`tools/fleet/fleet.py`; private network = restricted SSH tunnels, D48): tested end to end on Vast (`experiments/fleet-e2e.md`). **Cost forecast through G1, v0** (2026-10-08, `docs/design/cost-forecast-g1.md`, `tools/fleet/forecast.py`): plan at measured prices ≈ $8.8K vs $9.0K; levers ≈ $6.5K; pessimistic ≈ $10.8K; the learner (≈ 85K hanchan/h) binds once actors scale. CPU-only test **deferred** (D47, 2026-10-08) |

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
- Multi-process actors + reduced inference matrix on one EPYC 9684X box ($0.62; `bench_matrix.sh`): 1 / 2 / 3 concurrent processes give 52.6K / 77.8K / 85.3K hanchan/h, so 2–3 actor processes per box is the biggest throughput lever found (~$5.4–6.0 per million hanchan). Shadow pricing: up to b16c256 costs ≤ ~6% vs b8c192, b24c512 ~28% under AMP; torch.compile +8–11%. The shadow mechanism's own ~20% cost points at the observation path (np.stack + host-to-device copy) as a cheap optimization target.
- **Phase 0 exit, as `00` words it** (L0; σ_set, hanchan/h per box type and featurization samples/s logged with commit and config): all four are now logged. Still open among the 03 §10 Phase 0 deliverables: inference matrix, CPU-only verdict, Mahjax spike, VPN/health-check scripts, cost forecast through G1. Whether to declare exit now is the user's call.

## 2026-10-06 — Mahjax spike (03 §7, D12)

**Built**
- libriichi: `PlayerState.action_mask(at_kan_select)` from Python (6 lines in `state/getter.rs`; no behaviour change), test `tests/test_action_mask_py.py`.
- `tools/mahjax/`:
  - `parity.py` replays mjai logs through Mahjax with each kyoku's wall rebuilt from the log, and compares every seat's legal set at every decision, plus deltas, kyoku transitions and final scores;
  - `bench.py` and `net.py` measure throughput, with a JAX copy of the v4 network;
  - `select_patch.py` is the memory fix;
  - `box_mahjax.sh` is a cloud driver, not run.
- Conda env `mahjax` (JAX 0.11.2, Mahjax 0.1.6 editable from `workdir/mahjax-src`); the `mortal` env is untouched.

**Verified**
- Negative controls catch changed rules: a kuikae mutant, abortive draws off, and double ron off.
- Parity over 30K games (10K pilot A vs B, 10K hobbled, 10K Tenhou 2024-03): 19.5M decisions, 105 different, all explained.
  - Three Mahjax rule bugs vs Tenhou: furiten after passing a no-yaku winning tile, a honba ≥ 8 dealer cap, and riichi ankan of a waited tile.
  - One libriichi deviation: its arena pays a triple ron, which Tenhou aborts.
  - Mahjax's payouts equal Tenhou's on all 10K Tenhou games.
- Throughput on the 3060:
  - upstream Mahjax runs out of memory at 256 envs (22 MiB per env from `vmap` × `lax.cond`);
  - patched, env only: 39K steps/s;
  - env + observation + b8c192: 22.3K steps/s ≈ 127K hanchan/h (632 steps per hanchan), 5.8× libriichi on the same machine.
- `cargo test`, clippy (baseline 27), fmt and `pytest` (59) pass.

**Not done:** 5090/H100 throughput (needs a rental, ~$2–3; wouldn't change the verdict). Verdict recorded under D12 in `08`; new open decision D12a (triple ron in libriichi's arena).

## 2026-10-06/07 — SP ablation (hand-off `docs/design/sp-ablation-handoff.md`): done, verdict inconclusive

**Built (committed)**
- **Obs v5** = v4 with the 123 single-player-table rows (889–1011) zeroed and the SP calculator never called; same shape, so `Brain(version=5)` builds v4's modules and the same seed gives the same initial weights. libriichi (`consts.rs`, `obs_repr.rs`, oracle arms), `model.py`, `engine.py`.
- `tools/obs/hash_obs.py`: golden hash of `encode_obs` over 36 fixed arena/Tenhou logs; `--check-v5`.
- `bench_featurize.py --version`; `bench_1v3.py` reports network decisions per game.
- Configs `experiments/configs/sp-nosp-{a,b}.toml` (pilot configs with `version = 5` and new paths, nothing else) and the unattended chain `experiments/run_sp_ablation_chain.sh`.
- `tools/obs/sp_cost.py` (where v4's SP cost goes; cheaper-variant estimates), `tools/fleet/bench_learner.py` (P0 learner samples/s by obs version), `experiments/sp_ablation_pool.py` (the pre-registered decision rule).

**Verified**
- v4 bit-identical: golden hash `b9817059…` (22,794 decisions) identical before and after.
- v5 == v4 with the rows zeroed, masks equal: every decision of those 36 logs; every decision of every Rust state test (+ a dedicated test, debug build too); pytest on encoder, SL loader, init weights and a v5-vs-v4 arena game. `cargo test`, clippy (baseline 27), fmt, `pytest` (64) pass.
- **Speed (M2):** encode 3.4 µs vs 1,163 µs per decision; SL loader 90× per thread (74.5K vs 830 samples/s); SL pipeline (2 workers) 19.2K vs 4.6K samples/s; arena 54.4K vs 22.7K hanchan/h (2.8× per decision), CPU busy 2 vs 10, GPU 48% vs 22%.
- Training path unchanged since the pilots (trainer/dataloader files predate pilot A's start; v4 hash identical), so the pairs are matched.

**Results** (all in `experiments/sp-ablation.md`; registry rows exploratory)
- **Training:** no-SP A and B 8.6 h each (pilots ~25 h), mostly GPU-bound, peak RAM ~13.3 GB (system-wide, steady); both exhausted their data at step 157,869 like the pilots. BC loss: the early gap (0.06 nats at 2K steps) closes by ~100K steps; final 0.4880 vs 0.4888 (A), 0.4886 vs 0.4887 (B).
- **T-NI (δ = 0.01): both pass.** Pair 1 `20261007-064005-T-NI` (10K sets, −0.0007, LLR +3.66); pair 2 `20261007-164116-T-NI` (5K sets, +0.0044, LLR +4.37).
- **Effect size (fixed N, 20K sets each, family B):** pair 1 `20261007-173414-measure` −0.0060 [−0.0107, −0.0014]; pair 2 `20261007-203040-measure` +0.0012 [−0.0035, +0.0059]; pooled −0.0024 [−0.0056, +0.0009].
- **Pre-registered verdict: INCONCLUSIVE** (the pairs disagree, z = 2.16; consistency check added with the user before pair 2's result). Without the check: grey zone, keep v4. **v5 is not adopted; D12's reopen condition is not met.**
- **SP cost:** 3-shanten decisions are ~17% of decisions and ~83% of v4's SP time; shanten threshold 3→2 would save ~83%, threshold 1 ~98%, call/pass skipping ~12%.
- **P0 learner** (workstation): v4 3.6K samples/s (featurization-bound, matches the cloud's 3.75K), v5 8.4K (2.3×); GPU-only ceiling 15.1K.
- **Lesson:** two matched pairs differ by ~as much as the effect (run-to-run "luck" ~±0.003 u per network, rough); recipe claims at 0.005 resolution need ~4+ pairs, not 2.

**Open (user):** keep v4 (recommended; nothing changes) or adopt v5 as a deliberate trade (needs a decision-log entry stating the evidence was inconclusive); whether to record the result under D12 in `08`; the v4/v5 RL twin at G0 and the D47 draft only if SP is revisited. Lab quirk found: every lab run records `git.dirty = true` (the nonce ledger is appended before the git state is read); fix offered, not made.

## 2026-10-08: fleet scripts and private network (03 §10, D48)

The user chose SSH tunnels over WireGuard/Tailscale (D48; `03` §2 and the `00` risk register now name it).

**Built**
- `mortal/`:
  - `server.py` refuses a non-loopback bind unless `online.server.allow_non_loopback` is set;
  - `recv_msg` unpickles with `weights_only=True`, so a hostile actor host can't run code on the learner;
  - `train_online` saves checkpoints atomically;
  - `client.py` logs `blocked Ns waiting for params`, the soak's "actors blocked" figure.
- `tools/fleet/fleet.py`: up, provision, join (`--verify`), start, stop, health, pull, watch (`--replace`), down, status and ssh. It keeps a per-run lock, records `events.jsonl` and `health.jsonl`, runs a detached watchdog (`up --hard-hours`), and every full `down` ends with a label sweep.
- Box side:
  - `box_tunnel.sh`: the actor's reconnecting tunnel;
  - `box_supervise.sh`: a restart loop with backoff;
  - `box_learner.sh`: server and trainer as one unit, restarting together, because a fresh server would leave a running trainer waiting forever;
  - `box_health.py`: the probe.
- Tests: `tests/test_fleet.py` (pure logic) and `tests/test_transport.py` (+2: pickles refused, loopback only). `tests/test_fleet_tunnel.py` runs the real restrictions against an sshd; it's gated by `FLEET_TEST_SSHD` and has a negative control.

**Verified**
- `pytest tests/`: 77 pass, 4 skip (the gated tunnel tests). The tunnel tests pass against Ubuntu 22.04's OpenSSH 8.9 (`FLEET_TEST_SSHD`), including the negative control (all 3 violations caught) and the host-key pinning.
- Local P0 smoke under the box scripts: a killed `server.py` restarts both processes, the old trainer group (workers, respawned child) is gone, and training resumes. A stopped session leaves no processes and frees the GPU.
- **Cloud** (`experiments/fleet-e2e.md`, $0.41):
  - bring-up in 9.5 min;
  - all 4 restriction checks pass on a real learner;
  - first-minibatch ratio 0.99999 across machines;
  - recovery from a killed client (10 s), tunnel (5 s) and `server.py` (22 s);
  - a destroyed actor replaced by `watch --replace` 591 s after detection;
  - `down` left zero instances.
- P0 through the tunnel: ≈ 66K hanchan/h per box with 2 processes (short sample).
- One fix came out of the test: the image's sshd restarts seconds after first answering, which cost a box. `provision` now waits for two SSH answers 20 s apart and retries syncs.

**Not done:** learner replacement (in the code, never exercised); the eval box's GSPRT queue; interruptible actors. The fleet soak (03 §3.7, ≥ 48 h at T1 scale) needs the leading arm and the user's go-ahead.

## 2026-10-08 (later): Phase 1 readiness decisions (D49, D50) and budget checkpoints

**Decided by the user**
- **D49:** obs v4 is Phase 1's observation, not the project's. It is reviewed at G0 against a measured value of one doubling of RL compute; v5's break-even is ≈ 0.002–0.004 u per doubling. `05` §1 and §2.2 and the `00` G0 section are amended.
- **D50:** every fleet role runs on RTX 4090s through Phase 2, because the pinned torch 2.6 has no Blackwell kernels. `03`, `04`, `07` and the `00` planning unit are amended. D17a narrows to "4090 with ≥ 48 vCPU vs H100".

**Built**
- `fleet.py`:
  - `box_hours` (the billed compute budget);
  - `snapshot --label` (copied on the learner, hash-checked, read-only, never overwritten, logged in `snapshots.jsonl`);
  - `watch --budget-box-hours B --snapshot-at 0.25,0.5`, which freezes `p25`/`p50`, then `full` at the budget, then downs.
- Tests: `test_fleet.py` +2 (budget accounting; snapshot hash check and no-overwrite).
- `experiments/p0.md`: the P0 run plan, with the pre-registered doubling estimate (`measure` full vs p50, 100K sets, ≈ ±0.002 u).

**D51 (decided by the user): the arena aborts a triple ron.**
- **The deciding fact:** riichi.dev runs on riichienv-core. Its default rule (the Tenhou preset) aborts a triple ron, verified by staging one in riichienv 0.4.10 (`tools/rules/riichienv_sanchaho.py`, `01` §3).
- **The change:** `arena/board.rs`. Three rons end in a zero-delta `ryukyoku`; the dealer repeats with honba +1, a riichi declared on that discard is not accepted, and the three rons are recorded as not applied (D46).
- **Rust tests:** triple ron, riichi on the discard, and a double-ron control. With the abort disabled, both triple-ron tests fail (negative control).
- **Verified:**
  - `cargo test` passes, clippy is unchanged at 28 errors (the same as without the change), fmt is clean, and `pytest` gives 79 passed, 4 skipped;
  - the v4 golden hash is unchanged (`b9817059…`);
  - the deterministic A/A passes exactly for rule-based play (`20261008-150055-AA`) and neural true-FP32 play (`20261008-150119-AA`), 500 sets each at `6749abd`;
  - the three triple-ron games from the A/A′ run replay identically up to the discard, then end in the abortive draw, and every seat's entries join their sidecars. The declarers' decision comes out as pass (45) with no behaviour probability, as D51 states.

## 2026-10-09: P1 decided (D52) and the K1 critic learning rate chosen

- **D52 (user):** both P1 seeds on horizon H; critic b8c128, MSE onto λ-returns, γ 1, λ 0.95, AdamW with weight decay 0, grad clip 1.0, constant LR. The real config is `experiments/configs/p1.toml`. `train_online.py` gains an optional `control.seed`.
- **LR check** (`experiments/k1-lr-check.md`, pre-registered):
  - 100K fixed B0 games, replayed in identical drains to the unmodified trainer at each LR.
  - **Choice: 1e-4.** EV over the last 20%: 0.0088 vs 0.0070 for 3e-4.
  - The first attempt was killed by memory pressure under P1's cloud loader settings. The rerun used 1 worker and 16-log batches, recorded as an amendment, and peaked at 5.7 GB PSS.
- **Reading:** both critics explain < 1% of the H return's variance after 100K hanchan, still rising.
  - A post hoc ceiling: perfect knowledge of the current kyoku would explain only 14%, since later kyoku dominate the H return. In the game's last kyoku it would explain ~100%.
  - So a low EV is expected under H, but this doesn't show the critic is healthy.
  - **Suggested before the real B0:** log EV for last-kyoku decisions separately.
- **Open:** P1 seed 2's initialization (pilot A again, or pilot B with its own B0); the per-phase EV diagnostic.

**Open (user):** the riichi.dev bot account and adapter, P1 seed 2's init.

## 2026-10-09: arm P0 launched (run `p0-a`)

- **Approved by the user:** the tuning allowance (800-game batches, 2 clients per actor box, learner loader workers by a short check) and ≈ $200 of spend. The user wants to be told if spend heads above $500.
- **Learner check** (`p0-tune`, $0.49, EPYC 7C13): 4 loader workers give 4,841 samples/s against 3,601 with 2, so 4 was chosen.
  - The first `torch.compile` in `bench_learner.py` hung for 45 min on that box. The timing passes then ran eager, an amendment recorded in `experiments/p0.md` before any timing.
- **Health check of the K1 critic** on 4,000 fresh games: EV 0.063 in the last kyoku, 0.013 overall; the untrained control is ~0.
- **fleet.py fixes:** the allowlist now covers custom EPYC models (7B13 Zen 3, 7K62 Zen 2); `watch --allow-zen2` for replacement actors; `up --set` and `--no-start`.
- **P0 launched** 2026-10-09 22:22 UTC at `a8c51e9`:
  - learner EPYC 9684X; actors EPYC 9684X and 7C13, 2 clients each; `dataset.num_workers = 4`; ≈ $1.44/h;
  - all tunnel restrictions verified;
  - first drain of 1,600 logs, first-minibatch ratio 1.00005.
  - `watch` (on the workstation) replaces lost boxes, freezes `p25`/`p50` at 108/216 box-hours, then takes `full` at 432 and downs everything (≈ 2026-10-15 22:00 UTC). Safety stop at $450; watchdog at 156 h.
  - **The workstation must stay on and awake until then.**
- **After the run:** T-std full vs pilot A (G0) and the pre-registered doubling `measure`, full vs p50, 100K sets (`experiments/p0.md`).
- **2026-10-10 incidents** (details in `experiments/p0.md`):
  - the learner box was lost and auto-replaced in 19.6 min;
  - the trainer wedged on Vast's 1,024 open-file limit (fixed: limit raised, stall detection and restart);
  - **a bash precedence bug in `fleet.py start` replaced the trained checkpoint with pilot A on every learner start** (fixed in `d06a10d`, tested).
  - P0 effectively restarted from pilot A at 02:14:11 UTC. Its budget is rebased there: 12.6 box-hours, ≈ $6, discarded. It now ends ≈ 2026-10-16 02:15 UTC.

## 2026-10-10: P1 B0 warm-ups done (D53; `experiments/p1.md`)

- **Seed 1** (`p1a-b0`): 199K games, $5.93. Its final-window calibration is not monotone, so the registered rule calls for **one 4 box-hour extension** (pending).
- **Seed 2** (`p1b-b0`): 164K games, $5.60. **Passes.**
- In both, EV rises then plateaus (overall ≈ 0.011, last kyoku ≈ 0.04). B0 drained only ≈ 45K hanchan/h because of the oracle featurization, below the planned 0.3–0.5M.
- **P0 at 18:27 UTC:** 48.6 of 432 box-hours since the rebase; the learner trains ≈ 110K games/h; $30.00 so far. Snapshots: p25 ≈ Oct 11 14:15 UTC, p50 ≈ Oct 13 02:15, full ≈ Oct 16 02:15.
- **Cloud spend so far:** ≈ $47.8.

## Next steps

Read `docs/design/handoff-2026-10-10.md` first: it has the live runs, the exact commands and the
failure modes already met.

1. **Live runs.** Keep P0's watcher alive until it downs the fleet (≈ 2026-10-16 02:15 UTC). Then,
   on the workstation:
   - G0 T-std, P0 `full` against pilot A;
   - the pre-registered doubling `measure`, `full` against `p50`, 100K sets (`experiments/p0.md`).
2. **P1 (live, handoff §1b).** Keep both watchers alive until ≈ 2026-10-17 03:20 UTC. Then, per
   seed, G0's T-std against its own pilot, the head-to-head against P0 and the doubling estimate
   (`experiments/p1.md`), queued after P0's evaluations.
3. **Arm V** (`experiments/v.md`, `experiments/v-tuning.md`): when CQL model A ends (≈ 2026-10-12
   09:30 UTC) the queue freezes it and starts model B. Then: A's sanity checks, its hash in
   `v.md` and `tools/fleet/frozen.sha256`, `v_qgaps.py` on it to fix the sweep's ε, and the four
   sweep configs, committed. Then the sweep (rental approved 2026-10-11, ≈ $11–15); its
   evaluation runs on the workstation after model B (≈ 2026-10-13 11:00 UTC), or on the eval box
   if it is back first (all four on one machine). V's runs need their own approval.
   Arm H follows V.
4. **SL substrate (`04`):**
   - the secure-tier data path (D23) and the D17a measurement;
   - b24c512 AWR and BC teachers, then the students;
   - the riichi.dev smoke deploy (L1). The adapter and its parity tests are done
     (`experiments/riichidev-adapter.md`); the user decided its inputs on 2026-10-12 (the
     student; the shared 4090). The supervisor, daily summary and latency metrics are built; the
     100K-game illegal-action test runs on the eval box. Next: the validation game (needs the
     token).
5. **Search pilot (`06` §0):** the oracle stage reports by G0.
6. **Fleet:**
   - a fleet soak report from P0's `health.jsonl`;
   - ~~the eval-box queue~~ done 2026-10-12 (`tools/fleet/evalbox.py`, README "The eval box");
   - re-running the cost forecast with D50's 4090 prices.
