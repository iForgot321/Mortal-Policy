# STATUS

Running log of where the project is against `docs/reference/00_ROADMAP.md`. Newest first.
Update at the end of every session: what changed, verified vs only attempted, next step.

## Current position

**Phase 0 (Foundations). L0 met 2026-10-05.** The lab is calibrated and trustworthy. Phase 0 exit still needs the fleet throughput numbers (03), which need cloud boxes. The behaviour-probability design is decided (D46) and implemented on the actor side.
Everything runs on the workstation (RTX 3060 12 GB, 15 GB RAM); no cloud spend yet ($0 of $14.3K).

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
| Interfaces (05 §3, D37) | Raw-logit head, fixed reward units (opt-in), ordered trajectories + GAE: **done with tests**. Stored behaviour probs: **decided (D46) and done with tests** for actor sidecars + learner join; client/server transport and the trainer's ratio switch come with the P0/P1 wiring (`docs/design/behaviour-probs.md`) |
| Fleet (03): VPN, provisioning, benchmarks, Mahjax spike | Client reconnect/backoff done; the rest needs cloud accounts and spend |

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

## Next steps

Pilots A/B, A/A′, calibration and the L0 checks are all done (see the table).

1. Fleet (03): cloud accounts, VPN, provisioning, benchmarks, Mahjax spike → Phase 0 exit.
2. Phase 1 on the workstation: SL teachers and student toward L1 (`04`); riichi.dev smoke deploy.
3. P0/P1 wiring: the ordered-trajectory loader in training, sidecar transport, and stored-probability ratios (D37, D46).
