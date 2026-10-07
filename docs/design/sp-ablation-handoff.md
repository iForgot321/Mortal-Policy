# Hand-off: the SP-channel ablation (v4 without the single-player tables)

**For:** a fresh agent. **Written:** 2026-10-06, after commit `dde1299` (Mahjax spike).
**Status:** not started. Everything below runs on the workstation; no rentals (decided with the user).

## 0. Read first, in this order

1. `CLAUDE.md` (rules, machine limits, build steps), then `STATUS.md`.
2. `experiments/mahjax-spike.md`, §3 and the verdict: why this question came up.
3. `experiments/pilot-quick-students.md` and `experiments/configs/pilot-qs-{a,b}.toml`: the recipe you
   are matching.
4. `docs/reference/02_evaluation_lab.md` §6.1 (checkpoint vs recipe claims) and
   `docs/reference/08_decision_log.md` **D41**, **D44**, **D12** (its 2026-10-06 spike result).
5. `libriichi/src/state/obs_repr.rs` (the SP block is the tail of `encode_obs`, from
   `if self.version == 4 {`), `libriichi/src/state/agent_helper.rs` (`single_player_tables`),
   `libriichi/src/algo/sp/` (the calculator).

## 1. Background

**What the SP channels are.** 123 of v4's 1012 observation rows are not observations. They are the
output of a single-player expected-value calculator (upstream Mortal's v4, Equim 2023-10-31,
`2c662b3`; a Rust port of nekobean's mahjong-cpp). The calculator pretends the opponents don't
exist: nobody else wins, calls or deals in, and the player draws from the unseen tiles until the
wall runs out, always making the best choice for their own hand. It runs only within 3-shanten.

| Rows (from 889) | Content |
|---|---|
| 2 | Best achievable expected points (two scales) |
| 68 | For each discard: the tiles that would then improve the hand, split by whether the discard keeps or loses shanten |
| 2 | The discard that leaves the most improving tiles; at a non-discard decision, the current hand's improving tiles |
| 51 | For each discard and each of the next 1–17 draws: tenpai probability, win probability, expected points |

At a call/pass decision the tables describe the current hand, not each discard.

**What they cost** (`tools/fleet/bench_encode.py`, single thread, 200 pilot A-vs-B games, 130,710
decisions):

| Per decision | Time |
|---|---|
| 4 × `PlayerState.update` | ~3 µs per event |
| `encode_obs` v3 (no SP) | 3.4 µs |
| `encode_obs` v4 | **1,158 µs mean**, median 105 µs, p99 13.8 ms, max 50 ms |

That is ~0.76 CPU-seconds per hanchan. It is probably most of libriichi's actor CPU (by arithmetic
~57% of the busy CPU on the EPYC 9684X box; the py-spy profile sampled only the Python main
thread, while encoding runs on rayon workers). It also explains why the pilot students were
featurization-bound: the SL loader's 825 samples/s per thread is ~1.2 ms per decision.

**Why it matters.**
1. It is the blocker in D12. The rest of v4 ports to JAX easily; the SP search does not. D12
   reopens if a JAX-portable observation is non-inferior and actor throughput binds at G0.
2. If SP can go, libriichi actors and SL featurization get much cheaper too, with or without Mahjax.

**The question:** is a student trained without the SP channels non-inferior to one trained with them?
Nobody has measured it; as far as we know upstream added SP in v4 together with other changes.

## 2. Goal and definition of done

**Goal:** answer the question to the plan's standard for recipe claims (two matched pairs =
preliminary evidence, `02` §6.1), and measure what dropping SP is worth in speed. **Do not adopt
anything.** Changing the observation is a frozen-item decision for the user.

**Done when:**
- Obs **version 5** exists: v4 with the 123 SP rows zeroed and the SP calculator never called.
  v4's output is bit-identical to before (hash test), and v5 equals v4 with rows 889–1011 zeroed
  at every decision (test).
- The speed check (M2) is recorded: encoding cost, arena hanchan/h, SL samples/s, v4 vs v5.
- No-SP student **seed 1** is trained, frozen, and tested with **T-NI against pilot A** (a registry
  row, exploratory).
- If that passes: no-SP **seed 2**, trained and tested with **T-NI against pilot B**.
- `experiments/sp-ablation.md` holds the method, numbers, verdicts and a recommendation;
  `STATUS.md` is updated; the result is reported to the user.

## 3. What already exists

| Piece | Where |
|---|---|
| Control students (b8c192, v4, GN, BC, one epoch of quick500k) | Pilot A `workdir/pilot/qs-a/pilot-qs-a-final.pth` (seed 1, sha256 `15121432…`); pilot B `workdir/pilot/qs-b/pilot-qs-b-final.pth` (seed 2, `b267c026…`). Read-only. |
| Their configs | `experiments/configs/pilot-qs-{a,b}.toml` (differ only in seed and paths) |
| Training data | `workdir/pilot/quick500k.pth` (file index) + `~/mj-data/splits/masks.jsonl` (Q6/Q8). Tenhou data: workstation only (D23). |
| Trainer | `mortal/train_offline_phase1.py`, run from `mortal/` with `MORTAL_CFG=…`. `control.seed` seeds the init, the file order and the DataLoader workers (`:68–74`). |
| Unattended chain example | `experiments/run_pilot_chain.sh` (wait on PID, check the log, freeze with `chmod a-w` + sha256) |
| Lab | `python -m lab run --test T-NI --challenger <ckpt> --champion <ckpt> --family B --description …` (AMP by default, D43). Calibrated (D44), registry in `lab/registry/runs.jsonl`. |
| Mixed versions in the arena | Each Rust agent encodes with its own engine's `version` (`libriichi/src/agent/mortal.rs:276–287`); `lab/arena.load_engine` reads `cfg['control']['version']` from the checkpoint. |
| Benchmarks | `tools/fleet/bench_encode.py` (encoding cost), `tools/fleet/bench_1v3.py` (arena hanchan/h; benchmark-only seed key `0x62656e6368`, no nonces), `tools/fleet/bench_featurize.py` (SL samples/s) |
| Measured reference | Pilots: ~25 h each, ~3.6K samples/s, featurization-bound. A vs B: σ_set 0.328, recipe noise Δu +0.0015 [−0.0031, +0.0061]; T-NI A vs B passed at 6K sets (~1 h). Arena: ~22K hanchan/h on this workstation. |

## 4. Design: observation version 5

**Definition:** v5 = v4, except the SP block (rows 889–1011) is all zeros and
`single_player_tables()` is never called. Same shape, (1012, 34).

**Why zeroed rather than removed.** Same input shape means `Brain(version=5)` builds the same
modules as v4, so with the same seed the initial weights and the data order are identical to the
pilot's. The pair then differs only in the SP inputs: the cleanest matched pair available. (Zero
inputs give their first-layer weights no gradient, so it is functionally a removal.) Dropping the
rows for good is a later production decision.

**Sites to change** (grep for `version` to be sure; this list is from 2026-10-06):
- `libriichi/src/consts.rs`: `obs_shape(5) = (1012, 34)`; `oracle_obs_shape(5)` = v4's (the critic
  is unaffected); `MAX_VERSION`.
- `libriichi/src/state/obs_repr.rs`: every `matches!(self.version, … 4)` / `match self.version`
  arm that applies to v4 must also take 5 (about 12 sites). The SP block becomes "v4 computes,
  v5 skips the rows" (`self.idx += 123`, with a `debug_assert`).
- `libriichi/src/state/test.rs`: the version-dependent assertions.
- `mortal/model.py`: `Brain` (`case 3 | 4`, `case 2 | 3 | 4`) and `DQN` (`case 4`,
  `if self.version == 4`) must treat 5 as 4.
- `mortal/engine.py:63`: `case 2 | 3 | 4`.
- Anything else that branches on version (`player.py` passes it through; check
  `dataset/invisible.rs` and `dataset/gameplay.rs` too).

**Invariants:**
- v4 output bit-identical to before the change. Record the hash **before** touching the encoder (M0).
- Arena and training share the encoder, so they stay hash-identical (CLAUDE.md rule 2) by
  construction; v4 itself is not modified.
- v5's mask equals v4's mask (masks don't depend on SP).

## 5. Plan

Milestones in order; each code change is a small, single-purpose commit.

### M0: Research and baseline (≤ 1–2 h)

- **Golden hash:** before editing anything, hash `encode_obs(4, False)` and `encode_obs(4, True)`
  (kan-select decisions) over ~20 arena logs and a few Tenhou logs; keep the script and the hash.
- **Look for prior evidence** (timeboxed): Mortal's docs, GitHub issues and discussions, and the
  v4 commit for any v3-vs-v4 or SP ablation; nekobean's mahjong-cpp docs for what the tables
  are meant to capture. Note what you find (or that there is nothing) in the write-up.

### M1: Version 5 + tests

- Implement §4. Rebuild and install the `.so` by rename (CLAUDE.md).
- **Tests:**
  - Rust (`state/test.rs`): for a few hands, v5 rows 0–888 equal v4's, rows 889–1011 are zero.
  - pytest (reuse the `arena_run` fixture in `tests/conftest.py`): at every decision of every seat,
    kan select included, `encode_obs(5)` equals `encode_obs(4)` with rows 889–1011 zeroed; masks equal.
  - The golden hash from M0 still matches for v4.
- Pre-commit checks (CLAUDE.md): fmt, clippy (baseline: 27 errors; add none), `cargo test`, `pytest tests/`.

### M2: Speed check (~1 h, no training)

- `bench_encode.py --versions 3 4 5`: v5 should cost about what v3 does.
- Arena: `bench_1v3.py` for 15 min with pilot A (v4), then 15 min with a **copy** of pilot A whose
  config says `version = 5` (`workdir/sp-ablation/pilot-a-as-v5.pth`; never edit `workdir/pilot/`).
  The copy plays somewhat differently (its SP inputs are zero), so game lengths shift slightly;
  report decisions per game alongside hanchan/h.
- `bench_featurize.py` for v5 (it may need a `--version` option): SL samples/s at 1 and 16 threads.
- Record the numbers. They feed the D12 reopen condition and the actor-throughput plan.

### M3: Train no-SP seed 1 (~1 day)

- `experiments/configs/sp-nosp-a.toml` = `pilot-qs-a.toml` with `version = 5` and paths under
  `workdir/sp-ablation/nosp-a/`. **Everything else identical**, including `num_workers = 2`,
  `file_batch_size = 20` and `max_steps = 158000`: worker count changes the file interleaving, so a
  different value breaks the matched pair.
- Expect it to be faster than the pilots' ~25 h, since featurization was their bottleneck; check
  samples/s in the first hour and peak RAM (`free -g`).
- Run it unattended with a chain script like `run_pilot_chain.sh`: wait on the trainer's PID, check
  the log for errors and the final step count, freeze the final checkpoint (`chmod a-w`, sha256).
- Diagnostic only: compare its BC loss curve with pilot A's (`experiments/pilot-quick-students.md`
  lists both pilots' values). Loss curves are not verdicts (CLAUDE.md rule 1).

### M4: T-NI pair 1

- First a smoke: a few sets of v5 vs v4 through `lab.arena.play_sets` with the benchmark key
  (`0x62656e6368`), not a seed family, to confirm mixed versions run.
- Then `python -m lab run --test T-NI --challenger workdir/sp-ablation/nosp-a/nosp-a-final.pth
  --champion workdir/pilot/qs-a/pilot-qs-a-final.pth --family B --description "SP ablation pair 1:
  v5 (no SP) seed 1 vs pilot A (v4) seed 1, matched recipe"`.
- Also save the style panel (`python -m lab table`), e.g. whether shanten-backward rates or
  riichi/call rates move.

### M5: Branch on the verdict

- **Fail** (no-SP is inferior): stop. One pair is enough to keep SP. Write up; mention cheaper SP
  variants as possible follow-ups (skip SP at call/pass decisions; shanten threshold 3 → 2; a
  compute cap), but don't start them.
- **Pass:** that is a checkpoint claim only. Go to M6.
- **Inconclusive at the cap:** report it with the estimate and CI; the user decides.

### M6: Seed 2 and T-NI pair 2 (only if M5 passed)

- `sp-nosp-b.toml` = `pilot-qs-b.toml` with `version = 5`, paths `workdir/sp-ablation/nosp-b/`.
- T-NI against pilot B, same form as M4.
- Two passes = **preliminary** evidence that SP can be dropped (`02` §6.1). Confirmation (fresh
  family-C nonces, reserved panel configurations) is not implemented in the lab CLI (families A/B
  only); say so, don't build it unasked.

### M7: Write-up

- `experiments/sp-ablation.md`: method, the M2 speed table, training runs (time, samples/s, peak
  RAM, final loss vs the pilots), verdicts with registry run ids, style-panel differences, and a
  recommendation.
- `STATUS.md`: a dated section and next steps.
- Report to the user. Adopting v5, or a production version without the rows, needs a
  decision-log entry first; that is the user's call. So are any follow-ups (D12 reopening, cheaper
  SP variants, Phase 1 teachers on the new observation).

## 6. Rules and gotchas

- **RAM (15 GB):** never run an arena or lab job while training. The pilots peaked at ~8.6 GB with
  2 workers. Run M2 before M3, and M4 after M3 finishes.
- **Waiting on long jobs:** wait on the PID, not on a success string (a crash hangs a string
  waiter). `pgrep -f` matches its own shell; use the `'[x]yz'` pattern trick.
- **Build:** `PYO3_PYTHON` and `LD_LIBRARY_PATH` as in CLAUDE.md; install the `.so` by rename,
  never `cp` over it; rebuild after every Rust change.
- **Frozen items:** v4, `lab/lab.toml`, `lab/seeds.toml`, the splits, the pilots and GRP-v2 are
  not to be changed. v5 is a new version, not an edit of v4.
- **Registry:** rows are `exploratory`; describe the pair in `--description`.
- **Evaluation mode:** verdict runs use AMP (D43); don't pass `--no-amp`.
- **Commits:** small and single-purpose; the user allows committing finished work without asking.
  Never push.
- **Tenhou data** stays on the workstation (D23). No rentals for this work.

## 7. Time

| Step | Estimate |
|---|---|
| M0–M1 | ½–1 day |
| M2 | ~1 h |
| M3 + M4 | ≤ 1 day training (probably less) + ~1 h T-NI |
| M6 (if needed) | the same again |
| M7 | ~1 h |

About 2 days if SP turns out to matter, about 3 if it doesn't.
