# STATUS

Where the project is against `docs/reference/00_ROADMAP.md`. Read it whole at the start of a session
and update it at the end: what changed, verified vs only attempted, the next step. Keep it near 150
lines. Older session entries are in `STATUS-archive.md`, verbatim: search it, don't read it through.

## Current position (2026-11-18)

**Done: D64's anchor-release pilot** (`experiments/anchor-release.md`, exploratory, $173.62). **Both
candidates beat the fixed anchor's continuation at both tables, so T1 re-anchors to each new
champion** (the tie rule: A and B are level). Paired against `soak-a` full on 40K deals:

| | 3× pilot A (training table) | 3× pilot B (held out) |
|---|---|---|
| A, rising target | +0.0121 [+0.0076, +0.0164] | +0.0044 [+0.0000, +0.0087] |
| B, re-anchor | +0.0109 [+0.0066, +0.0150] | +0.0058 [+0.0014, +0.0101] |

- ≈ 60% of `k2b`'s whole gain again in 144 box-hours, where the soak's 192 bought nothing. Smaller at
  the held-out table (36% and 53% of the training table's), and A's pass there is on the line.
- Nothing is rented for it; the frozen snapshots are below.
- **The stage-timing check is done** (the note's last section): reading 1, **24 h stages**, but
  borderline. B's gain over the soak: +0.0061 at 12 h, +0.0069 at 24 h, +0.0109 at 48 h; the second
  24 h, +0.0040 [−0.0003, +0.0082], misses significance by 0.0003.

**Done: the student distillation `students-s1`** (`experiments/students-s1.md`): finished at step
1,142,584 (2.34B presentations) on 2026-11-17 09:14 UTC; pulled, wiped and destroyed by its driver at
09:25 UTC after 143.4 h, **≈ $72.65**. The four students' EMA and raw weights are in
`workdir/secure/runs/students-s1/box/students-s1/` (read-only; EMA ≠ raw checked for all four).
**Done: their qualification** (2026-11-17, `experiments/students-s1.md`; eval box ≈ $2.66, down):
**b12c256 and b16c256 qualify; b8c192 doesn't.**

| Student | T-NI vs BC s1 (δ 0.01) | Golden set (≤ 1 pp below BC s1) |
|---|---|---|
| b8c192 | **fail**, 12,500 sets, −0.0074 | pass |
| b12c256 | **pass**, 5,000 sets, +0.0047 | pass |
| b16c256 | **pass**, 10,000 sets, −0.0021 | pass |

- The control (D66): distillation beats it at b8c192 (T-std pass, +0.0075); it beats No-SP A by
  +0.0140; BC s1's EMA weights beat No-SP A by +0.0226.
- **Next by the plan:** b12c256 and b16c256 go to D31's curves (D65, D68), with T1's
  re-anchoring. The code they need is done and smoke-tested (2026-11-18, below). What's left is
  the controller's hold (the user), the stage checks and the scripts, then the rental.

Phase 0 is exited (D55). In Phase 1 the SL substrate is built: teachers done, two students
qualified. The early research has met G0's Select conditions; T1 and D31's curves run on
re-anchoring (D64, D68).

| Rung | State |
|---|---|
| L0, the lab is trustworthy | **Met** 2026-10-05: calibrated GSPRT (D44), σ_set 0.3277, exact A/A in true FP32 |
| L1, the baseline is deployed | Not yet: students distilled and two qualified (b12c256, b16c256); no ranked play |
| L2, RL beats its initialization | **Met** 2026-10-26: `k2b` full vs pilot A +0.0195 [+0.0174, +0.0216], 100K sets, confirmed |

**SL substrate (`04`)**
- **Data:** dataset-v1 is frozen (D45) and the splits are hashed. The encrypted bundle
  `45b6ffa43e2ceadd` is in the bucket `mortal-sl-0d00ec51`; its local shards were deleted on
  2026-11-10 (`manifest.json` kept, which `secure.py ship` reads).
- **Teachers:** b24c512 v5 on Hyperstack A6000s (D62), $123.23 for both. Validation top-1, EMA
  weights: BC s1 81.34%, AWR s1 80.99%.
  - AWR vs BC is even at T-std, so BC (D17). The ensemble is the two (D63).
  - BC s1 beats the quick student No-SP A: T-std pass, +0.0219 [+0.012, +0.031].
  - **Both T-stds played the raw weights, not the EMA weights** (a save bug, fixed 2026-11-10;
    `experiments/teacher-ema-export.md`). D67: the teachers are the real EMA weights from now on.
- **Students:** four distilled online from both teachers in one stream (D66, D67); two qualify
  (above, `experiments/students-s1.md`).

**RL (`05`)**
- P0, both P1 seeds and V lost to their initializations; the diagnosis found unanchored drift
  (`experiments/rl-diagnosis.md`). P1 + K2 (the KL anchor) passed G0's retry and confirmed: run
  `k2b`, $259.
- **The gain carries to other tables** (exploratory, `experiments/k2b-opponents.md`): +0.0177 at 3×
  pilot B, +0.0084 at 3× shanten-greedy.
- **It stopped growing at the fixed anchor:** the run's second half bought +0.0025, the 48 h soak
  +0.0016 (the soak itself passes its gate, `experiments/fleet-soak.md`).
- **D64 is satisfied:** the critic isn't the lever (`experiments/critic-ceiling.md`: K1 explains
  13.7% of the kyoku's GRP-Δu, supervised critics 8.4–9.0%), and re-anchoring beats the fixed
  anchor (above). D31's curves: two seeds per size, a third when close, 24 h stages (D65, D68).
- **Early-research spend:** ≈ $1.41K of the $2.2K cap with this pilot ($173.62), which leaves ≈ $785
  for D65's curves (≈ $630–690).
- The observation is v5 from G0 on (D61); its checks and its recipe check pass
  (`experiments/v5-switch.md`).

**Search (`06` §0):** the pilot fails on recovery (0.157 on pilot A, 0.055 on `k2b`, against 0.5).
Parked (the user, 2026-10-21).

**riichi.dev (`07`):** adapter, supervisor and daily summary built. The validation game and the
100K-game parity test pass. Ranked play waits for π_SL (the user, 2026-10-12).

**Spend:** ≈ $1.45K of ≈ $14.3K, summed from the notes' figures; there is no ledger total.

## Frozen artifacts (read-only; never overwrite)

| What | Path under `workdir/` | sha256 |
|---|---|---|
| Pilot A / B (b8c192 BC, v4) | `pilot/qs-{a,b}/pilot-qs-{a,b}-final.pth` | `15121432…` / `b267c026…` |
| No-SP A / B (the v5 quick students) | `sp-ablation/nosp-{a,b}/nosp-{a,b}-final.pth` | `498aabc8…` / `3180f2f0…` |
| CQL A / B | `cql/qs-{a,b}/cql-qs-{a,b}-final.pth` | `f8a867da…` / `c0ea3b96…` |
| GRP-v2 | `grp-v2/grp-v2-frozen.pth` | `db28beac…` |
| `k2b` full / p50 (P1 + K2, v4) | `fleet/runs/k2b/snapshots/{full,p50}.pth` | `f5a22c5c…` / `7d5937a7…` |
| `soak-a` full; `k2v5` full (v5) | `fleet/runs/{soak-a,k2v5}/snapshots/full.pth` | `9c0b7b34…`; `0510c68c…` |
| `release-a` / `release-b` full (the pilot, v4) | `fleet/runs/release-{a,b}/snapshots/full.pth` | `a57f195f…` / `be94d8e8…` |
| Heads-quick (b8c192 v5 with heads) | `teacher/heads-quick/final.pth` | `ac9cc916…` |
| BC s1 / AWR s1, the trainer's file | `teacher/teacher-{bc,awr}-s1/final.pth` | `6762492e…` / `bcb2987f…` |
| BC s1 / AWR s1, **raw** weights for the lab | `teacher/teacher-{bc,awr}-s1/ema.pth` (misnamed) | `2f200d7c…` / `381ce65f…` |
| BC s1 / AWR s1, **EMA** weights for the lab | `teacher/teacher-{bc,awr}-s1/ema-from-state.pth` | `59d94636…` / `75e55855…` |

The teachers' training states (the only source of their EMA) are in
`workdir/secure/runs/teacher-{bc,awr}-s1/box/`. Lab seeds: `lab/seeds.toml`; ledger `lab/registry/`.

## Waiting on the user

0. **The disk:** after the 2026-11-17 deletion of 19 finished lab runs' game logs (90.7 GB; each
   keeps its registry row, `games.npz` and `panel.json`), C: had 12 GB free. On 2026-11-18 it had
   57 GB, so the vhdx compaction looks done. If C: runs low again: `ext4.vhdx` isn't sparse, and
   `fstrim` doesn't help with `discard` on, so only a compaction returns space (CLAUDE.md, Machine).
1. *(Approved, recorded so it isn't asked again:)* the eval box for the students' qualification
   when they exist (2026-11-11, `experiments/students-s1.md`); the eval box for the pilot's measures
   (2026-11-12); no rerun of AWR vs BC on the EMA weights (the user: skip).
2. **Ranked riichi.dev play before π_SL.** It reverses the 2026-10-12 decision and `07` §1's smoke
   deploy, so it needs an `08` entry first. `origin/main` is far behind; a deployed version's source
   must be public before it plays.
3. **`CLAUDE.md`'s Machine section** still says how space returns to Windows (fstrim, shutdown,
   compact). The pending compaction is gone from this file (the user, 2026-11-12); that sentence
   describes the machine and was left.

## Last session: 2026-11-18, the curves' prerequisites (`experiments/size-curves.md`)

State at the start: clean tree at `6a61eff`, nothing rented, no watchers needed, `.so` current,
C: at 57 GB free.

**Changed** (`764dbc1`..; nothing rented): the note's items 1–3 and the configs, each with tests
(`experiments/size-curves.md`, "Before the first rental", has the detail).
- **In-run re-anchoring:** `fleet.py reanchor` / `watch --reanchor` → the trainer's `reanchor.json`,
  picked up at its next save with the coefficient carried; the anchor's weights live in the
  trainer's state, and the watcher confirms each pickup from `reanchor.jsonl`.
- **`server.py`** (the audit's finding 9): `submit_param` and `drain` need a per-start token only
  the learner's user can read.
- **Price limit:** `up --max-dph` (fleet and eval box) holds every offer a run takes, retries and
  replacements included.
- `stage_init.py --from-policy` (the B0s' inits); `curves-{b12,b16}-s{1,2,3}.toml`; the two
  students in `frozen.sha256`.

**Verified**
- `env -u MORTAL_CFG pytest tests/`: 219 passed, 5 skipped (14 new). No Rust change.
- **The re-anchoring smoke passed** on the workstation (`experiments/curves_smoke.py`, b12c256):
  - pickup at the next save, the coefficient carried, the KL to the new anchor 0.0070 → 0.0009;
  - a restart with the anchor's file moved away resumed on it;
  - a bad hash was refused, and a second re-anchor went through;
  - a tokenless `submit_param` was denied.
- **v5 monitoring states** for both students (`workdir/diag/credit-b12c256`, `credit-b16c256`):
  15,635 and 15,331 decisions, entropies 0.5166 and 0.5177.
- Found on the way:
  - loopback port 5004 times out on this workstation (WSL mirrored networking), so the smoke
    uses 5099;
  - a run from a B0 snapshot keeps the B0's run clock (`k2v5` began at t = 226,800), so the
    curves' inits drop it.

**Not done**
- The token's check on a box (with the B0s).
- The stage-boundary checks (the stop check, the drift limits) and the launch, B0 and measure
  scripts.
- The rental itself: the B0s, then the two batches, need the user's yes on a live quote.

**Waiting on the user:** the controller's hold after each re-anchor (`experiments/size-curves.md`,
"Open"). With `k2.toml`'s 5 windows, the coefficient hits its floor ≈ 10–15 min into every stage, as
in every run so far, so D68's start of 0.11 and its carry change nothing. Proposed:
`hold_windows = 100`.

**Next:** the user's answer on the hold, then the stage checks and the scripts. Then the quote for
the B0s and the first batch.
