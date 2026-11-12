# STATUS

Where the project is against `docs/reference/00_ROADMAP.md`. Read it whole at the start of a session
and update it at the end: what changed, verified vs only attempted, the next step. Keep it near 150
lines. Older session entries are in `STATUS-archive.md`, verbatim: search it, don't read it through.

## Current position (2026-11-12)

**Live: D64's anchor-release pilot, candidate A (`release-a`)** on Vast, rented 2026-11-12 18:43 UTC
(`experiments/anchor-release.md`): learner 60391930 (machine 54271), actors 60391934 and 60391935,
$1.76/h, ≈ $84 for its 144 box-hours. `full` ≈ 2026-11-14 19:00 UTC, then its watcher downs it;
watchdog deadline 2026-11-14 22:43 UTC; stop $95. Launch log `workdir/fleet/release-a-launch.out`.
- **After a reboot or a WSL restart:** `bash experiments/release_launch.sh a --resume` (the watcher
  and the watchdog, each only if missing).
- **Candidate B (`release-b`)** launches at up to $2.10/h (the user, 2026-11-12: ≈ $100, ≈ $185 for
  both, stop $105). Launch log `workdir/fleet/release-b-launch.out`; resume as A, with `b`.

**Live: the student distillation `students-s1`** on Hyperstack VM 1154755 (RTX A6000, $0.507/h), rented
2026-11-11 10:03 UTC; training since 10:28 UTC at 4,580 samples/s. At 2026-11-12 06:00 UTC: step
156,000 of ≈ 1,142,600 (13.7%), ≈ $10 so far; it should end ≈ 2026-11-17 08:30 UTC (≈ 142 h, ≈ $72).
The watchdog (PID 1127, re-armed 2026-11-12 06:05) downs it at 2026-11-18 15:03 UTC (177 h, ≈ $89.7).
Driver PID 1122 (`experiments/secure_teacher.sh --attach`), log `workdir/secure/runs/students-s1.session.log`.
- **The workstation's driver and watchdog died** between 12:29 and 14:29 UTC on 11-11 (the last
  logged pull is 12:29; WSL restarted 23:03 UTC) and were restarted 11-12 06:05. The box trained on
  unsupervised meanwhile, with no loss.
- **After a reboot or `wsl --shutdown`:** `TRAINER=train_students.py setsid -f bash experiments/secure_teacher.sh students-s1 students-s1 --attach >> workdir/secure/runs/students-s1.session.log 2>&1 < /dev/null`.
- Validation at step 140,000 (200 in-window heldout_game logs, EMA, exploratory): top-1 b8c192 81.16%,
  b12c256 81.40%, b16c256 81.44%, the control 80.51%; the ensemble 81.61%.

**Done: D64's critic check** (`experiments/critic-ceiling.md`): K1 explains 13.7% of the kyoku's
GRP-Δu, above supervised critics on 100K games (8.4–9.0%); by the pre-registered reading the critic
isn't the lever. The audit's "1.2–1.8%" was of the H return, mostly unpredictable later kyoku.

Phase 0 is exited (D55). In Phase 1 the SL substrate is being built: teachers done, students
training. The early research has met G0's Select conditions; T1 waits on D64.

| Rung | State |
|---|---|
| L0, the lab is trustworthy | **Met** 2026-10-05: calibrated GSPRT (D44), σ_set 0.3277, exact A/A in true FP32 |
| L1, the baseline is deployed | Not yet: teachers trained, students not distilled, no ranked play |
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
- **Students (running):** b8c192, b12c256, b16c256 and the hard-label control, distilled online from
  both teachers in one stream, 2.34B presentations (D66, D67; `experiments/students-s1.md` has the
  pre-registered qualification). The workstation smoke run passed (1,525 samples/s → ≈ 137 h).

**RL (`05`)**
- P0, both P1 seeds and V lost to their initializations; the diagnosis found unanchored drift
  (`experiments/rl-diagnosis.md`). P1 + K2 (the KL anchor) passed G0's retry and confirmed: run
  `k2b`, $259.
- **The gain carries to other tables** (exploratory, `experiments/k2b-opponents.md`): +0.0177 at 3×
  pilot B, +0.0084 at 3× shanten-greedy.
- **It stopped growing at the fixed anchor:** the run's second half bought +0.0025, the 48 h soak
  +0.0016 (the soak itself passes its gate, `experiments/fleet-soak.md`).
- **D64: T1 and D31's curves wait** for a critic check (workstation, $0) and an anchor-release pilot
  (a rental). Drafts: `docs/design/t1-preconditions-draft.md`. **D65:** D31's curves run two RL seeds
  per size.
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
| Heads-quick (b8c192 v5 with heads) | `teacher/heads-quick/final.pth` | `ac9cc916…` |
| BC s1 / AWR s1, the trainer's file | `teacher/teacher-{bc,awr}-s1/final.pth` | `6762492e…` / `bcb2987f…` |
| BC s1 / AWR s1, **raw** weights for the lab | `teacher/teacher-{bc,awr}-s1/ema.pth` (misnamed) | `2f200d7c…` / `381ce65f…` |
| BC s1 / AWR s1, **EMA** weights for the lab | `teacher/teacher-{bc,awr}-s1/ema-from-state.pth` | `59d94636…` / `75e55855…` |

The teachers' training states (the only source of their EMA) are in
`workdir/secure/runs/teacher-{bc,awr}-s1/box/`. Lab seeds: `lab/seeds.toml`; ledger `lab/registry/`.

## Waiting on the user

1. **D64's pilot** (`docs/design/t1-preconditions-draft.md`): its base, size and candidates (≈ $175),
   after the critic check's reading.
2. *(Approved 2026-11-11, recorded so it isn't asked again:)* the eval box for the students'
   qualification when they exist (`experiments/students-s1.md`); the critic check (done, above); no rerun
   of AWR vs BC on the EMA weights (the user: skip).
3. *(spare)*
4. **Ranked riichi.dev play before π_SL.** It reverses the 2026-10-12 decision and `07` §1's smoke
   deploy, so it needs an `08` entry first. `origin/main` is far behind; a deployed version's source
   must be public before it plays.

## Last session: 2026-11-10, the audit's seven changes (the user: "settle all 7")

**Changed**
- **Trainer file order** (`mortal/teacher.py`, `train_teacher.py`): the file list is shuffled by the
  seed before the loader's workers split it (`[dataset] file_order`, default `'shuffled'`). The
  teachers' order is `'sorted'`, now written into their three configs. A state resumes only in the
  order it started in.
- **Validation:** `[dataset] val_start` keeps the trainer's sample inside the train window.
  `tools/eval/validate.py` scores any checkpoint on `heldout_game` or `heldout_time` on the
  workstation, with the trainer's own validation pass (`experiments/validation-samples.md`).
- **Checkpoint loading:** every load goes through `mortal/checkpoint.py` (`load_checkpoint`:
  weights-only, plus numpy scalars). A test fails on an unrestricted load anywhere in the code.
- **The EMA save bug, found on the way:** `final.pth`'s `ema_weights` were the raw weights (the save
  kept live tensors across the swap back). Fixed (`teacher.ema_weights`), and `export_ema.py` now
  rebuilds the EMA from a run's last `state.pth` and refuses a file whose EMA equals its raw weights.
- **Decisions** (`08`): D64 (T1 waits for the critic check and the anchor-release pilot), D65 (two RL
  seeds per size), D66 (the hard-label control student); D63a opened. `00`, `04` §6 and `05` §4–6
  amended to match; the distillation draft amended; `docs/design/t1-preconditions-draft.md` written.
- **This file:** cut from 760 lines to this; the rest is in `STATUS-archive.md`.

**Verified**
- `pytest tests/`: 191 pass, 5 skipped (10 new). No Rust change, so no cargo checks.
- The recovered EMA weights reproduce the trainer's own final EMA validation on the same 200 logs:
  BC s1 81.336% / NLL 0.4750 (trainer: 81.337% / 0.4750), AWR s1 80.989% / 0.4863 (80.989% / 0.4863).
- Every existing checkpoint loads as data: of 92 scanned, 87 hold only tensors and plain values and
  5 v1-era ones add numpy scalars, which the loader allows. The lab, the anchor and a trainer state
  were loaded through it.

**Not done**
- `mortal/train_students.py`, the control's code and `students-s1.toml` (they follow the user's
  choices on the draft).
- No lab run, no rental. The two teacher T-stds haven't been repeated on the EMA weights.
- `server.py` still takes `submit_param` and `drain` from any tunnel peer (the audit's finding 9).

**Next:** the user's choices above. Then the student trainer and its smoke run on the workstation.
