# STATUS

Where the project is against `docs/reference/00_ROADMAP.md`. Read it whole at the start of a session
and update it at the end: what changed, verified vs only attempted, the next step. Keep it near 150
lines. Older session entries are in `STATUS-archive.md`, verbatim: search it, don't read it through.

## Current position (2026-11-12)

**Live: D64's anchor-release pilot** on Vast (`experiments/anchor-release.md`, pre-registered and
launched 2026-11-12): two fleets of a learner and 2 actors, each continuing `k2b` full for 144
box-hours. `full` ≈ 2026-11-14 19:00 UTC for both, then each watcher downs its run. ≈ $165 projected
for both (approved ≈ $175).

| Run | What | Boxes (learner; actors) | $/h | Watcher / watchdog PID | Deadline (UTC) | Stop |
|---|---|---|---|---|---|---|
| `release-a` | A: pilot A's anchor, target rising 0.02 → ≈ 0.06 | 60394006; 60391934, 60391935 | 1.63 | 107231 / 103391 | 11-14 22:43 | $95 |
| `release-b` | B: re-anchored to `k2b` full, 0.02 | 60394073; 60394075, 60397192 | 1.77 | 108414 / 105869 | 11-14 23:09 | $105 |

- **After a reboot or a WSL restart:** `bash experiments/release_launch.sh a --resume`, then the same
  with `b` (the watcher and the watchdog, each only if missing). Then the two chains below if they
  are gone; both skip what is already done.
- **Don't edit trainer or engine code while these run:** a replacement box gets the working tree as
  it is at that moment.
- **On the workstation, detached:**
  - `experiments/release_soak_chain.sh` (since 19:10 UTC, log `workdir/lab/release-soak.log`): the
    comparator, `soak-a` full at tables of pilot A and pilot B, 40K sets each, done ≈ 11-13 11:00 UTC.
    **No other lab or GPU job meanwhile.** Its registry rows are uncommitted until it ends.
  - `experiments/release_measure_chain.sh` (since 19:45 UTC, log
    `workdir/fleet/runs/eval/release-measures.log`): waits for both `full` snapshots, rents the eval
    box (approved, ≤ $0.75/h, ≈ $8–10), plays the candidates' four measures, pulls and downs it.
- **Then:** `python experiments/release_paired.py` gives the pre-registered reading. Also owed: the
  drift report at p25, p50 and `full` (`experiments/rl_drift.py`, in the note), which needs the GPU
  free of the lab.

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
- **D64: T1 and D31's curves wait** for the anchor-release pilot (running, above); the critic check
  is done. What the review changed in the pilot's draft: `docs/design/t1-preconditions-draft.md`.
  **D65:** D31's curves run two RL seeds per size.
- **Early-research spend:** ≈ $1.24K of the $2.2K cap by the fleet's own records; ≈ $1.41K after
  this pilot, which leaves ≈ $790 for D65's curves (≈ $630–690).
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

1. *(Approved, recorded so it isn't asked again:)* the eval box for the students' qualification
   when they exist (2026-11-11, `experiments/students-s1.md`); the eval box for the pilot's measures
   (2026-11-12); no rerun of AWR vs BC on the EMA weights (the user: skip).
2. **Ranked riichi.dev play before π_SL.** It reverses the 2026-10-12 decision and `07` §1's smoke
   deploy, so it needs an `08` entry first. `origin/main` is far behind; a deployed version's source
   must be public before it plays.
3. **`CLAUDE.md`'s Machine section** still says how space returns to Windows (fstrim, shutdown,
   compact). The pending compaction is gone from this file (the user, 2026-11-12); that sentence
   describes the machine and was left.

## Last session: 2026-11-12, the pilot's draft reviewed, pre-registered and launched

**Changed**
- **The review of `docs/design/t1-preconditions-draft.md`** found the pilot not ready as drafted: the
  candidates' room wasn't matched, the pass rule compared with `k2b` full, not the soak, and a new
  run clock would have restarted the LR, clip and entropy schedules. The user took the review's
  picks ("Go with your picks, I give the okay for the box rental as well").
- **Pre-registered** (`experiments/anchor-release.md`, D64's note in `08`, `05` §5.1's pointer) and
  the draft rewritten to the decided design.
- **Code:** `tools/fleet/stage_init.py` (a snapshot without its run clock, and with `--new-anchor`
  without its anchor state); `fleet.py` ships the anchor's file to the learner;
  `release-{a,b}.toml` and their smoke configs; `experiments/release_launch.sh`,
  `release_soak_chain.sh`, `release_measure_chain.sh`, `release_paired.py`.
- **Launched** both fleets (above). B's live quote passed the $1.90/h limit, the user chose "Now, up
  to $2.10/h", and it rented at $1.77/h.
- **The disk compaction** is no longer waiting on the user (the user: not needed).

**Verified**
- `pytest tests/`: 202 pass, 5 skipped (4 new). No Rust change, so no cargo checks.
- Both configs smoke-ran on the workstation before the rental, and both learners logged the right
  anchor, coefficient and clock on their boxes. A at t = 4,800: KL 0.0195, agreement 95.2%.

**A correction to the review:** it said candidate A as drafted would end near 0.027 nats. `k2b`'s
own start and A's smoke run show that once a cut starts the coefficient falls to its floor within
minutes and the KL climbs to the band's upper edge. A is expected near 0.05 nats and B near 0.027
from `k2b`; the note says so, and the user was told.

**Not done**
- The drift report at each snapshot, the candidates' measures and the reading (they follow the
  runs). No strength claim yet.
- `server.py` still takes `submit_param` and `drain` from any tunnel peer (the audit's finding 9).

**Next:** read `workdir/lab/release-soak.log` and commit its registry rows (≈ 11-13 11:00 UTC); the
drift report at p25 (≈ 11-13 07:00 UTC) and p50; after ≈ 11-14 19:00 UTC the eval-box chain, then
`release_paired.py` and the result in the note. The students end ≈ 11-17.
