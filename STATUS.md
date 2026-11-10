# STATUS

Running log of where the project is against `docs/reference/00_ROADMAP.md`. Newest first.
Update at the end of every session: what changed, verified vs only attempted, next step.

## 2026-11-09 (later) — the audit's three checks (the user brought them back)

State at the start: clean tree, nothing rented, `.so` current, splits frozen, nothing running.

- **Check 1, running on the workstation:** the `k2b` opponent check, pre-registered in
  `experiments/k2b-opponents.md` (`dfa65a8`) before any run. Three 20K-set family-A measures, one
  at a time: `k2b` full vs 3× shanten-greedy, pilot A vs 3× shanten-greedy (new block `floor`,
  nonces [54200, +20000)), `k2b` full vs 3× pilot B (block `aa-prime`, the A/A′ run's deals). Each
  table is read as a pair on the same deals, `k2b` − pilot A; the pilot A vs pilot B half is the
  existing `20261005-124255-measure`. The pilot A floor run is an addition to the audit's two.
  - Started 2026-11-10 05:19 UTC: `experiments/k2b_opponents_chain.sh` (detached, pid 83705), log
    `workdir/lab/k2b-opponents.log`. ≈ 2 h + 2 h + 3.5 h, so ≈ 13:00 UTC. It ends with
    `experiments/k2b_opponents_paired.py` → `workdir/lab/k2b-opponents-paired.json`.
  - Verified before the start: a 100-set smoke run on a dev key (not a result) played `k2b` against
    the rule-based bot cleanly, ≈ 10K sets/h. **No result yet.**
  - **Paused by the user at 05:43 UTC** (`kill -STOP` on the chain 83705 and its measure 83710, at
    5,500 of the first run's 20,000 sets). Resume with `kill -CONT 83710 83705`. The stopped measure
    keeps ≈ 2 GB of RAM and its GPU memory. A reboot loses it: restart the chain then (it skips
    finished measures; the interrupted one starts over on the same block, with no registry row).
  - **No lab or training job on the workstation until it ends** (RAM).
- **Check 2, drafted, not run:** BC s1 (EMA) vs No-SP A, T-std (`experiments/teacher-vs-quick.md`).
  The eval box needs the user's yes (≈ $0.50) and can't overlap the chain (the lab lease), so it
  comes after it. The same test runs on the workstation for free, slower.
- **Check 3, the user's call:** whether ranked riichi.dev play starts before π_SL. It reverses the
  2026-10-12 decision ("no pilot goes on the board first") and `07` §1's smoke deploy, so it needs
  an `08` entry first. Not decided; `origin/main` is 133 commits behind, and a deployed version's
  source must be public before it plays.

- **Disk (the user, 2026-11-10): C: was down to 2.2 GB free** of 931 GB. The WSL disk file was 245 GB
  of it (Linux used 242 GB); Windows itself holds ≈ 684 GB. That is why the chain is paused.
  - **Removed at the user's word (verified, Linux 242 → 210 GB used):** the pip, npm and cargo
    `target/` caches (the next Rust build is a full rebuild); `data/raw` (12 GB, the v1-era
    2020–2024 logs and their zips; `data/stats` kept); the local encrypted bundle's 65 `.gpg` files
    (9.3 GB). Before that, every bundle file was matched with the bucket copy by size and multipart
    MD5, and with the manifest by sha256. `manifest.json` stays, since `secure.py ship` reads it;
    the key is in `~/.config/mortal-secure/`. A new `bucket.py push` would need `pack.py` to
    rebuild the files.
  - **Not removed:** the Playwright browsers in `~/.cache` (1.1 GB; installed browsers the MCP
    server uses, not a download cache), the `mahjax` conda env (5.7 GB), the lab game logs (89 GB,
    the user's decision).
  - **The space isn't back on C: yet:** that needs `sudo fstrim -av`, `wsl --shutdown` and a
    compact of the disk file, which ends the paused run. C: had 23 GB free at 09:10 UTC from the
    user's own cleanup on Windows.
  - **Guard added:** `lab/diskguard.py` (6 tests; `pytest tests/` 181 pass, 5 skipped). A lab run
    refuses to start, before reserving nonces, unless C: and the lab's filesystem keep 10 GB free
    after its estimated logs; a running one waits before a chunk under 5 GB (`LAB_MIN_FREE_GB`).
    The chain's remaining two measures pick it up; the paused first one predates it. Cargo checks
    not run (no Rust change).

**Next:** resume the chain at the user's word (`kill -CONT 83710 83705`; ≈ 5.5 GB of logs to come);
read its result into `experiments/k2b-opponents.md`; then check 2 by the venue the user picks.

## 2026-11-09 — repository audit (read-only session; findings not yet reviewed by the user)

Nothing was trained, rented or measured in the lab; no code changed. State at the start: clean tree,
nothing rented, `.so` current, splits frozen.

**Verified:** `pytest tests/` 175 pass, 5 skipped; `cargo test` passes; `cargo fmt` clean; clippy at 27
errors (CLAUDE.md says 28). The AWR teacher's final weights come from one uninterrupted trainer (the
duplicate's only resume is its own start at step 57,015).

**Findings from the logs and code (evidence in the session's report; each is a proposal, not a decision):**
- `k2b`'s gain is measured only against pilot A, the opponent it trained against; no registry row
  plays it against pilot B, shanten-greedy or a mixed table (`02` §4.2's panel was not run at G0).
- `k2b` sat at the anchor's equilibrium from its first tenth (coefficient fixed at 0.0759, KL easing
  0.025 → 0.019); its second half bought +0.0025, the soak +0.0016 (not distinguishable from 0).
- The K1 critic explains 1.2–1.8% of the return's variance (6–10% in the last kyoku) and its loss never
  fell; the SL kyoku-Δ head reaches ≈ 10% from public information alone.
- BC s1 (42.7M parameters, 2.34B presentations) is +0.9 pp top-1 over the b8c192 heads-quick student
  (81.34% vs 80.46%); no lab row compares a teacher with a quick student.
- `train_teacher.py` splits the date-sorted file list between workers, so every batch is one ≈ 2.6-year
  era, and the last ≈ 9.5K steps had no 2022-11 → 2025-06 games (that worker ran out first; the loss
  steps 0.469 → 0.477 there). The 200 validation games include 47 from 2009–2014, outside the train window.
- Checkpoints written on Vast learner boxes are loaded on the workstation with `weights_only=False`
  (`lab/arena.py`, `mortal/kl_anchor.py`, `mortal/player.py`).

**Next (the user's call):** `k2b` full vs pilot B and vs shanten-greedy (workstation); BC s1 vs No-SP A
(eval box); whether ranked riichi.dev play starts before π_SL; a seeded shuffle before the worker split
and a hard-label control student before the distillation run.

## Current position (2026-10-28 14:45 UTC)

**Phase 0 is exited** (D55, the user, 2026-10-11; the bar was met 2026-10-05). **Phase 1 early
research is under way.**

- **The fleet soak passes** (2026-10-28 07:46 UTC, `soak-a`, `experiments/fleet-soak.md`): 47.8 h from
  `k2b` full, learner + 3 actors (no eval box, the user's choice), $125.26. Queue bounded, lag stable
  (D60; p95 12.3, ratio p99 ≤ 1.145), 100% drained, actors blocked 1%, no box losses. The learner was
  supply-bound at ≈ 4.9K samples/s (benchmark 7.5K). Tracking vs `k2b` full: p25 +0.0026, p50
  −0.0036, `full` +0.0016 [−0.0050, +0.0082]: no measurable change. The T1 soak gate (`03` §3.7) is met.
- **The v5 recipe check passes** (2026-10-28 14:38 UTC, `k2v5` full, `experiments/v5-switch.md`):
  against No-SP A, reverse KL 0.014 (≤ 0.10), argmax agreement 95.7% (≥ 90%), Δu at argmax +0.0132 ±
  0.0052 on 4,000 diagnostic sets (CI above 0). `k2b` at p25 (v4): 0.025, 94.2%, +0.012. With the v5
  A/A and frozen hash, D61's checks are met: v5 results count. $81.94 (B0 $9.95 + stage 2 $71.99).
  Nothing is rented.
- **Next: the SL substrate** (the user, 2026-10-28, G0's option 1: teachers → students → the actor-size
  curves → π_SL → T1). First piece drafted for review: the secure data path (D23,
  `docs/design/secure-data-path-draft.md`). The training set is 7.9 GiB encrypted, decrypted on the
  secure box's disk (the user: fine), wiped after each run, with a ledger. b24c512 trains at 1.9K
  samples/s on the 3060 (≈ 14 days per teacher), so teachers rent. Providers compared at the user's
  request: about $65–140 per teacher on a Vast 4090, Hyperstack/Massed Compute A100 80GB ($1.35/h) or
  A6000-class (estimates). The user chose Hyperstack, with its own object storage for the bucket.
  - Hyperstack (2026-10-29): $55 credit; single-GPU VMs only in CANADA-1; on-demand A100/H100 sold
    out, A100 spot $1.08, L40 $1.00, A6000 $0.50 in stock.
  - Bucket `mortal-sl-0d00ec51`: private, presigned GETs work.
  - Encrypted bundle `45b6ffa43e2ceadd`: 1.85M logs, 9.32 GiB, uploaded.
  - Tools: `tools/secure/{hyperstack,bucket,pack}.py`.
  - The box tools are built and tested: `secure.py`, `unpack.py`, `bootstrap_vm.sh`,
    `bench_teacher.py`.
  - **The measurement session ran (2026-10-29, $1.74):** the data path worked end to end on three
    VMs (shipped, wiped, destroyed, ledgered). The real trainer: A6000 4,995 samples/s (130 h,
    **$66 per teacher**), L40 5,346 (122 h, $122). **D62 (the user, 2026-10-29): the teachers train on
    Hyperstack RTX A6000s**, resolving D17a.
  - **The teacher trainer is built** (`mortal/train_teacher.py`; opponent labels in libriichi; tests
    pass). The user confirmed the defaults (β 1.0, LR 1e-4, no quick sweeps, GN).
  - **The heads check passed** (2026-10-30, `experiments/teacher-heads-check.md`): the heads-on quick
    student vs No-SP A, T-NI pass at 5,000 sets, Δu +0.0082 (`20261030-075954-T-NI`). The teachers
    keep the heads attached.
  - **The teachers are training** (the user's yes, 2026-10-30): `teacher-awr-s1` (VM 1133199) and
    `teacher-bc-s1` (VM 1133202) on Hyperstack A6000s since ≈ 20:00 UTC. At ≈ 6K
    samples/s (GPU 100%), BC ends ≈ 2026-11-04 06:00 UTC. AWR ran 41 h at half speed beside a
    second trainer that the driver started by mistake (fixed, `e14f422`), so it ends ≈ 2026-11-05
    10:30 UTC, inside its 150 h watchdog.
  - **BC s1 finished** (2026-11-04 07:25 UTC, $54.72): 2.34B presentations, validation top-1 81.24% raw /
    81.34% EMA (`workdir/teacher/teacher-bc-s1/final.pth`). **AWR s1 finished** (2026-11-05 10:31 UTC,
    $68.51): validation top-1 80.99% EMA. Both VMs are destroyed; the teachers cost $123.23. **The AWR vs
    BC T-std fails** (2026-11-05, `20261105-105720-T-std`): Δu −0.0007 ≈ [−0.009, +0.008] at 6,500 sets,
    so **BC is the teacher objective** (D17 resolved). **D63 (the user): the ensemble is BC s1 + AWR s1**, no second
    seed; BC s1 is the best single teacher. The distillation draft is written (`docs/design/distillation-draft.md`):
    one A6000 stream, both teachers online, three students, ≈ 128 h, ≈ $65; it waits for the user's choices. Then T-std AWR vs BC on
    the workstation, and the winner's s2 (a yes).
- **G0's Select conditions are met** (`00` Gate G0): the arm (P1 + K2) passed T-std against its
  initialization on the retry's one seed (D59), L2 confirmed, the fleet soak passed. Left for the
  user: the actor size (D31) and the start of T1.

- **The observation is v5 from G0 on** (D61, the user, 2026-10-26): about even on strength, much cheaper.
  **The v5 checks pass** (15:04 UTC, `experiments/v5-switch.md`): v5 = v4 minus SP at 22,794 decisions,
  frozen v5 hash `c0ec1825…`, deterministic v5 A/A (0 divergent sets), the drift tool on v5. The
  recipe check is drafted and waits for the user's yes.
  Phase 1's results stay v4 evidence. Before a v5 result counts: a v5 A/A and frozen v5 hash (workstation),
  and a recipe check (k2-style from No-SP A to p25, ≈ $65, needs a yes). No-SP A/B are frozen inputs now.
- **Post-G0 queue done** (2026-10-26 15:00 UTC): the diagnosis on `full` (drift 0.019, 95% agreement;
  RL's gain in its own sampled play +0.067 ± 0.008 vs P0's +0.022; `experiments/k2-retry.md`), and the
  search pilot on `k2b`: belief +0.0005 (z 1.1), recovery 0.055 (`experiments/search-pilot.md`).
- **L2 is met: the G0 retry confirms** (2026-10-26 07:33 UTC, `20261025-060641-T-confirm`): `k2b` full vs
  pilot A, Δu +0.0195 [+0.0174, +0.0216], 100K sets on fresh family-C nonces. The first confirmed RL
  gain. Left before G0's Select: the fleet soak (a rental, the user), the actor size (D31), and the
  observation review (draft D61, `docs/design/d61-draft.md`, with the user).
- **The doubling estimate is grey** (2026-10-25 13:04 UTC, `20261024-122034-measure`): `k2b` full vs p50
  Δu +0.0025 [+0.0004, +0.0046], 100K sets. By its pre-registered reading, no v4 → v5 switch on this
  alone; the user decides at the G0 review (`docs/design/observation-review-d49.md`). L2's T-confirm
  started 13:06 UTC (`20261025-060641-T-confirm`, family C), ≈ 17.5 h.
- **Fleet-soak preparation** (2026-10-24/25, `experiments/fleet-soak.md`): D60 makes policy lag
  non-blocking while learning stays stable (the user). The learner check found `k2b`'s learner path
  bound by copies in its main process (≈ 4.6K samples/s against 6.0K for 3 actors). Two optimization
  rounds (`fa61520`, `4c30a83`; same tensors, identity tests) bring it to ≈ 7.5K on an EPYC 9474F.
  ≈ $1.40. The soak itself (a rental, ≈ $130) waits on the user.
- **The G0 retry passes** (2026-10-24 19:20 UTC, `20261024-110559-T-std`; `experiments/k2-retry.md`):
  `k2b` full vs pilot A, Δu +0.0114 (≈ [+0.003, +0.019]), T-std pass at 7,000 sets, LLR +3.88. The
  first online arm-seed above its initialization. `k2b` ended at 18:04 UTC ($259.09; nothing rented).
  Next: the doubling estimate (running, ≈ 15 h), then L2's T-confirm on family C and the fleet soak,
  both waiting on the user.
- **`k2b`'s p25 early check: continue** (2026-10-20 07:01 UTC; `experiments/k2-retry.md`). Reverse
  KL 0.025 (stop > 0.10), argmax agreement 94.2% (stop < 90%), Δu at argmax vs pilot A +0.012 ±
  0.0055 on 4,000 diagnostic sets: the first online checkpoint above its initialization.
  Exploratory; G0's T-std at `full` (≈ 2026-10-24 18:00 UTC) decides.
- **The search pilot ran on pilot A and fails on recovery** (2026-10-19 19:24 → 10-20 00:27 UTC,
  the user's go-ahead; `experiments/search-pilot.md` → "Pilot run on pilot A"). 20,000 riichi-vs-dama
  decisions from 1,200 games (family-A block `search-pilot`). Oracle +0.0112 u per decision (z
  12.7); belief +0.0018 (z 3.4); **recovery 0.157 [0.071, 0.24]** against D32's 0.5. p99 latency
  1.95 s on the 3060 (passes). By D32's rule the fix is in the belief; what to try is the user's
  call.
- **The workstation was off ≈ 06:50–19:15 UTC on 2026-10-19.** `k2b` trained on unsupervised; its
  watcher and watchdog were restarted at 19:16 UTC with `resume-supervision.sh`. KL ≈ 0.025 nats
  (target 0.02), 84K games/h, $45.50 at 75.9 box-hours.
- **D49 memo** (`docs/design/observation-review-d49.md`): P0's doubling estimate meets its bar,
  but it measures recovery from drift. The user approved a `k2b` full vs p50 measure (2026-10-21,
  pre-registered in `experiments/k2-retry.md`); the v4/v5 call waits for it.
- **v5's search speed** (2026-10-22, `docs/design/observation-review-d49.md`): batched rollouts 1.33×
  faster than v4 (1.46× per event), one-at-a-time search 1.6× at p50 and 2.2× at p99 (2.00 → 0.90 s,
  limit 2.5 s). Below the arena's 2.4–2.8×; for riichi vs dama the belief, not rollout count, binds.

- **P0 fails G0** (2026-10-17 04:08 UTC, `20261017-033700-T-std`): `p0-a` full vs pilot A, Δu
  −0.0268 (≈ [−0.038, −0.016]), stopped at 5,000 sets with LLR −10.57. Per 05 §2.3.6 a failing
  arm goes to diagnosis (§8) first; it isn't evidence against the algorithm family (D28).
  `experiments/p0.md` → "G0 result".
- **`p1-b` fails G0** (04:39 UTC, `20261017-040806-T-std`): full vs pilot B, Δu −0.0131 (≈ [−0.024,
  −0.003]), stopped at 5,000 sets with LLR −6.17. So P1 can't pass §2.3 step 1, whatever `p1-a`
  does (`experiments/p1.md` → "G0 results"). P0's doubling measure runs next (04:39 UTC, ≈ 10 h).
- **07's 100K-game parity test passes** (2026-10-18): 20/20 chunks, 100,000 games, 63.7M requests, 0
  mismatches, fallbacks or adapter failures (`experiments/riichidev-adapter.md`).
- **Eval box 56337703 destroyed** (2026-10-18 17:03 UTC). Its queue had drained, and its registry
  matched the workstation's (27 run rows, 29 nonce rows). The lab lease is released. The full
  `evalbox.py down` couldn't run over the 60%-loss home link, so the parity reports and `labq/`
  were copied one file at a time.
- **The home internet link** loses 50–90% of packets (to 1.1.1.1; 0% to the router) since
  ≈ 11:20 UTC on 2026-10-18. Uploads run ≈ 200 KB/s and drop. The fleet's rsync now resumes dropped
  transfers (`--partial-dir`, 12 tries for box syncs), and the watcher won't judge boxes
  unreachable while its own link is down.
- **`p1-a` fails G0** (2026-10-18 05:26 UTC, `20261018-044407-T-std`): Δu −0.0209 (≈ [−0.032, −0.010]),
  5,000 sets. Every online arm-seed has now failed. P1 (mean −0.017) beats P0 (−0.0268), so the G0
  retry with K2 runs P1 seed 1 (D59).
- **K2** (`experiments/k2-retry.md`, the user's go-ahead):
  - Built: the KL anchor, the LR/clip/entropy schedules on a saved run clock, and the automated p25
    early check (`fleet.py watch --early-check`, `--anchor-alert`). Tuning (2 × 4 h, $14.12) chose the
    adaptive anchor: 100% of KL readings in band, against 95% for the fixed rule.
  - Box setup took 40 min on 256-thread hosts (pytest threads), fixed in 675c3db. That also explains
    p1-a's lost actor.
  - The retry: approved (≈ $300, stop $320). Its first launch, `k2`, was downed at 12:39 UTC on 10-18 by
    the bad home link ($2.30). **Relaunched as `k2b` at 17:59 UTC on 2026-10-18**, automatically once the
    link recovered and the quote was $1.80/h (≈ $259). p25 early check ≈ 2026-10-20 06:00 UTC; full
    ≈ 2026-10-24 18:00 UTC; watchdog 2026-10-25 05:59 UTC.
- **P0's doubling estimate** (16:03 UTC, `20261017-043916-measure`): full vs p50 Δu +0.0071 [+0.0047,
  +0.0093], 100K sets. By the pre-registered reading the CI's lower bound > 0.004, so the G0 review
  weighs a v5 switch. Caveat from the diagnosis: both checkpoints are below the init, so part of it
  is likely recovery from drift (`experiments/p0.md`).
- **Diagnosis of the online arms** (the user, 2026-10-17; `experiments/rl-diagnosis.md`): the credit
  attaches correctly. Training-game gains are less than sharpening alone: at matched entropy, RL's
  other changes cost 0.040 ± 0.008 (P0) and 0.018 ± 0.008 (P1). Every run moved 20–25% of its argmax
  decisions from the SL policy by p25, with reverse KL ≈ 0.4 nats, ≈ 20× K2's anchor target. P1's
  entropy collapsed (05 §8.1). The roadmap's branch for this: one bounded G0 retry of the best arm
  with K2, 2 seeds, ≤ $0.6K from the reserve.
- **The eval box was lost with unpulled results** (2026-10-17 00:47 UTC).
  - Instance 56069526's host contract ended (`end_date`), and Vast stopped it.
  - `vastai start` is queued ("resources unavailable"), and `vastai copy` off it fails.
  - On it: P0's G0 T-std and doubling measure, and the 100K parity rerun (r2, chunks c01–c12 at
    least). I hadn't pulled it in its 40 h up.
  - Vast restarted it on 2026-10-18 at 16:01 UTC with an empty disk: the results are gone. It was
    destroyed at 16:10 (`retired` in `workdir/fleet/runs/eval/fleet.json`).
  - Rule for which result counts (`experiments/p0.md`): the rerun decides once it starts. The lost
    family-B ranges are burned in the ledger (`b8f27d2`).
  - Fix (`e0281e5`): offers whose contract ends within 7 days are skipped, and `evalbox.py status`
    shows the contract end. The running watchers still have the old code.
- **New eval box** (the user, 2026-10-17: rent one if the old one isn't back when `p1-b` ends):
  instance 56337703, EPYC 7C13 (32 cores), $0.639/h, rented 03:23 UTC, holding the lab lease.
  - Queue: P0's G0 T-std (`full` vs pilot A), `p1-b`'s G0 T-std (`full` vs pilot B), then P0's
    doubling measure.
  - ≤ 10 h each at ≈ 40K hanchan/h.
  - **The 100K parity test runs again (r3)**, the user's yes (2026-10-17): 20 chunks, queued at
    04:53 UTC at priority 9 behind the doubling measure (`workdir/riichidev/submit-100k-r3.sh`,
    reports `parity-100k-r3-cNN.json`). That's ≈ 1 h per chunk, from ≈ 14:40 UTC. The adapter
    suite passes in the box's riichidev venv (14 tests). `p1-a`'s T-std (priority 0) goes ahead of
    the chunks still waiting when it's submitted.
- **`p1-b` ended** at 03:21 UTC on 2026-10-17: 432.26 box-hours, step 4,474,000, $239.89. `full.pth` is
  pulled and read-only, and the boxes are destroyed.
- **`p1-a` lost actor1 at 11:04 UTC on 2026-10-16.** Three replacements failed (2 setups over
  40 min, 1 GPU error), and the watcher gave up.
  - Replaced by hand at the user's go-ahead: EPYC 7713, $0.494/h, joined 02:03 UTC on 10-17.
  - Now learner-bound at ≈ 80K games/h: the replacement learner is a Zen 3 7B13, and the replay
    queue is full.
  - `full` ≈ **2026-10-18 04:00 UTC** at three boxes, ≈ $264 in all (cap $275).
  - P0 + P1 ≈ $722, against ≈ $705 approved. The user approved the overrun with these
    replacements.

- **`p1-a` (P1 seed 1) resumed 2026-10-14 20:22 UTC** (p50 snapshot taken 23:07 UTC, step 2,907,600) after ≈ 20 h idle (`experiments/p1.md` →
  "Incident"). Its learner died at 23:47 UTC on Oct 13; the replacement's checkpoint copy failed,
  and the health check reported the never-started learner as `ok` for 19 h.
  - **Fixed** (`9e0e989`): failed bring-ups are reported and retried, checkpoint first.
    `fleet.py exclude` and `deadline --extend-hours` were added. The flaky setup test that
    failed the first new learner was fixed in `5ceefe6`.
  - **Resumed** from the workstation's mirror (step 2,807,600). The idle 59.08 box-hours are left
    out of the budget (206.1 counted at the resume). The hard deadline moved 21 h, to **2026-10-18
    12:23 UTC**, and `full` is expected ≈ 2026-10-17 23:00 UTC.
  - **Spend (approved by the user, 2026-10-14):** `p1-a`'s safety stop raised to **$275** (≈ $257
    projected). P0 + P1 come to ≈ $705, against ≈ $640 approved on 2026-10-11.
- **The workstation rebooted at ≈ 19:00 UTC on 2026-10-14.** All three runs' watchers and
  watchdogs are back, every watcher on the fixed code (`p0-a` and `p1-b` restarted at 19:49 UTC
  at the user's go-ahead).
- **Arm V stopped by its pre-registered stop rule** (2026-10-14; `experiments/v-tuning.md` →
  "Results"). The sweep's 4 measures (2026-10-13 18:01 – 2026-10-14 01:16 UTC, workstation) put
  every online checkpoint far below the CQL model it started from: e1 −0.729 u (CI entirely below
  0), e0 extended −0.874, e2 −0.922, e1-lr −1.168. The decision rule keeps e1; the stop rule
  fires, so V's runs don't launch. `v_qgaps.py`: **the Q function collapsed in every run** (all
  legal actions within ≈ 0.5 u, median best–second gap 0.013 u against A's 1.81; argmax = human
  21–38% against 81%). **The interface checks are clean** (2026-10-14,
  `experiments/v_online_check.py`): actions align in the 1v3 path, and the targets have the same
  scale as on Tenhou. Leading hypothesis: online drops the CQL term, and with the dueling head
  each correction of the overestimated argmax raises the other actions' Q. **The replay test
  confirms it** (2026-10-14, `experiments/v_replay_test.py`; A fine-tuned on its own greedy play
  with the online step, no exploration): without CQL the median gap falls 1.81 → 0.034 u in 10K
  steps; with CQL 5 it widens to 8.8 u and the policy stays pinned to A. V needs an online step
  between the two (a small CQL weight, Cal-QL or mixed offline batches: a design change, so `08`
  first). **The replay sweep over the weight** (2026-10-14 11:34–13:04 UTC): 0.1, 0.3 and 1
  all widen the gap (4.3 / 5.6 / 6.9 u at 10K steps, still growing) and pin the policy to A
  (≈ 97%). No fixed weight holds it on greedy data: the untried actions have no target, so the
  weight sets the speed of the push-down, not an equilibrium. **Cal-QL and weight 0.03**
  (2026-10-14 19:34–20:35 UTC): Cal-QL levels the gap at ≈ 5.5 u but inflates the taken move
  2.3 u above its return (½·MSE 2.15), so it fails as written. **0.03 comes closest:** A's argmax
  96.5%, near-ties and exploration rate as A's (1.4% vs 1.5%), but the median gap still drifts
  1.81 → 2.94 u in 10K steps. **60K steps at 0.03** (2026-10-14 22:59 – 10-15 01:59 UTC, at the
  user's go-ahead): the median gap slows but doesn't level off (1.81 → 4.01 u; +0.15 u per 10K
  steps over the last 30K, against +0.90 in the first 10K). The near-ties close slowly: the
  10th-percentile gap goes 0.26 → 0.16 u, and exploration 1.5% → 2.4%. Argmax = A stays at
  95%, and human agreement at 0.797. The pre-set bar (gap levels off, exploration holds) is not
  met. **60K at 0.01** (2026-10-15 02:59–05:59 UTC, the user asked for it): the expected trade.
  The median gap stays nearer A (2.15 u), but the near-ties close faster: 10th percentile 0.07 u,
  exploration 1.5% → 4.7% and still rising, argmax = A 94%. By the pre-set rule, **0.03 is
  the setting.** The replay is greedy-only, the worst case; online exploration gives the untried
  actions targets. **D57 (the user, 2026-10-15): V's online step keeps CQL at 0.03.**
  - **Online test `v-cql03`** (`experiments/v-tuning.md` → "Online test at CQL 0.03"):
    e1's settings plus the term, 06:52–08:44 UTC, 73,200 online steps.
    - Cost $5.42: prices were ≈ 2× the quoted ≈ $3, and the user was told.
    - `v_qgaps.py`: no collapse (median gap 4.76 u, off-argmax 2.4%, argmax = human 0.767), as
      the replay predicted.
    - **Measured against A: −0.0431 u, CI [−0.0505, −0.0357]**, against e1's −0.729. By D57's
      reading, the term doesn't rescue online V at this length: **V stays stopped**, and the
      next design is the user's choice.
- **Halfway checks: every arm is below its start** (the user asked for them, 2026-10-15;
  exploratory; `p0.md` and `p1.md` → "Halfway check"). Each run's `p50` against its own
  starting policy, 10K sets on the shared block `halfway`:
  - `p0-a` −0.0334 u [−0.0410, −0.0257];
  - `p1-a` −0.0262 [−0.0337, −0.0187];
  - `p1-b` −0.0216 [−0.0291, −0.0142].
  - So every online checkpoint measured so far, V's included, loses to the frozen model it
    started from and trains against.
- **The user (2026-10-15): wait for the full P0 and P1 runs before more measurements, and work
  on the search code meanwhile.**
- **Search pilot** (`experiments/search-pilot.md`, 06 §0).
  - **Oracle stage: built and tested.** `libriichi/src/search/` replays a kyoku from a latent
    history to a decision, checking every event against the log, redeals the undealt tiles and
    rolls branches out to the end of the kyoku. `search/` runs self-play, the riichi-vs-dama
    decisions, the oracle rollouts, the GRP valuation and the report (`python -m search`).
    - Branching with the logged action reproduces the logged kyoku exactly, through riichi,
      wins, chi, pon and all three kans.
    - Smoke run (dev key, pilot A, not a result): 256 decisions × 130 rollouts in 95 s on the
      3060, and 256/256 true-world replays reproduce the self-play game. 20K decisions take
      ≈ 2.2 h on the workstation, with no cloud box.
    - Before the run: G0's blueprint, and the design choices in the note for the user to
      confirm (above all what the oracle knows, and the primary decision rule).
  - **Belief stage: built, and the learned hand model passes D58's test** (the user chose it,
    D58, 2026-10-15).
    - D38's v1 belief (uniform particles reweighted by action likelihoods) predicted opponents'
      tenpai worse than a constant rate.
    - A model of opponents' concealed hands, trained on 4,000 pilot-A self-play games (37 min),
      does well on held-out decisions: tenpai log loss 0.258 against 0.415 for the base rate,
      a predicted rate of 14.1% against 14.6% true, and waits log loss 0.047 against 0.060.
    - Reweighting it by likelihoods makes it worse, so by D58 the model alone is the belief.
    - One belief search takes p50 0.68 s and p99 2.21 s on the 3060, under the 2.5 s criterion.
    - Before the run: G0's blueprint, and a hand model trained on its self-play (≈ 1.5 h).
      The pilot then takes ≈ 5–6 h on the workstation.
- **Eval box stall (2026-10-15):** the 100K test's first chunk named the riichidev venv, which
  this rental never had. The queue runner died on it at 11:34 UTC and the box sat idle until
  18:25 UTC (≈ $4.4); I hadn't checked that the first chunk started. Fixed in `8df7b39`: a job
  that can't start now fails with a note, and `submit` builds the venv on first use. The venv
  is built, the runner restarted at 18:25 UTC, and c00 was requeued after c19. The 20 chunks now
  end ≈ 2026-10-16 14:30 UTC, later once P0's G0 measures go ahead of them.
- **P0 ended on schedule** (2026-10-16 02:18 UTC; its watchdog destroyed the 3 boxes; $218.50 in
  all). `full.pth` is copied back, read-only. **Its G0 evaluations are queued on the eval box**
  (the user, 06:20 UTC: "queue them now"), at priority 0 ahead of the 100K test's last chunks:
  - G0's T-std, `full` against pilot A (family B);
  - the pre-registered doubling measure, `full` against `p50`, 100,000 sets on family B
    (`experiments/p0.md`, ≈ 10 h).
  Both were queued with `--no-sync` on the box's commit `8df7b39` (the lab code is unchanged
  since). They push the 100K test's end to ≈ the evening of 2026-10-16.
- **Eval box up again** (the user, 2026-10-15: "rerun the full 100000 game test"). Instance
  56069526, EPYC 9684X, $0.65/h, rented 08:42 UTC, holding the lab lease.
  - The halfway checks of P1 and V's measure ran there first.
  - The 20 chunks of the 100K test (`workdir/riichidev/submit-100k-r2.sh`, reports
    `parity-100k-r2-cNN.json`) run at background priority since 18:25 UTC (the stall above).
    ≈ 1 h each, so done ≈ 2026-10-16 14:30 UTC.
  - Then `evalbox.py pull` and `down`. P0's G0 evaluations go through its queue meanwhile.
- **Eval box closed** (the user, 2026-10-14 11:30 UTC: "close the eval box and only rent it
  when ready"). It ran 06:17–11:30 UTC (instance 55876675, $0.65/h, ≈ $4.2). The lab lease is
  released, so lab jobs run on the workstation again. Rent it again only when a job is ready for
  it, and ask first.
  - **The riichi.dev validation game passed** (2026-10-14 07:34 UTC; pilot A; 0 failures, 0
    timeouts, p50 ≈ 10 ms). `main` was pushed first at the user's word (`b7eecce`). Ranked
    play waits for the π_SL student (decided 2026-10-12).
  - **The 100K-game test found two adapter bugs in 25K games** (5 of 20 chunks finished,
    `experiments/riichidev-adapter.md` item 4). No illegal move reached the server, but:
    chankan (robbing a kan) is missed, because the adapter's fallback passes instead (seeds
    20281 and 20282); and libriichi offers a chi that riichienv doesn't (seed 14176). **Both
    fixed 2026-10-15** (`e127b7f`): riichienv's `kakan > dora` order after an open kan, and the
    four-kan abort's discard. Regression tests replay the recorded requests, and 100 full-check
    games pass. The 100K test runs again on the next eval-box rental, with the user's go-ahead.
  - The token is in `~/.config/riichidev/token` (mode 600); the box's copy went with the box.
- **CQL model B frozen** (2026-10-13 18:01 UTC by hand, step 156,000, sha256 `c0ea3b96…`, in
  `tools/fleet/frozen.sha256`). Sanity checks pass (data out at step 157,869 at 11:26 UTC; losses
  finite and within 0.005 of A's). `v_qgaps.py` on B gives ε 0.0502 / 0.1944, so the sweep's
  values carry over to `v-b.toml`.
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
    approved dollars. **e1-lr rerun** (D56, approved by the user) ran clean 18:04–19:56 UTC; **the sweep's four runs are done: $14.91, 31.5 box-hours.** **e0 extended** to e1's step count (the user; D56): done 2026-10-13 02:49 UTC, $1.09. **The sweep is done: $16.00, 33.9 box-hours**; every comparison is with e1's final. Step-matched
    comparisons (amendments written before any measurement). **Measured 2026-10-14: the stop
    rule fired** (above).
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

Read `docs/design/handoff-2026-10-18.md` first: it has the live run, what to do at each of its
events, and the failure modes met since 2026-10-10.

1. **The G0 retry passed** (2026-10-24, `experiments/k2-retry.md`). The doubling estimate (`full` vs
   p50, `20261024-122034-measure`) runs on the workstation until ≈ 2026-10-25 10:30 UTC
   (`experiments/k2b_g0_chain.sh`, log `workdir/fleet/runs/k2b/g0-chain.log`); read it by its
   pre-registered rule. Then `rl_drift.py` / `rl_diag_sampling.py` on `full` for the write-up.
2. **After the pass** (the user, 2026-10-24): queued on the workstation by
   `experiments/k2b_post_g0_chain.sh` (log `workdir/fleet/runs/k2b/post-g0-chain.log`), after the
   doubling estimate:
   - L2's T-confirm, 100K sets on fresh family-C nonces (pre-registered in `experiments/k2-retry.md`),
     ≈ 10-25 10:30 → 10-26 01:00 UTC;
   - the diagnosis's drift and sampling runs on `full` (≈ 1.6 h);
   - the search pilot on `full` (≈ 6 h; a hand model on its self-play, then the pilot on block
     `search-pilot`), done ≈ 10-26 09:00 UTC.
   Then the fleet soak (`03` §3.7, ≥ 48 h at T1 scale, a rental): the user's call. Known risk:
   policy lag p95 ≈ 9–11 versions against its ≤ 2 (`experiments/p0.md` → "Policy lag").
3. **Search pilot (`06` §0): parked** (the user, 2026-10-21), except the rerun on `k2b`'s `full`,
   queued (the user, 2026-10-24; item 2). It fails on recovery (0.157 vs 0.5);
   a bigger hand model doesn't move it (`experiments/search-pilot.md`). Not pursued for now: the
   ceiling test (can any belief reach 0.5 on riichi vs dama?) and a hand model aimed at late-kyoku
   threat. If `k2b` passes G0, the pilot runs again on its blueprint, on the same block.
4. **SL substrate (`04`):**
   - the secure-tier data path (D23) and the D17a measurement;
   - the b24c512 teachers, then the π_SL student;
   - the riichi.dev smoke deploy (L1). The adapter's 100K parity test passed on 2026-10-18, and
     ranked play waits for the student.
5. **G0's observation review (D49):** v4 or v5, weighing P0's doubling estimate and its caveat.
6. **Arm V:** stopped (D57's test failed). A fix needs a design change and a new approval.
