# D64's anchor-release pilot: pre-registered 2026-11-12, before it runs

The second of D64's two conditions before T1 (the first, the critic check, is done:
`experiments/critic-ceiling.md`). The user chose this design and approved the rental on 2026-11-12
("Go with your picks, I give the okay for the box rental as well"), after the review of
`docs/design/t1-preconditions-draft.md` §2. Exploratory; nothing here has run.

## Why

`k2b` beat pilot A by +0.0195 and then stopped improving at K2's constant anchor:

- Its coefficient settled at 0.0759 in the first hour (t = 18,400 of 3.66M steps) and never moved.
  The KL to pilot A eased from 0.025 to 0.019 over the run as the schedules decayed, and to 0.018
  over the soak, at the same coefficient (`anchor.jsonl` of both runs).
- The run's second half bought +0.0025 [+0.0004, +0.0046]. The soak's 47.8 h more at the same anchor
  bought +0.0016 [−0.0050, +0.0082] (`experiments/fleet-soak.md`).

**Question:** does loosening the anchor let the policy improve past the soak, and which way of
loosening it should T1 use?

## Candidates

Both continue `k2b` full (`f5a22c5c…`, v4, global step 3,880,000) for 48 h on a learner and 2
actors, with three pilot As at the table as before.

| | **A, rising target** | **B, re-anchor** |
|---|---|---|
| Anchored to | pilot A, as `k2b` | `k2b` full itself |
| Target | 0.02 · (1 + t/10K)^0.228 nats: 0.031 at 60K steps, 0.044 at a quarter of the run, 0.051 at half, ≈ 0.060 at the end (1.22M steps at `k2b`'s 25.4K steps/h) | 0.02 nats, constant |
| Controller | `k2b`'s, continued: the coefficient starts at 0.0759 | `k2b`'s, from its start: coefficient 0.1, floor 0.01 |
| Config | `experiments/configs/release-a.toml` | `experiments/configs/release-b.toml` |
| Init | `workdir/release/init-a.pth` (`db453c45…`): `k2b` full without its run clock | `workdir/release/init-b.pth` (`77f06b16…`): also without its anchor state |
| Run | `release-a` | `release-b` |

- **What the controller is expected to do (an expectation, not a criterion).** It lowers the
  coefficient when a 400-step window's KL is below target / 1.5 and raises it above target × 1.5.
  The KL answers over thousands of steps, so once a cut starts the coefficient falls to its 0.01
  floor within a few windows, and the KL then climbs until it meets the band's upper edge, where
  the coefficient steps back up and holds it. `k2b`'s own start went this way (floor by t = 5.2K,
  the KL over 0.03 near t = 16K, the coefficient back to 0.0759 by t = 18.4K, the KL then at
  ≈ 0.027), and so did `k2-tune-a`'s (0.114).
  - **A:** with the KL at 0.019 the first cut comes when the target passes 0.0285, near 37K steps
    (≈ 1.5 h). The KL to pilot A should then climb to the upper edge, ≈ 0.05 nats within the first
    few hours, and stay near there. The rule guarantees at least 0.04 by the end and allows up to
    0.09.
  - **B:** it replays `k2b`'s start around the new centre, so the KL to `k2b` should settle near
    0.027, not 0.02.
- **Their distances from pilot A are measured, not assumed equal.** KL isn't additive: ≈ 0.027
  around `k2b` on top of 0.019 can be anywhere up to ≈ 0.09 from pilot A, and ≈ 0.046 if the new
  step is orthogonal to the old one. A's expected ≈ 0.05 sits inside that range.

## Held equal to the soak

The soak (`soak-a`, `9c0b7b34…`) is the fixed anchor's continuation from the same checkpoint, so it
is the comparator. Besides the anchor, the candidates differ from it in two ways:

- **Schedules: constants at the soak's values.** A staged init restarts the run clock, which would
  restart `k2.toml`'s power laws at LR 2e-5, clip 0.2 and entropy weight 1e-3. The soak ran at LR
  6.7e-6 to 6.0e-6, clip 0.1 (its floor) and entropy weight 1.6e-4 to 1.4e-4. Both configs hold
  LR 6.4e-6, clip 0.1 and entropy weight 1.5e-4.
- **Size: 144 box-hours each** (3 boxes × 48 h) against the soak's 192 (4 boxes). The candidates get
  fewer games, which only makes passing harder.

The learner's settings are the soak's (4 loader workers, 4 trainer threads), written into the
configs. The critic (K1) is as `k2b` left it, by the critic check's reading.

## Fleet and cost

```bash
# experiments/release_launch.sh X runs, for X = a or b:
F="python3 tools/fleet/fleet.py"
$F up --run release-$X --config experiments/configs/release-$X.toml --init workdir/release/init-$X.pth \
    --actors 2 --procs 2 --hard-hours 52 --verify --yes
setsid -f $F watch --run release-$X --replace --allow-zen2 --budget-box-hours 144 --snapshot-at 0.25,0.5 \
    --max-dollars 95 --anchor-alert 0.10 >> workdir/fleet/runs/release-$X/watch.out 2>&1 < /dev/null
```

After a workstation restart: `bash experiments/release_launch.sh a --resume` (and `b`), which
restarts the watcher and the watchdog if they are missing.

- **Snapshots:** p25 (≈ 12 h), p50 (≈ 24 h) and `full` (144 box-hours, ≈ 48 h), then the watcher downs
  the run. Hard deadline 52 h.
- **Quote:** ≈ $86 per run at `k2b`'s $1.80/h, ≈ $175 for both (approved). Each run stops at $95. It
  is rented only if the live quote is at most $1.90/h per fleet; above that the user is asked.
- **Where the early-research cap stands** (D65, D24): the fleet's own records sum to ≈ $1.24K of
  the $2.2K for every RL run so far (the arms, their tuning, the K2 retry, the soak and the v5
  checks; the eval box's $81 and the teachers' $123 are SL substrate). After this pilot ≈ $1.42K,
  which leaves ≈ $780 for D65's curves (≈ $630–690).
- **No automatic stop** besides the budget and the deadline. `--anchor-alert` logs an event if a
  run's KL to its own anchor averages above 0.10 nats over 2 h.

## Reported along the way (not criteria)

At each snapshot, on the workstation, on the diagnosis's states (`experiments/rl_drift.py --logs
workdir/diag/credit-a --init` pilot A): the reverse KL to pilot A, argmax agreement with it and the
policy's entropy. `k2b` full on these states: 0.0194, 95.4%, 0.468 (pilot A's entropy 0.522). For B
this is the only view of its distance from pilot A: its own `anchor.jsonl` logs the KL to `k2b`.
Also each run's coefficient trace, learner steps and games.

Both known failure modes come after early gains (an entropy collapse when an anchor is annealed too
fast, forgetting when it moves), which is why all three snapshots are read.

## Measures

Six runs of `python -m lab measure --sets 40000 --family A --block release`, AMP on, exploratory:
A `full`, B `full` and `soak-a` full, each as the challenger at a table of three pilot As (the
training table) and at a table of three pilot Bs (the held-out one). All six play the same 40,000
deals (a new family-A block, `release`).

- **`soak-a` full's two runs** don't depend on the pilot, so they play on the workstation while
  the fleets train (≈ 15 h, ≈ 13 GB of logs).
- **The candidates' four runs** play on an eval box once both `full` snapshots exist
  (`experiments/release_measure_chain.sh`).
- The two machines' GPUs differ, which acts like AMP's batch-composition noise (near-tie argmaxes
  flip, zero-mean; CLAUDE.md's gotcha): it adds to D's variance, not to its mean.

For a candidate at a table, per set:

    D = set score of the candidate − set score of soak-a full      (same deal, same opponents)

with the lab's set-level bootstrap 95% CI. The expected standard error is ≈ 0.0021 (the SD of such
a difference was 0.417 at pilot B's table, `experiments/k2b-opponents.md`), so a CI of ≈ ±0.004.

**Reading, fixed now:**

1. **A candidate passes** if D's CI is entirely above 0 at both tables.
2. **Both pass:** T1's mechanism is the one with the larger D at pilot B's table, if the 95% CI of
   the paired difference between the two candidates there excludes 0. Otherwise it is B: it has no
   schedule to tune, and it is what a chain of champions repeats.
3. **One passes:** that one.
4. **Neither passes:** D64's "reopen" goes to the user. The report then says what this pilot can't
   separate: no mechanism, the decayed schedules (held at the soak's values here), or 48 h being
   too short.

Also reported: each run's Δu, CI, average rank and style panel, and the paired difference between
the candidates at each table.

**Cost:** 240,000 sets: 80,000 on the workstation ($0) and 160,000 on the eval box, ≈ 14 h at the
≈ 11K sets/h its b8c192 measures ran at, ≈ $8–10 (approved at ≈ $8; rented at ≤ $0.75/h when both
`full` snapshots exist, down when the queue drains).

## Code (2026-11-12)

- `tools/fleet/stage_init.py`: a snapshot without its run clock and, with `--new-anchor`, without
  its anchor state (`tests/test_stage_init.py`, 3 tests).
- `tools/fleet/fleet.py` `inputs()`: the learner also gets `policy.kl_anchor.state_file` when a run
  names one (`tests/test_fleet.py`).
- The two configs, and `release-smoke-{a,b}.toml` for the smoke runs below.

## Smoke runs (2026-11-12, the workstation; not results)

`experiments/configs/release-smoke-{a,b}.toml`: each config's anchor and schedules on k2-smoke's
plumbing (32-game drains, the controller every 5 steps), from the staged inits.

- **A** (155 steps, no errors):
  - The trainer logged a new clock ("origin at step 3,880,000, t = 0") and "loaded the kl anchor
    (coef 0.0759375)".
  - The KL to pilot A started at 0.019 with 95% agreement, `k2b`'s figures.
  - The LR, clip and entropy weight logged 6.4e-6, 0.1 and 1.5e-4 throughout.
  - The target followed its formula (on the smoke's 20-step scale: 0.0210 at t = 5, 0.0289 at
    t = 80), and the coefficient was first cut in the window where the KL fell below target / 1.5
    (t = 60). It reached the 0.01 floor five cuts later (t = 110) with the KL still at 0.019–0.021:
    the controller outruns the policy, as the expectation above says.
- **B** (125 steps, no errors):
  - The trainer logged "kl anchor: adaptive, pi_SL …/fleet/runs/k2b/snapshots/full.pth, coef 0.1"
    and a new clock, and loaded no anchor state, with pilot A still the table's opponent.
  - The first window's KL was 0.00000 with 99.96% agreement: the anchor is the policy's own
    starting point, not pilot A (which would read 0.019 and 95%).
  - The controller held for 5 windows, then cut the coefficient each window to the 0.01 floor
    (t = 60), as `k2b`'s start did. The KL had reached 0.0007 by t = 120 at the soak's LR.

## Launch (2026-11-12)

- **A, `release-a`:** rented 18:43 UTC at a live quote of $1.81/h (limit $1.90/h). Its first
  learner host (machine 54271) never finished loading its image and the fleet replaced it after
  25 min. Training began 19:23 UTC: learner 60394006 (EPYC 7713), actors 60391934 (EPYC 7C13) and
  60391935 (EPYC 7B13), **$1.63/h**, ≈ $78.
  - On the learner: "run clock: origin at step 3,880,000, t = 0" and "loaded the kl anchor (coef
    0.0759375)". At t = 4,800 the KL to pilot A was 0.0195 with 95.2% agreement, `k2b` full's
    figures.
- **B, `release-b`:** by 19:05 UTC the three cheapest remaining machines quoted ≈ $2.09/h, above
  the limit, so the user was asked and chose "Now, up to $2.10/h" (≈ $100 for B, ≈ $185 for both;
  stop $105). It rented at 19:09 UTC, and the boxes it got cost less: learner 60394073
  (Threadripper PRO 5995WX), actors 60394075 (EPYC 7713) and 60397192 (EPYC 7Y83, after the first
  actor2 host failed), **$1.77/h**, ≈ $85. Training began 19:43 UTC.
  - On the learner: "kl anchor: adaptive, pi_SL …/fleet/runs/k2b/snapshots/full.pth, coef 0.1" and
    the same new clock.
- **Projected: ≈ $165 for both,** under the approved ≈ $175. `full` ≈ 2026-11-14 19:00 UTC for
  each (144 box-hours); watchdogs at 22:43 and 23:09 UTC.
- **The boxes report their commit as `-dirty`.** At each sync the tree held uncommitted `STATUS.md`
  text (A's actors, at `7ef2a0a`) or the lab's new registry rows (the rest, at `acec88f`). The two
  commits differ in notes and one workstation script, so every box runs the same code.
  **A replacement box gets the working tree as it is at that moment: no trainer or engine code is
  edited until both runs are down.**
- **The comparator's chain** (`release_soak_chain.sh`) has run on the workstation since 19:10 UTC
  at ≈ 5.1K sets/h (both measures done ≈ 2026-11-13 11:00 UTC). Its first run made the block
  `release`: family-A nonces [74200, +40000).
- **The eval-box chain** (`release_measure_chain.sh`) has waited, detached, since 19:45 UTC for
  both `full` snapshots (log `workdir/fleet/runs/eval/release-measures.log`). When the measures
  are in: `python experiments/release_paired.py` gives the reading.

## Progress (2026-11-13 19:50 UTC): both runs past p50, nothing lost

| | `release-a` (A) | `release-b` (B) |
|---|---|---|
| p25 | 06:47 UTC, step 4,138,800 (`eff4d3db…`) | 07:12 UTC, step 4,171,200 (`dbac46cc…`) |
| p50 | 18:44 UTC, step 4,412,000 (`ce61bfc8…`) | 19:13 UTC, step 4,540,000 (`54548b31…`) |
| Learner steps/h (games/h, last hour) | 23.0K (84K) | 28.3K (103K) |
| Billed | $40.89 at 75.2 box-hours | $43.57 at 73.9 box-hours |

No box lost, no `anchor_alert`. B's hosts are faster, so at equal box-hours B has had ≈ 23% more
learner steps than A; that sits beside any A-versus-B reading.

**The comparator** (workstation, block `release`, 40,000 sets each; exploratory):

| Table | `soak-a` full, Δu | 95% CI | Run |
|---|---|---|---|
| 3× pilot A | +0.0190 | [+0.0158, +0.0223] | `20261112-121013-measure` |
| 3× pilot B | +0.0249 | [+0.0215, +0.0282] | `20261112-192915-measure` |

**What the controllers did** (each learner's `anchor.jsonl`), against the expectation written
before launch:

- **A:** one cut at a time, not a fall to the floor. The coefficient went 0.0759 → 0.0506 at
  t = 42.4K, → 0.0338 at t = 106K and → 0.0225 at t = 539K, each when the rising target / 1.5
  reached the KL. After each cut the KL rose fast enough to stay above that edge, so **the KL
  trails target / 1.5**: 0.024 at t = 100K, 0.032 at 200K, 0.034–0.035 from 300K to 550K. The
  expectation ("≈ 0.05 within the first few hours") was wrong. The fall to the floor in `k2b`'s
  start came from starting at KL 0, far below the band, and in the smoke run from its 5-step
  windows.
- **B:** as expected up to the overshoot: the floor by t = 4.8K, the KL to `k2b` at 0.03 near
  t = 190K, and four steps up to 0.0506. But at that coefficient the KL then fell to ≈ 0.0155 and
  has stayed there, inside the band, so nothing moves it: B holds ≈ 0.016 nats from `k2b`, not
  0.027.

**Drift from pilot A** (`experiments/rl_drift.py`, 15,215 decisions of `workdir/diag/credit-a`;
output in `workdir/diag/release-drift.txt`; pilot A's entropy 0.522):

| Checkpoint | Reverse KL to pilot A | Agreement | Entropy |
|---|---|---|---|
| `k2b` full | 0.0194 | 95.4% | 0.468 |
| `soak-a` full | 0.0181 | 95.4% | 0.470 |
| A p25 / p50 | 0.0348 / 0.0354 | 93.8% / 93.8% | 0.430 / 0.430 |
| B p25 / p50 | 0.0418 / 0.0415 | 93.5% / 93.4% | 0.415 / 0.420 |

- The two candidates sit at similar distances from pilot A (B ≈ 1.2× A's), each about twice
  `k2b`'s. B's 0.016 from `k2b` adds up to 0.042 from pilot A, so its step partly continues the
  old direction.
- Entropy is ≈ 8% (A) and ≈ 11% (B) below `k2b`'s and level between p25 and p50: lower, not
  collapsing.

## Training done (2026-11-14)

| | `release-a` (A) | `release-b` (B) |
|---|---|---|
| `full` | 18:48 UTC, step 4,958,400, 144.24 box-hours (`a57f195f…`) | 19:15 UTC, step 5,316,000, 144.27 box-hours (`be94d8e8…`) |
| Learner steps in the run | 1.08M | 1.44M (1.33× A's) |
| Cost | **$78.35** | **$85.15** |
| Box losses | actor1 lost 17:11 UTC, replaced, recovered after 1,745 s (within 30 min) | none |

Both watchers downed their fleets at `full`; **$163.50 for both** (approved ≈ $175). No
`anchor_alert` in either run.

**The anchors at the end** (the learners' last 50 windows): A at KL 0.049 to pilot A (coefficient
0.0225 since t = 539K; target 0.060 at t = 1.08M, so target / 1.5 = 0.040); B at KL 0.018 to `k2b`
(coefficient 0.0506 since t = 192K), up from 0.0155 at half-time.

**Drift from pilot A at `full`** (same states and script as above):

| Checkpoint | Reverse KL to pilot A | Agreement | Entropy |
|---|---|---|---|
| `k2b` full | 0.0194 | 95.4% | 0.468 |
| A p25 / p50 / **full** | 0.0348 / 0.0354 / **0.0505** | 93.8% / 93.8% / **93.1%** | 0.430 / 0.430 / **0.399** |
| B p25 / p50 / **full** | 0.0418 / 0.0415 / **0.0445** | 93.5% / 93.4% / **93.0%** | 0.415 / 0.420 / **0.427** |

- A moved most of its distance in the second half, after its third cut; it ends 14% further from
  pilot A than B.
- **A's entropy fell again in the second half**, 0.430 → 0.399 (−15% from `k2b`'s), while B's held
  (0.415–0.427). That is the direction of the entropy-collapse failure mode named above; it isn't a
  criterion, and whether it cost strength is what the measures say.

**The measures** started on the eval box at 19:24 UTC (machine 13450, EPYC 7B13, $0.556/h): the
four 40K-set runs queued in order (A at pilot A, A at pilot B, B at pilot A, B at pilot B), at
≈ 9K sets/h, so ≈ 18 h, done ≈ 2026-11-15 13:30 UTC. The chain pulls every hour and downs the box
when the queue drains (forced at 22 h, ≈ 11-15 17:25 UTC).

## Result (2026-11-15): both candidates pass; T1's mechanism is B, the re-anchor

The eval box played the four candidate measures 2026-11-14 19:24 → 11-15 13:26 UTC and was downed
by the chain ($10.12, 18.2 h). `experiments/release_paired.py` → `workdir/lab/release-paired.json`.
Exploratory; family A, block `release` (nonces [74200, +40000)), AMP on; the lab is calibrated.

**Each run against three copies of the table's pilot** (Δu, 40,000 sets):

| Engine | 3× pilot A | 3× pilot B | Run ids (A table, B table) |
|---|---|---|---|
| `soak-a` full | +0.0190 [+0.0158, +0.0223] | +0.0249 [+0.0215, +0.0282] | `20261112-121013`, `20261112-192915` |
| A `full` | +0.0311 [+0.0275, +0.0345] | +0.0293 [+0.0259, +0.0327] | `20261114-192413`, `20261114-235914` |
| B `full` | +0.0299 [+0.0264, +0.0333] | +0.0307 [+0.0272, +0.0342] | `20261115-043647`, `20261115-085704` |

**The pre-registered differences, D, paired on the same deals** (95% set-level bootstrap):

| D | 3× pilot A (training table) | 3× pilot B (held out) |
|---|---|---|
| **A − soak** | **+0.0121 [+0.0076, +0.0164]** | **+0.0044 [+0.0000, +0.0087]** |
| **B − soak** | **+0.0109 [+0.0066, +0.0150]** | **+0.0058 [+0.0014, +0.0101]** |
| A − B | +0.0012 [−0.0031, +0.0056] | −0.0014 [−0.0057, +0.0028] |

SD of D 0.44 (0.417 assumed), so each CI is ±0.0043, as planned; paired correlation 0.15–0.20.

**Reading, as fixed:**
1. **A passes** (both CIs above 0; at pilot B's table the lower bound is +0.00003, on the line).
2. **B passes** (lower bounds +0.0066 and +0.0014).
3. **Both pass, and A − B at pilot B's table includes 0, so T1's mechanism is B: re-anchoring to
   the new champion.**

**What else the data say (beyond the rule):**
- **Releasing the anchor paid; the fixed anchor didn't.** At the training table each candidate
  gained ≈ +0.011 over the soak in 144 box-hours ($78–85), against the soak's +0.0016 over `k2b`
  in 192 ($125). That is ≈ 60% of `k2b`'s whole gain over pilot A (+0.0195), in a third of its
  compute.
- **The gain is smaller at the held-out table:** 36% of the training table's for A, 53% for B. For
  A the gap is ≈ 2.5 SE (0.0077 ± 0.0031); for B 1.6 SE. Part of what was learned is specific to
  pilot A, more so for A, whose anchor and opponent are both pilot A. Pilot B is pilot A's twin
  (seed 2), so neither table says much about other kinds of opponent.
- **The style moved as `k2b`'s did** (lab panels, challenger seat, 160K games each):

  | At 3× pilot A | Avg rank | 1st % | 4th % | Win % | Deal-in % | Call % | Riichi % |
  |---|---|---|---|---|---|---|---|
  | `soak-a` full | 2.4767 | 25.41 | 24.03 | 21.72 | 12.51 | 28.54 | 19.58 |
  | A `full` | 2.4630 | 25.49 | 23.36 | 21.92 | 11.89 | 30.38 | 18.79 |
  | B `full` | 2.4646 | 25.43 | 23.42 | 21.60 | 11.78 | 29.70 | 18.94 |

  The gain is almost all fewer fourths (−0.6 to −0.7 pp; the utility charges −1.3 u for a fourth),
  through fewer deal-ins and more calls; firsts are unchanged. The pilot B table shows the same.
- **No sign of a cross-machine artifact.** The soak's runs played on the workstation (RTX 3060),
  the candidates' on the eval box (RTX 4090). The share of sets with identical scores is the same
  across machines (A = soak 9.4%) as on one machine (A = B 9.0%), and the soak's +0.0190 at pilot
  A's table matches `k2b`'s confirmed +0.0195. Not proven: a same-machine replay of one pair would
  settle it ($0, ≈ 8 h on the workstation).

**Limits:**
- One run per candidate, and exploratory. Not a confirmation (`05` §5.2 wants matched pairs and
  family C).
- **A and B are tied, so B was chosen by the tie rule, not by a measured advantage.** B also had
  1.33× A's learner steps at equal box-hours (faster hosts), which, if anything, favours B.
- Only `full` was measured, so whether more release keeps paying, or where it turns, is unknown.
- A's entropy fell to 0.399 (−15% from `k2b`) with no measurable cost at `full`; B's held at 0.427.
- B's controller over-tightened after its overshoot and held 0.016–0.018 nats from `k2b`, below its
  0.02 target, so B had less room than designed and still tied A.

**For T1's pre-registration (not decided here):** how often to re-anchor (the pilot re-anchored
once, for one 48 h stage); the controller's start (B's coefficient floor-then-overshoot left it
too tight); whether to keep a weak second penalty to the SL policy against forgetting; and a
held-out table of another kind than pilot A's twin.

**Cost:** $163.50 (fleets) + $10.12 (eval box) = **$173.62**, approved ≈ $175 + ≈ $8.

## Stage-timing check: pre-registered 2026-11-16, before it runs

The user ran it on 2026-11-16 ("Start on 1"), the first of the open choices for T1's
pre-registration: **how long a re-anchoring stage should be.** The pilot measured only B's `full`
(48 h, 144 box-hours). If B's gain was mostly in by half-time, stages can be shorter and T1 gets
more re-anchors for the same money; if it kept coming, 48 h stays.

**Runs** (the workstation, $0, `experiments/release_timing_chain.sh`): B's frozen p50 (24 h,
72.17 box-hours, `54548b31…`) and then p25 (12 h, 36.12 box-hours, `dbac46cc…`), each as the
challenger at 3× pilot A, 40,000 sets on block `release` (the same deals as every pilot measure),
AMP on, `--no-logs` (C: has 22 GB free; only the per-set scores are needed). Only the training
table: the held-out gain (+0.006) is too small to time at this size. ≈ 7.3 h each at the
workstation's ≈ 5.5K sets/h; p50 first, since it decides.

**Quantities** (`experiments/release_timing.py`, paired on the same deals, 95% set-level bootstrap):
- G(s) = B's snapshot s − `soak-a` full, for s = p25, p50 (G(full) = +0.0109 [+0.0066, +0.0150]);
- R = B full − B p50: the gain of the stage's second half.

**Reading, fixed now:**
1. **R's CI includes 0 and G(p50)'s CI is entirely above 0:** the stage's gain was mostly in by
   24 h. T1's pre-registration proposes 24 h stages (72 box-hours on this fleet).
2. **R's CI is entirely above 0:** the gain was still coming in the second half. 48 h stays.
3. **Anything else:** grey, and 48 h stays (the length that was tested).

Also reported: G(p25), and G(p50) / G(full) as a point estimate. The reading proposes a length to
the user; T1's pre-registration decides it.

**What it can't show:** whether a 24 h stage followed by a re-anchor gains as much as the first
24 h of a 48 h one (the controller restarts at each re-anchor), or anything about v5 students.
