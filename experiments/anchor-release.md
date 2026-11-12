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

- **A's target is set by what the controller delivers.** It lowers the coefficient only when the KL
  is below target / 1.5, so a target ending at 0.06 enforces a KL of at least 0.04 by the end. With
  the KL at 0.019 the first cut comes when the target passes 0.0285, near 37K steps (≈ 1.5 h).
- **B replays `k2b`'s start around a new centre.** `k2b`'s coefficient fell to its floor in the first
  minutes and climbed back to 0.0759 on the overshoot (to 0.032); `k2-tune-a`'s ended at 0.114 the
  same way. Where B's settles is reported, not fixed.
- **Their distances from pilot A are measured, not assumed equal.** KL isn't additive: 0.02 around
  `k2b` on top of 0.019 can be anywhere up to ≈ 0.08 from pilot A.

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
F="python3 tools/fleet/fleet.py"
for X in a b; do
  $F up --run release-$X --config experiments/configs/release-$X.toml --init workdir/release/init-$X.pth \
      --actors 2 --procs 2 --hard-hours 52 --verify --yes
  setsid -f $F watch --run release-$X --replace --allow-zen2 --budget-box-hours 144 --snapshot-at 0.25,0.5 \
      --max-dollars 95 --anchor-alert 0.10 > workdir/fleet/runs/release-$X/watch.out 2>&1 < /dev/null
done
```

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

## Measures (after both `full` snapshots, on an eval box)

Six runs of `python -m lab measure --sets 40000 --family A --block release`, AMP on, exploratory:
A `full`, B `full` and `soak-a` full, each as the challenger at a table of three pilot As (the
training table) and at a table of three pilot Bs (the held-out one). All six play the same 40,000
deals (a new family-A block, `release`).

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

**Cost:** 240,000 sets. The eval box is approved (≈ $8 quoted; rented when both `full` snapshots
exist, down when the queue drains).

## Code (2026-11-12)

- `tools/fleet/stage_init.py`: a snapshot without its run clock and, with `--new-anchor`, without
  its anchor state (`tests/test_stage_init.py`, 3 tests).
- `tools/fleet/fleet.py` `inputs()`: the learner also gets `policy.kl_anchor.state_file` when a run
  names one (`tests/test_fleet.py`).
- The two configs, and `release-smoke-{a,b}.toml` for the smoke runs below.
