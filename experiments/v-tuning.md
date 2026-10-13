# Arm V tuning sweep (05 §2.2; written 2026-10-11, before any sweep)

- **Question:** V's exploration (ε and τ) and its learning rate, for both seeds' runs
  (`experiments/configs/v-a.toml`, `v-b.toml`).
- **Status:** step 1 done 2026-10-12 (model A frozen at 09:51 UTC; ε fixed below; the four
  configs committed with this line). The rental (≈ $11–15) was approved by the user on 2026-10-11.
  - **The runs ended 2026-10-12 12:37 UTC** ("The runs" below): e0, e1, e2 valid; e1-lr void (its
    boxes were destroyed by a `fleet.py` bug, now fixed). $12.12 and 25.5 box-hours in all, ≈ 1.5
    over the allowance; inside the approved dollars. Reported to the user, with the option of
    rerunning e1-lr (≈ $3, 6 box-hours).
  - **e1-lr rerun approved by the user** (2026-10-12; D56, past the allowance). It runs under the
    same name and config; the void run's records are in `workdir/fleet/runs/v-tune-e1-lr-void1`.
    - **Its comparison, registered before it starts:** e1's final checkpoint (59,600 online steps)
      against the rerun's checkpoint nearest 59,600 online steps.
    - The rerun is snapshotted every 10 min from 60 min after its trainer starts until 5 min
      before its deadline (`fleet.py snapshot --run v-tune-e1-lr --label m<minutes>`); its final
      checkpoint also counts.
    - If even its final has fewer steps, its final is compared with the e1 checkpoint nearest it,
      as for the others.
  - **The rerun ran clean** (18:04–19:56 UTC, $2.79): 57,600 online steps, within 10% of e1's, so
    the finals are compared.
  - **e0 extended to e1's step count** (the user, 2026-10-12; D56); **done 2026-10-13 02:49 UTC,
    $1.09, snapshot at exactly 215,600.** Registered before it started:
    - run `v-tune-e0-ext`, config `v-tune-e0.toml`, from e0's final checkpoint (192,800; its
      optimizer state is restored, as at e0's mid-run resume);
    - a poller reads the learner's log and snapshots it as soon as its step reaches 215,600, e1's
      final (`fleet.py snapshot --run v-tune-e0-ext --label s215600`), then downs the run;
    - **that snapshot is e0's checkpoint for the decision**, compared with e1's final. e1 `t1145`
      is no longer needed;
    - if the extension fails, e0's original final is compared with e1 `t1145` as registered
      above.
  - **Next: 4 measures on the workstation after model B** (≈ 2026-10-13 11:00 UTC; ≈ 6 h): e1
    final, e0 (extended) `s215600`, e2 final, the e1-lr rerun's final. Each of the last three is
    compared with e1 final.
  - Step-matched comparisons: amendment under the decision rule.
- **Budget:** V's tuning allowance, ≤ 1 box-day (`05` §2.2). Four runs × 3 boxes × 2 h = 24
  box-hours, setup included. That is ≈ $11 at P1's prices (≈ $1.39/h per 3-box run). It sits
  outside each seed's 432 box-hours. The evaluation runs on the workstation ($0).
- **Not a strength claim:** the lab's `measure` rows are exploratory, on a named family-A block
  (family A's role is tracking and checkpoint selection, `lab/seeds.toml`).

## Why exploration is the main question

Measured on model A at steps 78K–88K (`experiments/v.md` → "Measured: how much v0 explores";
script `experiments/v_qgaps.py`):
- CQL leaves a median gap of 1.8 u between the best and second-best action. The actions humans
  didn't take sit near −6 u.
- So at Equim's τ 0.05, an exploration draw leaves the argmax only ≈ 1.3% of the time. With ε 0.005
  that is ≈ 0.01 non-greedy actions per game.
- V learns Q only for the actions it takes. At v0 it can lower an overvalued greedy action but
  almost never finds an undervalued one.

The learning rate is the second question: v0's 2e-5 is P0's policy rate; Equim's example has 1e-4.

## The four runs

| Run | ε | τ | Learning rate | Non-greedy actions per game (target) |
|---|---|---|---|---|
| `v-tune-e0` | 0.005 | 0.05 | 2e-5 | ≈ 0.01 (Equim's values, as v0) |
| **`v-tune-e1`** (default) | set for the target | 0.5 | 2e-5 | 1 |
| `v-tune-e2` | set for the target | 1.0 | 2e-5 | 7 (≈ 5% of decisions) |
| `v-tune-e1-lr` | as `e1` | 0.5 | 1e-4 | 1 |

- **Why `e1` is the default:** it is the smallest of the three settings that explores in most
  games. `e0` is kept as the control: Equim's values, the arm as he ran it.
- **Setting ε:** `v_qgaps.py` on the frozen model A, the last 5 heldout games:
  ε = target ÷ (multi-option decisions per game × the share of a draw that leaves the argmax at
  τ). Rounded to 2 significant figures, capped at 0.5. On A at step 86,000 this gave 0.053
  (`e1`) and 0.20 (`e2`). The values used are written here and the four configs committed
  before the first box is rented. That commit is the pre-registration point.
  - **Used (frozen A, step 156,000, 2026-10-12):** 155 multi-option decisions per seat-game;
    off-argmax share 0.1303 at τ 0.5 and 0.2361 at τ 1.0; ε 0.0496 and 0.1915, so **`e1` and
    `e1-lr` ε = 0.050, `e2` ε = 0.19** (`experiments/configs/v-tune-*.toml`).
- **Everything else as v0** (`v-a.toml`): top-p 1, batch 512, constant learning rate, weight
  decay 0, clipping 0.5, γ 1, next-rank 0.2, 800 games per actor batch. Each run has its own
  paths (`workdir/v-tune-<run>/` on the box; `fleet.py` pulls the learner to
  `workdir/fleet/runs/v-tune-<run>/learner-run/`).
- **Initialization and opponents:** every run starts from frozen CQL model A and plays against
  it (D54). The chosen values then apply to both seeds.
- **Learner settings:** 4 loader workers and 8 torch threads, P1's learner-check result on the
  same box class (`--set dataset.num_workers=4 --set control.num_threads=8`). V's loader differs
  from P1's, so its throughput is measured here and reported.

## Procedure

1. **After model A is frozen** (the queue does it, ≈ 2026-10-12 09:30 UTC):
   - record its hash in `v.md` and in `tools/fleet/frozen.sha256`;
   - run `v_qgaps.py` on it with `--target 0.5:1 --target 1.0:7`, and fix ε;
   - write the four configs (`v-a.toml` with only the swept keys and the paths changed) and commit.
2. **`v-tune-e1` first:**

   ```bash
   python3 tools/fleet/fleet.py up --run v-tune-e1 --config experiments/configs/v-tune-e1.toml \
       --init workdir/cql/qs-a/cql-qs-a-final.pth --hard-hours 2 \
       --set dataset.num_workers=4 --set control.num_threads=8 --yes
   ```

   With a `fleet.py watch --replace` as for the main runs. This is also V's full-size check:
   - once its learner has drained logs and submitted parameters, the other three start;
   - if it fails, I fix it and restart it. Failed attempts count toward the 24 box-hours, and if
     the allowance would run out I stop and ask.
3. **The other three:** in parallel if Vast has the boxes, otherwise one after another. Each ends
   at its 2 h hard deadline: the watchdog pulls the learner's last checkpoint and destroys the
   boxes.
4. **Evaluation**, on the workstation after model B finishes (≈ 2026-10-13 11:00 UTC; nothing else
   runs there meanwhile), or on the eval box (`tools/fleet/evalbox.py`) if it is back first; it
   closed on 2026-10-12 until the riichi.dev token arrives. All four on the same machine. On the
   workstation, `python -m lab measure ...` with the same arguments. For each run:

   ```bash
   python3 tools/fleet/evalbox.py submit -- measure \
       --challenger workdir/fleet/runs/v-tune-<run>/learner-run/mortal.pth \
       --champion workdir/cql/qs-a/cql-qs-a-final.pth --arm V --sets 10000 \
       --family A --block v-tune --purpose "V tuning sweep (experiments/v-tuning.md)"
   ```

   - The first run allocates the block and the others reuse it. So all four play the same 10,000
     deal sets, and the differences between runs aren't diluted by different deals.
   - AMP on, the lab default. Its noise is zero-mean between different checkpoints (D43).
   - ≈ 1.5 h each at the 3060's ≈ 6.8K sets/h (≈ 6 h in all); the eval box's rate is measured
     by its first A/A. 95% CI ≈ ±0.0065 u per run.

## Decision rule

Δ(run) is the run's mean Δu against CQL model A over the 10,000 sets.

- **Void runs:** a run whose learner crashed or was lost, or whose losses went non-finite, is void.
  A void run can't be chosen. If `e1` is void, it is rerun once if the allowance has room;
  otherwise I ask.
- **Exploration:** choose `e0` or `e2` over `e1` only if its Δ is at least **0.01 u** above
  `e1`'s. If both are, choose the higher. Otherwise keep `e1`.
- **Learning rate:** choose 1e-4 only if Δ(`e1-lr`) is at least 0.01 u above Δ(`e1`) and
  `e1-lr` isn't void. Otherwise keep 2e-5. The chosen learning rate goes with the chosen
  exploration. The interaction between the two isn't tested.
- **Stop:** if the chosen setting's own Δ is significantly negative (its 95% CI is entirely
  below 0), V's runs don't launch. I report it to the user, with the diagnosis order of `05` §8.
- **Why a 0.01 u margin, leaning to the defaults:**
  - it is about twice the standard error of the difference between two runs at 10K sets, and the
    effect size the lab's T-std is built around (its bounds are 0 and 0.01 u);
  - 2 h is ≈ 1/70 of a run, and short sweeps favour settings that move fast early (a higher
    learning rate, more exploration), which may not hold over 6 days.

### Amendment: step-matched comparisons (2026-10-12 11:30 UTC, before any measurement)

The four runs didn't get the same amount of training:
- `e1-lr` lost 38 min: its learner and one actor were stuck loading on Vast for 25 min and were
  replaced. It trains ≈ 75 min to its hard deadline; the others ≈ 1 h 52 min.
- The actor boxes differed. In the first hour, e1 made ≈ 92K games/h; e0 and e2, each with a
  16-core Ryzen 9 5950X actor, ≈ 70–74K/h.

Each run's learner consumes every game once, so its step count measures how much it trained. So:
- e1 is snapshotted at 11:45, 11:55, 12:05 and 12:15 UTC (`fleet.py snapshot --run v-tune-e1
  --label t1145` …). 11:45 matches `e1-lr`'s training time.
- Each other run is compared with the e1 checkpoint (a snapshot or the final one) whose learner
  step count is nearest its own final count. In the rule above, Δ(`e1`) for that run means the
  matched checkpoint's Δ.
- The matched checkpoints are measured like the runs: same command, block and 10,000 sets.
- e1's final checkpoint is measured and reported in any case. The stop rule uses the chosen
  run's final checkpoint.
- If the step counts differ by less than 10%, the final checkpoints are compared as first written.

This adds up to three measures (≈ 1.5 h each on the workstation).

**Clarified and amended 2026-10-12 13:10 UTC, after the runs ended and before any measurement**
(what happened is in "The runs" below):
- "Step count" means online steps: the learner's counter minus the initialization's 156,000.
- A learner that was lost and **resumed from its own checkpoint** doesn't void its run. The run
  is a single training trajectory with fewer steps, and the step matching above covers that. The
  void rule still applies to a run whose final checkpoint can't be fairly compared: one with no e1
  checkpoint within 10% of its online steps.

## Reported, not decisive

- The realized non-greedy rate. From the behaviour sidecars: an action whose behaviour
  probability is below 1 − ε left the argmax.
- `v_qgaps.py` on each run's final checkpoint with `--reference` frozen A: how many decisions'
  argmax moved, and the Q gaps after online learning.
- Training curves: the Q loss (each sample is seen once, so it is out-of-sample), the logged CQL
  term (how far the untried actions' Q rose), policy lag, learner samples/s and steps, cost.
- The trainee's average score in its training games, which includes exploration's cost.

## The runs (2026-10-12)

All from frozen CQL model A (step 156,000), at `afd3c1f`/`ec2059d`, with a 2 h hard deadline that
includes setup. Final checkpoints `workdir/fleet/runs/v-tune-<run>/learner-run/mortal.pth`,
read-only.

| Run | Trained (UTC) | Online steps | Games drained | Cost | Final checkpoint sha256 | Compared with | Status |
|---|---|---|---|---|---|---|---|
| `e0` | 10:45–12:36 | 36,800 | ≈ 121K | $2.73 | `428933a46b91…` | e1 `t1145` (39,200) | valid; learner lost and resumed |
| `e1` | 10:29–12:22 | 59,600 | 203,200 | $2.85 (+ $0.51 failed) | `38c5f105ee1d…` | (the reference) | valid |
| `e2` | 10:45–12:37 | 65,600 | 228,800 | $2.82 | `8a50501711f7…` | e1 final (59,600) | valid |
| `e1-lr` | 11:23–11:54 | 11,200 | 41,600 | $3.21 | `23a5395da246…` | none within 10% | **void** |
| `e1-lr` rerun (D56) | 18:04–19:56 | 57,600 | 220,800 | $2.79 | `9d1db65f4b6e…` | e1 final (within 10%) | valid |
| `e0` extended (D56) | + 10-13 02:10–02:49 | 59,600 | + 81,600 | + $1.09 | `fd08435f4ffe…` (`s215600`) | e1 final (same step) | valid |

e1's snapshots (`workdir/fleet/runs/v-tune-e1/snapshots/`, read-only), by step:
- `t1145` 195,200 (`b44fdc860687…`);
- `t1155` 200,800;
- `t1205` 206,400;
- `t1215` 211,600.

In all: **$12.12, 25.5 box-hours** (the allowance is 24) before the rerun. **With it: $14.91, 31.5
box-hours** (D56). **With e0's extension: $16.00, 33.9 box-hours.**

e0's extension (`workdir/fleet/runs/v-tune-e0-ext/`):
- It resumed from e0's final checkpoint and ran from step 192,800 to 215,600 on 3 boxes.
- The poller snapshotted it at exactly 215,600 (`snapshots/s215600.pth`, read-only) and
  downed the run.
- Losses: DQN 0.025 → 0.022, logged CQL term 1.96 → 1.99, next rank 0.636 → 0.608; all finite.

The rerun's snapshots (`workdir/fleet/runs/v-tune-e1-lr/snapshots/`, read-only) weren't needed:
- `m60` 186,400; `m70` 191,600; `m80` 196,800; `m90` 202,400; `m100` 207,600.
- Its final has 57,600 online steps against e1's 59,600, 3.4% apart, so the finals are compared
  as first written.

What happened:
- **e1's first rental failed** setup on every box (1.1 box-hours, $0.51). Adding A to
  `frozen.sha256` broke a test that pinned the list; fixed in `ec2059d`.
- **e0's learner instance exited** at 11:42 (on Vast's side). The watcher replaced it at 11:53 and
  restored the 11:16 pull, at step 174,000. That lost ≈ 26 min of training, and the run waited
  20 min with the actors blocked. It resumed at 12:02 and ran to 192,800.
- **e1-lr's boxes were destroyed at 12:23 by e1's teardown.** `fleet.py`'s backstop sweep matched
  labels by the prefix `mp-v-tune-e1-`, which also matches `mp-v-tune-e1-lr-…`. Its last pull was
  at 11:54 (step 167,200), so it has 11,200 online steps, and no e1 checkpoint is within 10%. Void:
  the learning rate stays 2e-5 unless it is rerun. Fixed: a box name has no dashes, and the sweep
  matches `mp-<run>-<box>` exactly, with a test. The live P0/P1 runs' names don't prefix one
  another, so they were never exposed.
- Earlier, e1-lr's learner and one actor were stuck loading for 25 min and were replaced
  (≈ 0.9 box-hours).

Training, first logged step to last (online, so each sample is seen once):

| Run | DQN loss | Logged CQL term | Next-rank loss |
|---|---|---|---|
| `e0` | 0.026 (step 174K) → 0.025 | 1.97 → 1.95 | 0.71 → 0.64 |
| `e1` | 0.037 → 0.020 | 0.39 → 2.04 | 0.74 → 0.57 |
| `e2` | → 0.017 | → 2.02 | → 0.51 |
| `e1-lr` | → 0.014 (166,800) | → 2.07 | → 0.36 |
| `e1-lr` rerun | → 0.012 | → 2.10 | → 0.29 |

- **The logged CQL term rose from ≈ 0.4 to ≈ 2.0 in every run, e0 included.** e0 barely explores,
  so the untried actions' Q rise because the CQL penalty is gone, not because exploration
  visits them.
- All losses are finite.

## Timeline

| When (UTC) | What |
|---|---|
| 2026-10-12 09:51 | Model A frozen; step 1 done by ≈ 10:10 |
| 2026-10-12, after step 1 | The sweep: ≈ 2.5 h if the four runs go in parallel |
| 2026-10-13 ≈ 11:00 | Model B finishes, frozen; the evaluation starts (≈ 6 h on the workstation) |
| After the fourth measure | The decision; the chosen values go into `v-a.toml` and `v-b.toml` in one commit |
| Then | V's runs, after their own rental approval (≈ $210 each) |

P0's G0 evaluations need the workstation from 2026-10-16 02:15 UTC, and P1's from 2026-10-17
03:20. The sweep's evaluation finishes before them.
