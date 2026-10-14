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
  - **The 4 measures ran 2026-10-13 18:01 – 2026-10-14 01:16 UTC on the workstation** (model B
    ended 11:26 UTC and was frozen at 18:01), one after another, in this order: e1 final, e0
    (extended) `s215600`, e2 final, the e1-lr rerun's final. Each of the last three is compared
    with e1 final. ≈ 1 h 48 min each (≈ 5.6K sets/h).
  - **Result: the stop rule fires.** e1 is kept, but its own Δ is −0.729 u (95% CI entirely
    below 0), so V's runs don't launch. Every run's Q function collapsed ("Results" below).
    Reported to the user 2026-10-14.
    - Block `v-tune`: family A nonces 23,000–32,999, allocated by the first measure.
    - Queue `workdir/lab/v-tune/measures.sh` (not tracked); it logs each start with the
      checkpoint's hash, and each end, to `workdir/lab/v-tune/queue.log`, and stops at the first
      failure. Per-run logs `workdir/lab/v-tune/<run>.log`.
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

## Results (2026-10-14)

Each run's final checkpoint against frozen CQL model A, 10,000 sets on block `v-tune` (family A
nonces 23,000–32,999), AMP, 40,000 games each (`lab/registry/runs.jsonl`):

| Run | Lab run | Online steps | Δu | 95% CI | Avg rank | Avg pt |
|---|---|---|---|---|---|---|
| `e1` | `20261013-110149-measure` | 59,600 | **−0.729** | [−0.736, −0.722] | 3.44 | −73.8 |
| `e0` extended (`s215600`) | `20261013-124951-measure` | 59,600 | −0.874 | [−0.881, −0.867] | 3.59 | −89.0 |
| `e2` | `20261013-143617-measure` | 65,600 | −0.922 | [−0.928, −0.916] | 3.65 | −93.8 |
| `e1-lr` rerun | `20261013-162554-measure` | 57,600 | −1.168 | [−1.172, −1.164] | 3.89 | −120.3 |

- **Decision rule:** neither e0 nor e2 is 0.01 u above e1, and e1-lr isn't either, so e1's
  exploration and 2e-5 are kept.
- **Stop rule: fires.** e1's CI is entirely below 0. Each run's challenger finishes 3rd or 4th
  against three copies of the model it started from, so V's runs don't launch.

**`v_qgaps.py` on each final** (`--reference` frozen A; the same 5 heldout games, 3,096
decisions; `workdir/lab/v-tune/qgaps.txt`), against frozen A from `experiments/v.md`:

| | A | `e1` | `e0` ext | `e2` | `e1-lr` |
|---|---|---|---|---|---|
| argmax = human | 0.806 | 0.381 | 0.351 | 0.316 | 0.214 |
| argmax differs from A | — | 0.613 | 0.639 | 0.686 | 0.785 |
| Q of legal actions, pct 1/50/99 | −11.3 / −6.3 / 0.49 | −0.31 / −0.08 / 0.14 | −0.24 / −0.09 / 0.07 | −0.30 / −0.11 / 0.06 | −0.49 / −0.15 / 0.14 |
| best − second best, median | 1.81 | 0.014 | 0.013 | 0.012 | 0.013 |
| off-argmax share of a draw at τ 0.05 | 0.015 | 0.68 | 0.69 | 0.71 | 0.67 |

- **The Q function collapsed in every run.** The CQL model's spread of values (untried actions
  ≈ 6 u down) is gone: all legal actions' Q sit within ≈ 0.5 u of each other, and the median gap
  from the best to the second best is ≈ 0.013 u. The argmax is close to noise. It matches the
  human's action in 21–38% of decisions (A: 81%).
- **So the exploration targets no longer hold.** ε was set for gaps of ≈ 1.8 u. With gaps of
  ≈ 0.01 u, a Boltzmann draw leaves the argmax ≈ 70% of the time even at τ 0.05, and in e0 too.
  The trainee's training games become close to random play, which is a feedback loop.
- The logged CQL term rising from ≈ 0.4 to ≈ 2.0 in every run (above) was the same collapse
  seen from the training side. e0 was already at 1.97 at step 174K.

**Diagnosis, in `05` §8's order** (units or interface bug → KL too low → LR too high → critic
diverging → masking bug; for a value learner, the critic is the policy):
1. **Units: checked at the config level, clean.** Offline and online use the same `pts`
   ([90, 50, −10, −130], u = pts / 100), the same data loader (`value_targets = True`) and the
   same loss code. The online DQN loss (≈ 0.02) is ≈ 10× lower than the offline one (≈ 0.2), which a
   flat Q fitting lower-variance 1v3 targets could explain, but it isn't checked yet.
2. **The one designed difference: online drops the CQL term** (`min_q_weight = 0`, as Equim's
   online mode does). Without it nothing holds the untried actions down. Gradient flows only
   through the taken action, so the trunk drifts and the rest of the Q head follows. e1-lr's
   larger collapse at 5× the learning rate fits this.
3. **The interface half of step 1: checked 2026-10-14, clean** (the user asked for the checks;
   `experiments/v_online_check.py`, output `workdir/lab/v-tune/online-check.txt`). The script
   reads e1's measure logs (`20261013-110149-measure`; the first 200 logs, which the arena writes as the
   training actors do) through the trainer's own loader (`FileDatasetsIter`, `value_targets`),
   and 100 Tenhou heldout games. FP32, no TF32.

   | Seats | Decisions | Q target sd (u) | mean | e1: argmax = logged | A: argmax = logged | MSE e1 | MSE A |
   |---|---|---|---|---|---|---|---|
   | 1v3 challenger (e1 played) | 29,654 | 0.250 | −0.070 | **0.987** (1.000 where gap > 0.05) | 0.333 | 0.058 | 9.66 |
   | 1v3 champion (A played) | 96,067 | 0.248 | +0.026 | 0.371 | **0.9996** (1.000 where gap > 0.05) | 0.060 | 0.135 |
   | Tenhou heldout (humans) | 62,634 | 0.256 | +0.002 | 0.363 | 0.810 | 0.064 | 0.376 |

   - **Alignment:** each model reproduces its own logged moves. Every miss is at a near-tie
     (gap ≤ 0.05 u), where the lab's AMP and this FP32 pass can flip the argmax. So
     observations, actions and masks line up in the 1v3 path.
   - **Units:** the targets have the same scale on 1v3 seats and on Tenhou (sd ≈ 0.25 u).
   - **The 10× lower online loss** is explained without a bug. The trainer logs ½·MSE. Offline,
     A's loss (0.376 / 2 ≈ 0.19, as logged) is dominated by human moves that CQL pushed ≈ 6 u
     down. On greedy play the taken action is the argmax, and e1's ½·MSE on its own seats is
     ≈ 0.03, as logged online. e1 explains little of the target variance (MSE 0.058 against
     variance 0.062).
   - **A overestimates its own greedy moves:** MSE 0.135 against a target variance of 0.062 on
     the seats it played. Its Q(best) has a median of 0.21 u, against a mean target of +0.03.

**So step 1 of `05` §8 is clean, and the collapse comes from the online design.** The Q head is
dueling: Q = V + A(a) − mean of A over the legal actions (`model.DQN`). Online, only the taken
action has a target, and it is almost always the argmax, whose Q starts too high. Each correction
lowers V and that action's A, and through the mean term raises every other legal action's Q.
The CQL term held the others down offline; online it is dropped (`min_q_weight = 0`), so the
spread shrinks until the argmax is noise. Exploration then leaves the argmax most of the time,
and the trainee learns from near-random play. A hypothesis that fits the measurements, not yet
tested.

### Replay test (2026-10-14, the user asked for it; `experiments/v_replay_test.py`)

- **Setup:** frozen A fine-tuned on fixed data of its own greedy play: the champion seats of
  14,000 logs drawn from the four sweep measures, ≈ 5.1M decisions.
  - The online trainer's step: ½·MSE on the taken action + CQL × weight + 0.2 × next rank; AMP;
    AdamW without weight decay; clipping 0.5; constant lr 2e-5; batch 512; a fresh optimizer.
  - 10,000 steps per arm. Both arms get the same files in the same order (seed 1).
  - Every 1,000 steps a probe on `v_qgaps.py`'s set: the last 5 Tenhou heldout games, 3,096
    decisions with ≥ 2 legal actions.
  - ≈ 31 min per arm on the workstation. Outputs in `workdir/v-replay/mq0`, `mq5` (read-only):
    `probe.tsv`, `run.log`, and `final.pth` (weights only).
- There is no exploration and no feedback: the data never changes, and every taken action is A's
  argmax.

| Steps | weight 0: median gap (u) | median legal Q | argmax = A | off-argmax at τ 0.05 | logged CQL term | weight 5: median gap | median legal Q | argmax = A | off-argmax at τ 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 0 (A) | 1.81 | −6.27 | 1 | 0.015 | — | 1.81 | −6.27 | 1 | 0.015 |
| 1,000 | 0.63 | −2.43 | 0.951 | 0.040 | 0.50 | 7.07 | −13.6 | 0.968 | 0.002 |
| 2,000 | 0.29 | −1.11 | 0.925 | 0.087 | 0.95 | 7.60 | −14.2 | 0.969 | 0.003 |
| 4,000 | 0.11 | −0.43 | 0.876 | 0.205 | 1.48 | 8.34 | −14.9 | 0.970 | 0.003 |
| 6,000 | 0.068 | −0.26 | 0.822 | 0.290 | 1.64 | 8.37 | −15.1 | 0.972 | 0.002 |
| 8,000 | 0.044 | −0.17 | 0.787 | 0.374 | 1.70 | 8.71 | −15.2 | 0.974 | 0.002 |
| 10,000 | **0.034** | −0.13 | 0.735 | 0.437 | 1.74 | **8.83** | −15.3 | 0.972 | 0.002 |

argmax = human: weight 0, 0.806 → 0.668; weight 5, 0.806 → 0.802. ½·MSE: weight 0 ≈ 0.024;
weight 5 ≈ 0.090. The median best legal Q falls from 0.21 to ≈ 0.01–0.06 u in both arms
within 1,000 steps: A's overestimation of its greedy moves is corrected first.

- **The hypothesis holds.** Without the CQL term, the online step alone collapses the Q
  spread, with no exploration and no feedback. In 10,000 steps the median gap falls 1.81 → 0.034
  u, and it is still falling (the online e1 reached 0.013 at 59,600 steps). The logged CQL term
  rises 0.50 → 1.74, the online runs' signature (≈ 2.0). The online runs add exploration's
  feedback on top, from a starting point that already decays.
- **Full-strength CQL (5) does the opposite.** When every taken action is the argmax, the term
  pushes the others down without limit: the median gap widens to 8.8 u and the untried actions
  sink to ≈ −15 u. The argmax stays A's (97%), and exploration at τ 0.05 almost never leaves it
  (0.2%). The model can't learn anything new, and its taken-action fit is worse (½·MSE 0.090
  against 0.024).
- **So:** in `05` §8's terms, the arm's design, not its units, interface or learning rate. V's
  online step needs something between the two. The standard remedies for offline-to-online
  fine-tuning: a small CQL weight that holds the gap steady, calibrated CQL (Cal-QL: the
  push-down floored at a reference value), or mixing offline data with the CQL term into the
  online batches. Any of them changes V's design, so a `08` entry comes first, and V's runs need
  a new approval.

### Replay sweep over the CQL weight (2026-10-14 11:34–13:04 UTC, the user asked for it)

The same replay at weights 0.1, 0.3 and 1: the same data, order and seed as the weight 0 and 5
arms, ≈ 30 min each. Outputs are in `workdir/v-replay/mq0.1`, `mq0.3` and `mq1` (read-only). The
question was whether some weight holds the gap near A's 1.81 u over 10,000 steps.

| Weight | median gap at 1K steps (u) | at 10K | change over the last 3K | median legal Q | argmax = A | argmax = human | off-argmax at τ 0.05 | logged CQL term | ½·MSE |
|---|---|---|---|---|---|---|---|---|---|
| A (step 0) | — | 1.81 | — | −6.27 | 1 | 0.806 | 0.015 | ≈ 0.50 | — |
| 0 | 0.63 | 0.034 | −0.018 | −0.13 | 0.735 | 0.668 | 0.437 | 1.74 | 0.024 |
| 0.1 | 3.19 | 4.31 | +0.20 | −7.71 | 0.969 | 0.804 | 0.008 | 0.151 | 0.030 |
| 0.3 | 4.35 | 5.62 | +0.22 | −9.18 | 0.971 | 0.804 | 0.005 | 0.110 | 0.036 |
| 1 | 5.59 | 6.92 | +0.10 | −11.2 | 0.969 | 0.803 | 0.003 | 0.084 | 0.050 |
| 5 | 7.07 | 8.83 | +0.17 | −15.3 | 0.972 | 0.802 | 0.002 | 0.067 | 0.090 |

- **No weight in the sweep holds the gap.** Every positive weight widens it, more for larger
  weights. It widens most in the first 1,000 steps and is still growing at 10,000. Even 0.1
  more than doubles A's gap (1.81 → 4.31 u). The switch from collapse to widening lies between
  0 and 0.1.
- **Why there's no stable weight on this data:** every taken action is the argmax, so the
  untried actions have no target. Only the CQL push-down acts on them, and nothing stops it. The
  weight sets the speed, not an equilibrium. Online, explored actions get targets, but at
  e1's rate that's ≈ 1 action per game, and a wider gap makes a Boltzmann draw leave the argmax
  less often (0.008 at τ 0.05 for weight 0.1, against A's 0.015).
- **Every CQL arm keeps A's policy.** They change ≈ 3% of A's argmaxes, all within the first
  1,000 steps, with argmax = human unchanged at 0.80. All five arms correct A's overestimate of
  its greedy moves (median best legal Q 0.21 → 0.01–0.03 u).

### Cal-QL and weight 0.03 (2026-10-14 19:34–20:35 UTC, the user asked for them)

Two more replay arms, the same data, order and seed: **Cal-QL** (weight 5, with the push-down
floored at the kyoku's return; `--calql`, added in `79a4f7f`) and plain CQL at
**0.03**. Outputs are in `workdir/v-replay/calql5` and `mq0.03` (read-only). The logged CQL term
is the plain one in every arm.

| Arm | median gap at 1K (u) | at 10K | change over the last 3K | gap, 10th percentile | median legal Q | median best Q | argmax = A | argmax = human | off-argmax at τ 0.05 | ½·MSE |
|---|---|---|---|---|---|---|---|---|---|---|
| A (step 0) | — | 1.81 | — | 0.26 | −6.27 | 0.21 | 1 | 0.806 | 0.015 | — |
| weight 0 | 0.63 | 0.034 | −0.018 | 0.005 | −0.13 | 0.009 | 0.735 | 0.668 | 0.437 | 0.024 |
| **weight 0.03** | 2.08 | **2.94** | +0.26 | **0.25** | **−6.21** | 0.016 | **0.965** | **0.802** | **0.014** | 0.027 |
| weight 0.1 | 3.19 | 4.31 | +0.20 | 0.49 | −7.71 | 0.024 | 0.969 | 0.804 | 0.008 | 0.030 |
| weight 5 | 7.07 | 8.83 | +0.17 | 1.66 | −15.3 | 0.030 | 0.972 | 0.802 | 0.002 | 0.090 |
| Cal-QL 5 | 5.04 | 5.55 | −0.04 | 1.12 | −7.40 | **2.31** | 0.971 | 0.803 | 0.003 | **2.15** |

- **Cal-QL as written fails:** the gap levels off (≈ 5.5 u, flat over the last 6K steps), but
  the taken move's value is inflated about 2.3 u above its return, and the fit to the
  returns is ≈ 100× worse (½·MSE 2.15).
  - Why: once the other actions sit at the floor, the logsumexp no longer falls when they are
    pushed down. The term's only remaining gradient is the push-up on the taken action, held
    back only by the TD loss. At equilibrium, Q(taken) − return ≈ weight × (1 − its softmax
    share), ≈ 1.2 u with 2 legal actions and ≈ 2.5 u with 14.
  - So with a discrete logsumexp, the floor needs a small weight, or a push-down term without
    the matching push-up. That's a new design, not this arm.
- **Weight 0.03 comes closest to holding A.**
  - Over 10K steps it keeps A's argmax (96.5%) and agreement with humans (0.80).
  - It keeps the near-ties as A had them: the 10th-percentile gap is 0.25 against 0.26, and the
    exploration rate at τ 0.05 is 1.4% against 1.5%, flat from step 1K to 10K.
  - It corrects A's overestimate of its greedy move (0.21 → 0.02 u), and the untried actions
    stay where A put them (median −6.2 u).
  - **But the median gap still widens:** 1.81 → 2.94 u, +0.1 u per 1K steps, not slowing.
    The switch from collapse to widening lies between 0 and 0.03.
- **What 10K steps doesn't show:** the online runs took ≈ 60K steps, and this replay's 10K is
  one pass over its 14,000 logs (≈ 5.1M decisions). Whether 0.03's drift levels off, or keeps
  widening until exploration dies (as at 0.1 and above), needs a longer run.

**Proposed next (free; the user's decision): a 60K-step replay at weight 0.03** on 84,000 of
the 160,000 sweep logs. That's one pass with no repeats, about the online runs' length, ≈ 3 h on
the workstation.
- If the median gap levels off and the exploration rate holds near A's: an `08` entry (V's
  online step gets the CQL term at 0.03, not 0), then the user's approval for one short online
  cloud run (≈ $3), measured against A as in the sweep.
- If it keeps widening: weight 0.01 next, or a push-down-only conservative term (the near-ties
  held, no push-up on the taken action).

V stays stopped until then.

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
