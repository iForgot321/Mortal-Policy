# Arm V tuning sweep (05 §2.2; written 2026-10-11, before any sweep)

- **Question:** V's exploration (ε and τ) and its learning rate, for both seeds' runs
  (`experiments/configs/v-a.toml`, `v-b.toml`).
- **Status:** step 1 done 2026-10-12 (model A frozen at 09:51 UTC; ε fixed below; the four
  configs committed with this line). The rental (≈ $11–15) was approved by the user on 2026-10-11.
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

## Reported, not decisive

- The realized non-greedy rate. From the behaviour sidecars: an action whose behaviour
  probability is below 1 − ε left the argmax.
- `v_qgaps.py` on each run's final checkpoint with `--reference` frozen A: how many decisions'
  argmax moved, and the Q gaps after online learning.
- Training curves: the Q loss (each sample is seen once, so it is out-of-sample), the logged CQL
  term (how far the untried actions' Q rose), policy lag, learner samples/s and steps, cost.
- The trainee's average score in its training games, which includes exploration's cost.

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
