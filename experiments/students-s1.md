# The student distillation, `students-s1` (04 §6, D66, D67): pre-registered 2026-11-11

Written before the students exist. The run, its launch rules and the qualification below are fixed;
the results are added under them and nothing above "Results" changes once the run starts.

## The run

- **Config:** `experiments/configs/students-s1.toml`; trainer `mortal/train_students.py` (losses in
  `mortal/distill.py`, tests in `tests/test_students.py`).
- **Teachers (frozen, D67):** the real EMA weights of BC s1 (`workdir/teacher/teacher-bc-s1/ema-from-state.pth`,
  `59d94636…`) and AWR s1 (`…/teacher-awr-s1/ema-from-state.pth`, `75e55855…`). π_ens is the mean of
  their masked softmax probabilities at T = 1; the trainer refuses files that don't match the hashes.
- **Students:** b8c192, b12c256, b16c256 (KL to π_ens + 0.1 × the human move's CE + the heads toward
  the ensemble's), and the control `control-b8c192` (D66: the teachers' loss on the true labels, no
  teacher term). The control and the distilled b8c192 start from the same initialization.
- **Data:** the teachers' train window (`train_E ∪ train_O`, 1,805,376 games), one epoch × m↔p
  augmentation, 2.34B presentations at batch 2048, the file list shuffled by seed 1 before the
  loader's 4 workers split it.
- **Optimizer, per student:** AdamW (weight decay 0.1 on conv and linear weights), peak LR 3e-4 after
  5,000 warm-up steps, cosine to 1e-5, gradient clip 1.0, AMP, EMA 0.999.
- **Validation every 20,000 steps:** 200 heldout_game logs from 2015 on. Per student, raw and EMA: the
  human move's NLL and top-1, each head against the labels, KL to π_ens, argmax agreement with π_ens.
- **Where:** one Hyperstack RTX A6000 (D62) through `tools/secure/` and
  `experiments/secure_teacher.sh` with `TRAINER=train_students.py`. Outputs per student:
  `ema-<name>.pth` and `raw-<name>.pth`, lab-ready and read-only.

## The smoke run (workstation), before the rental

`experiments/configs/students-smoke.toml`: the same trainer, teachers, students and batch, on 3,000
train files for 1,500 steps, 2 loader workers, 20 validation logs.

**It passes if all hold:**
1. it runs to its end and writes every student's files; its validation pass completes;
2. every student's training loss falls: the mean of its last 300 steps is below that of steps 100–400;
3. GPU memory peaks below 10 GB;
4. the control's share of the GPU time is at most 15% (D66's reopen condition).

**Projection for the A6000:** the 3060's samples/s (median of the logged lines after the first 3) ×
3.12, the teachers' measured A6000 / 3060 ratio (6,050 / 1,938 samples/s). The run is launched if the
projection finishes 1,142,600 steps in at most 165 h. Otherwise the result goes to the user.

## Launch (approved, D67)

- `secure.py up` quotes the A6000. The watchdog's deadline is the $90 stop at that price
  (`floor(90 / price)` hours, ≈ 176 h at $0.509/h).
- `MIN_RATE = 3850` samples/s: below it, 2.34B presentations don't fit in 170 h. The driver checks
  the median after the trainer's first 3 timing lines; a slower box is pulled and downed (≈ $1).
- The teachers' weights ship with `--input`; the trainer checks their hashes on the box.

## Qualification (04 §6, D31, D40, D67): fixed now

For each candidate's EMA weights (`ema-b8c192.pth`, `ema-b12c256.pth`, `ema-b16c256.pth`):

1. **T-NI against BC s1** (its real EMA weights, `ema-from-state.pth`): δ = 0.01, family B, the
   calibrated GSPRT of `lab/lab.toml`, cap 60,000 sets. Hitting the cap counts as a fail.
2. **Golden-set top-1 within 1 pp of BC s1 in every category** (`tools/eval/golden_agreement.py`,
   the six categories of `04` §4, BC s1's EMA weights as the reference), on the workstation.

**A candidate qualifies if it passes both.** The two smallest qualifying candidates go to D31's RL
curves (two seeds per size, D65), on the recipe D64's pilot selects; that is its own
pre-registration and rental. **If none qualifies:** diagnosis first (longer distillation, a larger
student); nothing goes to RL.

The T-NIs run on a Vast eval box (weights only). The user approved it in advance (2026-11-11): it is
rented once the students' files are pulled, and downed when its queue drains.

## The control's readings (D66), exploratory

On the same eval box, after the T-NIs:

1. **Distilled b8c192 against the control** (EMA weights both), T-std, family B: does distillation
   add strength at b8c192? Pass: yes, by the lab's δ. Fail or cap: not measurably.
2. **The control against No-SP A**, a 20,000-set fixed-N measure, family A: what the full data adds
   to a b8c192 trained without a teacher (No-SP A: the same shape on 500K games).
3. **BC s1's EMA weights against No-SP A**, T-std, family B: the reference's own lab row. The
   earlier one (`20261110-145755-T-std`) played BC s1's raw weights.

## Reported, not judged

- Every student's final validation, in-window and on heldout_time (`tools/eval/validate.py`), beside
  the teachers' (`experiments/validation-samples.md`).
- Golden-set top-1, top-3 and log-likelihood per category for every student and both teachers.
- The run's cost, hours, samples/s and each part's share of the GPU time.

## Results

### The golden-set reference (2026-11-11, workstation, before any student exists)

`tools/eval/golden_agreement.py` on the full golden set (120,000 decisions, 20,000 per category; AMP;
`workdir/eval/golden-teachers.json`), top-1:

| Category | BC s1 (EMA, the reference) | AWR s1 (EMA) | Heads-quick (b8c192, 500K games, raw) | Heads-quick − BC s1 |
|---|---|---|---|---|
| routine | 85.50% | 84.97% | 84.39% | −1.11 pp |
| riichi_dama | 84.20% | 83.58% | 81.68% | −2.52 pp |
| call_pass | 94.88% | 94.67% | 94.24% | −0.64 pp |
| push_fold | 80.27% | 79.70% | 78.59% | −1.68 pp |
| oorasu | 78.29% | 77.45% | 76.37% | −1.92 pp |
| early | 78.11% | 77.82% | 77.29% | −0.82 pp |
| all | 83.54% | 83.03% | 82.09% | −1.45 pp |

So a student qualifies on the golden set only with top-1 ≥ BC s1's − 1 pp in each row: 84.50%,
83.20%, 93.88%, 79.27%, 77.29% and 77.11%. A b8c192 trained directly on 500K games misses four of the
six; riichi-vs-dama is where the size and data show most.

### The smoke run (2026-11-11 08:40–09:16 UTC, the workstation): passes

`workdir/students-smoke/run/` (train.log, timing.jsonl). 1,500 steps, 3.07M presentations, 36 min.

| Criterion | Measured | |
|---|---|---|
| 1. Runs to its end, writes every student's files, validation completes | `final.pth`, `ema-`/`raw-` for all 4; validation at steps 1,000 and 1,500 | pass |
| 2. Every training loss falls (last 300 steps vs steps 101–400) | KL to π_ens: b8c192 0.503 → 0.149, b12c256 0.447 → 0.130, b16c256 0.434 → 0.128; the control's CE 1.109 → 0.679 | pass |
| 3. GPU memory below 10 GB | 4.2 GB peak | pass |
| 4. The control's share of the GPU time ≤ 15% | 7.8% (teachers 50.7%, b8c192 7.8%, b12c256 14.7%, b16c256 19.0%) | pass |

- **Rate:** 1,525 samples/s on the 3060 (median after the first 3 lines; data wait 0.0%, GPU-bound),
  the draft's estimate with the control (1,531). **Projection: × 3.12 = 4,758 samples/s, 136.6 h** for
  1,142,600 steps: inside the 165 h limit, so the run launches.
- **Validation at step 1,500** (20 logs, EMA weights, exploratory): top-1 b8c192 73.7%, b12c256 74.9%,
  b16c256 74.9%, control 72.5%; the ensemble 81.6%. KL to π_ens 0.246 / 0.214 / 0.216 / 0.291. After
  1,500 of 1.14M steps this says only that the pipeline learns.
- RAM: the main process 3.0 GB and each of the 2 workers ≈ 3.5 GB (PSS); 2 GB stayed available.
- `tools/eval/golden_agreement.py --limit 30`: all 280 decisions matched the golden set's labels and
  masks.

### The run's end (2026-11-17)

Done at step 1,142,584 (2,340,012,032 presentations) at 09:14 UTC; the driver pulled, wiped and
destroyed VM 1154755 at 09:25 UTC after 143.37 h, **≈ $72.65** (the watchdog's stop was $89.7).
The outputs are in `workdir/secure/runs/students-s1/box/students-s1/`, read-only. Each student's EMA
weights differ from its raw weights (max |EMA − raw| 0.0010–0.0014 over the trunk), so the
teachers' save bug didn't recur.

| Student | EMA weights for the lab | sha256 |
|---|---|---|
| b8c192 | `ema-b8c192.pth` | `6fd237ae66b5530c…` |
| b12c256 | `ema-b12c256.pth` | `8f8031f44aa95479…` |
| b16c256 | `ema-b16c256.pth` | `8aedb80b92390df3…` |
| control (b8c192, hard labels) | `ema-control-b8c192.pth` | `529ff579dc5eeb51…` |

### Qualification launch (2026-11-17; the user: "Start it")

- **The lab half** (`experiments/students_qual_chain.sh`, log `workdir/fleet/runs/eval/students-qual.log`):
  a Vast eval box at ≤ $0.75/h, the six jobs in the order fixed above (the three T-NIs, then the
  control's T-std, measure and T-std), pulled hourly, downed when the queue drains. Worst case,
  every test to its cap, is 400K sets (≈ 40 h, ≈ $30), which is also the forced-down ceiling; the
  GSPRTs usually stop far sooner (the teacher T-stds stopped at 5,000–6,500 sets).
- **The golden-set half** (the workstation, `tools/eval/golden_agreement.py`, AMP as for the
  reference): the four students on the full golden set → `workdir/eval/golden-students.json`,
  judged against BC s1's row in `workdir/eval/golden-teachers.json` (above).
- **Reported, not judged:** `tools/eval/validate.py` on every student and both teachers' EMA weights,
  in-window (`heldout_game --start 2015-01-01`) and on `heldout_time`.
- **First launch, aborted (2026-11-17 17:49–17:58 UTC, ≈ $0.07):** the box (machine 41250) set up,
  then the first job's upload died on a dropped connection, and the chain downed it as designed. No
  job ran. The chain now tries each submit 3 times, 3 min apart, and the machine is marked bad for
  the eval box. Relaunched at once.
- **The golden set ran (17:50–17:51 UTC);** its reading follows below with the T-NIs. The validation
  reports failed on a `tools/eval/validate.py` bug: a student's file keeps only its heads' shape
  under `[teacher]`, not the loss weights (`KeyError: 'huber_delta'`). Fixed (`loss_config`,
  `tests/test_validate.py`) and rerun.

### The golden-set half (2026-11-17): all three students pass

`tools/eval/golden_agreement.py` on the full golden set (120,000 decisions, AMP, EMA weights;
`workdir/eval/golden-students.json`), top-1 against BC s1's reference row, in pp. The rule: no
category more than 1 pp below BC s1.

| Category | BC s1 | b8c192 | b12c256 | b16c256 | control b8c192 (not a candidate) |
|---|---|---|---|---|---|
| routine | 85.50% | 85.08% (−0.43) | 85.41% (−0.10) | 85.38% (−0.13) | 85.04% (−0.46) |
| riichi_dama | 84.20% | 83.86% (−0.34) | 84.17% (−0.03) | 84.19% (−0.02) | 83.05% (**−1.16**) |
| call_pass | 94.88% | 94.77% (−0.11) | 94.90% (+0.02) | 94.86% (−0.02) | 94.68% (−0.20) |
| push_fold | 80.27% | 79.92% (−0.35) | 80.14% (−0.13) | 80.05% (−0.22) | 79.83% (−0.45) |
| oorasu | 78.29% | 77.64% (−0.64) | 77.79% (−0.50) | 77.80% (−0.49) | 77.36% (−0.92) |
| early | 78.11% | 77.98% (−0.13) | 78.20% (+0.09) | 78.11% (+0.00) | 77.84% (−0.27) |
| all | 83.54% | 83.21% (−0.34) | 83.43% (−0.11) | 83.40% (−0.15) | 82.97% (−0.58) |

- **b8c192, b12c256 and b16c256 pass the golden half;** their qualification now rests on the T-NIs.
  Their worst category is oorasu (−0.5 to −0.6 pp).
- The control would miss on riichi_dama (−1.16 pp) and is the weakest of the four in every row.
  At b8c192, the teachers' targets add 0.2–0.8 pp of human agreement over training on the labels
  (exploratory; the control's lab readings follow).
- b12c256 and b16c256 are level with each other, and within 0.15 pp of BC s1 overall, at 1/6 and
  1/5 of its parameters.

### Validation (reported, not judged; 2026-11-17)

`tools/eval/validate.py`, EMA weights, AMP, 200 logs each (`workdir/eval/validate-students-*.json`):

| | In-window (heldout_game from 2015; 128,821 decisions) top-1 / NLL | heldout_time (2025-07 on; 128,393) top-1 / NLL |
|---|---|---|
| BC s1 | 81.70% / 0.4689 | 82.72% / 0.4450 |
| AWR s1 | 81.37% / 0.4790 | 82.43% / 0.4547 |
| b8c192 | 81.44% / 0.4748 | 82.56% / 0.4511 |
| b12c256 | 81.56% / 0.4715 | 82.64% / 0.4478 |
| b16c256 | 81.60% / 0.4710 | 82.62% / 0.4471 |
| control b8c192 | 81.33% / 0.4796 | 82.37% / 0.4554 |

The order matches the golden set's: every student above AWR s1 and below BC s1 (within 0.1–0.3 pp),
the control last. The heads' losses are within 0.5% of BC s1's for placement and kyoku-Δ, and
within 4% for tenpai and waits. heldout_time scores higher for every model; it is a different sample of players and
dates, not an easier version of the same one.
- **Second launch, failed in setup (2026-11-17 18:00–18:32 UTC, ≈ $0.55 with the first):** two boxes
  in a row failed `setup_box.sh`'s test suite on `tests/test_value.py` (a workstation path to the
  GRP), and a third host was offline; no job ran. The cause was the new `tests/test_validate.py`:
  importing `validate.py` at collection left its workstation `MORTAL_CFG` in the environment, which
  every later test then used. It passed here, where those paths exist. Fixed (the test restores the
  variable; a test checks it), the suite passes with no inherited config (205 passed, 5 skipped),
  and the chain was relaunched. `fleet.py`'s provisioning retries also took a $1.038/h host after
  the first setup failure, above the chain's $0.75/h limit, for 25 min.

### The lab half and the qualification (2026-11-17): b12c256 and b16c256 qualify; b8c192 doesn't

Eval box 61110696 (Threadripper PRO 5995WX, $0.511/h), 18:45 → 22:32 UTC; pulled and destroyed by
the chain at 22:43 UTC. **≈ $2.66** with the three failed attempts. Family B (T-NIs and T-stds, the
calibrated GSPRT) and A (the measure); AMP; EMA weights throughout. A sequential test's mean is
biased away from the bound it didn't cross, and the ≈ 95% intervals below are mean ± 1.96 · sd/√n,
not part of the test.

| Test | Challenger vs champion | Verdict | Sets | Δu | ≈ 95% | Run |
|---|---|---|---|---|---|---|
| T-NI (δ 0.01) | b8c192 vs BC s1 | **fail** | 12,500 | −0.0074 | [−0.0130, −0.0019] | `20261117-184513-T-NI` |
| T-NI | b12c256 vs BC s1 | **pass** | 5,000 | +0.0047 | [−0.0037, +0.0131] | `20261117-193933-T-NI` |
| T-NI | b16c256 vs BC s1 | **pass** | 10,000 | −0.0021 | [−0.0080, +0.0037] | `20261117-200117-T-NI` |
| T-std | distilled b8c192 vs the control | **pass** | 14,000 | +0.0075 | [+0.0021, +0.0129] | `20261117-204544-T-std` |
| measure | the control vs No-SP A | measured | 20,000 | +0.0140 | [+0.0093, +0.0186] (bootstrap) | `20261117-212358-measure` |
| T-std | BC s1 (EMA) vs No-SP A | **pass** | 5,000 | +0.0226 | [+0.0131, +0.0321] | `20261117-221715-T-std` |

**Qualification, as fixed:**
- **b12c256 qualifies** (T-NI pass, golden set within 1 pp everywhere).
- **b16c256 qualifies** (the same).
- **b8c192 doesn't:** its T-NI crossed the lower bound (LLR −3.07 against −2.944), so it is about
  0.007 u worse than BC s1, although its golden-set agreement was within 1 pp.
- **The two smallest qualifying candidates, b12c256 and b16c256, go to D31's RL curves** (two seeds
  per size, D65), on the recipe the anchor-release pilot selected. That is its own
  pre-registration and rental.

**The control's readings (D66, exploratory):**
1. **Distillation adds strength at b8c192:** the distilled b8c192 beats the hard-label control on
   the same data and schedule (T-std pass, ≈ +0.0075). The golden set showed the same order.
2. **The full data adds to a b8c192 trained without a teacher:** the control beats No-SP A (500K
   games) by +0.0140.
3. **BC s1's own row on its EMA weights:** +0.0226 against No-SP A, as on the raw weights (+0.0219,
   `20261110-145755-T-std`).

**Beyond the rule:**
- Size matters between b8c192 and b12c256 and not visibly beyond: b12c256 and b16c256 are level
  with BC s1 in play, as on the golden set, at 1/6 and 1/5 of its parameters.
- Golden-set agreement within 1 pp did not imply equal strength: b8c192 was within 0.65 pp
  everywhere and still lost ≈ 0.007 u in play. The T-NI carried the qualification.
