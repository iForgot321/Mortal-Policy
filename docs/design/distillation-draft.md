# Draft: distilling the student candidates (`04` §6), for the user's review

**Status: draft, 2026-11-05.** The third piece of the SL substrate, after the secure data path and
the teachers. It implements `04` §6 with D63's ensemble (BC s1 + AWR s1, EMA weights). It deviates
from D40 in one place, flagged below: the ensemble's targets are computed online, not cached. Nothing
here rents.

**Amended 2026-11-10** (the user: the audit's changes, D64–D66): a fourth student, the hard-label
control (D66); the file list shuffled before the workers split it; validation inside the train
window, with heldout_time scored on the workstation; D31's curves with two RL seeds per size (D65),
after D64's pilot. The premise holds: BC s1 beats No-SP A by +0.0219 (`experiments/teacher-vs-quick.md`).

**Found 2026-11-10: the teachers' `ema.pth` files hold the raw weights** (a save bug, fixed:
`experiments/teacher-ema-export.md`). The real EMA weights are `ema-from-state.pth` in each teacher's
directory. Wherever this draft says "EMA weights", which file that means is decision 4 below.

## What `04` §6 asks for

- **Candidates:** b8c192, b12c256, b16c256 (`03` §4.1: 3.6M / 6.8M / 8.5M parameters, 82 / 189 / 242M
  MACs per inference).
- **Loss:** KL(π_ensemble ‖ π_student) at T = 1, plus 0.1 × the human action's cross-entropy, plus MSE
  toward the teachers' placement, kyoku-Δ and auxiliary outputs. The students carry every head: the
  critic warm-start (`05` §4) and the belief sampler (`06` S-1) need them.
- **Qualification (D31, D40):** T-NI against the best single teacher (δ = 0.01), which is **BC s1**
  (D63); and golden-set top-1 within 1 pp of that teacher in every category.
- **A control (D66):** a b8c192 trained in the same stream on the true labels only. It shows what the
  teachers' targets add at student size. It isn't a candidate.
- **Selection is by RL curves (D31, D65):** the two smallest qualifying students go through RL at 6
  and 12 box-days, two RL seeds each, on the recipe D64's pilot selects (not K2's fixed anchor). Pick
  by the mean Δu per dollar at 12; the larger net wins only if its lead is more than the seed spread.
- **The chosen student is π_SL:** the RL initialization, K2's anchor, the permanent pool opponent and
  the riichi.dev incumbent (L1).

## The design: one stream, both teachers online, the three students and the control

Each batch is featurized once and run through both teachers (forward only). The three students and
the control then train on it, each with its own optimizer.

**Measured on the 3060** (synthetic batches of 1024, AMP, `torch.compile`):

| Step | Samples/s |
|---|---|
| the two b24c512 teachers, forward only | 3,055 |
| the three students training | 3,563 |
| **one stream: both teachers + three students** | **1,630** |
| for scale: b24c512 teacher training | 1,938 |

Scaled by the teacher's measured A6000 rate (6,050 samples/s with the real pipeline), one stream runs
at ≈ 5,100 samples/s on an A6000. That is **≈ 128 h and ≈ $65 for all three students**.

**With the control:** one more b8c192 trains. A b8c192 is ≈ 16% of the three students' step (82 of
513M MACs), so the stream slows by ≈ 7%: **≈ 137 h, ≈ $70**. An estimate from the timings above; the
smoke run measures it (D66: above 15%, the control is reconsidered).

**The alternatives, costed the same way:**

| Option | Time on an A6000 | Notes |
|---|---|---|
| **One stream, online targets (recommended)** | **≈ 128 h** | no cache, one run |
| Cache the ensemble's outputs, then one student stream | ≈ 68 h + ≈ 59 h ≈ 127 h | ≈ 0.7 TB of cache (FP16, all heads) on the secure tier; only pays off if students are distilled again later |
| Three separate runs, online targets | ≈ 225 h | each run repeats the teachers' forward passes |

**Deviation from D40.** D40 says "ensemble targets: cached per shard". That avoided two teacher
passes per state per epoch, when one run per student was assumed. With one stream for all students
and one epoch, each state passes through the teachers exactly once either way, so a cache saves no
compute and adds ≈ 0.7 TB of secure storage. An `08` note amends that line if the user agrees.

**The cost of one run:** a failure stops all three students. Resume from the last checkpoint covers it
(`TeacherDataset`'s cursor, as the teachers used). It's also a clean comparison: the three students
see identical batches.

## Targets and loss (per student)

- **π_ens:** the mean of the two teachers' masked softmax probabilities (T = 1).
- **Heads:** the targets are the two teachers' mean outputs, as probabilities (placement, tenpai,
  waits) and values (kyoku-Δ).
- **L = KL(π_ens ‖ π_s) + 0.1·CE(human action) + 0.5·MSE(placement) + 0.5·MSE(kyoku-Δ)
  + 0.2·[MSE(tenpai) + MSE(waits)].**
  - The waits term is weighted by the ensemble's tenpai probability for each opponent, as the
    teacher's waits loss counted only tenpai opponents.
  - The head weights follow the teacher's loss (`04` §5); `04` §6 names the terms but not their
    weights.
- **The control's loss** is the teachers' (`teacher.teacher_loss`, BC): the human action's
  cross-entropy plus the heads against the true labels (final placement, the kyoku's Δ, opponents'
  tenpai and waits), with `04` §5's weights. No teacher term.
- **The stream carries the labels, for the control:** the students' targets come from the teachers
  except the human action, but the control's heads need the GRP and the opponent replay. That is the
  teachers' pipeline: its 4 workers gave 12–17K samples/s on the A6000, above the ≈ 5K the stream needs.

## Data and schedule

- **Data:** the same train window as the teachers (`TeacherDataset`, seed 1): 1 epoch × m↔p
  augmentation, **2.34B presentations** (D40's accounting), ≈ 1,142,600 steps at batch 2048.
  - **Not the teachers' order:** `file_order = 'shuffled'` (the trainer's default since 2026-11-10).
    The teachers' workers each held one ≈ 2.6-year slice of the date-sorted list, so every batch was
    one era, and their last ≈ 9.5K steps had no games from 2022-11 on (that worker ran out first).
    The list is now shuffled by the seed before the workers split it.
  - Option: half (1.17B, no augmentation) ≈ 64 h, ≈ $33. Recommended: the full 2.34B, for parity with
    the teachers.
- **Per student:** AdamW (fused, weight decay 0.1 on conv and linear weights), peak LR 3e-4 (the quick
  students' value for these sizes), 5,000 warm-up steps, then cosine to 1e-5. AMP, gradient clip 1.0,
  EMA 0.999.
- **Validation every 20K steps** on 200 heldout_game logs inside the train window
  (`val_start = '2015-01-01'`; 47 of the teachers' 200 were from 2009–2014), per student:
  - human-move NLL and top-1;
  - KL to π_ens and argmax agreement with the ensemble;
  - each head against the true labels.
- **heldout_time** (2025-07 on) isn't in the bundle. Each student's final weights are scored on it on
  the workstation after the run (`tools/eval/validate.py`, 200 logs outside the golden set), beside
  the teachers' figures (`experiments/validation-samples.md`).
- **GPU memory:** 2.3 GiB at batch 1024 on the 3060, so ≈ 5 GiB at 2048: no constraint.

## Hardware and path

One Hyperstack RTX A6000 (D62) through `tools/secure/`:
- the encrypted bundle from the bucket, shipped and unpacked on the box;
- the driver: `experiments/secure_teacher.sh`, generalized to take the trainer script;
- the watchdog, ledger and wipe as before;
- the teachers' EMA weights go to the box as frozen inputs (`tools/fleet/frozen.sha256`; weights, so
  allowed anywhere).

## New code (before any rental)

**Built 2026-11-11** (`experiments/students-s1.md` has the run's pre-registration): `mortal/train_students.py`
and `mortal/distill.py` (`distill_loss` and the ensemble there, not in `teacher.py`), `tests/test_students.py`,
`tools/eval/golden_agreement.py`, `experiments/configs/students-s1.toml` and `students-smoke.toml`; the
driver takes `TRAINER`, `INPUTS` and `MIN_RATE`. The teachers' weights ship with `secure.py setup --input`
rather than through `frozen.sha256`, which every Vast box would also receive (≈ 340 MB each).

1. **`mortal/train_students.py`:**
   - the stream from `TeacherDataset` (no labels), with the two frozen, compiled teachers;
   - N students, each with its own optimizer, scheduler, scaler and EMA, all in one checkpoint file;
   - resume, validation and timing as in `train_teacher.py`, plus its lock and its side-stream
     prefetch;
   - at the end, a `final.pth` per student with its EMA weights;
   - the control as a fourth student whose loss is `teacher_loss` on the labels.
2. **`mortal/teacher.py`:** `distill_loss()`, with tests: KL is 0 when the student matches, masked
   actions don't count, the waits weighting.
3. **`tools/eval/golden_agreement.py`** (workstation only: the golden set is heldout_time Tenhou
   data):
   - for a checkpoint, top-1, top-3 and the human action's mean log-likelihood per category, over the
     120,000-decision golden set (`~/mj-data/splits/golden.parquet`, 6 categories × 20K);
   - it also gives `04` §9's "golden-set per-category table logged" for the teachers.
4. **`experiments/configs/students-s1.toml`;** the teachers' EMA files added to `frozen.sha256`.
5. **A smoke run on the workstation** with the real teachers on a small index: the rate, memory and
   falling losses.

## Qualification and selection (pre-registered before the students exist)

- **For each student's EMA weights:**
  - T-NI against BC s1's EMA weights (δ = 0.01, family B), on the Vast eval box (weights only; up to
    60K sets each, a few dollars in all);
  - golden-set top-1 within 1 pp of BC s1 in every category, on the workstation.
- **The control's two readings (D66), exploratory:**
  - the distilled b8c192 against the control, T-std (family B): does distillation add strength at
    this size?
  - the control against No-SP A, a 20K-set measure: what the full data adds to a b8c192 without a
    teacher. With BC s1's +0.0219 over No-SP A, that separates the teacher's size from its data.
- **Then:** the two smallest that qualify go to D31's RL curves: 6 and 12 box-days, two RL seeds each
  (D65), ≈ 48 box-days on the 4090 fleet, on the recipe D64's pilot selects. That's a rental with its
  own pre-registration and the user's yes, and its result picks π_SL.
- **If none qualifies:** diagnose before anything else. Candidates: longer distillation, or a larger
  student.

## Cost and time

- Distillation ≈ 137 h, **≈ $70** (A6000) with the control; the T-NIs and the control's readings
  ≈ $5–20; D31's RL curves later, separately (≈ $630–690, D65).
- Calendar: the code and the smoke run ≈ 1–2 days, the run ≈ 5.7 days, qualification ≈ 1 day.

## Decisions for the user

1. **One stream with online targets (recommended)**, with an `08` note amending D40's
   "ensemble targets cached per shard"; or the cache.
2. **Presentations:** 2.34B (recommended) or 1.17B.
3. **All three candidates in the one run (recommended).** `04` §6's "two smallest that qualify"
   needs to know which ones do, and each extra student costs only its own small share of the stream.

4. **The teachers' weights (D63a).** The real EMA weights (`ema-from-state.pth`: the better
   human-move fit, +0.10 to +0.13 pp top-1, but never played in the lab), or the raw weights
   (`ema.pth` as exported: what both teacher T-stds measured). Recommended: the real EMA weights, as
   D63 intended, with the teacher-vs-quick T-std run again on BC s1's (≈ $0.30) so the reference the
   students qualify against has a lab row.

Settled 2026-11-10 (the user): the control rides in the same run (D66); the shuffled file order and
the in-window validation are the trainer's settings for it.

## Not in this draft

D31's RL curves and the choice of π_SL (their own pre-registration); the riichi.dev smoke deploy (L1,
`07` §1); `04` §8's free-gain checks.
