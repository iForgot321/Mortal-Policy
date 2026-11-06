# Draft: distilling the student candidates (`04` §6), for the user's review

**Status: draft, 2026-11-05.** The third piece of the SL substrate, after the secure data path and
the teachers. It implements `04` §6 with D63's ensemble (BC s1 + AWR s1, EMA weights). It deviates
from D40 in one place, flagged below: the ensemble's targets are computed online, not cached. Nothing
here rents.

## What `04` §6 asks for

- **Candidates:** b8c192, b12c256, b16c256 (`03` §4.1: 3.6M / 6.8M / 8.5M parameters, 82 / 189 / 242M
  MACs per inference).
- **Loss:** KL(π_ensemble ‖ π_student) at T = 1, plus 0.1 × the human action's cross-entropy, plus MSE
  toward the teachers' placement, kyoku-Δ and auxiliary outputs. The students carry every head: the
  critic warm-start (`05` §4) and the belief sampler (`06` S-1) need them.
- **Qualification (D31, D40):** T-NI against the best single teacher (δ = 0.01), which is **BC s1**
  (D63); and golden-set top-1 within 1 pp of that teacher in every category.
- **Selection is by RL curves (D31):** the two smallest qualifying students go through RL on the
  engine (P1 + K2) at 6 and 12 box-days each. Pick by Δu per dollar at 12; if the larger trails at 6
  but leads at 12, pick the larger.
- **The chosen student is π_SL:** the RL initialization, K2's anchor, the permanent pool opponent and
  the riichi.dev incumbent (L1).

## The design: one stream, both teachers online, all three students

Each batch is featurized once and run through both teachers (forward only). All three students then
train on it, each with its own optimizer.

**Measured on the 3060** (synthetic batches of 1024, AMP, `torch.compile`):

| Step | Samples/s |
|---|---|
| the two b24c512 teachers, forward only | 3,055 |
| the three students training | 3,563 |
| **one stream: both teachers + three students** | **1,630** |
| for scale: b24c512 teacher training | 1,938 |

Scaled by the teacher's measured A6000 rate (6,050 samples/s with the real pipeline), one stream runs
at ≈ 5,100 samples/s on an A6000. That is **≈ 128 h and ≈ $65 for all three students**.

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
- **The stream needs no labels:** every target comes from the teachers except the human action. So
  the loader skips the GRP and the opponent replay, a lighter pipeline than the teachers'. The 4
  workers gave 12–17K samples/s on the A6000 even with labels.

## Data and schedule

- **Data:** the same train window and order as the teachers (`TeacherDataset`, seed 1): 1 epoch ×
  m↔p augmentation, **2.34B presentations** (D40's accounting), ≈ 1,142,600 steps at batch 2048.
  - Option: half (1.17B, no augmentation) ≈ 64 h, ≈ $33. Recommended: the full 2.34B, for parity with
    the teachers.
- **Per student:** AdamW (fused, weight decay 0.1 on conv and linear weights), peak LR 3e-4 (the quick
  students' value for these sizes), 5,000 warm-up steps, then cosine to 1e-5. AMP, gradient clip 1.0,
  EMA 0.999.
- **Validation every 20K steps** on the teachers' 200 heldout_game logs, per student:
  - human-move NLL and top-1;
  - KL to π_ens and argmax agreement with the ensemble;
  - each head against the true labels.
- **GPU memory:** 2.3 GiB at batch 1024 on the 3060, so ≈ 5 GiB at 2048: no constraint.

## Hardware and path

One Hyperstack RTX A6000 (D62) through `tools/secure/`:
- the encrypted bundle from the bucket, shipped and unpacked on the box;
- the driver: `experiments/secure_teacher.sh`, generalized to take the trainer script;
- the watchdog, ledger and wipe as before;
- the teachers' EMA weights go to the box as frozen inputs (`tools/fleet/frozen.sha256`; weights, so
  allowed anywhere).

## New code (before any rental)

1. **`mortal/train_students.py`:**
   - the stream from `TeacherDataset` (no labels), with the two frozen, compiled teachers;
   - N students, each with its own optimizer, scheduler, scaler and EMA, all in one checkpoint file;
   - resume, validation and timing as in `train_teacher.py`, plus its lock and its side-stream
     prefetch;
   - at the end, a `final.pth` per student with its EMA weights.
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
- **Then:** the two smallest that qualify go to D31's RL curves (P1 + K2 at 6 and 12 box-days each,
  ≈ 24 box-days on the 4090 fleet). That's a rental with its own pre-registration and the user's yes,
  and its result picks π_SL.
- **If none qualifies:** diagnose before anything else. Candidates: longer distillation, or a larger
  student.

## Cost and time

- Distillation ≈ 128 h, **≈ $65** (A6000); the T-NIs ≈ $5–15; D31's RL curves later, separately.
- Calendar: the code and the smoke run ≈ 1–2 days, the run ≈ 5.4 days, qualification ≈ 1 day.

## Decisions for the user

1. **One stream with online targets (recommended)**, with an `08` note amending D40's
   "ensemble targets cached per shard"; or the cache.
2. **Presentations:** 2.34B (recommended) or 1.17B.
3. **All three candidates in the one run (recommended).** `04` §6's "two smallest that qualify"
   needs to know which ones do, and each extra student costs only its own small share of the stream.

## Not in this draft

D31's RL curves and the choice of π_SL (their own pre-registration); the riichi.dev smoke deploy (L1,
`07` §1); `04` §8's free-gain checks.
