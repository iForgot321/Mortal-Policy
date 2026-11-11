# The teachers' exported "EMA weights" were the raw weights (found and fixed 2026-11-10)

**In one line:** `train_teacher.py` wrote the raw weights into `final.pth` twice, once as
`ema_weights`. Every export from it (`ema.pth`), and both lab tests that played those files, used
the raw weights. The real EMA weights survive in each run's last `state.pth` and are now exported.

## How it was found

`tools/eval/validate.py` (new that day) scored BC s1's `final.pth` with `--weights ema` on the
trainer's own 200 validation logs. It gave top-1 81.24%, NLL 0.4770: the trainer's final validation
of the **raw** weights (81.24%, 0.4770), not of the EMA weights (81.34%, 0.4750).

## Evidence

- In `final.pth` of `teacher-bc-s1`, `teacher-awr-s1` and `heads-quick`, every tensor under
  `ema_weights` equals the raw one bit for bit (max difference 0).
- `workdir/teacher/teacher-{bc,awr}-s1/ema.pth` are bit-identical to those raw weights too.

## Cause

`save(final=True)` built the checkpoint dict from `module.state_dict()`, which returns the live
parameter tensors, not copies. It then swapped the EMA in (`Ema.swap` copies in place), took
`state_dict()` again for `ema_weights`, and swapped the raw weights back. Both dict entries pointed
at the same tensors, which held the raw weights again by the time `torch.save` ran.

## What it touched

| Thing | Effect |
|---|---|
| `teacher-{bc,awr}-s1/ema.pth` (`2f200d7c…`, `381ce65f…`) | Hold the raw weights. Kept, read-only: registry rows cite their hashes |
| AWR vs BC T-std (`20261105-105720-T-std`, `experiments/teacher-ab-tstd.md`) | Compared raw with raw. Pre-registered on EMA weights |
| Teacher vs quick T-std (`20261110-145755-T-std`, `experiments/teacher-vs-quick.md`) | BC s1's raw weights played |
| D63 ("the ensemble is BC s1 + AWR s1 (EMA weights)") | The files it names hold raw weights; D63a is open |
| The heads check (`20261030-075954-T-NI`) | None: it judged the raw weights by design |

**Not touched:** the training itself; the raw weights; the trainer's in-run and final validation
figures (it swaps the EMA in, validates and swaps back, which is right); the EMA shadow in every
`state.pth`.

## Recovery

The trainer saves `state.pth` just before `final.pth`, at the same step, with the EMA shadow.
`tools/secure/export_ema.py` now rebuilds the EMA weights from it, in the trainer's parameter order
(`teacher.param_groups`).

| Run | Source (`workdir/secure/runs/<run>/box/<run>/state.pth`, step 1,142,584) | EMA weights | sha256 |
|---|---|---|---|
| BC s1 | `teacher-bc-s1` | `workdir/teacher/teacher-bc-s1/ema-from-state.pth` | `59d9463626125707…` |
| AWR s1 | `teacher-awr-s1` | `workdir/teacher/teacher-awr-s1/ema-from-state.pth` | `75e55855c2effa1c…` |

**Check:** on the trainer's 200 validation logs (133,278 decisions), the rebuilt weights give the
trainer's own final EMA figures.

| Weights | Top-1 | NLL | The trainer's final EMA validation |
|---|---|---|---|
| BC s1, rebuilt EMA | 81.336% | 0.4750 | 81.337%, 0.4750 |
| AWR s1, rebuilt EMA | 80.989% | 0.4863 | 80.989%, 0.4863 |
| BC s1 `final.pth`'s `ema_weights` (the same tensors as `ema.pth`) | 81.243% | 0.4770 | raw: 81.241%, 0.4770 |

The last digits differ by AMP's batch-composition noise.

## Fix

- `teacher.ema_weights` copies the tensors while the EMA is swapped in; `train_teacher.py` uses it.
- `export_ema.py` exports from a `state.pth`'s shadow, and refuses a file whose EMA weights equal
  its raw weights.
- Tests (`tests/test_teacher.py`): the copies are the average and the raw weights return; the CPU
  smoke run's `final.pth` EMA differs from its raw weights and equals what the export rebuilds from
  `state.pth`; the old bug's file is refused.

## Open (the user)

- **D63a:** which weights the teachers are for the distillation and for student qualification. The
  real EMA weights fit human moves better (+0.10 pp top-1 for BC, +0.13 pp for AWR) and are what D63
  meant; the raw weights are what the lab has measured.
- **Whether the two T-stds run again on the real EMA weights** (an eval box, ≈ $0.80 for both). If
  the EMA weights are the teachers, BC s1's is the reference students qualify against, and it has no
  lab row yet.
