# Validation samples: in-window and held-out-by-time (2026-11-10)

**Exploratory** (CLAUDE.md rule 1): human-move agreement is a diagnostic, not a strength verdict.
No lab nonces, no registry rows. The audit of 2026-11-09 found that the trainer's 200 validation
logs span 2009–2025 while training starts in 2015, and that `heldout_time` was never scored.

## What changed

- `train_teacher.py` takes `[dataset] val_start`: only logs from that day on enter its sample.
- `tools/eval/validate.py` scores any checkpoint on a fixed sample of `heldout_game` or
  `heldout_time`, with the trainer's own validation pass (`teacher.validate`). Workstation only:
  `heldout_time` isn't in the secure bundle. Its sample leaves out the golden set's 12,122 logs.

```
python tools/eval/validate.py --split heldout_game                       CKPT ...   # the teachers' sample
python tools/eval/validate.py --split heldout_game --start 2015-01-01    CKPT ...   # in-window
python tools/eval/validate.py --split heldout_game --end 2014-12-31      CKPT ...   # before the window
python tools/eval/validate.py --split heldout_time                       CKPT ...
```

## Results: policy top-1 / NLL, 200 logs each

| Sample | Decisions | BC s1, raw | BC s1, EMA | AWR s1, raw | AWR s1, EMA | Heads-quick (b8c192), raw |
|---|---|---|---|---|---|---|
| The teachers' sample: `heldout_game`, 2009-02 → 2025-05 (47 logs before 2015) | 133,278 | 81.24% / 0.4770 | 81.34% / 0.4750 | 80.86% / 0.4890 | 80.99% / 0.4863 | 80.41% / 0.5060 |
| **In-window:** `heldout_game` from 2015-01-01 | 128,821 | 81.59% / 0.4721 | 81.70% / 0.4689 | 81.21% / 0.4830 | 81.37% / 0.4790 | 80.63% / 0.4993 |
| Before the window: `heldout_game` to 2014-12-31 | 135,203 | 80.78% / 0.4905 | n/a | 80.33% / 0.5028 | n/a | 79.57% / 0.5246 |
| **`heldout_time`:** 2025-07 → 2026-01, outside the golden set | 128,393 | 82.58% / 0.4482 | 82.72% / 0.4450 | 82.24% / 0.4589 | 82.43% / 0.4547 | 81.69% / 0.4754 |

Raw weights: each run's `final.pth`. EMA weights: `ema-from-state.pth`
(`experiments/teacher-ema-export.md`; the first row reproduces the trainer's own final validation,
raw and EMA). Heads-quick trained on 500K games of 2020–2025.

## Reading

- **The old sample understated the fit inside the window by ≈ 0.35 pp** (BC raw: 81.24% → 81.59%).
  Games before 2015 are ≈ 0.8 pp harder to predict for a model trained on 2015 on.
- **No loss going forward in time.** On `heldout_time`, half a year past the last training game,
  every model predicts better than inside the window (BC raw 82.58% against 81.59%). The newest games
  are the most predictable, which fits the teachers' training loss by era (the audit).
- **The comparisons between models don't depend on the sample.** BC over AWR: 0.29–0.45 pp on every
  row. The teacher over the quick student: 0.8–1.2 pp on every row. EMA over raw: 0.10–0.19 pp.
- **For the next runs:** `val_start = '2015-01-01'` in the trainer's config, and `heldout_time`
  scored on the workstation after the run, as the distillation draft now says.
