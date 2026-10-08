# Ablation table

Generated from `runs.jsonl` by `python -m lab table`; do not edit by hand. Δ avg rank is relative to 2.5. Rows with Calibrated = NO used uncalibrated boundaries and do not count toward any rung (02 §4.1).

| Run | Mode | Description | Test | Family (nonces) | Sets | Δu | 95% CI | LLR | Verdict | Δ avg rank | avg pt | Calibrated | Commit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261003-014550-A/A | exploratory | deterministic A/A harness check | A/A | A [0, +500) | 500 | -0.0000 |  |  | FAIL | +0.0000 | +0.00 | NO | a0b6e5f2+dirty |
| 20261003-014642-A/A | exploratory | deterministic A/A harness check | A/A | A [0, +500) | 500 | +0.0000 |  |  | pass | +0.0000 | +0.00 | NO | a0b6e5f2+dirty |
| 20261003-014722-A/A | exploratory | deterministic A/A, phase1-v0 b40c192 (v1-era checkpoint, harness check only) | A/A | A [500, +500) | 500 | -0.0001 |  |  | FAIL | -0.0010 | -0.04 | NO | a0b6e5f2+dirty |
| 20261003-020344-AA | exploratory | deterministic A/A, FP32, phase1-v0 b40c192 (v1-era checkpoint, harness check only) | A/A | A [500, +500) | 500 | +0.0000 |  |  | FAIL | +0.0000 | +0.00 | NO | a0b6e5f2+dirty |
| 20261003-022054-AA | exploratory | deterministic A/A, FP32 without TF32, phase1-v0 b40c192 (harness check only) | A/A | A [500, +500) | 500 | +0.0000 |  |  | pass | +0.0000 | +0.00 | NO | a0b6e5f2+dirty |
| 20261005-124255-measure | exploratory | A/A': pilot quick students A (seed 1) vs B (seed 2), same recipe | measure | A [1000, +20000) | 20000 | +0.0015 | [-0.0031, +0.0061] |  | measured | -0.0019 | +0.16 | NO | a0b6e5f2+dirty |
| 20261005-163506-T-std | exploratory | L0 synthetic hobbled patch: pilot A with eps=0.1 random actions vs pilot A; must FAIL (02 §6) | T-std | B [1000000000, +100000) | 5000 | -0.3957 |  | -132.81 | fail | +0.5372 | -39.70 | yes | a0b6e5f2+dirty |
| 20261005-173018-T-NI | exploratory | L0 A/A' pair: pilot A vs pilot B, same recipe; must pass or be inconclusive (02 §6) | T-NI | B [1000100000, +60000) | 6000 | +0.0010 |  | +3.28 | pass | -0.0001 | +0.10 | yes | a0b6e5f2+dirty |
| 20261007-064005-T-NI | exploratory | SP ablation pair 1: v5 (no SP) seed 1 vs pilot A (v4) seed 1, matched recipe | T-NI | B [1000160000, +60000) | 10000 | -0.0007 |  | +3.66 | pass | +0.0003 | -0.11 | yes | 349fd71b+dirty |
| 20261007-164116-T-NI | exploratory | SP ablation pair 2: v5 (no SP) seed 2 vs pilot B (v4) seed 2, matched recipe | T-NI | B [1000220000, +60000) | 5000 | +0.0044 |  | +4.37 | pass | -0.0051 | +0.46 | yes | f72ec9a1+dirty |
| 20261007-173414-measure | exploratory | SP ablation effect size, pair 1: v5 (no SP) seed 1 vs pilot A (v4) seed 1, fixed N | measure | B [1000280000, +20000) | 20000 | -0.0060 | [-0.0107, -0.0014] |  | measured | +0.0079 | -0.59 | yes | 153887f2+dirty |
| 20261007-203040-measure | exploratory | SP ablation effect size, pair 2: v5 (no SP) seed 2 vs pilot B (v4) seed 2, fixed N | measure | B [1000300000, +20000) | 20000 | +0.0012 | [-0.0035, +0.0059] |  | measured | -0.0005 | +0.15 | yes | 92ce86e5 |
| 20261008-150055-AA | exploratory | D51 re-check: deterministic A/A after the triple-ron rule change (rule-based) | A/A | A [0, +500) | 500 | +0.0000 |  |  | pass | +0.0000 | +0.00 | yes | 6749abd7+dirty |
| 20261008-150119-AA | exploratory | D51 re-check: deterministic A/A after the triple-ron rule change (neural, true FP32, D43) | A/A | A [500, +500) | 500 | +0.0000 |  |  | pass | +0.0000 | +0.00 | yes | 6749abd7+dirty |
