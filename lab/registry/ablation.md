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
| 20261012-070327-AA | exploratory | deterministic A/A on the eval box (new hardware, evalbox.py first job) | A/A | A [21000, +2000) | 2000 | +0.0000 |  |  | pass | +0.0000 | +0.00 | yes | 051f641d+dirty |
| 20261013-110149-measure | exploratory | V tuning sweep: v-tune-e1 vs CQL model A | measure | A [23000, +10000) | 10000 | -0.7289 | [-0.7359, -0.7219] |  | measured | +0.9379 | -73.79 | yes | 2d348870 |
| 20261013-124951-measure | exploratory | V tuning sweep: v-tune-e0-ext vs CQL model A | measure | A [23000, +10000) | 10000 | -0.8740 | [-0.8807, -0.8675] |  | measured | +1.0901 | -89.04 | yes | 2d348870+dirty |
| 20261013-143617-measure | exploratory | V tuning sweep: v-tune-e2 vs CQL model A | measure | A [23000, +10000) | 10000 | -0.9216 | [-0.9277, -0.9155] |  | measured | +1.1491 | -93.75 | yes | 2d348870+dirty |
| 20261013-162554-measure | exploratory | V tuning sweep: v-tune-e1-lr vs CQL model A | measure | A [23000, +10000) | 10000 | -1.1683 | [-1.1721, -1.1645] |  | measured | +1.3868 | -120.34 | yes | 2d348870+dirty |
| 20261014-234559-measure | exploratory | p0-a p50 vs pilot A (halfway check) | measure | A [33000, +10000) | 10000 | -0.0334 | [-0.0410, -0.0257] |  | measured | +0.0454 | -3.44 | yes | 1645b88f+dirty |
| 20261015-085354-measure | exploratory | p1-a p50 vs pilot A (halfway check) | measure | A [33000, +10000) | 10000 | -0.0262 | [-0.0337, -0.0187] |  | measured | +0.0415 | -2.68 | yes | 4262550f+dirty |
| 20261015-094729-measure | exploratory | p1-b p50 vs pilot B (halfway check) | measure | A [33000, +10000) | 10000 | -0.0216 | [-0.0291, -0.0142] |  | measured | +0.0482 | -2.04 | yes | 4262550f+dirty |
| 20261015-104045-measure | exploratory | v-cql03 final (73,200 online steps) vs CQL model A | measure | A [23000, +10000) | 10000 | -0.0431 | [-0.0505, -0.0357] |  | measured | +0.0635 | -4.32 | yes | 4262550f+dirty |
| 20261017-033700-T-std | exploratory | G0: p0-a full vs pilot A (05 §2.3); rerun, the first was lost with eval box 56069526 | T-std | B [1000520000, +100000) | 5000 | -0.0268 |  | -10.57 | fail | +0.0432 | -2.73 | yes | b8f27d21+dirty |
| 20261017-040806-T-std | exploratory | G0: p1-b full vs pilot B (05 §2.3) | T-std | B [1000620000, +100000) | 5000 | -0.0131 |  | -6.17 | fail | +0.0256 | -1.32 | yes | b8f27d21+dirty |
| 20261017-043916-measure | exploratory | p0-a full vs p50 (doubling estimate) | measure | B [1000720000, +100000) | 100000 | +0.0071 | [+0.0047, +0.0093] |  | measured | -0.0020 | +0.80 | yes | b8f27d21+dirty |
| 20261018-044407-T-std | exploratory | G0: p1-a full vs pilot A (05 §2.3) | T-std | B [1000820000, +100000) | 5000 | -0.0209 |  | -8.93 | fail | +0.0408 | -2.02 | yes | b8f27d21+dirty |
| 20261024-110559-T-std | exploratory | G0 retry: k2b full vs pilot A (D59) | T-std | B [1000920000, +100000) | 7000 | +0.0114 |  | +3.88 | pass | -0.0167 | +1.12 | yes | 8833d7a5+dirty |
| 20261024-122034-measure | exploratory | k2b full vs p50 (doubling estimate, D49) | measure | B [1001020000, +100000) | 100000 | +0.0025 | [+0.0004, +0.0046] |  | measured | -0.0046 | +0.25 | yes | 7c2dab4f |
| 20261025-060641-T-confirm | confirmed | L2 confirmation: k2b full vs pilot A, 100K sets (D59) | T-confirm | C [2000000000, +100000) | 100000 | +0.0195 | [+0.0174, +0.0216] |  | pass | -0.0268 | +1.94 | yes | d65aa590 |
| 20261026-080110-AA | exploratory | D61: deterministic A/A with a v5 model (No-SP A, true FP32, D43) | A/A | A [500, +500) | 500 | +0.0000 |  |  | pass | +0.0000 | +0.00 | yes | b76423dd+dirty |
| 20261026-124738-measure | exploratory | fleet soak tracking: soak-a p25 vs k2b full | measure | A [44200, +10000) | 10000 | +0.0026 | [-0.0039, +0.0092] |  | measured | +0.0000 | +0.34 | yes | 5b9433cb+dirty |
| 20261027-004806-measure | exploratory | fleet soak tracking: soak-a p50 vs k2b full | measure | A [44200, +10000) | 10000 | -0.0036 | [-0.0103, +0.0030] |  | measured | +0.0069 | -0.35 | yes | 8eed01f6+dirty |
| 20261028-004949-measure | exploratory | fleet soak tracking: soak-a full vs k2b full | measure | A [44200, +10000) | 10000 | +0.0016 | [-0.0050, +0.0082] |  | measured | -0.0001 | +0.15 | yes | 87cc1082+dirty |
| 20261030-075954-T-NI | exploratory | teacher heads check (04 §5's guard): heads-on quick student vs No-SP A | T-NI | B [1001120000, +60000) | 5000 | +0.0082 |  | +5.84 | pass | -0.0055 | +0.87 | yes | 48afa9b4+dirty |
| 20261105-105720-T-std | exploratory | teacher objective (04 §5): AWR s1 vs BC s1, EMA weights | T-std | B [1001180000, +100000) | 6500 | -0.0007 |  | -3.27 | fail | -0.0088 | -0.23 | yes | 29c50642+dirty |
| 20261109-221907-measure | exploratory | k2b opponent check: k2b full vs 3x shanten-greedy | measure | A [54200, +20000) | 20000 | +0.4214 | [+0.4174, +0.4253] |  | measured | -0.5709 | +42.16 | yes | 94b0c7cb+dirty |
| 20261110-030053-measure | exploratory | k2b opponent check: pilot A vs 3x shanten-greedy (the pair's baseline) | measure | A [54200, +20000) | 20000 | +0.4130 | [+0.4091, +0.4170] |  | measured | -0.5573 | +41.36 | yes | 94b0c7cb+dirty |
| 20261110-041102-measure | exploratory | k2b opponent check: k2b full vs 3x pilot B | measure | A [1000, +20000) | 20000 | +0.0193 | [+0.0145, +0.0240] |  | measured | -0.0264 | +1.91 | yes | 94b0c7cb+dirty |
| 20261110-145755-T-std | exploratory | teacher vs quick student: BC s1 (EMA) vs No-SP A | T-std | B [1001280000, +100000) | 5000 | +0.0219 |  | +7.20 | pass | -0.0198 | +2.34 | yes | 94b0c7cb+dirty |
| 20261112-121013-measure | exploratory | anchor release: soak-a full at 3x pilot A | measure | A [74200, +40000) | 40000 | +0.0190 | [+0.0158, +0.0223] |  | measured | -0.0233 | +1.92 | yes | ed73d0a8+dirty |
| 20261112-192915-measure | exploratory | anchor release: soak-a full at 3x pilot B | measure | A [74200, +40000) | 40000 | +0.0249 | [+0.0215, +0.0282] |  | measured | -0.0299 | +2.51 | yes | ed73d0a8+dirty |
| 20261114-192413-measure | exploratory | anchor release: release-a full at 3x pilot A | measure | A [74200, +40000) | 40000 | +0.0311 | [+0.0275, +0.0345] |  | measured | -0.0370 | +3.14 | yes | 5093fc33+dirty |
| 20261114-235914-measure | exploratory | anchor release: release-a full at 3x pilot B | measure | A [74200, +40000) | 40000 | +0.0293 | [+0.0259, +0.0327] |  | measured | -0.0342 | +2.96 | yes | 5093fc33+dirty |
| 20261115-043647-measure | exploratory | anchor release: release-b full at 3x pilot A | measure | A [74200, +40000) | 40000 | +0.0299 | [+0.0264, +0.0333] |  | measured | -0.0354 | +3.02 | yes | 5093fc33+dirty |
| 20261115-085704-measure | exploratory | anchor release: release-b full at 3x pilot B | measure | A [74200, +40000) | 40000 | +0.0307 | [+0.0272, +0.0342] |  | measured | -0.0352 | +3.13 | yes | 5093fc33+dirty |
| 20261116-135055-measure | exploratory | anchor release timing: release-b p50 at 3x pilot A | measure | A [74200, +40000) | 40000 | +0.0259 | [+0.0225, +0.0294] |  | measured | -0.0358 | +2.56 | yes | 89c8ca56+dirty |
| 20261116-205523-measure | exploratory | anchor release timing: release-b p25 at 3x pilot A | measure | A [74200, +40000) | 40000 | +0.0251 | [+0.0217, +0.0284] |  | measured | -0.0331 | +2.50 | yes | 50a85599+dirty |
| 20261117-184513-T-NI | exploratory | students-s1 qualification: b8c192 (EMA) vs BC s1 (EMA), T-NI | T-NI | B [1001380000, +60000) | 12500 | -0.0074 |  | -3.07 | fail | +0.0058 | -0.84 | yes | 5c3389ca+dirty |
| 20261117-193933-T-NI | exploratory | students-s1 qualification: b12c256 (EMA) vs BC s1 (EMA), T-NI | T-NI | B [1001440000, +60000) | 5000 | +0.0047 |  | +5.28 | pass | -0.0091 | +0.32 | yes | 5c3389ca+dirty |
| 20261117-200117-T-NI | exploratory | students-s1 qualification: b16c256 (EMA) vs BC s1 (EMA), T-NI | T-NI | B [1001500000, +60000) | 10000 | -0.0021 |  | +3.18 | pass | -0.0014 | -0.29 | yes | 5c3389ca+dirty |
| 20261117-204544-T-std | exploratory | students-s1 control: distilled b8c192 vs the control (EMA both), T-std | T-std | B [1001560000, +100000) | 14000 | +0.0075 |  | +3.33 | pass | -0.0124 | +0.71 | yes | 5c3389ca+dirty |
| 20261117-212358-measure | exploratory | students-s1 control: the control (EMA) vs No-SP A, 20K measure | measure | A [114200, +20000) | 20000 | +0.0140 | [+0.0093, +0.0186] |  | measured | -0.0174 | +1.49 | yes | 5c3389ca+dirty |
| 20261117-221715-T-std | exploratory | students-s1 control: BC s1 (EMA) vs No-SP A, T-std | T-std | B [1001660000, +100000) | 5000 | +0.0226 |  | +7.53 | pass | -0.0276 | +2.39 | yes | 5c3389ca+dirty |
