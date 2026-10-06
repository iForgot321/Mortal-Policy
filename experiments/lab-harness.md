# Lab harness checks (Phase 0, 02 §6)

Toward L0: the deterministic A/A must give Δu = 0 exactly before any verdict is trusted.
Registry rows are in `lab/registry/runs.jsonl`; game logs under `workdir/lab/<run_id>/`.

## Deterministic A/A, rule-based (2026-10-03)

`python -m lab aa --model builtin:shanten-greedy --sets 500 --block aa-shanten-greedy`

- First attempt: every set's 4 games identical (0 divergent), but |set score| up to 5.6e-17,
  because 0.9 + 0.5 − 0.1 − 1.3 ≠ 0 in binary floating point. Fixed by summing utilities in
  integer hundredths (`lab/gsprt.py:set_scores`).
- Rerun: **pass**. 500 sets, all set scores exactly 0, s² = 0, 0 divergent sets.
- Conclusion: seat rotation, seed handling and per-set bookkeeping are exact.

## Deterministic A/A, neural net with AMP (2026-10-03)

`python -m lab aa --model workdir/phase1/phase1-v0-final.pth --sets 500 --block aa-harness-nn`
(v1-era b40c192 AWR checkpoint; harness check only, not a strength claim.)

- **FAIL as specified:** 21/500 sets (4.2%) had divergent games; mean −0.0001, set SD 0.027,
  max |set score| 0.3.
- Diagnosis (first divergent event of 10 sets): every divergence is an argmax flip at a near-tie,
  top-2 probability gap ≤ 0.004, often equal at fp16 resolution (e.g. 0.47946 vs 0.47946).
  The event histories up to that point are identical, so inputs are identical. The cause is AMP
  fp16 numerics depending on batch composition: challenger and champion engines run separate
  batches (1 vs 3 seats per game).
- Not a bookkeeping, seat-indexing or rule-asymmetry bug (those would also break the rule-based
  A/A, which is exact).
- Consequence for A/B tests: zero-mean noise at near-ties, no bias. Consequence for the harness
  check: run it in FP32 (`--no-amp`, deterministic cuDNN).

## Deterministic A/A, neural net FP32 (2026-10-03)

Same checkpoint, same nonce block (aa-harness-nn, family A [500, 1000)).

- **FP32 with TF32** (`--no-amp` before the TF32 fix): 2/500 divergent sets, all set scores still 0.
  Both divergences are near-tie flips, top-2 gaps 4e-5 and 2e-4. Ampere runs cuDNN convolutions in
  TF32 (10-bit mantissa) by default, so "FP32" still had fp16-like precision.
- **True FP32** (`--no-amp` now also disables TF32 and sets deterministic cuDNN): **pass**. 500 sets,
  0 divergent sets, every set score exactly 0, s² = 0.

**Conclusion:** the harness is exact; all observed divergence was inference numerics. Recorded as D43:
the deterministic A/A runs with `--no-amp`; verdict runs keep AMP, the deploy mode (02 §3).

## A/A′, calibration and L0 (2026-10-05)

- **A/A′** (run `20261005-124255-measure`): pilot A vs pilot B, 20K sets on family A, AMP.
  σ_set = 0.3277 (unpaired would be 0.415: duplicate benefit 37.7%), Δu = +0.0015 [−0.0031, +0.0061].
  Per-seat avg rank E/S/W/N 2.494 / 2.512 / 2.498 / 2.488.
- **§4.1 calibration:** nominal ±2.944 boundaries kept for every test type; worst error 4.77% (T-fine). D44.
- **Hobbled patch** (run `20261005-163506-T-std`): pilot A with ε = 0.1 random actions vs pilot A, T-std on
  family B. **Fail** at the first look (5,000 sets): Δu = −0.396, LLR −132.8. As required.
- **A/A′ pair under T-NI** (run `20261005-173018-T-NI`): pilot A vs B on family B. **Pass** at 6,000 sets:
  Δu = +0.0010, LLR +3.28; set SD 0.330 on fresh nonces, consistent with the calibrated σ_set.

**L0 met (2026-10-05):** deterministic A/A exact (D43), σ_set measured, the sequential procedure calibrated by
simulation (D44), the hobbled patch fails, and the A/A′ pair passes non-inferiority.
