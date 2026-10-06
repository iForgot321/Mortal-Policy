# Experiment: pilot quick students A and B (Phase 0, D41)

- **Plan items:** 04 §7 (pilot quick students), 02 §6 (A/A′), 05 §2 (P0/P1 initializations).
- **Purpose:** two students from the **same recipe** with **different training seeds**:
  1. A/A′ over 20K sets on family A gives σ_set, the duplicate benefit ρ, and the recipe noise |Δu|;
  2. their set-score distribution feeds the §4.1 calibration;
  3. they are the policy initializations for engine arms P0/P1.
- **Recipe** (`experiments/configs/pilot-qs-{a,b}.toml`; they differ only in `control.seed` and paths):
  - b8c192, obs v4, GroupNorm, policy head only;
  - **BC** (`policy.objective = 'bc'`, uniform weights; the BC prior is also τ_human for the piKL dial);
  - data: a 500K-game deterministic subset of `train` dated 2020-01-01 → 2025-06-30 (sha1 order, key
    `quick500k`), one epoch, no augmentation, Q6/Q8 masks applied;
  - AdamW, peak 3e-4, 2K warm-up, cosine to 1e-5, weight decay 0.1 on conv/linear weights, grad clip
    1.0, batch 2048, AMP + compile, fp16 obs storage (lossless under AMP).
- **Deviations from 04 §5/§7 to record:**
  - no EMA of weights (not implemented in the trainer yet);
  - no augmentation (to fit the 3060 box in about a day per student).
  - Both deviations are part of the recipe, identical for A and B.
- **Seeds:** A = 1, B = 2.
- **Hardware:** RTX 3060 12 GB workstation. Run A and B sequentially, never alongside arena jobs.

## Results

### Pilot A (seed 1), 2026-10-03 10:31 → 2026-10-04 11:19 (24.8 h)

- Data exhausted at step 157,869 (estimate 158,000). The last checkpoint is step 156,000; the unsaved
  1,869-step tail ran at the floor LR (~1e-5), so the loss is negligible.
- Frozen: `workdir/pilot/qs-a/pilot-qs-a-final.pth` (read-only, sha256 prefix `15121432d2268ae7`).
- BC loss (nats per decision, 2K-step means): 0.830 at 2K, 0.567 at 10K, 0.541 at 20K, 0.525 at 40K,
  0.510 at 80K, 0.495 at 120K, 0.489 at 156K. Still edging down at the end of the cosine.
- Throughput: ~1.75 steps/s steady (~3.6K samples/s), CPU-bound featurization. No errors, no skipped logs.

### Pilot B (seed 2), 2026-10-04 11:20 → 2026-10-05 12:42 (25.4 h)

- Started by `experiments/run_pilot_chain.sh`; last checkpoint step 156,000, like A.
- Frozen: `workdir/pilot/qs-b/pilot-qs-b-final.pth` (read-only, sha256 prefix `b267c0267f1538c6`).
- BC loss: 0.826 at 2K, 0.566 at 10K, 0.543 at 20K, 0.526 at 40K, 0.509 at 80K, 0.497 at 120K, 0.489 at 156K.

### A vs B

A/A′ over 20K sets gives Δu = +0.0015 [−0.0031, +0.0061] and σ_set = 0.328. A vs B under T-NI passes at
6K sets. Details: `experiments/lab-harness.md`, D44.
