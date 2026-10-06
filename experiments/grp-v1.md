# Experiment: grp-v1 — GRP reward model (Stage 0 prerequisite)

- **Date started:** 2026-07-07
- **Code commit:** a0b6e5f (+ local fix: removed `pin_memory` from `train_grp.py` collate — CUDA cannot re-init in forked DataLoader workers; the `packed_inputs.pin_memory()` call was a discarded-return no-op upstream anyway)
- **Config:** `mortal/config.toml` `[grp.*]` sections (this commit's copy is authoritative)
- **Hardware:** RTX 3060 12GB, WSL2, 16 cores, 15GB RAM

## Purpose

Global reward predictor (GRU h64×2, 24-class final-ranking permutation from per-kyoku
score trajectories). Required by the Phase 1 AWR dataloader (`RewardCalculator`).
Per doc 05 Finding 3: trained **once on unfiltered data, then frozen** for the whole
project so every later curation/algorithm ablation keeps the same reward function.

## Data

- **Train:** `data/raw/{2020,2021,2022,2023}/*.mjson` + `2024/20240[1-6]*` — 836,255 games (unfiltered)
- **Val:** `2024/20240[7-9]*` + `2024/20241[0-2]*` — 90,702 games (= future held-out-by-time period; GRP never trains on it)
- ~12 kyoku/game → ~10M train samples/epoch, ~19.6k steps/epoch at batch 512

## Hyperparameters (deltas from example config)

- `lr = 1e-3` (example dummy was 1e-5; standard AdamW for a tiny GRU, monitored for instability)
- batch 512, save_every 2000, val_steps 400 (unchanged)

## Stopping rule

Train until val acc/loss plateaus across ≥2 consecutive save points (tensorboard
`workdir/grp/tb`). Then freeze `workdir/grp/grp.pth` — copy to `workdir/grp/grp-v1-frozen.pth`
and never overwrite.

## Results

- **Frozen at step 80,000** (~4 epochs) as `workdir/grp/grp-v1-frozen.pth` (read-only),
  2026-07-07 01:12. Config `[grp].state_file` points at the frozen copy.
- LR anneal: 1e-3 (0–40k) → 1e-4 (40k–60k) → 1e-5 (60k–80k).
- **Val acc (24-class ranking permutation, chance 4.2%):** ~22.6% plateau at 1e-3,
  ~22.9% after 1e-4, ~23.0% final. Val loss 2.454–2.458 at freeze.
- Anneals gave small consistent gains (+0.4pp, +0.05pp) → converged; further tuning
  not warranted for a reward-shaping component.
- Throughput: ~67 batches/s @ batch 512 (GPU), ~6 min/epoch, ~30 min total train time.
- Fix required en route: removed `pin_memory` from collate (fork-CUDA crash, see header).

