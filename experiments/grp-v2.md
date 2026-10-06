# Experiment: grp-v2 — GRP reward model on dataset-v1 `train` (Phase 0)

- **Plan item:** 05 §1 item 2: train the GRP on `train`, then freeze it for Phases 1–2.
- **Replaces:** grp-v1 (`experiments/grp-v1.md`), which trained on 2020–2024H1 of an older data
  revision. Some of those games are now in `heldout_game`, and its validation period (2024H2) is now
  inside `train`.
- **Config:** `experiments/configs/grp-v2.toml`. Same network (GRU h64 × 2) and batch (512) as grp-v1.
  The LR schedule reproduces grp-v1's hand annealing from config: 1e-3 → 1e-4 at 40K → 1e-5 at 60K,
  stopping at 80K steps.
- **Data:** `train_E` + `train_O` (2015-01 → 2025-06 minus `heldout_game` and quarantine); validation on
  `heldout_game`. Index built by `tools/data/make_index.py grp`.
- **Stopping rule:** as grp-v1: the fixed 80K-step schedule, then confirm val loss/acc plateaued.
  Freeze as `workdir/grp-v2/grp-v2-frozen.pth` (read-only) and point every later config's `[grp].state_file` at it.

## Results (2026-10-03)

- Ran 80K steps in 17 min (10:04 → 10:22), ~80 batches/s at batch 512, on 1,805,376 train games;
  validation on 48,388 heldout_game games.
- Validation accuracy (24-way rank permutation, chance 4.2%) by step:
  10K 22.6%, 40K 22.7%, 60K 23.0%, 80K 22.8%.
- Validation loss: 2.467 at 10K, 2.458 at 60K, 2.460 at 80K. Plateaued over the last 3 save points.
- Matches grp-v1 (val acc ≈ 23.0%, loss 2.454–2.458), as expected for a small reward model.
- **Frozen** as `workdir/grp-v2/grp-v2-frozen.pth` (read-only), sha256 `db28beaca25305b95eec7c74d6abe0441442fb6f7f0ba9bf3ee9a0d2875e9157`.
  Every later config's `[grp].state_file` points here.
