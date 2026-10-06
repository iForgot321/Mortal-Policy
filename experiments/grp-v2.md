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

## Ablation: GRP-v2-nobye, without disconnect games (2026-10-05)

Question (user): would the GRP benefit from training without the ~9.5% of train games that contain a BYE?

- `experiments/configs/grp-v2-nobye.toml`: identical to grp-v2 except train excludes every BYE game
  (1,633,535 of 1,805,376 games). Same 80K-step schedule, 18 min.
- `experiments/compare_grp.py`: both models on held-out `heldout_game` games.

| Held-out games | Placement cross-entropy, v2 → nobye | Reward correlation | Mean \|Δ reward\| (u × 100) | Reward SD |
|---|---|---|---|---|
| clean (8,000 games, 339K player-kyoku) | 2.46194 → 2.46173 (−0.0002, ~1 SE) | 0.9979 | 1.20 | 25.5 |
| BYE (4,000 games, 178K player-kyoku) | 2.44391 → 2.44422 (+0.0003) | 0.9978 | 1.19 | 25.0 |

**Conclusion: no benefit; GRP-v2 stays frozen.**
- Excluding disconnect games does not measurably improve placement prediction on clean games.
- The reward difference between the two models is zero-mean and equally large on clean and BYE games, so it is run-to-run training noise, not a disconnect effect.
- Disconnects are short and the GRP reads score trajectories, which already reflect any disconnect.
- The ablation checkpoint (`workdir/grp-v2-nobye/grp.pth`) is not frozen and not used.
