# Experiment: phase1-v0-unfiltered — Stage 0 AWR baseline

- **Date started:** 2026-07-07 01:12 (launched after grp-v1 freeze)
- **Code commit:** a0b6e5f (+ local train_grp.py collate fix, see grp-v1.md)
- **Config:** `mortal/config.toml` at launch time
- **Depends on:** frozen `workdir/grp/grp-v1-frozen.pth` (see `grp-v1.md`)

## Purpose

Stage 0 baseline reproduction: offline Phase 1 AWR on the **unfiltered** 2020–2024H1
archive with the lighter substrate. This checkpoint is the fixed comparison target for
every Stage 1 curation ablation. Pipeline proof > numbers this week (doc 05).

## Setup

- **Network:** b40c192, version 4, GroupNorm — 10,787,456 Brain params + CategoricalPolicy
- **Data:** same unfiltered train globs as grp-v1 (836,255 games; Jul–Dec 2024 held out by time,
  never in training). All 4 seats per game (empty whitelist).
- **Sizing probe:** 723.3 samples/file → ~605M samples/epoch → ~1.18M steps @ batch 512
- **Schedule:** AdamW, peak 1e-4 → cosine → final 1e-5, warm-up 1000, max_steps 1,180,000
- **AMP + torch.compile on** (triton 3.2.0 verified), cudnn benchmark on
- **save_every 2000** (~230MB checkpoint, limits SSD churn), **test_every 100,000**
- **test_play baseline:** random-init b40c192 checkpoint (`workdir/phase1/baseline.pth`,
  seed 20260707) — no real prior baseline exists in Stage 0; test_play only proves the
  arena path and tracks best.pth, numbers vs a random net are not a strength metric.
- **num_workers 2**, file_batch_size 15 (RAM budget: ~1.5GB buffer/worker of fp32 obs
  (1012, 34) — box has 15GB total)

## Results

- **Launch verified 2026-07-07 01:21:** first 2,000 steps in 8:03 (ramp 2.75→4.62 it/s
  during compile/cudnn warmup), GPU-bound at ~94% util, 4.0/12.3GB VRAM, RAM ~3GB
  available (tight but stable). First checkpoint written.
- **Projected:** ~4.6 it/s steady → 1.18M steps ≈ 71h ≈ 3 days/epoch.
- **First test_play @ 100k steps (07:04):** avg rank 1.0, avg pt 90.0 over 1,000 games
  vs the random-init baseline — 1st place in every game, as expected vs a random net.
  Arena path proven; best.pth written. Test took ~6 min.
- **Incident 07:10–07:25:** launching the standalone 1v3 A/A proof *concurrently* with
  training OOM-killed a dataloader worker (kernel OOM, 15GB box), wedging the trainer at
  ~step 104,600. Trainer killed and restarted from the last checkpoint. Cost: restart
  reshuffles the file list (no mid-epoch resume in this trainer), so ~9% of the epoch
  is double-seen. **Operational rule: never run arena jobs concurrently with training
  on this box.**
- **Incident 2026-07-09 ~07:45 (step ~832k):** dataloader worker crashed on
  a 2022 file (its name is in the local quarantine list, not published) — file passes gzip+JSON checks
  but is semantically invalid (kakan of 5sr onto pon [5s,5s,5s] → libriichi rule
  violation "fifth 5sr witnessed"; converter bug in the source archive). First such
  file in ~60% of the archive. Fix: dataloader.py now retries failed batches
  file-by-file and skips only offenders with a logged warning. Restarted from the 832k
  checkpoint (another file-list reshuffle). Full loader-oracle scan of the archive
  deferred to Stage 1 hygiene (GPU-idle time); grep train.log for 'skipping bad log'
  to collect offenders found en route.
- **COMPLETE 2026-07-10 06:48.** Stopped deliberately at the **1,180,000-step**
  checkpoint (LR schedule fully consumed = one epoch-equivalent of samples; letting the
  restarted leg run its full file pass would have gone to ~2.0M steps at floor LR on
  mostly re-seen data). Frozen read-only as `workdir/phase1/phase1-v0-final.pth`.
- **Coverage caveat from the two restarts:** each restart reshuffles the full file
  list; estimated ~25% of files never seen, ~majority seen once, rest twice. Acceptable
  for the Stage 0 baseline (Stage 1 retrains from scratch on curated data); noted for
  honesty in later comparisons.
- **Curves (tensorboard):** AWR loss 1.85 → 0.85 (still slowly declining at stop),
  exp_adv stable ~2.09 all run, reward-normalization σ ≈ 0.687 stable, ~39M reward
  samples. All 12 test_plays vs the random-init baseline: 1.0 avg rank / 90.0 pt
  (saturated as expected; pipeline check only).
- **Bad logs:** full loader-oracle scan completed 2026-07-11
  (`scripts/scan_bad_logs.py`): **exactly 1 invalid file in all 926,957**
  (the 2022 file with the 5sr kakan). List frozen in
  `data/stats/bad_files.tsv`; dataset-v1 excludes it.
- **Wall time:** ~76h GPU total over 3 legs (07-07 01:12 → 07-10 06:48), ~4.3 steps/s
  sustained at batch 512 with AMP+compile.
