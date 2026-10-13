# Arm V: value-based Mortal (05 §2.1, D29)

- **Status (2026-10-11):**
  - the offline trainer is ported, and quick CQL model A is training on the workstation;
  - quick CQL model B (seed 2, D54) is queued to start when A finishes (below);
  - online value learning is ported and tested on tiny nets on the CPU (below);
  - a full-size check on the 3060 waits until model A finishes (RAM).
- **Arm:** Equim's value-based Mortal (`mortal/train.py`, removed from this fork's lineage in
  a24399b). Offline CQL on the pilots' data (`min_q_weight` 5, next-rank auxiliary 0.2), then
  online value learning without CQL, with the same GRP reward, `pts` = u and the same
  client/server fleet. 2 seeds, 3 boxes × 6 days each (`05` §2.2).
- **Seeds (D54):** seed 1 starts from CQL model A and plays against frozen A (`v-a.toml`); seed 2
  starts from CQL model B and plays against frozen B (`v-b.toml`), as P1 does with the pilots
  (D53).
- **Labels:** quick-model evidence (D41); exploratory rows.

## Quick CQL model (04 §7; registered 2026-10-11, before training)

- **Code:** `mortal/train_value.py`, a port of `a24399b^:mortal/train.py`'s offline mode.
  `FileDatasetsIter(value_targets=True)` in `mortal/dataloader.py` gives each decision its Q
  target and next rank. `AuxNet` is back in `mortal/model.py`.
- **Kept from Equim's trainer:**
  - Brain + dueling DQN head (v4: one linear layer to V and A) + next-rank AuxNet, one AdamW;
  - Q target: the decision's kyoku GRP-Δu, Monte-Carlo, γ = 1 (Equim's config; with γ = 1 his
    `steps_to_done` discount is 1);
  - loss: 0.5·MSE(Q(a), target) + 5·(logsumexp Q − Q(a)) + 0.2·CE(next rank).
- **Matched to pilot quick student A (`experiments/pilot-quick-students.md`), "same subset and
  SL budget":**
  - b8c192, obs v4, GroupNorm;
  - the `quick500k` subset, one epoch, Q6/Q8 masks, no augmentation, fp16 obs storage;
  - batch 2048, AdamW peak 3e-4, 2K warm-up, cosine to 1e-5 at 158K steps, weight decay 0.1,
    grad clip 1.0, AMP + compile, seed 1, 2 loader workers.
  - Equim's example config marks its optimizer values as placeholders, so the pilots' are used.
- **Differences from the pilots, recorded as part of the recipe (`05` §2.2):**
  - value head and auxiliary head in place of the policy head;
  - rewards in fixed u units (pts / 100, `05` §3.2), never standardized. Upstream standardized
    them; Equim's trainer didn't;
  - the loader runs the frozen GRP (`grp-v2-frozen.pth`) on every seat-game, which the BC
    pilots skipped. It costs CPU, so the run may take longer than the pilots' ≈ 25 h;
  - the reward scale fixes how strong the CQL term is relative to the MSE. Equim's absolute
    scale is unknown (his example's `pts` is a placeholder); `05` fixes u.
- **Frozen checkpoint:** the last save before the data runs out, as for the pilots (expected
  step 156,000 of ≈ 157,869), copied to `workdir/cql/qs-a/cql-qs-a-final.pth`, read-only, hash
  recorded here.
  - **Model A, frozen 2026-10-12 09:51 UTC** (by the queue) at step 156,000: sha256
    `f8a867da0d26fdeaca81114a52e48e0483d8f2f4b71ee902cec78e72a66c8362`, also in
    `tools/fleet/frozen.sha256`. Sanity checks pass: the data ran out at step 157,869, the
    pilots' count; every logged loss is finite; from step 20,000 to 156,000 the CQL term fell
    0.547 → 0.495, the DQN loss 0.240 → 0.205, and the next-rank loss 0.791 → 0.755 (below ln 4).
    It trained for ≈ 25.5 h on the workstation. Model B started at 09:51:48 UTC.
- **Sanity checks before it becomes V's initialization (not strength claims):** the data runs
  out at the pilots' step count (± 1%); every logged loss is finite; the CQL term falls and the
  next-rank loss falls below ln 4 ≈ 1.386.
- **Smoke test (400 logs, eager, 2026-10-11):** 127 steps, ≈ 1.5 steps/s; dqn 0.034 → 0.25,
  cql 1.90 → 1.59, next rank flat at 1.386 (still in warm-up). Ran end to end.
- **Launch:**

  ```bash
  cd mortal && MORTAL_CFG=../experiments/configs/cql-qs-a.toml setsid -f \
      python train_value.py > ../workdir/cql/qs-a/train.log 2>&1 < /dev/null
  ```

## Quick CQL model B (D54; queued 2026-10-11)

- **Config:** `experiments/configs/cql-qs-b.toml`, model A's with seed 2 and its own paths
  (`workdir/cql/qs-b/`), as pilot B is pilot A with seed 2. Same frozen checkpoint rule and sanity
  checks as model A; frozen as `workdir/cql/qs-b/cql-qs-b-final.pth`.
  - **Model B, frozen 2026-10-13 18:01 UTC** (by hand) at step 156,000: sha256
    `c0ea3b963de2d6121997952791d0908dc15d0126c7c54f46428a3e5d8a6b981f`, also in
    `tools/fleet/frozen.sha256`. Sanity checks pass: the data ran out at step 157,869 (11:26 UTC),
    the pilots' count; every logged loss is finite; from step 20,000 to 156,000 the CQL term fell
    0.547 → 0.494, the DQN loss 0.240 → 0.205, and the next-rank loss 0.789 → 0.750 (below
    ln 4). It trained for ≈ 25.6 h, as A did, and its losses track A's to the third decimal.
- **Queue:** `workdir/cql/queue-b.sh` (not tracked), started 2026-10-11 20:53 UTC with
  `setsid -f bash workdir/cql/queue-b.sh 2197357` (PID 2332387). It waits until model A's
  session (the trainer and its workers) has exited, then:
  - if A's log has `data exhausted at step`: freezes A as `cql-qs-a-final.pth` (read-only; never
    overwrites one), logs its hash and step, and starts B with the launch line above
    (`cql-qs-b.toml`, log `workdir/cql/qs-b/train.log`);
  - otherwise starts nothing and leaves the GPU idle for a diagnosis.
  - It refuses to start B over an existing `workdir/cql/qs-b/mortal.pth`.
  - Its log is `workdir/cql/queue-b.log`. Dry-run on fake directories, both paths, before it was
    armed.
- **Timing:** A ends ≈ 2026-10-12 09:30 UTC; B then takes ≈ 25 h, to ≈ 2026-10-13 11:00 UTC.
  No lab or arena jobs on the workstation until then.

## Online value learning (V's fleet runs; ported 2026-10-11)

- **Ported from Equim:**
  - **Learner:** `train_value.py` with `control.online = true`. It drains the 1v3 logs, learns
    from the trainee's seat only, and uses the same Q and next-rank targets as offline.
  - **No CQL term online**, as in Equim's trainer. It is still logged as a diagnostic.
  - **Parameters** are submitted every `submit_every` steps and after each drain.
  - **Starting from the CQL model:** a fresh optimizer and schedule, and its steps are kept.
  - **Actors:** `engine.ValueEngine`. It is greedy on Q, except that with probability ε it
    draws from softmax(Q/τ) restricted to its top-p nucleus (Equim's `sample_top_p`).
    `behaviour_dist` gives that exact distribution, so the D46 sidecars stay correct. They
    are written but unused, because the value loss doesn't use behaviour probabilities.
- **Changed from Equim:**
  - No test_play, and so no respawning wrapper: evaluation is the lab's job, and test_play is
    what hung his online trainer.
  - Checkpoints are written atomically.
  - The trainer logs the policy lag of each drain (`data/lag_mean`, `data/lag_max`).
- **Plumbing:**
  - **`control.head = 'value'`.** `client.py` and `player.TrainPlayer` build a DQN and a
    `ValueEngine` from it, and `fleet.py start` runs `train_value.py` on the learner
    (`box_learner.sh`'s 5th argument).
  - **Frozen baselines.** `player.load_frozen` loads value checkpoints (`current_dqn`) as
    well as policy ones, both played greedily.
- **Tested** (`tests/test_value.py`; CPU, tiny nets):
  - the sampler matches `behaviour_dist`;
  - frozen baselines load from both kinds of checkpoint;
  - a 1v3 game with `ValueEngine` as trainee feeds the value loader (the trainee's seat only,
    every action legal);
  - `server.py` + `train_value.py` online: drains the logs, trains, submits parameters (the
    server then serves the DQN) and saves.
- **Config v0: `experiments/configs/v-a.toml`.**
  - **Same as P0:** batch 512, save and submit every 400 steps, 800 games per actor batch,
    capacity 1600, fp16 obs, the fleet layout.
  - **Engine-agnostic choices from P0 (`05` §2.4):** weight decay 0 and gradient clipping 0.5.
  - **Equim's values, marked as placeholders in his example config:** next-rank 0.2 and γ = 1.
  - **Opponents (D54):** each seed's own frozen initialization (`05` §2.4's `baseline.train`
    row), playing greedily on Q: CQL model A for `v-a.toml`, CQL model B for `v-b.toml`.
- **Tuning** (V's allowance, ≤ 1 box-day, written down before the runs, `05` §2.2):
  - **Learning rate:** constant. v0 is P0's 2e-5; Equim's example has 1e-4.
  - **Exploration:** v0 is Equim's ε 0.005, τ 0.05, top-p 1. Q here is in u units, so his
    temperature's scale doesn't carry over.
  - **The sweep's plan and decision rule: `experiments/v-tuning.md`.**
- **Measured: how much v0 explores** (2026-10-11; model A's checkpoint at step 78,000, half
  trained; 3 heldout games, 1,830 decisions with ≥ 2 legal actions; CPU;
  `experiments/v_qgaps.py`). To redo on the frozen model.
  - CQL pushes the actions the humans didn't take far down. The best legal action's Q has a median
    of 0.26 u; the median of all legal actions' Q is −5.8 u. A kyoku can't move u by more than
    2.2, so these are not values, just "not in the data".
  - Gap from the best to the second-best action: median 1.8 u, 25th percentile 0.76, 10th
    percentile 0.26.
  - Argmax Q matches the human's action in 80% of decisions.
  - Share of an exploration draw that lands off the argmax, by τ: 1.5% at 0.05; 2.9% at 0.1; 5.5%
    at 0.2; 13% at 0.5; 23% at 1.0.
  - So at v0 (ε 0.005, τ 0.05), a game's ≈ 152 multi-option trainee decisions carry about 0.01
    non-greedy actions, one per ≈ 90 games. The trainee learns Q only for the actions it takes,
    so v0 can correct overvalued greedy actions but almost never finds undervalued ones.
- **Redone on frozen model A** (2026-10-12, step 156,000; the last 5 heldout games, 20 seat-games,
  3,096 decisions with ≥ 2 legal actions, 155 per seat-game; GPU). It barely moved from step
  78,000:
  - argmax Q matches the human's action in 80.6% of decisions;
  - the best legal action's Q: median 0.21 u (10th–90th percentile −0.01 to 0.49); all legal
    actions' Q: median −6.3 u;
  - best minus second best: median 1.81 u; 25th percentile 0.70; 10th 0.26;
  - off-argmax share of a draw: 1.5% at τ 0.05; 2.8% at 0.1; 5.6% at 0.2; 13.0% at 0.5; 23.6% at 1.0;
  - ε for the sweep's targets: 0.0496 (1 per game at τ 0.5) and 0.1915 (7 per game at τ 1.0),
    used as **0.050 and 0.19** (`experiments/v-tuning.md`).
- **Redone on frozen model B** (2026-10-13, step 156,000; the same 5 heldout games, 3,096
  decisions; `--reference` A). B explores as A does:
  - argmax Q matches the human's action in 80.4% of decisions, and A's argmax in 94.9%;
  - the best legal action's Q: median 0.20 u (10th–90th percentile −0.02 to 0.47); all legal
    actions' Q: median −6.3 u;
  - best minus second best: median 1.85 u; 25th percentile 0.71; 10th 0.27;
  - off-argmax share of a draw: 1.5% at τ 0.05; 2.8% at 0.1; 5.5% at 0.2; 12.9% at 0.5; 23.3% at 1.0;
  - ε for the sweep's targets: 0.0502 and 0.1944, which round to A's 0.050 and 0.19. So the
    values the sweep picks on A carry over to `v-b.toml` unchanged.

## Open before V's fleet runs

1. **Decided 2026-10-11 (D54):** CQL model B for seed 2 (queued), and each seed plays against
   its own frozen CQL model.
2. **After each CQL model finishes** (the queue freezes A):
   - the sanity checks; record the hash here;
   - add it to `tools/fleet/frozen.sha256` (boxes check it, and `fleet.py` needs it as the
     baseline input);
   - redo the exploration measurement above on the frozen model.

   Done for A on 2026-10-12 and for B on 2026-10-13.
3. **The tuning sweep: planned in `experiments/v-tuning.md`** (2026-10-11); the user approved its
   rental (≈ $11–15) the same day. Its first run is also V's full-size check.
4. **Rental approval** for V's two runs (≈ $210 each at P1's prices), after the sweep.
