# Arm V: value-based Mortal (05 §2.1, D29)

- **Status (2026-10-11):**
  - the offline trainer is ported, and quick CQL model A is training on the workstation;
  - online value learning is ported and tested on tiny nets on the CPU (below);
  - a full-size check on the 3060 waits until model A finishes (RAM).
- **Arm:** Equim's value-based Mortal (`mortal/train.py`, removed from this fork's lineage in
  a24399b). Offline CQL on the pilots' data (`min_q_weight` 5, next-rank auxiliary 0.2), then
  online value learning without CQL, with the same GRP reward, `pts` = u and the same
  client/server fleet. 2 seeds, 3 boxes × 6 days each (`05` §2.2).
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
  - **Opponents:** the arm's frozen initialization (`05` §2.4's `baseline.train` row), i.e. CQL
    model A playing greedily on Q.
- **Tuning** (V's allowance, ≤ 1 box-day, written down before the runs, `05` §2.2):
  - **Learning rate:** constant. v0 is P0's 2e-5; Equim's example has 1e-4.
  - **Exploration:** v0 is Equim's ε 0.005, τ 0.05, top-p 1. Q here is in u units, so his
    temperature's scale doesn't carry over.

## Open before V's fleet runs

1. **V's second seed.** `00`/`05` say "a quick CQL model". P1 gave each seed its own
   initialization (D53: seed 2 from pilot B). The same for V means a CQL model B (seed 2,
   ≈ 25 h more on the workstation). The user's call; a D53-style entry if so.
2. **After model A finishes:**
   - freeze it and record its hash here;
   - add it to `tools/fleet/frozen.sha256` (boxes check it, and `fleet.py` needs it as the
     baseline input);
   - a full-size local check (server, trainer and one client on the 3060 for a few minutes).
3. **The tuning plan** (learning rate, exploration), written here before any sweep.
4. **The opponents.** v0 plays against CQL model A, the arm's own frozen initialization (as P0
   and P1 play against pilot A). `05` §2.2 also says "same opponents for every arm". If that
   means pilot A for V too, it is a one-line change to `baseline.*.state_file`.
5. **Rental approval** for V's two runs (≈ $210 each at P1's prices) and for the tuning
   allowance.
