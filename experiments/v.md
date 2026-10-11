# Arm V: value-based Mortal (05 §2.1, D29)

- **Status:** the quick CQL model's offline trainer is ported (2026-10-11) and model A is
  training on the workstation. Online value learning (V's fleet runs) is not ported yet.
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

## Open before V's fleet runs

1. **V's second seed.** `00`/`05` say "a quick CQL model". P1 gave each seed its own
   initialization (D53: seed 2 from pilot B). The same for V means a CQL model B (seed 2,
   ≈ 25 h more on the workstation). The user's call; a D53-style entry if so.
2. **Online value learning:** not ported. It needs the DQN engine with Equim's exploration
   (Boltzmann ε, temperature, top-p) in the actors, the next-rank target from drained logs,
   no CQL term, and `fleet.py` launching the value trainer. Equim's exploration values are
   placeholders in his example config, so they belong in V's tuning allowance (≤ 1 box-day,
   written down before the runs, `05` §2.2).
3. **Rental approval** for V's two runs (≈ $210 each at P1's prices).
