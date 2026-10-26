# Draft D61: switch the observation to v5 after G0 (for the user's review)

**Status: approved by the user on 2026-10-26.** The entry is D61 in `08_decision_log.md` (with the
soak `soak-a` added to the v4 evidence); the edits below are made. This file is the draft's history.

---

**D61 (2026-10-__): from G0 on, the observation is v5 (v4 without the single-player tables); Phase 1's results stay v4 evidence.** Amends D49 (the review it schedules), `05` §1 item 4 (v4 frozen through Phase 1) and CLAUDE.md rule 2's observation item. Decided by the user.
- **Why** (G0's observation review, D49):
  - **Strength is about even.** `k2b`'s doubling estimate is +0.0025 [+0.0004, +0.0046] (grey by its pre-registered reading). v5 buys ≈ 1.2–1.5 extra doublings of RL compute at equal cost, ≈ +0.003–0.004 u if the gain per doubling holds. v5's SL cost is −0.0024 [−0.0056, +0.0009], inconclusive (the SP ablation's pairs disagree). The net, ≈ +0.001 u, is inside the noise of both.
  - **v5 is cheaper:** SL training 2.9× faster (the loader 90× per thread), actors 2.4–2.8×, search rollouts 1.33× batched and 1.6–2.2× one at a time.
  - **Now is the cheapest time:** the b24c512 teachers aren't trained yet, so nothing is retrained.
- **What moves to v5:** the teachers and students (`04`), Phase 2's RL (`05`; the engine P1 + K2 starts from a v5 initialization, with π_SL the same v5 checkpoint, which K2's anchor requires), search (`06`; the hand model follows the blueprint's version), and the deployed student (`07`).
- **What stays v4 evidence, labelled so:** P0, P1, V and their diagnosis; the SP ablation; the search pilot on pilot A; G0's pass (`k2b` full vs pilot A), L2's confirmation and the doubling estimate. The engine selection (P1 + K2) carries over as a recipe, not as a network.
- **Before a v5 result counts:**
  1. a deterministic A/A with a v5 model (`02` §6), and a frozen v5 featurizer hash beside v4's (the arena and training paths already agree: `experiments/sp-ablation.md` §2);
  2. **the recipe check on v5:** a `k2`-style run from No-SP A (`workdir/sp-ablation/nosp-a/nosp-a-final.pth`, the SP ablation's v5 counterpart of pilot A), stopped at p25 (108 box-hours, ≈ $65), with D59's early-check measures against No-SP A. It must clear the same thresholds `k2b` cleared at p25 (reverse KL ≤ 0.10, argmax agreement ≥ 90%, Δu CI not entirely below 0). Pre-registered before it runs; a rental, so the user's go-ahead;
  3. T1's final-student check (D41) then tests the engine on the final v5 student, as planned.
- **Unchanged:** the utility vector, GRP-v2, the rules config, the seed families, the lab's test definitions, the data splits, and K2's settings.
- **Cost:** the checks ≈ $66 (the recipe check, a v5 throughput re-benchmark ≈ $1); the A/A on the workstation. Teacher runs get cheaper.
- **Not in this entry:** obs v6 (the 889 rows without the zeroed block, a smaller input) and the cheaper shanten-threshold-1 variant. Either would need its own entry.
- **Reopen if:** the recipe check fails a threshold, the v5 A/A fails, or the first v5 student is measurably weaker than its v4 counterpart beyond the SP ablation's interval.

---

## Edits once approved (after the `08` entry)

- `05` §1 item 4 and §2's "same inputs" line; `00`'s G0 observation-review note; `04`'s teacher plan
  (v5, and D17a's hardware question, since featurization stops binding); `07` (the deploy student is
  v5); CLAUDE.md rule 2 ("observation schema: v5 from G0, D61").
- `tools/fleet/frozen.sha256`: add No-SP A and B, so every box gets them (setup checks their hashes,
  and the tests expect every frozen file).
- New configs for the recipe check (`experiments/configs/k2-v5.toml`: `control.version = 5`, the init
  and anchor No-SP A).

## Already done (2026-10-25, before any decision)

The tools that hard-coded v4 now follow the model's version, with v4 the default, so nothing changes
for v4: the search hand model (`search/handmodel.py`: its data, root views and saved models carry the
blueprint's version; older saves load as v4), `tools/fleet/bench_1v3.py`'s shadow network, and
`tools/fleet/bench_learner.py`'s K2 anchor (pilot A for v4, No-SP A for v5, or `--anchor`).
Test: `tests/test_search.py::test_hand_model_follows_the_blueprint_version`.
