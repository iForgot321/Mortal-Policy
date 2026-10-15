# Search pilot: riichi vs dama (06 §0, D32)

- **Status (2026-10-15):**
  - the oracle stage's code is built and tested (below). It runs on the workstation, with no
    cloud box;
  - the run waits for G0, which picks the blueprint ("family-A self-play of the leading arm").
    The user asked to wait until the P0 and P1 runs finish (2026-10-15);
  - the belief stage (S-1 particles, likelihood weights, the latency harness) is not started.
- **Labels:** the oracle stage is a lab-only diagnostic: never a strength result, never deployed
  (06 §0). The pilot's verdict is the recovery ratio, belief gain ÷ oracle gain.

## Code

- **Rust, `libriichi/src/search/`** (Python module `libriichi.search`):
  - `replay`: plays a kyoku's latent history (a `Board`) through its logged events, forcing each
    player's logged reaction. It stops where the board waits for the root's action. Every event
    the board emits must equal the logged one, so this is also the consistency check that
    particles will need (06 §2).
  - `redeal_undealt`: deals the undealt tiles again in random order: the rest of the yama, the
    rinshan, the unrevealed dora indicators and the ura indicators. Hands, past draws and revealed
    indicators stay.
  - `rollout`: plays many branches to the end of their kyoku in one batch, with one agent at every
    seat. This is the `branch()` interface of 06 §3.
  - `rollouts`, `root_observations` and `self_play` are the Python entry points. `self_play` plays
    each seed once with one engine at all four seats; the 1v3 arena would play it four times,
    identically.
  - `Board` and `BoardState` are now `Clone`; nothing else in the arena changed.
- **Python, `search/`:** `value.py` gives the GRP-v2 expected utility after a kyoku, in u
  (05 §3.2). Its kyoku-to-kyoku transition mirrors `arena/game.rs`. `pilot.py` holds the
  stages, and `python -m search` runs them (`search/__main__.py`).
- **Tests:**
  - `cargo test search::` (5): every kyoku of rule-based self-play replays from its seed;
  - a wrong seed fails at `start_kyoku`;
  - branching with the logged action reproduces the logged rest of the kyoku (06 §3's test);
  - a redeal keeps the dealt tiles.
  - `tests/test_search.py` (8): the same through Python, with chi, pon and all three kan types
    (a small random network); `next_kyoku` predicts every next `start_kyoku` and the final
    scores; the GRP value is a utility.
  - Bugs found on the way, both fixed:
    - an open kan's dora is logged before the discard that reveals it;
    - a riichi's acceptance is logged before a call on the riichi discard.

## Design as built (to confirm before the run)

These fill in what 06 §0 leaves open. Each is a choice, not a fact, and the run should not start
until they are fixed.

1. **What the oracle knows.** The opponents' hands and every past draw are the true ones; the
   undealt tiles are dealt again for each rollout.
   - Why: the wall's order is chance, and no inference can recover it. An oracle that also knew
     the order would know the evaluation's own outcome (below), ura indicators included, which
     favours riichi. The ratio would then measure clairvoyance, not belief quality.
   - That bound is still reported, as `clairvoyant_gain`: the better of the two candidates in
     the true world, at no extra cost.
2. **Candidates:** riichi, with the blueprint's riichi discard after it, and the policy's best
   non-riichi action. The decision is left out of the class when that action isn't a discard
   (a win, a kan or an abortive draw: 1 of 756 in the smoke run).
3. **Rollouts:** to the end of the kyoku, with the blueprint at all four seats, greedy, in true
   FP32. Each is valued by the GRP-v2 expected utility after the kyoku, the K-surrogate units of
   05 §3.2. There is no truncation and no critic, so no bootstrap error.
4. **Redeals:** R = 64 per candidate. Both candidates share the same 64 deals (common random
   numbers).
5. **Evaluation:** each candidate is played once in the true world, the same world for every
   condition. Gain = V_true(the search's choice) − V_true(the policy's choice); a decision where
   they agree contributes exactly 0. The standard error is clustered by game.
6. **Decision rule** (06 §4 lists two; the primary one must be named before the run):
   - plain: switch away from the policy's choice when the paired z of the estimated gain
     exceeds a threshold (0 and 2 are reported);
   - regularized: π·exp(Q̂/λ) over the two candidates, played by its argmax (λ = 0.05 and 0.1 u
     are reported).
7. **Decisions:** all riichi-legal own-turn decisions in the self-play logs, shuffled with seed 0;
   the first 20K in the class are kept. That needs about 1,100 games at about 19 per game, on a
   named family-A block.

## Smoke run (2026-10-15, dev key, pilot A; not a result)

- `workdir/search/dev-smoke`: 40 games of pilot A self-play (22 s), 756 riichi-legal decisions,
  256 sampled.
- The oracle stage ran 256 decisions × 130 rollouts in 95 s on the 3060 (≈ 350 rollouts/s), peak
  RAM 2.2 GB. At that rate, 20K decisions take ≈ 2.2 h.
- **Reproduced 256/256:** the policy's own action, played in the true world, gives back the
  self-play kyoku exactly (06 §3's test, on real decisions).
- The gains in its report come from 256 dev decisions and mean nothing yet.
