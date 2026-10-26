# Search pilot: riichi vs dama (06 §0, D32)

- **Status (2026-10-24): the rerun on `k2b`'s `full` is queued** (the user, after G0's pass;
  `experiments/k2b_post_g0_chain.sh`): a hand model on 4,000 games of its self-play (dev key, held
  out: 40 games at nonces 0–39), then the pilot on the block `search-pilot`, in `workdir/search/pilot-k2b`.
- **Status (2026-10-21): parked by the user.** A bigger hand model (5× data, 3× steps) doesn't
  move recovery (below). Not pursued for now: the ceiling test and a threat-aimed hand model.
- **Status (2026-10-20): the pilot ran on pilot A and fails on recovery** (below, "Pilot run on
  pilot A"): oracle +0.0112 u per decision (z 12.7), belief +0.0018 (z 3.4), recovery 0.157
  [0.071, 0.24] against D32's 0.5; p99 latency 1.95 s on the 3060.
- **Status (2026-10-15):**
  - the oracle stage's code is built and tested (below). It runs on the workstation, with no
    cloud box;
  - the run waits for G0, which picks the blueprint ("family-A self-play of the leading arm").
    The user asked to wait until the P0 and P1 runs finish (2026-10-15);
  - **the belief stage is built and passes D58's calibration test.** D38's v1 (uniform particles
    reweighted by action likelihoods) was worse than a base rate. The user chose a learned hand
    model (D58), and on held-out decisions it predicts opponents' tenpai and waits well
    (below). The likelihood correction makes it worse, so by D58 the model alone is the belief.
    One belief search takes 0.68 s p50 and 2.21 s p99 on the 3060.
  - the pilot's run waits for G0's blueprint, then a hand model trained on that blueprint's
    self-play (≈ 45 min of games, ≈ 40 min of training). The whole pilot is then ≈ 5–6 h on
    the workstation.
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

## Design as built (the user confirmed 1 and 6, D58)

These fill in what 06 §0 leaves open. The user fixed the oracle's knowledge (1) and the primary
rule (6, the plain rule with z > 2) on 2026-10-15 (D58).

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

## Belief stage: S-1 particles, v1 (2026-10-15)

- **Code:** `libriichi/src/search/particle.rs` (`Layout`) and `belief.rs` (`Beliefs`, the Python
  class).
  - `Layout` reads the deciding seat's view of the kyoku. Every event is masked first
    (`visible`), so no hidden tile can be read.
  - It records the slots whose tiles the seat has seen: its own hand and draws, opponents'
    tsumogiri draws and revealed dora indicators.
  - For each opponent, it records the tiles the public events say it held, and by when:
    - each discard from hand, before that turn's draw;
    - each called or kanned tile, before the call.
  - In the arena a discard of the tile just drawn is always marked tsumogiri, so a draw followed
    by a discard from hand can't be the discarded tile.
- **Sampler (the uniform proposal of D38):**
  - It deals each opponent's unknown slots in time order from the unseen tiles. Each draw is
    proportional to the tile's count, among the choices that keep every requirement possible
    (Hall's condition on the deadlines, and enough copies left).
  - The importance weight is the product of the possible share at each slot. The undealt slots
    are shuffled.
  - A particle is accepted when the kyoku's public events replay through it as the seat saw
    them (`replay_seen_by`). That covers what the sampler doesn't enforce, such as a riichi hand
    being tenpai.
- **Tests** (Rust, on rule-based games and a random network's calls fixture,
  `tests/data/search`):
  - the true history always fits the layout;
  - every accepted particle gives back the seat's exact root observation and all 136 tiles;
  - masked and full logs give identical particles (no leak).
- **Acceptance on the smoke run's 256 riichi decisions** (64 particles per root, at most 20K
  tries, 8 s in all):

  | Opponents in riichi | Roots | Accepted / tried | Roots short of 64 |
  |---|---|---|---|
  | 0 | 189 | 21% | 0 |
  | 1 | 59 | 0.13% | 48 |
  | 2 | 8 | 0 | 8 |

  - Uniform tiles almost never make a riichi player's hand tenpai.
  - The proposal's log weights also spread by a median of 27 within a root, so the effective
    sample size is about 1. Opponents with many discards from hand force tiles late in some
    samples and not in others.
  - **Next:** D38's constraint-preserving swap moves. Start from one valid particle (repairing
    riichi hands to tenpai), then run a Metropolis chain of tile swaps between unknown slots
    whose target is uniform over consistent histories. Particles then need only the likelihood
    weights.

## Belief stage: S-1 particles, swap-move chain (2026-10-15)

- **Chain** (`Layout::chain_until`, D38's constraint-preserving moves):
  - It swaps the tiles of two unseen slots, the first in an opponent's hand.
  - A swap is rejected when it breaks a constraint: what an opponent had to hold, and by when;
    the arena's tsumogiri rule; a riichi hand being tenpai after its riichi discard.
  - The proposal is symmetric, so the chain's stationary distribution is uniform over the
    histories consistent with the public events: the prior D38 asks for. Particles then carry
    no proposal weight; only the action likelihoods will weight them.
  - The undealt tiles of each particle are shuffled, an exact step for them.
  - The chain starts from one v1 sample whose riichi hands are repaired to tenpai by greedy swaps
    (the best of 32 per step). A 2,000-swap burn-in forgets the start, and there are 200 swaps
    between particles.
  - Each particle must still replay the seat's view. The one rule the chain doesn't model, a
    riichi player's later ankan keeping its waits, is caught there.
- **Fixes on the way:**
  - The v1 sampler's look-ahead now runs an exact bipartite matching of needs to slots.
  - Counting arguments missed two kinds competing for one slot, typically a pair cut from hand
    over two turns. Under the arena's tsumogiri rule, the draw between the two discards can't be
    that tile.
- **Smoke run's 256 riichi decisions, 64 particles each:** 0.2 s in all on 16 cores.

  | Opponents in riichi | Roots | Valid / drawn | Roots short of 64 |
  |---|---|---|---|
  | 0 | 189 | 12,096 / 12,096 | 0 |
  | 1 | 59 | 3,718 / 3,969 | 1 (a riichi ankan of 9m) |
  | 2 | 8 | 512 / 512 | 0 |

- **The chain samples the right distribution** (`the_chain_agrees_with_importance_sampling`):
  - The test uses 6 roots at turns 4–6, where the uniform proposal's importance weights are
    reliable (ESS 475–1,889 of 12,000).
  - The mean count of each tile kind in each opponent's hand agrees with the chain's over 612
    comparisons: mean z² 1.04, max |z| 3.91.
  - With only 50 swaps between particles it was 1.22 and 4.34: autocorrelation, so the default
    is 200.
  - Later in a kyoku the importance weights collapse (ESS often below 10 of 4,000), which is
    why the chain is needed.

## Belief stage: likelihood weights, and why v1 fails (2026-10-15)

- **Code:** `likelihood.rs`, and `Beliefs.weigh` (`belief.rs`).
  - It replays each particle as the seat saw the kyoku and records every opponent decision.
  - For each decision it records the actions consistent with the log. A claim that lost to a
    higher-priority claim never reaches the log, so for the overridden player every action
    weaker than the logged claim counts.
  - It encodes the opponent's observation in the particle and runs the blueprint. The particle's
    weight is the sum of log Σ π(consistent actions), floored at 1e-8.
  - Decisions where every action the opponent could take is consistent are skipped (factor 1).
  - `true_log_likelihoods` and `root_shanten` score the true history the same way. They're
    lab-only and read the seed.
- **The weights point the right way** (48 smoke-run roots, pilot A): the true history's median
  log-likelihood is −84 against −157 for a typical particle (best of 16: −117). The truth beats
  every particle at 71% of roots.
- **But they collapse:** the median ESS is 1.0 of 64 at every root. Weighing costs 0.79 s per
  root on the 3060 (≈ 60 opponent decisions per particle to encode).
- **Calibration** (128 roots, 341 non-riichi opponents, "is this opponent tenpai at the root"),
  with the likelihood tempered by β:

  | β | Median ESS of 64 | Log loss | Brier |
  |---|---|---|---|
  | 0 (prior only) | 64 | 0.764 | 0.168 |
  | 0.03 | 34 | 0.756 | 0.167 |
  | 0.1 | 5.4 | 0.759 | 0.164 |
  | 1 | 1.0 | 0.778 | 0.164 |
  | constant base rate (17.6%) | | 0.465 | 0.145 |

- **Why:** real players build structured hands, and uniform tiles consistent with the public
  events don't look like them.

  | Shanten at the root (non-riichi opponents) | Truth | Prior particles |
  |---|---|---|
  | 0 (tenpai) | 17.6% | 0.5% |
  | 1 | 45.5% | 3.9% |
  | 2 | 26.4% | 23.6% |
  | 3+ | 10.3% | 67.3% |

  The posterior sits far out in the prior's tail (likelihood gaps of e^70 or more), and 64–256
  reweighted prior samples can't reach it.

### What next (a design choice for the user)

1. **D38 as written: sequential filtering.** Update weights at every opponent decision,
   resample at ESS < 0.3N, and rejuvenate with swaps accepted by the likelihood ratio.
   - Each rejuvenating swap re-scores the opponent's later decisions, ≈ 20 network calls.
   - Offline, every pilot root pays the whole kyoku: on the order of 10⁵ calls per root, so
     days for 20K roots on one GPU.
   - At play time the cost spreads over the kyoku.
   - It is also unclear that local swaps can climb from prior hands to the posterior.
2. **A learned proposal** (D38 allows it once a joint sampler with a computable density exists):
   an autoregressive model of each opponent's concealed hand, given the public information.
   - Training data is free: self-play logs, where the truth is known.
   - Particles would be drawn from it, completed into consistent histories by the chain (which
     keeps the hand fixed), and weighted by likelihood × prior ÷ proposal.
   - This is what LBS does, and the most promising route, but it is a model to build, train and
     validate.
3. **A cheap opponent model inside the chain:** for example, tile efficiency (a discard that
   keeps shanten is likely), so swaps climb towards structured hands at microseconds a step.
   - It isn't the blueprint, so it departs from D38's likelihood and would need an `08` entry.

## Belief stage: the learned hand model (D58, 2026-10-15)

- **Model** (`search/handmodel.py`):
  - Pilot A's encoder (b8c192, v4 observation) reads the deciding seat's observation.
  - A GRU then gives each other seat's concealed counts, kind by kind from 1m to C, in turn
    order after the decider. It carries the hand built so far, so shapes are modelled jointly.
  - Each kind's count is drawn among the counts still possible: no more than the copies the
    decider hasn't seen and earlier seats haven't taken, and no more than the tiles left, while
    leaving room for the rest.
  - The density is the product of those renormalized probabilities, so it is exact (tested: a
    sample's log q equals the training path's log-probability).
  - Red fives are dealt uniformly among the fives a seat holds.
- **Data:** `hand_examples` replays a full self-play log and gives, at every decision, the
  decider's observation and the others' true hands. 4,000 games of pilot A self-play (dev key,
  nonces 100,000+) train it; the smoke run's 40 games (nonces 0–39) are held out.
- **Particles** (`Layout::complete`, `Beliefs.from_hands`):
  - Each sampled hand goes, with the tiles the seat must have held, into its unseen entries by a
    bipartite matching (needs in time, excluded draws kept clear), and swaps within the seat
    spread the arrangement.
  - A riichi hand must be tenpai, or the draw is dropped; the stage draws twice the particles
    it needs. The undealt tiles are shuffled.
  - Tested: the true hands at 300+ roots always complete into a history that replays the seat's
    view and its root observation.
- **Pilot stage:** `python -m search belief` plays both candidates on the same particles.
  `report` scores the oracle and belief conditions on the same true worlds and gives D32's
  recovery ratio under the primary rule. `python -m search latency` times one search at a time,
  request to answer.

### Results (2026-10-15, pilot A, dev keys; D58's test)

- **Training:** 20,000 steps of 256 examples (every 2nd decision of 4,000 games), ≈ 37 min on
  the 3060 at 2,270 examples/s. Held-out loss went from 31.5 nats per hand (step 0) to 17.64,
  still falling slowly (17.70 → 17.66 → 17.64 over the last 5,000 steps).
  `workdir/search/hands-a/handmodel.pth` (read-only, sha256 `cf371ba71b6588fe…`).
- **Calibration** on the smoke run's 256 riichi decisions (held-out games), 64 samples per root:

  | | Learned model | Constant base rate | Uniform particles (v1) |
  |---|---|---|---|
  | Tenpai, non-riichi opponents (693): log loss | **0.258** | 0.415 | 0.76 |
  | Tenpai: Brier | **0.077** | 0.125 | 0.168 |
  | Tenpai: predicted rate (truth 14.6%) | 14.1% | | 0.5% |
  | Waits, opponent × kind (26,112 pairs): log loss | **0.047** | 0.060 | |

  - Riichi opponents (75): only 68% of samples are tenpai. The rest fit no history and are
    dropped, and the stage draws twice the particles it needs. 10 of 256 roots got fewer than
    64 (minimum 19).
- **The likelihood correction makes it worse** (128 roots, particles weighted by exp(β · log
  likelihood)). D58 keeps it only if it helps, so the model alone is the belief.

  | β | Median ESS of 64 | Tenpai log loss | Waits log loss |
  |---|---|---|---|
  | 0 (model alone) | 64 | **0.287** | **0.053** |
  | 0.1 | 4.5 | 0.369 | 0.072 |
  | 0.3 | 1.4 | 0.597 | 0.101 |
  | 1 | 1.0 | 0.775 | 0.123 |
  | constant base rate | | 0.465 | 0.068 |

- **Latency** (06 §0's criterion is p99 < 2.5 s on deploy-class hardware; this is the 3060, 60
  decisions one at a time, 64 particles): total **p50 0.68 s, p99 2.21 s** (max 2.35 s).
  Sampling and completing the hands takes p50 0.10 s, rollouts p50 0.58 s and p99 2.06 s,
  valuation < 0.02 s.
- **End to end** on the 256 smoke-run decisions (130 s; dev data, far too few for a verdict):
  - oracle, primary rule: +0.009 u (z 1.5), switching at 12% of decisions;
  - belief: −0.002 u (z −0.8), switching at 2%;
  - clairvoyant bound +0.075 u (z 4.5).
  - The recovery ratio means nothing at this size; the pilot's 20K decisions decide it.

## Pilot run on pilot A (2026-10-19/20, the user's go-ahead): oracle passes, belief fails

- **Why pilot A:** no RL arm has passed G0, so the user chose to run the pilot on the SL blueprint
  while the G0 retry (`k2b`) runs (2026-10-19). If `k2b` fails, the SL incumbent plus search is
  the main path and this is the blueprint that matters. If it passes, the pilot runs again on
  `k2b`'s blueprint, on the same block.
- **Run:** `workdir/search/pilot-a` (results read-only). Self-play: 1,200 games of pilot A
  (`15121432…`) on the new family-A block `search-pilot` (nonces 43,000–44,199,
  `20261019-122435-search-selfplay`), 4.5 min. 25,149 riichi-legal decisions; 20,000 sampled
  (seed 0; 373 excluded as non-discards). Oracle 2.2 h, belief 2.8 h on the 3060, 19:24 →
  00:27 UTC. The hand model is the one from D58's test (`hands-a`, `cf371ba7…`). Stages in order:
  `selfplay --block search-pilot --games 1200`, `decisions --count 20000`, `oracle --redeals 64`,
  `belief --hand-model workdir/search/hands-a/handmodel.pth --particles 64`, `report`.
- **Reproduced 20,000/20,000:** the policy's action in the true world gives back the self-play
  kyoku.

| Per decision, u (paired, SE clustered by game) | Oracle | Belief |
|---|---|---|
| **plain z > 2 (primary, D58)** | **+0.0112 ± 0.0009 (z 12.7), switches 10.3%** | **+0.0018 ± 0.0005 (z 3.4), switches 3.9%** |
| plain z > 0 | +0.0112 ± 0.0016 (z 7.1), 32.5% | −0.0003 ± 0.0015 (z −0.2), 29.9% |
| regularized λ = 0.05 | +0.0095 ± 0.0010 (z 9.6), 8.9% | +0.0025 ± 0.0007 (z 3.3), 6.0% |
| regularized λ = 0.1 | +0.0063 ± 0.0007 (z 8.5), 5.3% | +0.0016 ± 0.0005 (z 3.0), 3.4% |
| Clairvoyant bound (the better candidate in the true world) | +0.0774 ± 0.0017 | |

- **Recovery ratio (primary rule): 0.157, 95% CI [0.071, 0.24]** (game-clustered bootstrap,
  10K resamples, `experiments/search_pilot_breakdown.py`; none reached 0.5).
- **Latency** (`python -m search latency`, 100 decisions one at a time, 64 particles, the 3060):
  total **p50 0.82 s, p99 1.95 s**, max 2.28 s. Rollouts p50 0.71 s and p99 1.86 s; beliefs p50
  0.10 s. `latency.json`.
- **D32's go criteria for scaling search:** the oracle gain is significant (passes); **recovery
  ≥ 0.5 fails**; p99 latency < 2.5 s passes on the 3060, so it would on a deploy-class 4090.
  **The pilot fails on recovery.** By D32's diagnosis rule (oracle passes, belief fails), the fix
  is in reconstruction, calibration or the opponent model.
- **Where the gains are** (`search_pilot_breakdown.py`):

  | Policy's choice (share) | Oracle: gain, switches | Belief: gain, switches |
  |---|---|---|
  | riichi, the search may go dama (41%) | +0.0100, 8.4% | +0.0004, 1.4% |
  | dama, the search may riichi (59%) | +0.0121, 11.6% | +0.0027, 5.7% |

  - Almost all of the belief's gain is in riichi where the policy went dama. When the policy
    riichis and the oracle would go dama, the belief recovers nearly nothing. A guess, not
    tested: that choice depends most on what the opponents hold.
  - The belief's switches overlap the oracle's little: 262 of its 786 are also oracle switches
    (33%), and it finds 13% of the oracle's 2,055. Per switch it gains +0.045 u, the oracle
    +0.109.
- **Not a strength result.** The oracle condition is a lab-only diagnostic. The belief gain is
  one-decision paired Δu in the true worlds, not a full-game T-std (06 §0, D33a).
- **Open (the user):** whether to work on the belief now (D32's rule: reconstruction,
  calibration, opponent model), and what that would be. Candidates: the hand model's
  calibration on riichi → dama decisions specifically, more particles against noise (plain z > 0
  collapses to ≈ 0, a sign of noise), and a stronger hand model (its held-out loss was still
  falling at 20K steps).

### Why belief fails: noise or bias? (2026-10-20, exploratory)

- **The belief's gain is search, not "riichi more often"** (`search_pilot_riichi_more.py`). A
  no-search rule that riichis on the same number of dama decisions (667), picked by the policy's
  own riichi share, gains +0.0004 ± 0.0004 against the belief's +0.0018 ± 0.0005. Only 61 of its
  picks are the belief's.
- **Accuracy** (`search_pilot_accuracy.py`): per decision, the belief's estimate of
  riichi − dama against the oracle's.

  | Decisions | Correlation | Without sampling noise | Belief − oracle (u) | Share of the disagreement that is noise |
  |---|---|---|---|---|
  | All 20,000 | 0.39 | 0.52 | +0.0091 | 28% |
  | Early in the kyoku (first third by event index) | 0.46 | 0.65 | +0.0035 | 43% |
  | Middle third | 0.39 | 0.52 | +0.0076 | 29% |
  | Late third | 0.31 | 0.39 | +0.0165 | 17% |

  - "Without sampling noise" divides by the split-half reliabilities (oracle 0.87, belief 0.66).
    Even with unlimited particles, the belief would track the oracle at ≈ 0.5.
  - Most of the gap is systematic: **the belief rates riichi above the oracle**, by more the
    later in the kyoku. That fits its switches: 667 to riichi and 119 to dama.
  - A guess, not tested: the sampled hands are less threatening than the true ones late in the
    kyoku (opponents near tenpai), which makes riichi, after which the player can't fold, look
    better than it is. 822 decisions had fewer than 64 particles, because sampled hands for
    riichi opponents often weren't tenpai (68% were, in D58's test).
- **The hand model's training curve** (`workdir/search/hands-a/train.log`): held-out loss 31.5
  nats per hand at step 0, 18.49 at 2,500 steps, 17.90 at 10K, 17.64 at 20K (cosine schedule to
  0). Train ≈ held-out, so no overfitting. Only the final model was kept, so recovery against
  model quality hasn't been measured.

### Recovery against hand-model quality (2026-10-20, exploratory; the user's go-ahead)

- **Setup:** a weaker hand model, the same recipe and data as `hands-a` stopped at 2,500 steps
  (cosine schedule to 0 there): held-out 18.55 nats per hand against 17.64.
  `workdir/search/hands-a-2500/handmodel.pth` (read-only, sha256 `35614b30370f3667…`), 5 min.
  The belief stage with it on pilot-a's first 4,000 decisions (a random subset; ids follow the
  seed-0 shuffle), 34 min: `workdir/search/pilot-a-sub4k-hands2500`. Compared on the same
  decisions and true worlds with `experiments/search_pilot_handmodel_slope.py` (output
  `slope.txt` there).

  | 4,000 decisions, primary rule | Weak model (18.55) | Current model (17.64) |
  |---|---|---|
  | Belief gain, u per decision | −0.0005 ± 0.0011 | +0.0025 ± 0.0010 |
  | Recovery (oracle +0.0089 on this subset) | −0.06 [−0.38, 0.17] | 0.28 [0.07, 0.50] |
  | Correlation with the oracle, without noise | 0.45 | 0.53 |
  | Riichi overrating, all / late third | +0.0148 / +0.0274 | +0.0089 / +0.0166 |
  | Decisions with < 64 particles | 1,040 | 168 |

- **Paired difference, current − weak: +0.0030 ± 0.0012 u per decision (z 2.5).**
- **Reading:** hand-model quality is the lever. 0.9 nats per hand took the belief gain from
  nothing to the full run's level, cut the riichi overrating by 40% and the particle shortfall by
  6×. The current model's recovery on this subset (0.28) is above the full run's 0.157; that is
  subset noise, and the full run's number stands.
- **Extrapolation, two points only:** if the gain rose linearly with the loss, recovery 0.5 on the
  full set would need ≈ 1.2 nats more, against the 0.9 the last 17,500 steps of the current recipe
  gave. So the same training run longer is unlikely to be enough on its own. A better model
  (more self-play data, which is cheap; a larger head; a longer schedule; more weight on late-kyoku
  and tenpai opponents) is the direction.

### A bigger hand model: lower loss, same search (2026-10-20, exploratory; the user's go-ahead)

- **Setup:** the same architecture, learning rates, stride and held-out games as `hands-a`, with
  5× the data and 3× the steps: 16,000 more games of pilot A self-play on the dev key (nonces
  104,000–119,999, 55 min) plus `hands-a`'s 4,000; 60,000 steps (1.9 h). Held-out loss **17.21**
  nats per hand (17.68 at 20K steps, 17.39 at 40K; still falling slowly: 17.25 → 17.21 over the
  last 10K). `workdir/search/hands-a-big/handmodel.pth` (read-only, sha256 `7d38d4035dd59329…`). The belief
  stage on all 20,000 decisions, 2.8 h: `workdir/search/pilot-a-handsbig`
  (`slope-full.txt`, `slope-sub4k.txt`).

  | 20,000 decisions, primary rule | Current (17.64) | Bigger (17.21) |
  |---|---|---|
  | Belief gain, u per decision | +0.0018 ± 0.0005 | +0.0016 ± 0.0005 |
  | Recovery | 0.157 [0.071, 0.240] | 0.140 [0.058, 0.220] |
  | Correlation with the oracle, without noise | 0.52 | 0.52 |
  | Riichi overrating, all / late third | +0.0091 / +0.0165 | +0.0069 / +0.0138 |
  | Decisions with < 64 particles | 822 | 472 |

- **Paired difference, bigger − current: −0.0002 ± 0.0005 u per decision (z −0.4).** None.
- **Reading:** the trend from the weak model (18.55) to the current one doesn't continue. 0.43
  nats less loss cut the riichi overrating by a quarter and the particle shortfall by 40%, but the
  belief's verdicts track the oracle's no better, and the gain didn't move. So more data and steps
  of this recipe don't close the gap, and the riichi bias alone isn't what holds recovery down.
- **Open:** whether the gap can be closed at all. Part of the oracle's gain may come from
  information no inference from public events can recover (06 §0's caveat that the oracle is an
  optimistic bound). Untested.

## The pilot on `k2b`'s network (2026-10-26; the conditional rerun after G0's pass)

- **Run:** `workdir/search/pilot-k2b`, by `experiments/k2b_post_g0_chain.sh`. Blueprint `k2b` full
  (`f5a22c5c…`). Hand model on 4,000 games of its self-play (dev key, 40 held out), the current
  recipe: held-out 17.52 nats per hand at 20K steps (`hands-a`: 17.64). Self-play on the family-A
  block `search-pilot`, the same deals as pilot A's run; 20,000 decisions, all reproduced.

  | Per decision, u (primary rule: plain z > 2) | Pilot A | **`k2b`** |
  |---|---|---|
  | Oracle | +0.0112 ± 0.0009 (z 12.7), switches 10.3% | +0.0083 ± 0.0008 (z 9.7), 9.1% |
  | Belief | +0.0018 ± 0.0005 (z 3.4), 3.9% | **+0.0005 ± 0.0004 (z 1.1)**, 3.1% |
  | Recovery | 0.157 | **0.055** |
  | Clairvoyant bound | +0.0774 | +0.0779 |
  | Latency p99 (the 3060) | 1.95 s | 1.87 s |

- **Reading:** the stronger blueprint leaves the oracle less to gain (−26%), and the belief's gain
  is no longer distinguishable from 0. D32's go criteria fail on recovery again, more clearly.
  This fits the parked state: the belief, not the blueprint, limits riichi-vs-dama search.

