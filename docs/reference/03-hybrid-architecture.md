# Hybrid Transformer Architecture for Riichi Mahjong

## Overview

This is a domain-aware design for a riichi mahjong agent intended to run on consumer hardware (single high-end GPU) and reach competitive-with-Mortal strength. It processes game state through three stages: per-zone encoders that compress structured local information using strong inductive biases, a top-level transformer that performs cross-zone reasoning, and a set of action heads plus a value head that produce decisions and learned baselines.

The architecture's defining structural feature is a dedicated chronological action-history zone running in parallel with per-player discard-pile zones. This makes cross-player temporal-causal patterns (riichi-then-defense, kan-then-flip, pon-then-turn-skip) first-class in the input representation rather than something the model must reconstruct from spatial encodings. This is the main place the design differs from both Mortal (a 40-block ResNet on stacked feature planes) and Tjong (a Transformer-in-Transformer with a 4-step sliding-window memory).

The training pipeline is policy-based actor-critic throughout: supervised pretraining produces both a policy and a value function, and self-play refinement uses PPO with KL regularization to the pretrained policy and a global-reward value baseline. The design forks Mortal-Policy rather than Mortal main, since the value-based to policy-based transition is already done there.

## Implementation Strategy

Before describing the architecture, the implementation approach matters because it constrains some design decisions. The agent is built as a fork of Mortal-Policy, keeping:

- The full Rust simulator (`libriichi`) — rules, scoring, mjai protocol, arena
- The replay parsing and dataset ingestion pipeline (Tenhou and Mahjong Soul)
- The training infrastructure scaffolding (parameter server, replay buffer, GPU inference engine, checkpointing)
- The mjai protocol layer (for compatibility with the existing review ecosystem)

What gets replaced or substantially modified:

- The feature extraction layer in `libriichi::dataset` — refactored to emit per-zone tensors plus a chronological action stream rather than stacked feature planes
- The Python network module — fully rewritten
- The loss function and RL recipe — rewritten to fit the new architecture and head layout

This staging matters because it makes the project tractable. The simulator is the single most expensive piece to rebuild from scratch (battle-tested rules implementations are hundreds of hours of edge-case work), and it's exactly the part that's domain-shared with Mortal.

## Input Zones

The game state decomposes into the following zones. Symmetric zones (the four discard piles, the four meld sets, the four player-state vectors) share encoder weights across players, with player identity injected via a learned embedding added to the zone summary.

**Concealed hand zone.** The private 13- or 14-tile hand. Encoded as a 34×4 binary matrix (34 tile types, up to 4 copies each), augmented with derived features computed in the simulator and concatenated to the encoder input: shanten count, ukeire (tile acceptance count), per-tile waits if tenpai, available yaku flags, estimated hand value at tenpai, and red-five flags. Computing these in Rust is cheap and removes a significant burden from the network — the model shouldn't have to learn shanten from scratch.

**Per-player discard pile zones (×4).** Each player's discard pile in pile order, with per-tile flags for tedashi/tsumogiri, riichi declaration tile, and called-away marker. Per-tile danger features are also computed by the simulator and appended: genbutsu/suji/kabe flags relative to each living opponent, tile-fresh-vs-late timing. Variable length up to ~24 tiles, padded with attention masking.

**Per-player meld zones (×4).** Each player's called melds (chi, pon, kan, ankan), with per-meld type and the tile called from which player. Variable length, typically 0-4 melds.

**Per-player state zones (×4).** Per-player scalars: current score, riichi status, riichi turn, seat wind, recent call activity. Encoded by a small shared MLP.

**Global action-history zone.** The chronological interleaved stream of all actions across all players from the start of the round: discards, calls, riichi declarations, kans, dora reveals. Each action is a token tagged with player identity, action type, turn index, and event-timing features. Length is bounded by ~70 actions per round, typically 30-50 in practice.

Critically, the action-history zone uses **differentiated features** from the per-player pile zones to prevent pathway collapse during RL: the action-history tokens carry timing-and-event information (who acted when, what type of action, inter-event gaps, what got called away) but only low-resolution tile features (suit category, terminal/honor flag, rough number). The per-player pile zones carry full tile identity and danger features. This way the two views are genuinely complementary rather than redundant — the action-history zone learns the chronology, the pile zones learn the tile-level state.

**Global state zone.** Round wind, dealer, honba, riichi sticks, dora indicators, kan-dora indicators, wall tiles remaining, your seat, current turn number. Encoded as a vector via a small MLP.

## Per-Zone Encoders

Each zone is processed by a small encoder that produces one or more fixed-size summary vectors. Encoders for symmetric zones share weights; player identity is added via a learned embedding to the summary output.

**Hand encoder.** A small ResNet block operating on the 34×4 representation (treating it as a 2D grid where suit structure can be captured via 2D convolutions), followed by an MLP that concatenates and processes the derived hand features (shanten, ukeire, etc.). 2D convolutions across the 34×4 grid capture both within-suit shape (sequences, kanchan, suji) and across-copy patterns (pairs, triplets, kans) that pure 1D-along-tile-number convolutions miss. Output: one 256-dim summary vector.

This is a change from the original 1D-CNN design. Mahjong hand patterns aren't really 1D — pairs/triplets/kans don't have spatial locality at all, they're set-membership patterns that benefit from operating on the full 34×4 grid.

**Discard-pile encoder (shared, ×4).** A small transformer encoder, 2 layers at 128 dim with 4 heads, operating over the pile sequence with positional embeddings encoding pile order. Emits **three summary tokens per pile** by default — early-phase, middle-phase, and late-phase summaries via attention pooling with three learned query vectors. The original design treated multi-token piles as an ablation fallback; making it the default is cheap (top transformer easily handles a few extra tokens) and pile information is too strategically important to compress to a single vector. Output: three 256-dim tokens per pile (twelve total across four piles).

**Meld encoder (shared, ×4).** A small MLP per meld followed by mean-pooling across melds. Output: one 256-dim summary vector per player.

**Player-state encoder (shared, ×4).** A two-layer MLP mapping the per-player scalar features to a 256-dim vector.

**Action-history encoder.** A small transformer encoder, 3-4 layers at 256 dim with 8 heads, operating over the chronological action stream with positional embeddings encoding global turn index. Each input token combines learned embeddings for player identity, action type, low-resolution tile features, and timing flags. Output: one 256-dim summary via attention pooling, plus retention of the full sequence for the top transformer to attend into directly.

**Global-state encoder.** A two-layer MLP producing a 256-dim vector.

## Top-Level Transformer

After per-zone encoding, the input to the top transformer is a sequence of approximately 60-80 tokens:

- 1 hand summary
- 12 discard-pile summaries (3 per pile × 4 piles)
- 4 meld summaries
- 4 player-state summaries
- 1 action-history summary plus 30-50 retained action-stream tokens
- 1 global-state summary

Each token has zone-type and player-identity embeddings added. The action-stream tokens additionally carry their original turn-index positional embeddings, so the top transformer can reason about temporal positions within the history.

The top transformer is a 6-layer encoder at 384 dim with 8 heads. Standard pre-norm transformer blocks with GELU activations and dropout 0.1. All tokens are bidirectional — there's no causal masking, because at decision time the model has full access to everything that has happened.

The combined sequence is around 60-80 tokens, still cheap to process. The bulk of FLOPs are in the encoders and action heads.

## Action Heads and Value Head

Each action head uses a learned query vector that cross-attends into the top-transformer output, producing a context vector that a small MLP turns into logits. The MLPs are deliberately tiny (one hidden layer) to keep capacity in the shared trunk where it belongs.

**Discard head.** Cross-attention query produces 256-dim context; MLP produces 34 logits, masked to legal discards. Optionally separate logits for red-five variants.

**Riichi head.** Triggered only when riichi is legal. Binary logit (declare / don't).

**Call heads (chi, pon, kan).** One head per call type, each triggered only when legal. Chi has three sub-options (low/middle/high sequence position) when multiple chi configurations are possible.

**Win/draw heads (ron, tsumo, kyuushuu kyuuhai, ankan during riichi).** Binary heads triggered when applicable.

**Value head.** Cross-attends into the trunk output and produces two outputs: (a) round-end score delta, regressed via MSE; (b) hanchan-end placement probability distribution (4-class softmax over 1st/2nd/3rd/4th). The placement output is the more important of the two during RL — it's the signal that captures Mahjong's actual reward structure. Trained from supervised replay outcomes during stage 1 so the head is warm before RL begins.

The value head is non-negotiable for stage 2 training, and pretraining it during stage 1 is what makes self-play actually improve the agent rather than spend the first weeks of RL learning a value function from scratch on noisy returns.

**Auxiliary heads (training-only).** Three small heads attached to the trunk that don't run at inference but provide useful gradient signal during SL pretraining:

- *Opponent tenpai prediction:* per-opponent binary logit predicting whether they're in tenpai
- *Opponent waits prediction:* per-opponent 34-class distribution over likely waiting tiles
- *Opponent hand value prediction:* per-opponent regression over expected hand value if they win

These give the trunk concrete reasons to encode defensive information, which is otherwise underdetermined by SL on action prediction alone (the model can predict discards reasonably well without learning to model opponents). This is *not* oracle guiding in the Suphx sense — Suphx's oracle guiding is an RL technique involving gradual feature dropout — it's a multi-task learning auxiliary. Cheap, effective, and orthogonal to the main objective.

## Training: Stage 1 (Supervised Pretraining)

Train on Tenhou and Mahjong Soul replay data, on the order of millions of games. For each decision point in each replay, the model predicts the action the human took with cross-entropy loss summed across the heads that were active at that decision. The value head is trained simultaneously on the eventual round score and final placement. The auxiliary opponent-modeling heads are trained on ground-truth hidden information from the replay (which is visible in retrospect).

Total loss:
```
L_total = L_action + α · L_value + β · L_opponent_aux
```
with `α` around 0.5 and `β` around 0.2 (tune by watching individual loss curves; the action loss should dominate but the auxiliaries shouldn't be ignored).

Hyperparameters:
- AdamW with linear warmup and cosine decay
- Learning rate around 3e-4
- Batch size as large as memory allows
- Dropout 0.1, weight decay 0.01
- Gradient clipping at norm 1.0

Plan for **1-2 weeks of pretraining** on a single RTX 3090/4090, not "a few days." Mortal's full pipeline takes weeks on better hardware, and this architecture is more complex. Expect **low-dan accuracy targets** before RL: roughly 75-80% discard accuracy, 95%+ on call/pass decisions. These numbers are conservative against Tjong's published 81.5% discard accuracy on MCR data — riichi is harder for discard prediction because the action distribution is more concentrated, but the action-history zone should compensate.

## Training: Stage 2 (Self-Play RL Refinement)

Use Mortal's existing self-play infrastructure: parallel game-simulator instances generating trajectories, with the model running inference centrally on GPU. The training algorithm is **PPO with KL regularization to the pretrained policy and a learned value baseline**.

Loss components combined per update:
```
L_RL = L_PPO_clipped + γ · L_value_MSE + δ · L_entropy + ε · L_KL_to_prior
```

- **PPO clipped policy loss** with clip range 0.1-0.2 (start tight)
- **Value MSE loss** on actual returns, with the placement-distribution component weighted higher than the round-score component
- **Entropy bonus** with adaptive scheduling — log policy entropy throughout training, raise the coefficient if entropy collapses too fast, lower it once exploration has done its work. This is the single most fiddly hyperparameter; budget time for it.
- **KL-to-prior** keeping the current policy close to the SL pretrained policy. Generous coefficient early (0.1-0.5), decay slowly. This is what prevents the catastrophic regression most self-play projects suffer when RL noise destroys the carefully-trained SL policy.

Critical hyperparameters:
- Small learning rate (1e-5 to 5e-5, 10× smaller than pretraining)
- Gradient clipping at norm 0.5 (tighter than SL)
- GAE-λ around 0.95 for advantage estimation
- Roughly 4 PPO epochs per batch of trajectories

**Instrumentation is non-optional.** Log continuously:

- Win rate against frozen SL baseline (every N updates)
- KL divergence between current policy and SL policy
- Policy entropy per action type
- Value function calibration (predicted vs. actual returns, on a held-out set of self-play trajectories)
- Periodic 1v3 evaluation matches against stock Mortal as an external sanity check

If win rate against the SL baseline drops below ~50% and stays there, stop training — something has gone wrong and continuing wastes compute. Most common causes, in order: KL coefficient too low, learning rate too high, value function diverging.

Plan for **4-8 weeks of wall-clock self-play**. Periodic evaluation against the pretrained checkpoint and against frozen Mortal tells you whether RL is actually helping; it's easy for self-play to silently regress, especially in the first week.

## Why This Architecture

**Sample efficiency.** Per-zone encoders carry strong inductive biases (locality for the hand ResNet, set-aggregation for melds, sequence-awareness for piles, scalar processing for state vectors) that converge quickly with limited data. The top transformer only learns cross-zone integration, a smaller problem than learning everything from raw tile tokens. This matters in the 1M-10M-games regime accessible to a personal project.

**Cross-player temporal structure as first-class input.** The action-history zone makes interleaved chronology explicit, so patterns like riichi-then-defense, post-kan dora flips, and turn-skip-after-call are directly visible. Tjong's Transformer-in-Transformer uses a 4-step sliding-window memory; that's a much weaker temporal mechanism than the full chronological event stream with proper positional embeddings. Mortal's stacked feature planes don't represent chronology explicitly at all. This is the design's biggest architectural bet relative to the literature.

**Differentiated zone views prevent pathway collapse.** The action-history zone's timing-focused features and the pile zones' tile-focused features genuinely complement rather than duplicate each other, which matters under RL where attention can drift unpredictably to whichever pathway is working at any moment.

**Stability under RL.** The CNN and small-encoder portions act as regularizers — they don't change much during RL, concentrating policy updates in the top transformer where they're wanted. The KL-to-prior term reinforces this. Combined, these reduce the wild policy drift that pure-transformer architectures sometimes suffer in self-play (this is a hypothesis based on the recipe's structure, not a proven property).

**Modularity.** If a specific zone underperforms, the encoder for it can be upgraded without rearchitecting. Built-in instrumentation (per-zone attention rollout, per-head ablation) makes debugging tractable.

**Value head structure matches Mahjong's reward structure.** Predicting placement distribution rather than per-round score directly captures what actually matters at hanchan end, which is what every strong mahjong agent's training signal eventually has to converge on (Suphx via global reward prediction, Tjong via fan backward decomposition).

## Known Weaknesses

**Information bottleneck per zone.** Compressing each pile to three summary tokens is better than one but still lossy. The action-history zone retains its full sequence, which compensates for the most strategically important temporal information. If ablation shows pile information remains the bottleneck, the next step would be Perceiver-style cross-attention from the top transformer back into the per-pile token sequences rather than relying only on summaries.

**Implicit-action gaps.** The architecture encodes what happened, but not what could have happened but didn't — e.g., a player passing on a callable tile, which itself encodes information about their hand. This is shared with virtually all existing systems; treating as out of scope unless evaluation reveals it as a real weakness.

**Scaling ceiling.** With sufficiently large datasets (50M+ games), a pure token-level transformer would likely match or exceed this hybrid. The hybrid is the right answer for the data regime accessible to a personal project (1M-10M games), not the long-term ceiling.

**Architectural complexity.** More design decisions than a pure transformer or pure ResNet. Each one is something to get wrong, and encoder weight-sharing across symmetric zones requires careful implementation to avoid accidentally breaking parameter sharing during distributed training.

**RL is genuinely hard.** Policy-based RL with deep networks fails subtly and often silently. The instrumentation list above isn't optional; without it, the most likely outcome of stage 2 is "two weeks of training that quietly produced a worse agent."

## Approximate Parameter Budget

Reasonable starting configuration:

- Hand encoder (ResNet block + MLP): ~200K
- Discard-pile encoder (shared, multi-token output): ~500K
- Meld encoder (shared): ~50K
- Player-state encoder (shared): ~30K
- Action-history encoder: ~2M
- Global-state encoder: ~30K
- Top transformer (6L, 384d, 8h): ~10M
- Action heads (lean): ~500K total
- Value head: ~300K
- Auxiliary heads (training-only, dropped at inference): ~500K

**Total: ~14M parameters at inference, ~14.5M during training.** Comfortable on a single RTX 3090/4090 for both training and inference. Headroom to scale the top transformer to 8 layers / 512 dim if compute allows, or shrink the action-history encoder if memory pressures appear.

For comparison, this is in the same parameter scale as Tjong (15M) and substantially smaller than Mortal's 40-block 192-channel ResNet, which puts the design's burden where it should be: on architectural efficiency, not parameter count.

## Build Sequence

The architecture should not be built all at once. Each step below produces a working system that can be evaluated independently, so regressions are catchable.

1. **Fork Mortal-Policy. Get its existing architecture training on your dataset, end-to-end.** Don't change anything yet. This proves the environment, dataset, and infrastructure work, and gives you a baseline number to beat.
2. **Refactor feature extraction in `libriichi::dataset` to emit the per-zone schema.** Keep a simple model on top (modest CNN per zone, then concatenation, then existing heads). Confirm SL accuracy is at least as good as the baseline.
3. **Add the action-history zone with its transformer encoder.** This is the design's biggest architectural bet; validate it actually helps with ablation.
4. **Add the top transformer.** Compare against the concatenation baseline from step 2.
5. **Add the value head and auxiliary opponent-modeling heads. Confirm SL targets are met.**
6. **Begin RL with a small KL coefficient sweep first**, before any other RL hyperparameter tuning. KL is the most critical and the easiest to get wrong.
7. **Tune entropy schedule, learning rate, GAE-λ in that order.**

The worst failure mode is spending three months building the full architecture, having it underperform vanilla Mortal at this scale, and not knowing which of the architectural choices is responsible. Staged building means each addition is independently evaluable and the project always has a working fallback.

## Realistic Targets

- **End of stage 1:** match Tjong-on-MCR's ~81% discard accuracy, on equivalent riichi data. If significantly below, debug encoders and features before proceeding to RL.
- **End of stage 2:** competitive with stock Mortal in 1v3 duplicate matches (within ±1.5% win rate). Beating it is plausible with careful tuning but should not be the headline goal.
- **Realistic ceiling:** somewhere between stock Mortal and Suphx-level play. State-of-the-art is not the target; a strong tool for analyzing your own games is.

The path from "working stage-1 model" to "competitive with Mortal" is much longer than the path from "no model" to "working stage-1 model," and the gap is dominated by RL tuning rather than architectural cleverness. Budget accordingly.