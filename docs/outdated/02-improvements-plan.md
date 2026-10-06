# Mortal-Policy Improvements: Strategic Plan & Reference

## Purpose of This Document

Consolidates the algorithmic and infrastructure improvements identified as highest-value for the Mortal-Policy fork. Designed to preserve context for future Claude conversations on this project. Read alongside the existing project reference (`Mortal-Policy_Riichi_Mahjong_AI__Project_Reference`) and architecture document (`Hybrid_Transformer_Architecture_for_Riichi_Mahjong_`).

---

## Core Strategic Decision: Algorithm Before Architecture

**The hybrid transformer architecture should NOT be implemented in the same project phase as the algorithmic improvements below.** The cleanest experimental design is:

1. Take stock Mortal-Policy (existing ResNet) as the fixed substrate
2. Layer algorithmic improvements onto it, one at a time, with ablations
3. Lock in a hardened RL recipe based on measured deltas
4. ONLY THEN swap in the hybrid architecture as a single-variable test against the strengthened baseline

**Rationale:** Practitioners across the open mahjong AI lineage (Mortal discussion thread, Tjong, Suphx, Kanachan) report that architectures converge at different speeds but reach similar final strength under the same RL recipe. The RL algorithm and data dominate; architecture doesn't. Bundling architectural and algorithmic changes makes any performance delta unattributable.

The Mortal author's lighter network (used for online RL phase) is the right substrate for these experiments — faster than b24c512, already tested in self-play, simple to debug.

---

## The Improvement Stack (Priority Order)

Listed in order of decreasing certainty/impact-per-effort. Implement and ablate one at a time against the previous best baseline.

### Tier 1: High-Confidence Wins

#### 1. League Training (Highest single-change impact)

**Source:** AlphaStar (DeepMind), 2019.

**Problem addressed:** Single-pool self-play converges to a brittle equilibrium. Mahjong has rock-paper-scissors-ish playstyle space (pushers/folders/balanced), and naive self-play tends to find local equilibria that lose to playstyles outside the pool.

**Implementation:**
- Maintain a population of agents, not a single current-best
- **Main agents (2–3):** train via self-play against the full league
- **Main exploiters (1–2):** train specifically to beat the current main agents (frozen checkpoints), then retire to the league
- **League exploiters (1):** train to find global weaknesses across the entire league history
- Warm-start some agents from human-replay-pretrained policies for opponent diversity
- Sample opponents for each training game from a Nash-weighted distribution over the league

**Expected gain:** Most robust agent at modest extra compute cost. The single highest-impact change available for the same wall-clock budget.

**Effort:** 2–3 weeks of infrastructure work. Requires extending the existing parameter server / replay buffer setup to handle multiple concurrent agent identities and league sampling.

---

#### 2. Distributional Value Head (Highest impact-per-effort)

**Source:** C51 (Bellemare 2017), QR-DQN (Dabney 2017), IQN (Dabney 2018).

**Problem addressed:** Mahjong's return distribution is genuinely multimodal (mangan win vs. deal-in vs. draw) and heavy-tailed. A scalar value averages over distributions that no single outcome resembles, providing weak gradient signal and zero capacity for risk-sensitive decision-making.

**Recommended variant:** **IQN** (Implicit Quantile Networks). QR-DQN is the safe second choice. C51 is the worst fit for mahjong because point swings span −48,000 to +96,000 with state-dependent natural scales — fixed support hurts.

**Implementation:**
- Replace the scalar score-MSE head and 4-class placement softmax with quantile heads
- **Distributional placement head:** N quantiles (N=32 default, 64 for higher resolution) of agent's final hanchan score; placement derived deterministically from score-vs-opponents
- **Round-end distributional head:** N quantiles of round-end score delta via quantile regression
- IQN-specific: take quantile level τ as auxiliary input via cosine basis embedding, ~50K extra params
- Replace value MSE in PPO loss with quantile regression loss
- For PPO advantage: use quantile mean of predicted distribution as the scalar baseline (simpler than full distributional GAE; upgrade only if ablation justifies)

**New capability unlocked: Risk-sensitive action selection at inference.**
```
score(action) = (1 - risk_weight) * mean(value_dist) 
              + risk_weight * quantile(value_dist, target_q)
```
- `target_q = 0.05` when leading (worst-case-averse / CVaR on loss tail)
- `target_q = 0.95` when trailing in late rounds (upside-seeking)
- `target_q = 0.5` mid-game (mean)
- Heuristic on score gap × rounds remaining, OR learned auxiliary network

This is the closest thing in mahjong AI to what professional human players consciously do — shift between aggression and safety based on score situation.

**Calibration as diagnostic:** Distributional heads enable per-quantile calibration plots (predicted 10th percentile should be below actual return 10% of the time, etc.). Single most useful diagnostic for catching a diverging value function before win-rate metrics degrade.

**Effort:** ~1 week including tests. Main risk: subtle sign errors in asymmetric L1 quantile regression loss — copy from reference implementation, unit-test against empirical-quantile property.

**Critical sequencing note:** Introduce during **Stage 1 SL pretraining**, not just Stage 2 RL. The trunk needs to learn to produce useful distributional features from the start; if SL bakes in risk-neutral behavior, the policy can't act on quantile info at inference even if the head provides it.

---

#### 3. Off-Policy Correction via V-trace (Sample Efficiency)

**Source:** IMPALA (Espeholt et al., 2018). Related: Retrace (Munos et al., 2016).

**Problem addressed:** Vanilla PPO discards trajectories after a few epochs. In mahjong, strategically rich decision points (deep tenpai, defensive folds against riichi) are vastly outnumbered by trivial early-game discards. Throwing away trajectories wastes the rare gold.

**Implementation:**
- V-trace importance-weighted off-policy advantage estimation, replacing or supplementing GAE
- Allows reuse of trajectories across many gradient steps with proper importance correction
- Combine with **prioritized experience replay** weighted by TD-error magnitude to concentrate updates on high-information decisions

**Expected gain:** The Mortal author's 50M-games estimate likely drops to 10–20M with proper off-policy reuse. Comparable to the IMPALA-vs-A3C sample efficiency gains in Atari.

**Effort:** 2–3 weeks. Touches the replay buffer, the trajectory format, and the loss function. Highest implementation risk in Tier 1 because importance ratios can explode if not clipped properly — start with tight V-trace clipping (ρ̄=1.0, c̄=1.0) before relaxing.

---

#### 4. Belief-State Loop (Structural Gain on Hidden Information)

**Source:** Poker AI (DeepStack, Pluribus), Hanabi RL literature.

**Problem addressed:** The hybrid architecture's planned auxiliary heads (opponent tenpai prediction, opponent waits, opponent hand value) are trained but **not fed back as input to the policy**. The model has no explicit belief state representation.

**Implementation:**
- Train auxiliary opponent-prediction heads as already planned
- **Additionally:** feed their predictions back as input to the policy network on the next decision, creating a belief-conditioned policy
- Stop-gradient on the belief features fed forward to prevent the policy gradient from corrupting the belief representation
- Train belief heads via direct supervision against ground-truth hidden information from replays (which is visible in retrospect)

**Risk to manage:** The policy can learn to ignore belief features if they're noisy. Mitigate by: (a) high weight on the auxiliary belief loss during SL, (b) belief feature dropout during RL to force the policy to actually use them, (c) log mutual information between belief outputs and policy outputs as a sanity check.

**Expected gain:** Probably the biggest unexplored algorithmic direction in open-source mahjong AI. Belief networks were the key innovation in modern poker AI — explicit belief representation + reasoning over the belief is what separates DeepStack from naive DQN approaches.

**Effort:** 2 weeks. Mostly straightforward once auxiliary heads exist; the gradient flow management is the subtle part.

---

### Tier 2: Promising Bets, Higher Variance

#### 5. Search-at-Decision-Time with Distillation Back

**Source:** AlphaZero recipe adapted for imperfect information; Suphx's run-time policy adaptation as the closest mahjong precedent.

**Implementation:**
- At high-stakes decision points only (riichi declaration, push/fold against opponent riichi), run short MCTS-like search
- Use the value head to evaluate leaves
- For opponent moves at search nodes, sample from the belief network (Tier 1 #4) to weight opponent action probabilities
- Use search-improved actions as targets for an additional SL distillation pass during training

**Why selective:** Full search at every decision is too expensive and unnecessary. Mahjong has a small set of genuinely high-leverage decisions per round; concentrate search there.

**Effort:** 4–6 weeks. The belief-conditioned opponent modeling at search nodes is the hard part. Don't attempt before Tier 1 #4 is working.

---

#### 6. Oracle Distillation (Knowledge Distillation from Perfect-Information Teacher)

**Source:** Suphx (Microsoft Research, 2020), simplified.

**Problem addressed:** SL on action prediction alone underdetermines defensive reasoning — the model can predict discards reasonably without learning to model opponents.

**Implementation:**
- Train a separate "oracle" model with access to perfect information (all hidden tiles, opponents' hands, wall composition)
- Use auxiliary KL loss during normal SL: student's policy distribution should match teacher's, weighted high early in training and decayed later
- Simpler than Suphx's gradual feature dropout; Suphx's full recipe involves continuously dropping oracle features over training, which adds complexity without proportional benefit at small scale

**Effort:** 2 weeks once league training infrastructure exists (which already requires multi-model training).

**Note from project reference:** This is flagged as the highest-leverage research direction in the open Mortal lineage and is not fully reproduced in any open-source mahjong AI.

---

#### 7. Self-Distillation with EMA Teacher

**Source:** BYOL, MoCo, Mean Teacher recipes adapted to RL.

**Implementation:**
- Maintain an EMA-averaged copy of the model as a teacher
- Auxiliary loss: student matches teacher's output distribution with temperature annealing
- Essentially free to implement, stabilizes RL substantially in noisy gradient regimes

**Effort:** 2 days. Lowest-risk addition in either tier.

---

### Tier 3: Worth Bookmarking, Don't Pursue Soon

- **MuZero-style learned dynamics + latent-space planning.** Large engineering undertaking, mixed evidence on imperfect-information variants.
- **Mixture-of-Experts in top transformer.** Mahjong has clear phase/decision-type structure that should benefit. Wait until hybrid architecture is in play; doesn't apply to current ResNet substrate.
- **Decision Transformer / Trajectory Transformer for SL pretraining.** Reframes pretraining as conditional sequence modeling. Mixed results in sparse-reward domains. Worth a small ablation, not a full bet.

---

## Revised Build Sequence

Replaces/extends the staged plan in the original project reference.

### Stage 0: Baseline Reproduction (~1 week, $50–$150)
- Fork Mortal-Policy
- Train default architecture on NikkeTryHard 2020–2024 mjai archive with AWR
- Establish baseline numbers: discard accuracy on held-out replays, win rate vs. random, win rate vs. stock Mortal weights

### Stage 1: Data Curation (~1 week, mostly CPU)
- Compute per-player stable-rank stats across full Tenhou Phoenix 2009–2025
- Filter dataset by player skill thresholds
- Retrain Stage 0 model on curated data, measure delta

### Stage 1.5: Algorithmic Hardening on Stock Architecture (NEW — replaces direct jump to architecture)

This is where the Tier 1 improvements get implemented and ablated. **Each step is a single-variable change against the current best baseline.**

| Step | Change | Time | Validation |
|------|--------|------|------------|
| 1.5a | Distributional value head (IQN) added to SL pretraining | 1 week | Match SL accuracy of Stage 1, calibration curves stable |
| 1.5b | V-trace + prioritized replay in PPO | 2–3 weeks | Sample efficiency: same win rate at fewer self-play games |
| 1.5c | League training (3 main + 2 exploiters) | 2–3 weeks | Win rate vs. frozen Stage 1.5b agent and vs. stock Mortal |
| 1.5d | Belief-state loop (auxiliary heads → policy input) | 2 weeks | Win rate delta + mutual info diagnostic |
| 1.5e | Self-distillation with EMA teacher | 2 days | Stability of training curves |

**Hard rule:** Each step must show measurable improvement (or be discarded) before the next is added. If 1.5b doesn't improve sample efficiency, don't stack 1.5c on top of broken off-policy correction — debug or revert.

### Stage 2: Architecture Swap (Conditional)
Only after Stage 1.5 produces a hardened recipe. Implement the hybrid transformer architecture per the existing architecture document, swap into the hardened RL pipeline as a single-variable test.

If hybrid architecture wins: keep it. If it doesn't (as the cross-domain evidence suggests is plausible): the Stage 1.5 work is the project's contribution, and it's a real one.

### Stage 3: Tier 2 Research Bets (Optional)
- Oracle distillation
- Search-at-decision-time with distillation
- Pursue based on whichever Stage 1.5/2 result reveals the biggest remaining weakness

---

## What I'd Actually Do (Pragmatic Subset)

If compute budget is tight, the pragmatic minimum is:

1. **Stage 0 + Stage 1** (baseline + data curation) — non-negotiable foundation
2. **Distributional value head** (1.5a) — highest impact-per-effort, ~1 week
3. **League training** (1.5c) — highest single-change impact, ~3 weeks
4. **Belief-state loop** (1.5d) — biggest unexplored direction, ~2 weeks

This is roughly 7–8 weeks of focused work, produces measurable gains over public Mortal weights, and leaves clean exit points if compute runs out.

V-trace (1.5b) is the most valuable addition if you want to also reduce training cost; skip if compute is plentiful and time is tight.

The hybrid architecture is genuinely interesting and the design document is sound, but the Stage 1.5 evidence will tell you whether it's worth building. Possible outcomes:

- Stage 1.5 shows the algorithmic gains dominate → architectural complexity isn't justified → ship the hardened ResNet
- Stage 1.5 reveals specific encoding bottlenecks (e.g., chronological information genuinely matters) → targeted architectural changes (action-history zone) earn their complexity
- Stage 1.5 plateaus below stock Mortal → debug the recipe before changing anything else

---

## Instrumentation Requirements (Non-Optional)

Every Stage 1.5 experiment requires the following logged continuously:

- Win rate vs. frozen previous-stage baseline (every N updates)
- Win rate vs. stock Mortal (periodic external sanity check)
- KL divergence: current policy vs. SL pretrained policy
- Policy entropy per action type (discard, riichi, call separately)
- Value calibration curves per quantile (distributional head diagnostic)
- Per-zone attention rollout (when hybrid architecture is in play)
- League diversity metrics (Nash payoff matrix between agents)
- Belief network MI: mutual information between belief outputs and policy outputs

The single most common failure mode in self-play RL: training silently regresses for two weeks before anyone notices. Without these logs, that's the most likely outcome of any of the above.

---

## Honest Bottom Line (Updated)

The original project reference's bottom line stands: the bottleneck is patience and engineering rigour, not algorithmic novelty. The improvements above don't change that — they're concrete techniques with published evidence in adjacent domains, not magic. The value is that they're cheap to implement individually, ablatable, and address known weaknesses in the open mahjong AI lineage rather than chasing architectural fashion.

The Stage 1.5 plan exists because the project has a finite compute budget and a single person's time. Doing it well means knowing which changes earned their compute, not stacking ideas and hoping.

---

## Document Cross-References

- **Original project reference:** `Mortal-Policy_Riichi_Mahjong_AI__Project_Reference` (project files)
- **Hybrid architecture design:** `Hybrid_Transformer_Architecture_for_Riichi_Mahjong_` (project files)
- **Mortal-Policy repo:** https://github.com/Nitasurin/Mortal-Policy
- **AlphaStar league training paper:** Vinyals et al., "Grandmaster level in StarCraft II using multi-agent reinforcement learning," Nature 2019
- **IQN paper:** Dabney et al., "Implicit Quantile Networks for Distributional Reinforcement Learning," ICML 2018
- **IMPALA / V-trace paper:** Espeholt et al., "IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures," ICML 2018
- **Suphx paper:** Li et al., "Suphx: Mastering Mahjong with Deep Reinforcement Learning," arXiv:2003.13590
- **DeepStack (belief networks in poker):** Moravčík et al., Science 2017