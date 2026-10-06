# Project Roadmap: Building a Strong (Possibly Strongest) Riichi Mahjong AI
 
## Purpose of This Document
 
Master roadmap for the project. **The goal is maximum playing strength**: measurably surpass the public open-source frontier, contest the strongest deployed open agents (mjai.app / RiichiLab leaderboard, Mortal-lineage bots), and make a credible run at being the strongest open Riichi AI. Personal skill improvement and review tooling are byproducts, not drivers — they get built only where they cost nearly nothing.
 
Designed to preserve context for future Claude conversations. Read alongside:
 
- `Mortal-Policy_Riichi_Mahjong_AI__Project_Reference` — substrate, dataset situation, resource estimates
- `Dataset_Curation__Reference___Plan` — Stage 1 detail (four operations, validation protocol)
- `Mortal-Policy_Improvements__Strategic_Plan___Reference` — Stage 1.5 detail (Tier 1/2 improvement stack)
- `Hybrid_Transformer_Architecture_for_Riichi_Mahjong_` — Stage 2 detail (conditional architecture swap)
 
Where this roadmap and those documents conflict, this roadmap wins on **sequencing and go/no-go criteria**; the stage documents win on **implementation detail**.
 
---
 
## Strength Ladder (What "Strongest" Means Concretely)
 
There is no public strong reference model, so strength claims must be built from a ladder of falsifiable rungs. Each rung is a named milestone with a measurement protocol.
 
| Rung | Claim | Measurement | Realism |
|---|---|---|---|
| R0 | Beats trivial baselines | Arena vs. tsumogiri + shanten-greedy bots | Trivial (SL baseline clears it) |
| R1 | Beats akochan decisively | ≥55% win-rate-equivalent placement edge, 3,000+ duplicate hanchan, 2 seeds | Expected by end of Stage 1 |
| R2 | Strong-human decision agreement | Golden-set top-1 agreement trending toward ~80% discard; per-category (push/fold, riichi) agreement tracked separately | Expected by end of Stage 1 |
| R3 | Measurably beats own SL ceiling | Each Stage 1.5 step ≥50.8% vs. frozen previous best | The core of the project |
| R4 | Top tier on mjai.app / RiichiLab | Sustained leaderboard placement in the top group | The public, falsifiable "strongest open agent" arena |
| R5 | Exceeds Mortal-lineage deployed strength | Indirect: R4 + drift-detector + golden-set superiority; direct comparison impossible (weights withheld) | Hard but feasible per project reference |
| R6 | Suphx-class (Tenhou 10-dan equivalent) | Would require Tenhou deployment at scale | **Out of scope** without industrial compute; not a project criterion |
 
**Headline target: R4 by month ~9, credible R5 case by month 12.** R6 is the ceiling to respect, not chase.
 
---
 
## Strategic Principles (Unchanged, Now Load-Bearing)
 
1. **Algorithm and data before architecture.** Cross-domain evidence says the RL recipe and data dominate final strength; architectures converge to similar ceilings under the same recipe. The hybrid transformer is a Stage 2 conditional bet, not the plan.
2. **Single-variable changes, always.** Every improvement is ablated against the frozen previous best or reverted. A strength claim built on bundled changes is not a strength claim.
3. **Compute is the real budget.** With maximum strength as the goal, wall-clock self-play becomes the dominant cost. Sample-efficiency work (V-trace, prioritized replay) is **promoted from optional to core** — it directly converts dollars into rungs.
4. **Instrumentation is the difference between a strong agent and two wasted months.** Silent regression is the #1 failure mode in self-play RL.
5. **Exit points exist but the default is forward.** Under the strength goal, "good enough for review tooling" is not a stopping condition — only budget exhaustion or a hard negative result is.
 
---
 
## Phase Map
 
```
Stage 0     Baseline reproduction              ~1 wk      $50–150
Stage 1     Data curation (dataset-v1)         1–2 wk     minimal
Stage 1.5   Algorithmic hardening (Tier 1)     ~10–12 wk  $1,000–2,500
  1.5a        Distributional value head (IQN)    1 wk
  1.5b        V-trace + prioritized replay       2–3 wk    ← core, not optional
  1.5c        League training                    2–3 wk
  1.5d        Belief-state loop                  2 wk
  1.5e        EMA self-distillation              2 days
Gate G1     Go/no-go review                    1 wk
Stage 2     Architecture swap (conditional)    4–8 wk     varies
Stage 3     Research bets (Tier 2)             4–8 wk     varies
  3a          Oracle distillation                2–3 wk    ← default first bet
  3b          Search-at-decision + distillation  4–6 wk
Stage 4     Strength campaign                  6–10 wk    $1,000–3,000
Total       ~9–12 months part-time             $2,500–6,000 realistic envelope
```
 
---
 
## Stage 0 — Baseline Reproduction (~1 week)
 
Per project reference. Fork Mortal-Policy, build libriichi, train the lighter-ResNet with AWR on the NikkeTryHard 2020–2024 mjai archive.
 
**Deliverables:**
- Working end-to-end pipeline (data → features → training → arena)
- **The evaluation harness** — duplicate-hanchan protocol (≥3,000 hanchan, ≥2 seeds), automated, one command. Under the strength goal this harness is the single most important artifact of the entire project; every rung claim rests on it.
- Baseline numbers: per-category SL accuracy, arena results vs. tsumogiri + shanten-greedy, first golden-set agreement number.
 
**Exit criterion:** R0 cleared; numbers logged with commit hash and config.
 
## Stage 1 — Data Curation (1–2 weeks, hard ceiling 2)
 
Per curation plan: player-level skill filtering (stable rank ≥7, avg placement ≤2.45, ≥500 hanchan) on the full Phoenix 2009–2025 archive; suit-symmetry augmentation at mjai JSON level; game hygiene; decision-level weighting (3:1 on high-leverage decisions, ablate 2:1/4:1). Three-layer validation on every operation. Freeze held-out-by-player, held-out-by-time, and golden sets **before** anything else.
 
**Strength-goal adjustment:** none to the operations — the curation plan's stop criteria stand. The temptation under a strength goal is to extend curation ("more data ideas = more strength"); resist it. Past the four operations, marginal compute buys more in Stage 1.5.
 
**Exit criteria:** `dataset-v1` locked and deterministic; R1 (beats akochan) and R2 trajectory (golden-set agreement improving monotonically across operations) achieved; SL discard accuracy in the 75–80% band.
 
## Stage 1.5 — Algorithmic Hardening (~10–12 weeks)
 
Per strategic plan, with **one change in emphasis**: V-trace + prioritized replay (1.5b) is core, not skippable. The pragmatic subset in the strategic plan ("skip 1.5b if compute is plentiful and time is tight") was written for a bounded-ambition project. Under the maximum-strength goal, the long self-play campaigns of Stages 3–4 dominate total cost, and a 2–5× sample-efficiency multiplier compounds across every subsequent week of training. 1.5b pays for itself several times over.
 
**Order:** 1.5a → 1.5b → 1.5c → 1.5d → 1.5e. Rationale: the distributional head must be warm from SL before RL touches it; off-policy correction should be debugged before league training multiplies the number of concurrent policies generating data; the belief loop needs stable RL underneath it.
 
**Hard rule (unchanged):** each step shows ≥50.8% vs. the frozen previous best across 2 seeds, or is debugged/reverted before the next stacks on. 49.5–50.5% is noise regardless of intuition.
 
**Per-step detail** lives in the strategic plan. Key strength-goal notes:
 
- **1.5a (IQN):** introduce during SL retraining on dataset-v1, not bolted on during RL. The risk-sensitive inference capability (quantile selection by score situation) is deferred to Stage 4 tuning — during 1.5, use quantile-mean baselines only.
- **1.5b (V-trace + prioritized replay):** start with tight clipping (ρ̄=c̄=1.0). Success metric is explicit: equal win rate vs. baseline at ≤50% of the self-play games. If it can't demonstrate that, revert — broken off-policy correction poisons everything downstream.
- **1.5c (League):** 2–3 main agents, 1–2 main exploiters, 1 league exploiter; Nash-weighted opponent sampling; warm-start some agents from the SL policy for diversity. Log the league payoff matrix — diversity collapse is the failure mode.
- **1.5d (Belief loop):** stop-gradient on belief features into the policy; belief-feature dropout during RL; log belief↔policy mutual information.
- **1.5e (EMA teacher):** 2 days, essentially free stability.
 
**Exit criterion:** R3 — a hardened recipe with per-step measured deltas, an agent decisively above the SL ceiling, and calibrated value quantiles.
 
## Gate G1 — Architecture Go/No-Go (~1 week)
 
The decision the whole plan has been protecting. Run the full evaluation portfolio on the Stage 1.5 agent, then choose:
 
- **Path A — Stage 2 (architecture swap)** if Stage 1.5 surfaced *specific representational bottlenecks*: e.g., ablations or attention/saliency analysis showing chronological information is the binding constraint, belief heads starved by the feature representation, or per-category accuracy stuck on temporal-pattern decisions (riichi-response defense) while everything else improved.
- **Path B — Stage 3 first (research bets on the ResNet)** if gains came smoothly from the recipe and no representational bottleneck is identifiable. Under the strength goal this is the *default*: oracle distillation is flagged in the project reference as the highest-leverage unreproduced direction in the open lineage, and it works on any substrate.
 
Paths A and B are both taken eventually if budget allows; G1 only decides order. The decision rule for order: **whichever path attacks the weakest measured category of the current agent.**
 
## Stage 2 — Hybrid Architecture Swap (conditional, 4–8 weeks)
 
Per the architecture document's staged build sequence (fork feature extraction → zone schema with simple model → action-history zone ablation → top transformer → heads), swapped into the hardened Stage 1.5 recipe as a **single-variable test**.
 
**Strength-goal framing:** the architecture must beat the hardened ResNet under the identical recipe at Layer 2, or it ships as a documented negative result and the ResNet remains the vehicle. Sunk design effort is not a reason to carry a weaker model into Stage 4.
 
## Stage 3 — Research Bets (4–8 weeks)
 
**3a. Oracle distillation (default first, 2–3 weeks).** Train a perfect-information teacher (all hidden tiles visible — free from replays and self-play alike); distill via KL to the blind student, weighted high early, decayed. Simplified from Suphx's gradual feature dropout. This is the biggest known idea in the lineage that no open project has fully reproduced — the most plausible source of a genuine edge over Mortal-lineage agents, which is exactly what R5 requires.
 
**3b. Search-at-decision-time with distillation back (4–6 weeks, only after 1.5d works).** Short belief-sampled search at high-leverage decisions only (riichi declaration, push/fold vs. riichi); value head at leaves; distill improved actions back into the policy. Deployment note: mjai.app enforces a ~2-second move budget, so the deployed bot must be the distilled fast policy — search is a training-time amplifier, not an inference-time crutch.
 
Tier 3 items (MuZero-style planning, MoE, decision transformers) remain bookmarked, not scheduled.
 
## Stage 4 — Strength Campaign (6–10 weeks)
 
The endgame: convert the best recipe + best substrate into maximum strength via sustained compute.
 
1. **The long run.** Largest sustained self-play run of the project on the frozen final recipe. With 1.5b's efficiency gains, target 10–20M games; budget 32–64 CPU cores + 1 strong GPU for 4–8 weeks ($1,000–3,000 cloud). No recipe changes mid-run; checkpoint every N updates; full instrumentation.
2. **Risk-sensitive inference tuning.** Enable quantile-based action selection (CVaR-style when leading, upside-seeking when trailing late); tune the score-gap × rounds-remaining heuristic via arena ablation, or train the small auxiliary selector if the heuristic plateaus.
3. **Population hardening.** Final league pass against the full checkpoint history plus akochan and the SL policy, to sand off exploitable habits before public play.
4. **Public campaign.** Sustained mjai.app / RiichiLab submissions (R4). Iterate on observed losses: pull losing games, run the mistake classifier against the agent itself, feed systematic weaknesses back as targeted league exploiters.
5. **The R5 case.** Assemble the strongest defensible claim: leaderboard standing + golden-set superiority trend + drift-detector convergence + full ablation table. Publish weights, dataset recipe, and the ablation table — being the strongest *open* agent is only meaningful if it's verifiable.
 
---
 
## Compute & Cost Envelope
 
| Stage | GPU | CPU | Wall clock | Est. cost |
|---|---|---|---|---|
| 0–1 | 1× 4090 | light | 2–3 wk | $100–250 |
| 1.5 | 1× 4090 | 32 cores bursts | 10–12 wk | $800–1,500 |
| 2 and/or 3 | 1× 4090 (A100 if hybrid arch memory-bound) | 32 cores | 6–12 wk | $500–1,500 |
| 4 | 1× 4090/A100 | 32–64 cores sustained | 6–10 wk | $1,000–3,000 |
 
Realistic total: **$2,500–6,000 and 9–12 months part-time.** The strength goal roughly doubles the compute envelope of the bounded plan; almost all of the increase is Stage 4 self-play. Cloud spot pricing (4090 at $0.29–0.69/hr) is assumed; re-verify quarterly.
 
---
 
## Instrumentation (Non-Optional, Consolidated)
 
Logged continuously in every training run, one log file per experiment, with dataset version + commit hash + full config:
 
- Win rate vs. frozen previous-best (every N updates) and periodic floor checks (akochan, shanten-greedy)
- KL divergence to SL prior; policy entropy per action type (discard / riichi / call separately)
- Per-quantile value calibration curves (from 1.5a on)
- Golden-set agreement, per category; drift-detector accuracy
- League payoff matrix / diversity metrics (from 1.5c on)
- Belief↔policy mutual information (from 1.5d on)
- Sample-efficiency curve: strength vs. cumulative self-play games (the metric 1.5b is judged on)
 
**Kill rule:** if win rate vs. the frozen baseline drops below ~50% and stays for a sustained window, stop and diagnose (order of suspects: KL coefficient too low → LR too high → value divergence → importance ratios exploding).
 
---
 
## Risks (Strength-Goal Weighted)
 
1. **Silent RL regression** — unchanged #1; instrumentation is the only defense.
2. **Compute overrun** — the strength goal has no natural "good enough" stop; the envelope above is the commitment device. If Stage 4 exhausts budget short of R4, the exit deliverable is the hardened recipe + ablation table + weights, which is still a real contribution.
3. **Sample-efficiency shortfall** — if 1.5b fails, R4/R5 timelines stretch 2–4×; the fallback is a smaller/faster substrate (higher games/hour) traded against per-game quality.
4. **No direct Mortal comparison** — R5 remains partly circumstantial forever; mitigated by making everything public and letting the leaderboard arbitrate.
5. **Opponent-population mismatch** — an agent tuned in its own league can underperform against the mjai.app population; mitigated by warm-started diverse league members and iterating on real arena losses (Stage 4 step 4).
6. **Withheld Mortal-Policy infrastructure** — online-RL gaps may surface early in 1.5; budget discovery time.
7. **Ecosystem shift** — Mahjax-class GPU-vectorized simulators or newly released strong open weights could reshuffle cost-benefit mid-project; re-survey quarterly, re-plan only on hard evidence.
8. **Solo-developer burnout in flat-curve weeks** — the per-step ablation structure exists partly so there is always a recent, concrete, logged win.
 
---
 
## Milestone Summary
 
| Month | Milestone | Rung |
|---|---|---|
| 1 | Baseline + harness + dataset-v1 | R0–R1 |
| 2–4 | Tier-1 stack complete, each step ablated | R3 |
| 5 | Gate G1 decision, evidence-based | — |
| 5–7 | Architecture swap and/or oracle distillation | R3+ |
| 7–9 | Long run + risk-sensitive tuning + public submissions | R4 |
| 10–12 | Iterated hardening vs. arena losses; publish weights + ablations | R5 case |
 
---
 
## Honest Bottom Line
 
The strength ceiling for a solo developer with a few thousand dollars is **the strongest open, verifiable Riichi agent — not Suphx.** The path there is not a secret idea; it is the boring compounding of curated data, a hardened recipe where every component earned its place in an ablation, one genuinely under-exploited technique (oracle distillation), and more disciplined self-play compute than anyone else in the open lineage has published. The plan above spends its ambition exactly there. Everything that does not move a rung on the ladder is either free or cut.
