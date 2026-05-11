# Mortal-Policy Riichi Mahjong AI: Project Reference

## Purpose of This Document

Top-level reference for the Mortal-Policy fork project: what it is, what's known about the substrate, the dataset situation, and the overall strategic direction. Read alongside:

- **`Mortal-Policy_Improvements__Strategic_Plan___Reference`** — the detailed improvement stack, Stage 1.5 ablation plan, and tiered priority list. This is the operational planning document; the build sequence below is its summary.
- **`Dataset_Curation__Reference___Plan`** — the detailed Stage 1 curation plan: which operations matter, validation protocol, the suit-symmetry architecture verification, and stop criteria. The dataset section below is its summary.
- **`Hybrid_Transformer_Architecture_for_Riichi_Mahjong_`** — speculative architectural redesign, deliberately deferred until after the algorithmic recipe is hardened. See "Architecture vs. Algorithm" below for why.

---

## Project Overview

**Repository:** https://github.com/Nitasurin/Mortal-Policy

A policy-based fork of [Equim-chan's Mortal](https://github.com/Equim-chan/Mortal), originally based on Mortal V2 (2022) and migrated to Mortal V4 (2024). Switches from value-based RL to policy-based methods (AWR + PPO-style online training). Heavily inspired by Suphx (Microsoft Research). Written in Rust (libriichi emulator) + Python (PyTorch).

**Key context from author:** The open-source release deliberately omits trained weights, tuned hyperparameters, curated dataset, and parts of the online training infrastructure (opponent pools, replay buffers, multi-GPU parallelism). The released config defaults are explicitly described as "dummy placeholders."

---

## Architecture & Model (Current Substrate)

- **Default network:** 1D-ResNet, `conv_channels=512`, `num_blocks=24` (b24c512), with channel attention
- **Estimated parameters:** ~40–50M
- **Author's actual choice for policy version:** a lighter network than the b24c512 default, since online RL benefits more from throughput than from parameter count
- **Input:** mjai-format game state observations
- **Output:** action policy over the riichi action space

**This project treats the lighter ResNet variant as the fixed substrate** for the Stage 1.5 algorithmic experiments described below. Architectural changes are deferred — see "Architecture vs. Algorithm."

---

## Training Pipeline

Three phases, offline-to-online:

1. **Dataset preparation** — collect mjai-format game logs
2. **Offline Phase 1 (AWR — Advantage Weighted Regression)** — `train_offline_phase1.py`. Replaces Behavior Cloning from the original V2 implementation
3. **Offline Phase 2 (BPPO)** — optional, code "coming soon" per README
4. **Online Phase (PPO with importance sampling + clipping)** — `train_online.py`. Online-only training is possible but not recommended

---

## Resource Requirements

### Time & cost estimates (single-person reproduction)

| Scenario | GPU | CPU (self-play) | Wall clock | Cost |
|---|---|---|---|---|
| Offline Phase 1 only (baseline) | 1× RTX 4090 | minimal | ~2 weeks | ~$135 |
| Full offline → online (Mortal-V2-policy strength) | 1× A100/4090 | 32–64 cores | 6–12 weeks | $800–$2,500 |
| Multi-million hanchan ambitious run | Multi-GPU | Multi-node | months | $5,000+ |

### Key throughput numbers
- libriichi emulator: ~40K hanchans/hour per machine (Rust + Python NN inference)
- Policy version reportedly needs ~50M self-play games to surpass the value version (which itself needed ~10M)
- 50M games at 40K/hour = ~1,250 CPU-hours = ~52 days single-machine, ~6 days across 8 machines
- The 50M figure assumes vanilla on-policy PPO. With V-trace + prioritized replay (Stage 1.5b), the strategic plan estimates this drops to ~10–20M.

### Cloud GPU pricing reference (2026)
- RTX 4090: $0.29–$0.69/hr (Vast.ai, RunPod, Spheron)
- A100 80GB: $1.39–$2.99/hr
- H100 80GB: $1.99–$4.99/hr (specialized providers); $12+/hr on AWS

---

## Dataset Situation

### What Mortal uses (private)
- Source: Tenhou Phoenix room (鳳凰卓) logs, optionally supplemented with Mahjong Soul Jade/Throne logs
- Format: mjai JSON
- **Not released:** the curated/cleaned version. Author did "player-by-player cleaning" using historical PT/stable-rank statistics — exact filtering rules and player whitelist are private

### Public sources

**Tenhou official archives** at `tenhou.net/sc/raw/` (e.g. `scraw2009.zip` per year)
- Contains log IDs only; actual XML must be fetched per-ID
- Single-connection limit per Tenhou's policy (slow)
- Phoenix-room logs available 2009–present
- Recommended scraper: [Apricot-S/houou-logs](https://github.com/Apricot-S/houou-logs) (the older `MahjongRepository/phoenix-logs` is archived as incompatible with current licensing)

**Pre-converted public datasets**

| Dataset | Format | Coverage | Notes |
|---|---|---|---|
| [NikkeTryHard/tenhou-to-mjai](https://github.com/NikkeTryHard/tenhou-to-mjai) | mjai (drop-in for Mortal-Policy) | Phoenix room 2009–2025 | CC BY 4.0; **most directly useful** |
| [pjura/mahjong_board_states](https://huggingface.co/datasets/pjura/mahjong_board_states) | flattened feature vectors | 2009–2019, 650M+ states | Not in mjai format; needs conversion |

**Mahjong Soul:** No official archive. Use `tensoul-py-ng` + Amae-Koromo API to scrape, then convert protobuf → Tenhou JSON → mjai

### Curation operations (Stage 1)

Detailed in `Dataset_Curation__Reference___Plan`. Summary of the four operations, in priority order:

1. **Player-level skill filtering** (foundation, non-negotiable) — per-player stable-rank/placement/sample-size thresholds across the full 2009–2025 Phoenix archive. Closest open analog to the Mortal author's private cleaning step.
2. **Suit-symmetry augmentation** — permute the three number suits (manzu/pinzu/souzu) at the mjai JSON level. The architecture verification documented in the curation plan confirms suit equivariance is *partial and emergent*, not enforced, so augmentation provides genuine signal (effective multiplier ~1.2–2×, not literal 6×).
3. **Game-level hygiene** — drop disconnect/timeout/incomplete games, keep only full hanchan, verify rule consistency.
4. **Decision-level weighting** — per-decision loss weights on high-leverage decisions (riichi declarations, push/fold against opponent riichi, late-game dangerous discards, oorasu). The most underexplored of the four; addresses a known SL-on-action-prediction weakness.

**Time budget:** ~1 week focused, hard ceiling ~2 weeks. Past that, marginal compute buys more in Stage 1.5.

**Validation protocol** (mandatory for each operation): Layer 1 SL accuracy on frozen held-out-by-player and held-out-by-time sets, Layer 2 self-play head-to-head (≥3,000 duplicate hanchan, multiple seeds) against the previous baseline, Layer 3 action-agreement on a frozen golden set of stable-rank-9 decisions.

**No public stock Mortal weights exist.** The original "evaluate against stock Mortal" check is replaced by the portfolio of weaker anchors documented in the curation plan: action-agreement on the golden set (primary), periodic `mjai.app` submissions if active, akochan as floor check, heuristic shanten-greedy baseline, and a drift-detector classifier. See the curation plan's "No Public Stock Mortal" section for what this concretely changes.

---

## Strategic Direction

### Is surpassing Mortal feasible?

| Goal | Feasibility |
|---|---|
| Match/beat publicly available Mortal weights | Realistic for individual/small team |
| Beat strongest internal Mortal-Policy weights (`mjai.app`) | Hard but feasible — requires full online RL infrastructure |
| Beat Suphx-class systems | Not realistic without industrial resources |

The remaining strength gap above public state-of-the-art is set more by hardware budget and engineering rigour than by algorithmic ideas.

### Where wins actually come from (in order of impact-per-effort)

1. **Data quality** — per-player filtering by stable-rank/PT statistics. Negligible compute cost, meaningful strength gain
2. **RL recipe upgrades** — league training, distributional value head, off-policy correction, belief-state loop. These are the Tier 1 improvements detailed in the strategic plan; collectively they're the highest-leverage work available on the existing substrate
3. **Online RL infrastructure** — fast, parallel self-play with proper opponent pool diversity. Prerequisite to most of (2)
4. **Reward shaping / oracle distillation** (Suphx-style) — global reward prediction network + perfect-information teacher distilled to normal agent. Highest-leverage *research* direction; not fully reproduced in the open Mortal lineage. Tier 2 in the plan
5. **Training-side compounding** — AMP (`enable_amp`), `torch.compile`, EMA weights, batch size, LR schedule tuning. ~1.5–2× compute multiplier when stacked

### Architecture vs. Algorithm

The hybrid transformer design (`Hybrid_Transformer_Architecture_for_Riichi_Mahjong_`) is a deliberate, well-motivated redesign — and it is **deferred** until after the algorithmic recipe is hardened on the existing ResNet substrate.

Reasoning:

- Multiple practitioners (KimamanaNeko in the Mortal discussion thread, plus the Tjong / Kanachan / Suphx evidence) report that architectures converge at different speeds but reach similar final strength under the same RL recipe. **The RL algorithm and data dominate.**
- Bundling algorithmic and architectural changes makes any measured delta unattributable. Stage 1.5 in the strategic plan explicitly exists to isolate the algorithmic gains first.
- The hybrid design's biggest claimed wins (chronological action-history zone, belief-conditioned input, distributional value head) overlap substantially with Tier 1 algorithmic improvements that can be tested on the simpler substrate.

The hybrid architecture is revisited as a single-variable Stage 2 test against the hardened recipe. Possible outcomes:

- Stage 1.5 shows the algorithmic gains dominate → architectural complexity isn't justified → ship the hardened ResNet
- Stage 1.5 reveals specific encoding bottlenecks (e.g., chronological information genuinely matters) → targeted architectural changes earn their complexity
- Stage 1.5 plateaus below stock Mortal → debug the recipe before changing anything else

**Genuine candidates for architectural change once Stage 1.5 lands:**
- Smaller/faster network for higher self-play throughput (the author's own choice for online phase, already the default substrate here)
- Distributional value head (handled algorithmically in Stage 1.5a; doesn't require an architecture swap)
- Per-zone encoders with cross-zone attention (the hybrid design's main bet) — worthwhile only if Stage 1.5 surfaces evidence that the current feature representation is a bottleneck

---

## Build Sequence (Summary)

The detailed version with ablation criteria and validation checks lives in `Mortal-Policy_Improvements__Strategic_Plan___Reference`. Summary:

| Stage | Description | Time | Cost |
|---|---|---|---|
| 0 | Reproduce baseline: NikkeTryHard 2020–2024 archive + lighter ResNet + AWR. Establish measured numbers. | ~1 week | $50–150 |
| 1 | Data curation: four operations from `Dataset_Curation__Reference___Plan` — player filter, suit-symmetry augmentation, hygiene, decision-weighting. Locks in `dataset-v1` and frozen validation sets. | ~1 week (hard ceiling 2 weeks) | minimal |
| 1.5a | Distributional value head (IQN) added to SL pretraining | ~1 week | low |
| 1.5b | V-trace + prioritized replay in PPO | 2–3 weeks | low (debug) |
| 1.5c | League training (main agents + exploiters) | 2–3 weeks | $500–2,000 |
| 1.5d | Belief-state loop (auxiliary heads → policy input) | ~2 weeks | varies |
| 1.5e | Self-distillation with EMA teacher | ~2 days | negligible |
| 2 (conditional) | Hybrid transformer architecture swap, as single-variable test vs. hardened Stage 1.5 recipe | weeks | varies |
| 3 (optional) | Tier 2 research bets: oracle distillation, search-at-decision-time | weeks each | varies |

**Hard rule for Stage 1.5:** Each step must show a measurable improvement against the prior best baseline, or be reverted, before the next is stacked on top.

**Hobbyist subset** (if compute is tight): Stage 0 + Stage 1 + Stage 1.5a + Stage 1.5c + Stage 1.5d. Roughly 7–8 weeks of focused work, produces measurable gains over public Mortal weights, leaves clean exit points.

---

## Instrumentation Requirements (Non-Optional)

The single most common failure mode in self-play RL: training silently regresses for two weeks before anyone notices. Every Stage 1.5 experiment logs continuously:

- Win rate vs. frozen previous-stage baseline (every N updates)
- Win rate vs. stock Mortal (periodic external sanity check)
- KL divergence: current policy vs. SL pretrained policy
- Policy entropy per action type (discard, riichi, call separately)
- Value calibration curves per quantile (once distributional head is in place)
- League diversity metrics (Nash payoff matrix between agents) once 1.5c lands
- Belief network MI: mutual information between belief outputs and policy outputs once 1.5d lands

---

## Key Configuration Defaults (`config.example.toml`)

```toml
[control]
batch_size = 512
opt_step_every = 1
enable_amp = false      # turn ON for ~2× speedup
enable_compile = false  # turn ON for additional speedup

[resnet]
conv_channels = 512
num_blocks = 24

[policy]
awr_beta = 1.0
awr_clip = 100
entropy_weight = 1e-3
clip_ratio = 0.2
dual_clip = 3.0

[optim]
betas = [0.9, 0.999]
weight_decay = 0.1
[optim.scheduler]
peak = 1e-4
final = 1e-4
```

All values noted in config as "dummy placeholders, must be tuned" — hyperparameter search is on you. The strategic plan notes specific RL-side hyperparameters that need tuning at each Stage 1.5 step (KL coefficient, V-trace clipping, league sampling distribution, etc.).

---

## Key References

### Project documents
- **Strategic plan & improvement stack:** `Mortal-Policy_Improvements__Strategic_Plan___Reference`
- **Dataset curation plan:** `Dataset_Curation__Reference___Plan`
- **Hybrid architecture design (deferred):** `Hybrid_Transformer_Architecture_for_Riichi_Mahjong_`

### Code & related projects
- **Mortal-Policy:** https://github.com/Nitasurin/Mortal-Policy
- **Original Mortal:** https://github.com/Equim-chan/Mortal
- **Author's discussion thread (essential reading):** https://github.com/Equim-chan/Mortal/discussions/91
- **Mortal docs:** https://mortal.ekyu.moe
- **mjai-reviewer:** https://github.com/Equim-chan/mjai-reviewer
- **Phoenix logs scraper (current):** https://github.com/Apricot-S/houou-logs

### Papers
- **Suphx:** Li et al., "Suphx: Mastering Mahjong with Deep Reinforcement Learning," arXiv:2003.13590
- **AlphaStar league training:** Vinyals et al., Nature 2019
- **IQN (distributional RL):** Dabney et al., ICML 2018
- **IMPALA / V-trace:** Espeholt et al., ICML 2018
- **DeepStack (belief networks):** Moravčík et al., Science 2017
- **Tjong (transformer baseline):** Wiley CAAI Trans. on Intelligence Tech., 2024

---

## Honest Bottom Line

The bottleneck for almost everyone working on this is **patience and engineering rigour, not algorithmic novelty.** The strongest open-source mahjong AIs got there through careful data curation, robust training infrastructure, and long self-play runs — not new architectures.

The Stage 1.5 plan reflects this: it's a sequence of concrete, ablatable techniques with published evidence in adjacent domains, applied to the existing substrate so each one's contribution is measurable. The hybrid transformer is genuinely interesting and may pay off, but only as a deliberate single-variable test after the recipe underneath it has been earned. Plan accordingly.