# CLAUDE.md — Mortal-Policy Fork (Riichi Mahjong AI)

Read this in full at the start of every session. Detailed plans live in the four reference documents listed below; this file is the standing brief.

## What this project is

A single-person fork of [Mortal-Policy](https://github.com/Nitasurin/Mortal-Policy) (itself a policy-based fork of [Mortal](https://github.com/Equim-chan/Mortal)). Goal: produce an agent competitive with publicly deployed Mortal weights, with measurable algorithmic contributions, on a single high-end consumer GPU (RTX 3090/4090 class). Not chasing Suphx-level state of the art.

Stack: Rust (`libriichi` simulator, rules, mjai protocol, arena) + Python (PyTorch network and training loop). Inherits the simulator, replay parsing, mjai protocol layer, parameter server, replay buffer, and GPU inference engine from upstream Mortal-Policy.

## Current state

**The repo is a clean fork of Mortal-Policy with no changes made yet.** Stage 0 (baseline reproduction) has not started. No trained weights, no curated dataset, no validation sets frozen.

## Reference documents (read before doing anything substantive)

- `docs/reference/00-project-overview.md` — top-level project context, substrate facts, build-sequence summary.
- `docs/reference/01-dataset-curation.md` — Stage 1 plan, validation protocol, suit-symmetry architecture verification.
- `docs/reference/02-improvements-plan.md` — Stage 1.5 algorithmic stack with per-step validation criteria.
- `docs/reference/03-hybrid-architecture.md` — deferred Stage 2 architecture design.

When the user asks anything about Stage X, read the corresponding doc first rather than reasoning from this summary.

## Core strategic rules (do not violate without explicit user consent)

1. **Algorithm before architecture.** The hybrid transformer is Stage 2, conditional on Stage 1.5 evidence. Cross-domain evidence (Mortal author, Tjong, Suphx, Kanachan) is that the RL recipe and data dominate final strength; architectures converge to similar ceilings under the same recipe. Do not bundle architectural and algorithmic changes — it makes deltas unattributable.
2. **One variable at a time.** Every Stage 1 / Stage 1.5 change is ablated against the previous best baseline. Each step must show a measurable improvement at the validation layer specified for that step, or be reverted, before the next is stacked on top.
3. **Validation sets are frozen at the start of Stage 1 and never changed.** Held-out-by-player set, held-out-by-time set, and the golden set of stable-rank-9 decisions. Changing them mid-project invalidates all prior comparisons.
4. **Curation has a hard ceiling of ~2 weeks.** The first week of curation is one of the highest-leverage weeks in the project; the fourth would be one of the lowest. If a fifth curation idea appears, that's avoidance of Stage 1.5, not progress.
5. **Instrumentation is non-optional.** The most common self-play RL failure is silent regression for two weeks. Every experiment logs win rate vs. frozen previous baseline, KL to SL prior, per-action-type entropy, value calibration (once distributional head is in), league Nash payoffs (once league is in), belief MI (once belief loop is in).

## Verified substrate facts (do not re-derive from memory)

- **Default network in upstream Mortal-Policy:** 1D-ResNet, `conv_channels=512`, `num_blocks=24` (b24c512), ~40–50M params. The Mortal author's *actual* online RL substrate is a lighter network; **this project uses the lighter variant as the fixed substrate for Stage 1.5**, since throughput matters more than parameter count for self-play.
- **Tile-axis layout** (from `libriichi/src/tile.rs`):
  ```
  [0..9]   1m..9m   [9..18]  1p..9p   [18..27] 1s..9s
  [27..31] E,S,W,N  [31..34] P,F,C
  ```
- **Suit equivariance is partial and emergent, not enforced.** The shared Conv1d weights give translation equivariance along the 34-tile axis, but the kernel sees meaningless cross-suit windows (e.g. 8m–9m–1p) and the round/seat winds are written at wind-tile positions in `obs_repr.rs` lines 196–197. Explicit suit-permutation augmentation is therefore **not redundant** with the architecture; effective dataset multiplier is ~1.2–2× in the filtered-data regime, not literal 6×.
- **Released `config.example.toml` values are dummy placeholders per the upstream author.** Do not treat them as tuned. `enable_amp = false` and `enable_compile = false` in defaults — turn on for ~2× speedup.
- **No public stock Mortal weights exist.** The "evaluate against stock Mortal" check from the original strategic framing is impossible. Replacement portfolio: action-agreement on golden set (primary), periodic `mjai.app` arena submissions if active, akochan as floor check, heuristic shanten-greedy baseline, drift-detector classifier. See the curation plan's "No Public Stock Mortal" section.

## Build sequence (summary; the strategic-plan doc has ablation details)

| Stage | Description | Time | Status |
|---|---|---|---|
| 0 | Baseline reproduction: NikkeTryHard 2020–2024 mjai archive + lighter ResNet + AWR. Establish measured numbers. | ~1 wk | not started |
| 1 | Data curation: player filter, suit-symmetry aug, hygiene, decision-weighting. Freeze validation sets. Produce `dataset-v1`. | ~1 wk (cap 2) | not started |
| 1.5a | Distributional value head (IQN) added during SL pretraining (critical: must be in SL, not just RL). | ~1 wk | not started |
| 1.5b | V-trace + prioritized replay in PPO. | 2–3 wk | not started |
| 1.5c | League training (main agents + main/league exploiters). | 2–3 wk | not started |
| 1.5d | Belief-state loop: feed auxiliary opponent-prediction outputs back as policy input (stop-gradient on belief features). | ~2 wk | not started |
| 1.5e | EMA-teacher self-distillation. | ~2 d | not started |
| 2 | Hybrid transformer swap, single-variable test vs. hardened Stage 1.5 recipe. **Conditional.** | weeks | conditional |
| 3 | Tier 2: oracle distillation, search-at-decision-time. **Optional.** | weeks | optional |

Hobbyist-compute subset if budget is tight: Stage 0 + 1 + 1.5a + 1.5c + 1.5d. ~7–8 weeks total.

## Realistic targets

- End of Stage 1 vs. unfiltered baseline: +1 to +3 pp discard accuracy on held-out, 52–55% duplicate-hanchan win rate. Numbers much larger → suspect train/val leak. Much smaller → suspect curation isn't actually applying (silent feature-extraction failure is the usual culprit).
- End of Stage 1.5: measurably stronger than the Stage 1 baseline by every layer of the validation protocol, with action-agreement on the golden set monotonically improving.
- End of Stage 2 (if attempted): competitive with publicly deployed Mortal in `mjai.app` arena. Beating it is plausible, not the headline goal.

## Conventions

- **Code locations:** Rust simulator and feature extraction in `libriichi/`. Python network/training in `mortal/`. Feature extraction lives in `libriichi/src/state/obs_repr.rs` — the cleanest place to add per-zone tensors if/when Stage 2 starts.
- **Dataset format:** mjai JSON throughout. Suit-symmetry augmentation should happen at the mjai JSON level — preserves dora indicator handling automatically. The `obs_repr.rs`-level alternative is more invasive and must permute dora indicators explicitly.
- **Dataset versioning:** the curated output of Stage 1 is `dataset-v1`, regenerable deterministically from raw logs. Tag dataset version + code commit hash in every experiment log.
- **Experiment logging:** one log file per experiment with full hyperparameters, dataset version, commit hash. Future comparisons happen months apart; rigorous bookkeeping now is what makes them possible.

## When in doubt

- Pause and ask before bundling changes, deviating from the staged plan, or starting Stage 2 work.
- If a result looks too good (e.g. +10 pp discard accuracy from one curation step), assume train/val leak before believing it.
- If Stage 1.5 plateaus below the Stage 1 baseline, debug the recipe before changing anything else — that's the documented failure mode the staged plan exists to prevent.
- The bottleneck is patience and engineering rigour, not algorithmic novelty. Treat decisions accordingly.