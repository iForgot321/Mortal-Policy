# Mortal-Policy Dataset Curation: Reference & Plan

## Purpose of This Document

Consolidates dataset curation strategy for the Mortal-Policy fork: which operations matter, how to validate them, how much time to spend, and what's verified vs. assumed. Designed to preserve context for future Claude conversations on this project. Read alongside:

- `Mortal-Policy_Riichi_Mahjong_AI__Project_Reference` — overall project context
- `Mortal-Policy_Improvements__Strategic_Plan___Reference` — algorithmic improvements that follow curation
- `Hybrid_Transformer_Architecture_for_Riichi_Mahjong_` — conditional later-stage architecture work

Curation is **Stage 1** in the staged plan: roughly one week of focused work, completed before any algorithmic improvements (Stage 1.5) are layered on. Each curation operation must be independently testable; bundled changes destroy attribution.

---

## Strategic Framing

### Why curation matters

The Mortal author explicitly identifies "player-by-player cleaning using historical PT/stable-rank statistics" as a meaningful strength contributor. The exact filter rules and player whitelist are private. The project reference doc also identifies data quality as the highest impact-per-effort category, ahead of architectural changes.

### Why curation has a short high-value window

- The biggest lever (player filtering) has only a few knobs; the first reasonable filter captures most of the available gain
- Diminishing returns are visible in the testing protocol: expected gains are bounded (+1 to +3 pp discard accuracy, 52–55% win rate vs. unfiltered baseline)
- Each additional curation idea has weaker evidence than the basics
- Compute spent on dataset re-training grows faster than measurable improvement past the basic pass

### The headline rule

**Spend about a week of focused work, with a hard ceiling around two weeks.** Past that, time is better invested in algorithmic improvements where marginal compute buys more.

---

## The Four Operations (Ranked)

Listed in order of impact-per-effort. Implement and ablate one at a time against the previous best baseline.

### 1. Player-level skill filtering (foundation, non-negotiable)

This is the single biggest lever and the closest thing to what the Mortal author actually did.

**Concrete recipe:**
- Compute per-player statistics across the full Tenhou Phoenix 2009–2025 archive *before* any filtering (filtering first and computing stats on the filtered set is circular)
- Keep only players satisfying ALL of:
  - Stable rank ≥ 7 (roughly 7-dan stable)
  - Average placement ≤ 2.45
  - ≥ 500 hanchan in period (sample size floor)

**Why AND, not OR:** stable rank alone admits high-variance players whose estimate is unreliable; the sample-size floor screens those out.

**Expected effect:** cuts dataset by 60–80% but the remaining games are dramatically more consistent. Filtered data also makes every downstream RL experiment cleaner (less noise from weak players' bad decisions).

**Time cost:** 1–2 days to compute stats (parsing 16 years of replay logs is most of the work; the actual aggregation is cheap). Half a day to apply once stats exist.

### 2. Suit-symmetry augmentation (caveats — see verification section below)

**Conceptually:** Manzu, pinzu, and souzu are interchangeable in nearly all decisions (the only asymmetry is in specific yaku like sanshoku, which are handled correctly under any permutation). 3! = 6 permutations of the three number suits gives 6× literal training samples from every original game.

**Honest claim:** This is *not* a clean 6× data multiplier. Augmented examples aren't independent — they share the same underlying decision and opponent behavior, with only suit labels permuted. The effective dataset multiplier is realistically **1.2–1.5×**, possibly closer to 2× in the small filtered-dataset regime.

**Critical verification finding** (see "Mortal-Policy Architecture Verification" section): The existing Mortal-Policy architecture has *partial, emergent* suit equivariance via shared Conv1d weights along the 34-tile axis, but does NOT enforce true suit invariance. Augmentation is therefore **not redundant** with the architecture — it provides genuine signal — but the value is smaller than the raw permutation count suggests.

**Implementation approach:**
- Cleanest at the mjai JSON level: permute tile strings (`1m`↔`1p` etc.) before feeding to libriichi. Preserves the entire downstream pipeline including dora indicator handling automatically.
- Alternative: hook into feature extraction in `obs_repr.rs`. More invasive, harder to verify.
- Only permute the three number suits; honors (winds, dragons) must stay fixed.
- **Critical correctness check:** dora indicators must be permuted consistently with the rest of the game state. The JSON-level approach handles this for free.

**Time cost:** 1–2 days for implementation + unit test (verify yaku detection still resolves correctly, verify honors weren't permuted).

**Stance:** Worth implementing and ablating. Not "near-mandatory" — earlier framing was overconfident. The ablation cost is low; if Layer 1 stays flat-or-up and Layer 2 shows clear win-rate improvement, keep it. If not, discard.

### 3. Game-level hygiene (cheap, defensive)

Drop games that are noise rather than signal:
- Disconnect/timeout games (Tenhou flags these)
- Suddenly-terminated hanchan (incomplete final round, server issues)
- Tonpuu (east-only) games — keep only full hanchan, since the model needs to learn placement reasoning over the full game arc
- Verify rule consistency (aka-dora, kuitan settings) — Phoenix is consistent but worth checking after any cross-source merging

Expected effect: small but real. You're mostly buying robustness, not headline strength.

**Time cost:** half a day to apply once the filter logic exists.

### 4. Decision-level weighting on high-leverage decisions

Per-decision sample weights (NOT hard filters) in the SL/AWR loss:
- Riichi declarations
- Push/fold decisions when an opponent has declared riichi
- Last-few-turns dangerous-tile discards
- Oorasu (final round) decisions with placement implications

Starting ratio: roughly 3:1 between high-leverage and routine decisions. Ablate against uniform weighting; try 2:1 and 4:1 as well.

**This is the most underexplored of the four.** The Mortal author probably doesn't do it. Tjong doesn't. Directly addresses a known weakness of SL on action prediction (a model can hit good aggregate accuracy by nailing easy discards while being mediocre on the decisions that decide games).

**Important measurement note:** Layer 1 overall discard accuracy might *drop slightly* even when this change helps — because the loss is reweighted, the model intentionally spends less capacity on routine decisions. Track per-category accuracy (riichi decisions, push/fold, late-game discards) separately. Layer 2 head-to-head play is the trustworthy signal.

**Time cost:** 1–2 days to identify high-leverage decisions in the feature pipeline and add the weighting in the loss function.

---

## What to Skip (For Now)

- **Style stratification** — interesting but speculative on whether it improves the model versus changing what it overfits to. Defer until baseline exists.
- **Cross-source augmentation (Mahjong Soul Jade/Throne)** — adds volume but introduces meta drift. The four operations above give better signal-to-noise than dumping more games on the pile.
- **Temporal weighting / era cutoffs** — using the NikkeTryHard 2020–2024 archive is already implicitly a recent-meta filter. Don't need to add another layer.
- **Outcome-conditional filtering** ("only games where strong players won") — actively harmful. Biases toward survivor decisions and breaks the IID assumption.

---

## Testing Protocol (Three Layers)

Run all three on every change. Each curation operation must show measurable Layer 2 improvement before stacking the next on top, or be reverted.

### Layer 1: SL accuracy on fixed held-out sets (cheap, fast, suggestive only)

After each curation change, retrain SL pretraining and measure:
- Discard accuracy on **held-out-by-player** set
- Discard accuracy on **held-out-by-time** set (most recent 6 months never seen during training)
- Call/pass accuracy broken out separately
- Riichi-decision accuracy broken out separately (small fraction of decisions, high-leverage)

Held-out sets must be **frozen at the start of the project** and never changed. Changing them mid-project invalidates all previous comparisons.

What this layer tells you: whether the change moved the model toward predicting strong-player actions. What it doesn't tell you: whether the model is actually stronger.

### Layer 2: Self-play head-to-head against frozen previous baseline (medium cost, trustworthy)

The most important layer.

- **Duplicate mahjong**: same wall dealt to both sides with seats rotated. Eliminates ~80% of variance from random tile draws.
- **≥ 2,000 hanchan per comparison**, ≥ 3,000 in this protocol since there's no external anchor (see below). Mahjong variance is brutal; 500-game evaluations routinely flip sign with a different RNG seed.
- **Multiple seeds**: at least two different RNG seeds for wall-shuffling. If seeds disagree about whether a change helped, the effect is noise.
- Track win rate, average placement, deal-in rate, average score — not just win rate.

**Decision rule:** ≥ 50.8% win rate is meaningful improvement; 49.5%–50.5% is noise regardless of intuition.

### Layer 3: External anchors (degraded — see "No Public Stock Mortal" section)

**There are no public stock Mortal weights.** The Mortal repo deliberately withholds them. The original "evaluate against stock Mortal" protocol is impossible.

Replacement portfolio of weaker external anchors:

1. **Action-agreement on a frozen golden set** — held-out 5,000–10,000 decisions from games played by stable-rank-9 / top-100-on-leaderboard players. Measure top-1 and top-3 agreement with their actual decisions. Cheap to compute, stable across experiments. This is the **primary external check.** Tjong's 81.5% number is this metric on MCR data.

2. **Periodic submission to `mjai.app`** if the arena is active — slow, doesn't control match volume, useful as a slow background sanity check that the model isn't drifting in some direction that beats own baselines but loses to anyone outside the family. Verify availability before depending on it.

3. **Akochan as fixed weak opponent** — floor check, not ceiling check. If a curation change makes the model lose more to akochan than the previous version did, something is wrong.

4. **Heuristic baselines** — a simple "shanten-greedy" bot (always discard the tile that maximizes ukeire) is trivial to write and gives a fixed reference point that never changes. Catches catastrophic regressions.

5. **Drift detector** — train a small classifier to distinguish strong-human discards from model discards on the golden set. If classifier accuracy goes *up* over training, the model is getting more distinguishable from strong humans — useful early warning before win rate moves.

---

## No Public Stock Mortal: What It Changes

Without a public, strong, evaluation-compatible reference model:

- Layer 2 (head-to-head vs. own previous baseline) becomes more important, not less. Run more rigorously: 3,000+ duplicate hanchan, multiple seeds.
- Action-agreement on golden set replaces stock Mortal as primary external check. Track on every curation change. Monotonic improvement = probably correct direction. Stalls/regressions = signal.
- "Is my model good?" reduces to "measurably better than my previous version, decisions look more like strong-player decisions, doesn't lose to a known weak baseline." Weaker claim than "better than stock Mortal" but the strongest defensible claim given what's public.
- **Honest limitation:** if the final model is much weaker than the publicly deployed `mjai.app` version, this protocol will detect it via arena submissions but won't localize *why*. Closing that gap is what Stage 1.5 (algorithmic improvements) and possibly Stage 2 (architecture) are for, not curation.

---

## Mortal-Policy Architecture Verification (Critical for Suit-Symmetry Augmentation)

This was checked directly against the Mortal-Policy repo (cloned, source inspected) during this discussion. Documented here because it determines whether suit-symmetry augmentation is redundant.

### What the architecture actually does

The network in `mortal/model.py` uses `nn.Conv1d` with kernel size 3, stride 1, padding 1, along a length-34 axis. The encoder is built from `ResBlock` modules (with `ChannelAttention`) stacked into a `ResNet`. The flatten operates on `32 * 34`, confirming the spatial dimension is 34.

The 34 positions correspond to tile types in this fixed layout (from `libriichi/src/tile.rs`):

```
[0..9]   1m..9m  (manzu)
[9..18]  1p..9p  (pinzu)
[18..27] 1s..9s  (souzu)
[27..31] E,S,W,N (wind tiles)
[31..34] P,F,C   (dragon tiles)
```

Feature extraction (in `libriichi/src/state/obs_repr.rs`) writes everything to a `Simple2DArray<34, f32>` with channels on the row axis. Tile counts in hand: `self.arr.assign_rows(self.idx, tile_id, n, 1.)`. Discards: indexed by `tile.deaka().as_usize()`. The entire pipeline is keyed on raw tile ID.

### What this means for suit invariance

**Suit equivariance is partial, emergent, NOT architecturally enforced:**

1. **Translation equivariance along the tile axis is real.** The same kernel weights slide across all 34 positions, learning position-relative patterns rather than treating each slot independently. This is genuine parameter sharing.

2. **Cross-suit kernel windows are structurally meaningless.** The kernel sees content at positions like 8m–9m–1p (indices 7-8-9) — that window crosses the manzu/pinzu boundary, which has no semantic meaning. The network must learn to ignore those windows.

3. **Wind/dragon tiles get explicit positional treatment.** Lines 196–197 of `obs_repr.rs`:
   ```rust
   self.arr.assign(self.idx, state.bakaze.as_usize(), 1.);
   self.arr.assign(self.idx + 1, state.jikaze.as_usize(), 1.);
   ```
   The round wind and seat wind are written at the position on the tile axis corresponding to that wind tile. The network attaches semantic meaning to specific positions, not just relative ones.

### Implications

- **The network CAN learn approximate suit symmetry via shared kernel weights** but nothing forces it to.
- **With a 24-block, 512-channel ResNet and millions of examples**, it almost certainly converges on near-suit-symmetric behavior in the middle of each suit, but there's no formal guarantee.
- **Explicit suit-permutation augmentation is therefore NOT redundant** — it provides genuine signal that directly teaches the kernel to behave consistently across suits.
- **But the effective dataset multiplier is small** (1.2–1.5×, possibly closer to 2× in small filtered datasets) rather than literal 6× — because the architecture's parameter sharing means the model can reach the same property without help, given enough data.
- **Augmentation value is higher in the filtered-dataset regime** than in the unfiltered archive, because smaller datasets benefit more from inductive-bias-style augmentation.

### Implementation guidance

- Cleanest: at the mjai JSON level, permute tile strings (`1m`↔`1p`, etc.) before feeding to libriichi. Preserves all downstream handling including dora indicators.
- Alternative: hook into `obs_repr.rs`. More invasive.
- Honors (`E`, `S`, `W`, `N`, `P`, `F`, `C`) must stay fixed.
- Dora indicators must be permuted consistently — the JSON-level approach handles this automatically; the feature-extraction-level approach must do it explicitly.

---

## Order of Operations

1. **Compute per-player stats** across full archive (1–2 days, CPU-bound)
2. **Apply player filter + game hygiene** → measure dataset size
3. **Lock in validation sets:** held-out-by-player, held-out-by-time, golden set of stable-rank-9 decisions
4. **Train baseline SL model** on filtered set — Layer 1 + Layer 2 + Layer 3
5. **Add suit-symmetry augmentation** at the mjai JSON level, with unit test for yaku correctness and honor invariance — Layer 1 (expect flat-or-up; if dropped, has a bug) + Layer 2
6. **Add decision-level weighting** in loss function — track per-category accuracy, primary signal is Layer 2
7. **Final curated dataset is locked.** Version it (`dataset-v1` etc.) so it can be regenerated deterministically.

---

## When to Spend More Than a Week

Two scenarios justify extending into a second focused week:

**Scenario A: An operation doesn't validate.** Suit-symmetry augmentation drops Layer 1 accuracy when it should be flat → bug to find. Player filtering with reasonable thresholds doesn't improve Layer 2 win rate → filter logic is wrong. Debugging these is mandatory and can take time.

**Scenario B: Layer 1 numbers significantly below realistic targets** (e.g., discard accuracy clearly below 75% on held-out set after filtering and augmentation). Signals something fundamental is wrong — possibly feature extraction, hyperparameters, or over-aggressive filtering. Pause to investigate.

In both cases the extra time is debugging what's already there, not adding new curation operations.

## When to Stop Even If Curation Is "Going Well"

- All four operations validated at Layer 1 and Layer 2 — recipe is locked in
- Sketching a fifth, sixth, seventh curation idea (style stratification, era weighting, etc.) — this is engineering rigour turning into avoidance of the next stage
- Compute spent on dataset re-training growing faster than measurable Layer 2 improvement (diminishing returns; move to algorithmic improvements)

---

## Realistic Targets

After all four operations against the unfiltered baseline:

- **Layer 1 discard accuracy:** +1 to +3 percentage points
- **Layer 2 duplicate-hanchan win rate vs. unfiltered baseline:** 52–55%
- **Action-agreement on golden set:** monotonically improving across operations
- **Akochan floor:** never lose more to akochan than the previous version

Numbers dramatically larger than this → suspect a bug, most often a train/val leak where the same players/games appear on both sides of the split.

Numbers dramatically smaller → suspect curation isn't actually applying as intended (silent failure in feature extraction is the most common cause).

---

## Deliverable at End of Stage 1

- A versioned dataset (`dataset-v1`) that can be re-created from raw logs deterministically
- Documentation of filters and thresholds it uses
- Held-out-by-player and held-out-by-time validation sets, frozen
- Golden set of top-stable-rank decisions for action-agreement evaluation, frozen
- Layer 1 / Layer 2 / Layer 3 numbers comparing each operation against the previous baseline, durably logged
- Trained SL baseline on the final curated dataset, frozen, that becomes the substrate for Stage 1.5 algorithmic experiments

---

## Instrumentation (Non-Optional)

Every curation experiment requires the following logged continuously:

- Layer 1 metrics on all three held-out sets (per-category, not just aggregate)
- Layer 2 win rate vs. frozen previous-stage baseline (multiple seeds)
- Action-agreement on golden set
- Drift-detector classifier accuracy if implemented
- Periodic floor checks vs. akochan and heuristic baselines
- Dataset version tag, code commit hash, full hyperparameter set per experiment

**One log file per experiment.** Future-you will need to compare experiments months apart and rigorous bookkeeping now is what makes that possible.

---

## Honest Bottom Line

Curation is a high-impact-per-effort task with a *short* high-impact-per-effort window. The first week is one of the highest-leverage weeks in the entire project. The fourth week of curation is one of the lowest-leverage weeks. Knowing where the cliff is and getting off the curve at the right time matters more than executing any individual step perfectly.

The strategic-plan doc is right that algorithmic and infrastructure work dominates final agent strength past the basic curation pass. The case for the four-operation week: it's necessary infrastructure for everything downstream, the operations are individually cheap and ablatable, and skipping it means fighting noise from weak players' bad decisions while simultaneously debugging KL coefficients in Stage 1.5. The case against extending past two weeks: every additional week on curation is a week not spent on the distributional value head, league training, or the belief-state loop — where the compounding strength gains actually live.

---

## Document Cross-References

- **Original project reference:** `Mortal-Policy_Riichi_Mahjong_AI__Project_Reference` (project files)
- **Algorithmic improvements plan:** `Mortal-Policy_Improvements__Strategic_Plan___Reference` (project files)
- **Hybrid architecture design:** `Hybrid_Transformer_Architecture_for_Riichi_Mahjong_` (project files)
- **Mortal-Policy repo:** https://github.com/Nitasurin/Mortal-Policy
- **Recommended Tenhou scraper:** https://github.com/Apricot-S/houou-logs
- **Pre-converted mjai dataset:** https://github.com/NikkeTryHard/tenhou-to-mjai
- **Public mahjong AI arena:** https://mjai.app