# Experiments

One subdirectory per experiment. Each subdirectory is self-contained: anyone (you in six months, a collaborator, Claude Code in a fresh session) should be able to read the directory and know what was tried, what was measured, and whether it landed.

## Directory naming

```
YYYY-MM-DD-stage-shortname/
```

Examples:
- `2026-05-15-stage0-baseline/`
- `2026-05-22-stage1a-player-filter/`
- `2026-05-28-stage1b-suit-aug/`
- `2026-06-10-stage1.5a-iqn-head/`

The date is the start date. The stage tag matches the build-sequence table in `CLAUDE.md`. Shortname is descriptive enough to grep for later.

## Required contents

Every experiment directory contains at minimum:

```
YYYY-MM-DD-stage-shortname/
  README.md           # what, why, hypothesis, result, decision
  config.toml         # full hyperparameters used (committed, not symlinked)
  metadata.json       # commit hash, dataset version, hardware, start/end times
  metrics/            # raw logged metrics (whatever format your training loop emits)
  plots/              # diagnostic plots — calibration, training curves, etc.
```

If the experiment trains a model worth keeping, the checkpoint goes in artifact storage (not git), and `metadata.json` records the storage path and a hash.

## README.md template

Every experiment's README follows this structure. Filling it in honestly is what makes the project navigable in three months.

```markdown
# <stage tag>: <one-line description>

## Hypothesis
What single change is being tested, and what specific measurable outcome
would confirm it helps. State the baseline being compared against by
experiment directory name.

## Setup
- Dataset version: dataset-vN (or raw NikkeTryHard 2020–2024 if pre-curation)
- Code commit: <40-char hash>
- Substrate: lighter ResNet (or whatever)
- Hardware: <GPU model, count, host>
- Wall clock: <hours/days>
- Notable config deltas vs. previous experiment: <list, or "none">

## Validation layers run
- [ ] Layer 1: discard accuracy on held-out-by-player set
- [ ] Layer 1: discard accuracy on held-out-by-time set
- [ ] Layer 1: per-category breakdown (riichi, call/pass, push/fold)
- [ ] Layer 2: ≥3000 duplicate hanchan vs. <previous baseline name>
- [ ] Layer 2: ≥2 RNG seeds, results agree on sign
- [ ] Layer 3: action-agreement on golden set
- [ ] Layer 3: floor check vs. akochan and shanten-greedy

## Results
| Metric | Baseline | This run | Δ |
|---|---|---|---|
| Held-out-by-player discard acc | … | … | … |
| Held-out-by-time discard acc | … | … | … |
| Layer 2 duplicate-hanchan win rate | 50.0% | … | … |
| Golden set top-1 agreement | … | … | … |

(For Stage 1.5 runs, add: KL to SL prior, per-action-type entropy,
value calibration per quantile, league Nash payoffs, belief MI —
whichever apply at that stage.)

## Decision
One of:
- **Keep.** Becomes the new baseline. Next experiment compares against this.
- **Revert.** Measured improvement insufficient or absent.
- **Debug.** Result inconsistent with hypothesis in a way that suggests
  a bug rather than a real negative — describe what's being investigated
  and link to the follow-up experiment directory.

## Notes
Anything surprising, any bugs found mid-run, anything future-you should
know that doesn't fit the structured sections above. Keep terse.
```

## Hard rules

1. **One change per experiment.** If a run bundles two changes (e.g. new dataset + new loss term) the delta is unattributable and the experiment doesn't satisfy Stage 1 / Stage 1.5 discipline. Split it.
2. **Validation sets stay frozen.** Held-out-by-player, held-out-by-time, and the golden set are locked at the start of Stage 1. Don't regenerate them per experiment; that breaks all cross-experiment comparisons.
3. **Decision is binary.** Keep or revert. "Promising but unclear" with no follow-up plan is how a stack of half-evaluated changes accumulates and the project loses attribution.
4. **Layer 2 thresholds.** ≥50.8% duplicate-hanchan win rate over ≥3000 games with ≥2 RNG seeds agreeing in sign = real improvement. 49.5–50.5% = noise regardless of intuition.
5. **Too-good results are bugs until proven otherwise.** +10 pp discard accuracy from a single curation step is almost certainly train/val leak, not a genuine win. Investigate before celebrating.

## What lives outside this directory

- `docs/reference/` — the standing plans. Experiments reference them; experiments don't modify them.
- `docs/decisions/` — short ADR-style notes when you deviate from the planned build sequence (e.g. "extended curation to three weeks because of bug X"). Decisions are project-level; experiments are run-level.
- Model checkpoints and large artifacts — somewhere outside git. Record location and hash in `metadata.json`.

## Indexing

When the directory grows past ~10 experiments, add an `INDEX.md` here listing each experiment, its decision, and a one-line summary. Until then the directory listing itself is the index.