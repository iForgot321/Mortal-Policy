# 02: Evaluation Lab

**Phase 0 deliverable.** No training experiment starts until the L0 rung is met (`00_ROADMAP.md`). The only exception is the two pilot quick students needed for A/A′ (D41).

*Revised 2026-10-01 per D33–D36 and D41: calibrated sequential test, fixed-N confirmation rule, population panel, withdrawn VR-1, matched-pair recipe claims, exploratory vs confirmed labels.*

The model is Stockfish's Fishtest: every change is a **patch** that gets a sequential-test verdict on duplicate games. That makes verdicts cheap and trustworthy. Cheap verdicts mean more experiments, and more experiments mean more strength.

---

## 1. Principles

1. **One primary metric**, the one we optimize: Δu on the utility in §2.
2. **Test by a sequential procedure whose error rates are calibrated by simulation** (§4.1), not by fixed sample sizes. Confirm with fixed-N runs on fresh nonces.
3. **The unit of analysis is the duplicate seed-set** (4 games), never the individual game.
4. **Separate seed families for tracking, testing and confirmation.** This prevents the winner's curse.
5. **Measure the noise floor empirically** (A/A′) before trusting any verdict.
6. **Write verdicts to the ablation table** (§9), labelled *exploratory* or *confirmed*. Prose summaries don't count.
7. **Promotions and milestones also pass the population panel** (§4.2). The 1v3 duplicate test stays the per-patch test.

## 2. Utility and metrics

**Primary utility** (decision D2): **u = [+0.90, +0.50, −0.10, −1.30]** for 1st through 4th.
- It is matched to riichi.dev's OpenSkill Plackett–Luce update at equal strength (`01` §5).
- It sums to 0, so **the null for an equal-strength challenger is exactly 0**.
- Per-game SD ≈ 0.83 under a uniform placement split.
- Δu = 0.01 is roughly "1 pt/game" on a ×100 scale.

**Primary metric:** Δu = the challenger's mean utility per game in 1v3 duplicate against the named champion.

**Secondary metrics** (always reported, never gating):
- avg rank and the full rank histogram;
- **avg pt on [90, 45, 0, −135]** (Houou 7-dan; comparable with Mortal's published tables);
- tobi rate; the style panel (§7).

**Expected effect sizes** (for planning, from `01` §2):
- The first RL-over-SL step: roughly Δu 0.005–0.02.
- Mature incremental patches: roughly 0.002–0.01.
- Anything claimed above 0.05 per patch → suspect a bug, such as a leak or a seat-indexing error.

## 3. Duplicate format

- **Arena:** `libriichi` `OneVsThree`. For each seed, 4 games are played with the challenger rotating through East, South, West and North, against 3 copies of the champion. Walls, hands, dora/ura and rinshan are deterministic per (seed, kyoku, honba).
- **Action selection:** argmax on both sides by default.
  - **Argmax vs sampling** (from the policy, or from the search-updated policy) is an explicit **deployment choice**, tested by T-std (D33).
  - Evaluate the mode you will deploy.
- **Seed families:** three master keys with disjoint nonce ranges. Record them in the lab config, and never reuse a family's nonce range across roles.

  | Family | Role |
  |---|---|
  | **A** | Tracking evaluations and checkpoint selection. Noisy, frequent. |
  | **B** | GSPRT verdicts on patches |
  | **C** | Held-out confirmation at gates, and ablation-table numbers. Touched only at G0/G1/G2 and for publication. |

- **Duplicate caveat:** a call shifts the draw order, so correlation within a set decays after the first divergent call. Measure the real benefit (§6); don't assume one.

## 4. GSPRT specification

**Per-set score:** x_i = the mean over its 4 games of the challenger's utility, optionally corrected by control variates (§5).

**Statistic** (Fishtest `LLR_alt2` form, generalized to a Gaussian):
```
LLR = N / (2 s²) · [ (x̄ − μ0)² − (x̄ − μ1)² ]
```
- s² is the running sample variance of x_i, **floored** at 0.25·σ²_set (σ_set from A/A′).
- **Minimum sample: 5K sets before the first look.** Then look every 500 sets.
- **Nominal** boundaries: α = β = 0.05, so ±2.944. These are only the starting point; the boundaries actually used are those that pass §4.1.
- **Zero variance** (s² = 0 before the floor applies): no verdict; keep sampling.
- If the cap is hit, the verdict is **inconclusive**. That counts as a fail for superiority and non-inferiority tests.

**Why the extra rules** (REVIEW_V2 #1): plugging a running variance into a Gaussian likelihood does not guarantee Wald's error rates for bounded, discrete set scores. A sparse null stream at 500 sets can falsely pass non-inferiority about 8% of the time.

**Test types:**

| Type | H0 → H1 bounds (Δu) | Cap (sets) | Used for |
|---|---|---|---|
| **T-smoke** (non-regression) | [−0.02, 0] | 20K | Refactors, infrastructure changes, deploy builds |
| **T-NI** (non-inferiority) | [−0.01, 0] | 60K | Speed, distillation and stability patches, where "no worse" is the goal |
| **T-std** (superiority) | [0, +0.01] | 100K (400K games) | Default strength patch |
| **T-fine** (superiority) | [0, +0.005] | 250K (1M games) | Late-stage small patches |
| **T-confirm** (fixed-N, **fresh** family-C nonces) | **Pass rule:** 95% block-bootstrap lower bound > 0 (superiority) or > −δ (non-inferiority). N is fixed in advance. | 50K–250K | Gates, milestones, publication |

For non-regression and non-inferiority rows, "pass" means H0 (the regression) is rejected. For superiority rows, "pass" means H1 is accepted.

**Nonce retirement:** a family-C nonce range is **retired** once its result has informed any development decision. Confirmations always draw fresh ranges. Ordinary 95% intervals are reported only for fixed-N runs; sequential runs report the LLR and the calibrated error rates.

**Planning sample sizes** [calc]:
- Assumes σ_set = 0.42, i.e. no duplicate benefit, σ_game / 2. Replace it with the value measured in A/A′.
- At 40K games/h. Evaluation throughput is similar to self-play; measure it.

| δ | E[sets] at midpoint | Hours | E[sets] at μ1 | Fixed-N equivalent |
|---|---|---|---|---|
| 0.02 | 3.8K | 0.4 | 2.3K | 4.8K |
| 0.01 | 15.3K | 1.5 | 9.3K | 19.1K |
| 0.005 | 61K | 6.1 | 37K | 76K |

If duplicate play plus control variates bring σ_set down to 0.30, every row roughly halves.

**Eval capacity:**
- A **dedicated eval box** (`03` §2) running 24/7 at about 30–40K games/h handles about **10–15 T-std verdicts/day**, or 2–4 T-fine verdicts/day.
- It never competes with training.
- Queue order: kill-rule checks first, then gate tests, then candidate-lane verdicts, then tracking runs.
- If the queue backs up for more than 24 h, rent a second eval box from the reserve rather than shrinking test sizes.

### 4.1 Calibration of the sequential procedure (before L0; D36)

1. **Collect** empirical set-score distributions from A/A′ (§6), and from a pair with a known difference (the synthetic hobbled patch).
2. **Simulate** the *exact* procedure (minimum sample, look schedule, variance floor, caps) on streams resampled from those distributions, shifted to H0 and to H1. Use ≥ 10K simulated tests per test type.
3. **Include adversarial variants:** sparse (rare large losses), skewed, and heavy-tailed rescalings of the empirical distribution.
4. **Accept** a test type only if the empirical false-pass and false-fail rates are both ≤ 5%. Otherwise:
   - widen its boundaries until they pass;
   - or switch that test type to a bounded-score sequential method (empirical-Bernstein or betting confidence sequence; set scores are bounded in [−1.3, +0.9]).
5. **Record** the calibrated boundaries and error rates in the lab config and in `08`.

**Champion rule:** a patch is tested against the **frozen current best**. When it passes, it becomes the new champion, and the old champion joins the anchor ladder.

**Anchor ladder:** every ~2 weeks, run T-confirm-lite (20K sets on family A) of the current champion against π_SL and against each prior milestone. This catches non-transitive "beats the last champion, loses to older ones" drift.

### 4.2 Population panel (D34)

**Used for:** tranche gates, G0–G2 and publication. Not every patch.

**Configurations** (fixed at Phase 0; each is a table composition for the 3 opponent seats):

| Code | Opponents | Role |
|---|---|---|
| P-SL | 3× π_SL | Baseline |
| P-OLD | 3× an older champion (rotating milestone) | Drift check |
| P-STY | 1 each of first-seeking, 4th-avoiding and balanced variants | Style mix |
| P-MIX | π_SL + older champion + style variant | Mixed table |
| P-FLR | 3× shanten-greedy | Floor |
| **P-RES1, P-RES2** | **Reserved** mixed tables, e.g. BC prior + a variant + an older checkpoint | **Never used in training or routine tuning**; confirmation only |

**Score:** population-weighted Δu, with pre-registered weights (P-SL 0.3, P-MIX 0.3, P-STY 0.2, P-OLD 0.1, reserved 0.1).

**Severe-regression check:** Δu < −0.02 on any single configuration (fixed-N, 95% upper bound < 0) blocks promotion, whatever the average.

**Reporting:** every configuration's Δu is shown, so an average gain can't hide a large weakness.

**Cost:** mixed tables have higher variance than 1v3 duplicate, so panel runs use fixed-N (about 20K sets per configuration).

## 5. Variance reduction

**VR-0, duplicate + rotation:** always on.

**VR-1 (deal-luck sums) is withdrawn** (D36). Sums of starting-hand features and dealer indicators over the kyoku reached depend on the policy: dealer repeats and game length change them. So subtracting a fitted multiple of them biases Δu under a new candidate. Agreement on A/A′ can't establish unbiasedness for future policies.

**Allowed form, conditional chance residuals only:** at each chance event e (deal, draw, dora/ura reveal), add the term

```
E[V | pre-chance state] − V(post-chance state)
```

where the expectation is under the **true** chance distribution: uniform over the unseen wall tiles, given the full lab state. Each term is zero-mean by construction, for any policy.

**VR-2, draw-luck control variate** (AIVAT-lite)
- At each of the challenger's draws, add E_draw[V] − V(after the draw).
- V is the frozen GRP-plus-critic value; the expectation is over the unseen-tile distribution. Cost is up to 34 value evaluations per draw, so batch them.
- Only the chance-node terms apply, because argmax policies add no decision-node terms.

**Adoption rule** (for VR-2 or any chance-residual variant)
- Measured variance ratio ≤ 0.8 on A/A′ data.
- **Bias validation under policies that deliberately change** dealer repetition, game length and placement behaviour. Use the style variants and a hobbled policy. The corrected and raw means must agree within fixed-N CIs.
- Until it passes, **raw utility remains the promotion metric**.
- References: MJ-DLVAT reports 45.5% fewer games; a community riichi attempt got about 15% (`01` §7). Don't over-invest.

## 6. Calibration runs (Phase 0, before any B-side patch)

**A/A, deterministic (harness check)**
- The challenger is the same model as the champion, with argmax play.
- Every rotation within a set is then the same game, so **Δu must be exactly 0 and s² exactly 0**.
- Any deviation is a bookkeeping, seat-indexing or rule-asymmetry bug. Fix it before going on.

**A/A′ (noise floor)**
- Two SL students from the same recipe with **different training seeds**, over 20K sets on family A.
  - These are the **pilot quick students**, trained in Phase 0 as the explicit exception to "no training before L0" (D41).
  - They are reused as early-research initializations.
- Output:
  - σ_set, the real per-set SD;
  - the duplicate benefit ρ, from comparing against unpaired games;
  - "recipe noise", the |Δu| between equivalent recipes.
- If recipe noise is at or above δ = 0.01, then recipe-level claims need 2 training seeds (§6.1).

**Per-seat table:** East/South/West/North baselines for win, deal-in and placement, from the A/A′ runs. Later "worse as dealer" observations are compared against it.

**Synthetic patch:** before L0, verify GSPRT with the harness:
- a checkpoint deliberately hobbled by **random-action injection** (ε = 0.1 uniform over legal actions) must fail T-std. Temperature scaling preserves argmax, so it does not hobble deterministic play;
- an A/A′ pair must be inconclusive or pass non-inferiority.

### 6.1 Checkpoint claims vs recipe claims

- A GSPRT verdict says "**this checkpoint** is stronger". That is enough for day-to-day patches.
- A claim that "**this recipe change** helps" needs **matched patched/unpatched training pairs** (D41):
  - each pair trains from the same checkpoint at a declared compute budget;
  - the claim rests on the variation across paired effects.
  - Two candidate seeds against one realized champion don't isolate the recipe, because that control trajectory may have been unusually weak.
- **Two pairs = preliminary evidence.** Confirmation adds fresh family-C nonces and reserved panel configurations (`05` §5.2).
- Engine-arm claims at G0 follow `05` §2.3.

## 7. Style fingerprint panel (always logged)

Computed per evaluation run and per rolling 5K self-play hanchan, with **reference values in `01` §6** (LuckyJ, Suphx, NAGA, 特上 humans).

| Metric | Definition |
|---|---|
| Win / deal-in / call / riichi rate | Per kyoku |
| Dama share | Closed non-riichi wins ÷ wins |
| Avg win value, avg deal-in value | Points |
| Honitsu share, mangan+ share | % of wins |
| First-discard honour rate, first-discard 2–8 rate | First discard of each kyoku |
| **Shanten-backward rate, quiet** | Closed pre-riichi 14-tile hands with no opponent riichi: share of discards leaving shanten above the best achievable (standard shanten including chiitoi and kokushi) |
| Shanten-backward rate, facing riichi | Same, with an opponent in riichi |
| Tobi rate, 4th rate as oorasu leader | Placement discipline |
| Golden-set top-1/top-3 agreement, per category | Human-agreement diagnostic (`04` §4); **expected to fall** as RL improves |

**How to read it:**
- The panel is a **tripwire and a style compass**, not a target.
- Leaving the human band within the first 48 h of a run means a pathology (§8).
- Drifting toward LuckyJ's fingerprint later is fine; that is what flexible strong play looks like.

## 8. Tracking evaluations and kill rules during training runs

**Tracking**, every ~12 h:
- A fixed 2K sets on family A against the frozen π_SL, plus 2K sets against the current champion.
- For trends only; never used for verdicts.

**Kill rules** (automated; pre-registered):

| Signal | Action |
|---|---|
| Tracking Δu vs π_SL < −0.03 on two consecutive tracking evaluations | Pause; run the `05` §8 playbook |
| Discard-head entropy < 30% of its SL value for 3 consecutive windows | Entropy intervention (`05` §8.1) |
| Mean KL(π‖π_SL) > 0.3 nats/decision, sustained | Stop; KL coefficient or LR is wrong |
| Critic explained variance negative and falling for > 6 h | Stop; critic diverging |
| Any style-panel metric > 3σ outside the dataset band within the first 48 h | Pause; inspect games by hand |
| Illegal-action or chombo count > 0 in evaluation | Stop; masking bug |

## 9. Experiment registry and ablation table

**Registry:** one row per run, with:
- run ID, git commit, config hash, dataset hash, utility vector, **return horizon** (K/H), **engine arm** (P0/P1/V/H), **package** (K1–K3);
- **mode: exploratory or confirmed** (D35). Only confirmed rows count toward rungs;
- parent champion, cumulative self-play hanchan, wall-clock;
- a link to the log.

**Ablation table** (the project's permanent record): one row per verdict.

| Patch/package ID | Mode | Description | Test type | Seed family (nonce range) | Sets | Δu (±95% CI, fixed-N only) | LLR | Verdict | Panel (min config Δu) | Δ avg rank | Δ pt (90/45/0/−135) | Style-panel deltas | Commit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

Inconclusive and failed rows stay in the table. They are evidence too.

## 10. External and fixed anchors

- **Shanten-greedy bot:** discards to maximize ukeire; never calls; riichis at tenpai. Write it in Phase 0 and keep it frozen. It is a floor, checked monthly.
- **akochan:** CPU-heavy (10–60 min per game review; Equim stopped test-plays because of the cost). Use it **only at milestones L1 and L3**, over about 3–5K sets. Report Δu against it; never gate on it.
- **riichi.dev:** the external truth (`07` §2). Its noise is large, so never use it for patch verdicts.

## 11. Implementation notes

- Reuse `mortal/one_vs_three.py` and `libriichi::arena::OneVsThree`. Add:
  - `lab/gsprt.py`: running statistics, LLR, the stopping rule (minimum sample, floor), block bootstrap;
  - `lab/calibrate.py`: the §4.1 simulation harness;
  - `lab/panel_pop.py`: population-panel configurations and scoring (§4.2);
  - `lab/seeds.toml`: the family keys and nonce ranges;
  - `lab/cv.py`: control variates;
  - `lab/panel.py`: style metrics parsed from mjai logs, reusing `libriichi.stat.Stat` where it applies.
- Log every evaluation game's mjai JSON (compressed). Blunder mining (`07` §3) and the style panel both read these files.
- Make the lab callable as one command:

  ```
  lab run --challenger X --champion Y --test T-std --family B
  ```

  It should write its verdict row automatically.
