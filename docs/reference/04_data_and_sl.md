# 04: Data and the SL Substrate

**Phases 0–1.** This file covers two things:
- turning the only available houou dataset into a clean, frozen, legally handled training asset;
- building the SL **teacher → student** pair that every later phase depends on.

Curation is kept deliberately short. Past these steps, compute buys more in RL and search.

---

## 1. Dataset

**Source:** NikkeTryHard `tenhou-to-mjai` / HF `nikketryhard/tenhou-houou-mjai` (`01` §3).
- Houou 4-player **hanchan only**, 2009-02-01 → 2026-01-23, about 2.5–3.2M games.
- mjai `start_game.names` is present. Dan and rate are not in the mjai files, but they **are** in the bundled source DBs (`UN` tag). Disconnects are recorded there as `BYE`. See `04a` §0 and §3.
- **The step-by-step runbook is `04a_data_pipeline.md`.**

**Snapshot and hash it on day 1.** Tenhou's archive can't be re-downloaded. Store:
- the shard list, with each shard's SHA-256;
- a manifest hash;
- the download date.

**Legal handling:**
- The data license is a **disclaimer only**. Tenhou's terms prohibit redistributing logs and using them to build competing products.
- **Never redistribute** logs, shards or derived per-game data.
- Publish only the **recipe**: filters, thresholds, split rules and the manifest hash.
- Before publishing weights, re-read the terms and record the decision in `08`.
- **Cloud handling** (D23):
  - Store the snapshot only on encrypted volumes or object storage that you control.
  - Process it only on **secure-tier** providers: Runpod Secure, Lambda, Hyperstack, GCP or AWS. Never on community or peer hosts such as Vast community listings.
  - Delete the copies on burst instances after each SL run.
  - Self-play actors never receive Tenhou data.

**Hygiene** (half a day):
- Drop games with placeholder names (for example `EXAMPLE`) or fewer than 4 distinct names.
- Drop games that fail to parse or replay in libriichi, or whose final scores don't reconcile.
- Check rule consistency: aka on, kuitan on, and hanchan end conditions.
- The tonpuu filter is moot, because the dump is hanchan-only.

## 2. Player scoring (curation Op 1)

**The full method, measured numbers and sweep are in `04a` §6.** Summary:

- **Houou is already filtered by Tenhou's room gate** (7-dan and R ≥ 2000).
  - Measured true-skill SD across players is σ_b ≈ 0.02 u/game ≈ 2 pt/game. That is small against per-game luck (σ_w ≈ 0.83), but 2–4× Mortal's version-to-version gaps.
  - The shrinkage constant is k ≈ 1,100–1,700 games.
  - One season's split-half reliability is only about 0.11.
- **The best quarter of seats is only about +0.02 u/game better than average out of sample.** Raw in-sample averages overstate that about 3×.
- **Method:**
  - cross-fitted (game-hash halves E/O), multi-year, time-decayed empirical-Bayes skill;
  - combined with Tenhou R at game start (from the source DBs; leak-free for that game);
  - **games containing a disconnect are excluded from skill statistics** (`04a` §4.1).
- **Sweep arms:** A0 no filter (default), A1 bottom-20% trim, A2 soft loss weights, A3/A4 top 60%/40%, A5 R-only. Pre-registered T-NI/T-std rule; result recorded as D14a.
- **Mechanics:** a small loader extension carries per-group names or weights and seat × kyoku masks (`04a` §6.4).
- **Leakage rule (D40):**
  - Splits are frozen **before** scoring.
  - Skill estimates, empirical-Bayes parameters and arm thresholds are fitted on **`train` games only**, cross-fitted within `train`.
  - Held-out outcomes never influence which training seats are admitted.

**Disconnects (D27):** drop all seats' decisions in kyoku that overlap a disconnect window. That is about 1.2% of modern kyoku, versus 8.6% of games if whole games were dropped. Details in `04a` §4.1.

## 3. Splits (freeze before player scoring and before any training; hash the lists)

| Split | Rule | Use |
|---|---|---|
| **Held-out-by-time** | Games from 2025-07-01 → 2026-01-23 | Generalization; source of the golden set |
| **Held-out-by-game** | 2% of the remaining games, by game-ID hash, **all four seats excluded** from training | In-distribution validation, with no public-state leak across seats |
| **Train window** | 2015-01-01 → 2025-06-30, minus held-out-by-game | Default |
| Train window ablation | Add 2009–2014 | One quick-student T-NI/T-std check; the "modern meta" question |

## 4. Golden set (human-agreement diagnostic)

- **Source:** held-out-by-time games, seats of players in the top 5% of skill **computed only from pre-`heldout_time` games** (≥ 500 games).
- The player list is **frozen and hashed before any golden-set outcome is read** (D40).
- **Size:** ≥ 100K decisions, **stratified**, at least 10K per category:
  - routine discard;
  - riichi or dama decision;
  - call or pass;
  - push/fold facing riichi (at least one opponent in riichi, and not in tenpai yourself);
  - oorasu (South-4) decisions;
  - early discards (turns 1–6).
- **Metrics:** top-1 and top-3 agreement per category, plus mean log-likelihood of the human action.
- **Role:** a diagnostic only. It tracks SL quality. Under RL it is **expected to fall** as the agent goes beyond human play (`02` §7).

## 5. SL teacher

**Train both objectives at full scale.** Each teacher run costs under $100 (`03` §4.2), so the budget (D21) allows all of these:
- **AWR** (Mortal-Policy's Phase 1 with GRP advantages under the u pts);
- **BC** (uniform weights);
- **a second seed of whichever wins T-std** between them.

The winning pair (2 seeds) forms the **teacher ensemble** used for distillation (§6). The BC teacher is always kept as the **human prior τ_human** for the piKL dial (`07` §4), because an AWR policy isn't a pure human model.

**Optional larger teacher:** b40c512, about 70M params. Try it only if featurization can feed it (`03` §4.2) and if a T-std of student-from-b40 vs student-from-b24 is affordable.

**Network:** b24c512, v4 observation (1012×34), trained with AMP.
- Mortal-Policy's online code uses GroupNorm. The teacher can use either GN or BN; the student decision is in `03` §6 (item 4).

**Heads.** This is new code: Mortal-Policy has only `CategoricalPolicy`.

| Head | Target | Label source | Purpose |
|---|---|---|---|
| Policy (46) | Human action | Replay | Main |
| Placement (4-class) | Final hanchan placement | Replay end | Value warm-start, diagnostics |
| Kyoku-Δ (regression, Huber) | This kyoku's GRP-Δu, in **fixed u units** (D37) | GRP + replay | Critic warm-start (`05` §4, package K1); same fixed units as the critic (`05` §3.2) |
| Opponent tenpai (3 × sigmoid) | Is each opponent in tenpai now? | `Invisible` / replay hands | Belief sampler (`06` S-1) |
| Opponent waits (3 × 34 sigmoid) | Each opponent's winning tiles, if tenpai | Replay hands | Belief sampler; push/fold features |
| Opponent hand value (3 × bucket) | Expected win value if they win | Replay | Search-as-features (`06` S-5) |

**Loss:** L = L_policy + 0.5·L_placement + 0.5·L_kyokuΔ + 0.2·L_aux.

**Guard:** Equim found auxiliary tasks gave no strength. They exist here for the belief sampler and the critic warm-start.
- If a T-NI check shows the auxiliary heads hurt the policy, **detach them**: train them on stop-gradient trunk features, or as a separate small network.

**Augmentation:** the existing `enable_augmentation` (m↔p, 2×). Six-permutation augmentation is cut (`08`); it can only return as an ablation that excludes hands where ryuuiisou is live.

**Decision weighting** (optional; v1's Op 4):
- Weight high-leverage decisions 2–3×: riichi, push/fold, oorasu.
- AWR weights multiply on top of these, so test **jointly** with the AWR β at quick scale, with one T-std.

**Starting hyperparameters** (sweep at quick scale; the Mortal-Policy config values are placeholders):
- AdamW;
- peak LR ∈ {1e-4, 3e-4}, 2–5K warmup steps, cosine decay;
- weight decay 0.1 on conv and linear weights only;
- gradient clip 1.0;
- batch 2048 with AMP;
- EMA of weights at 0.999;
- 1–2 epochs over the train window.

**Budget and hardware** (D40: budget in **sample presentations**):
- Presentations = decisions × augmentation × epochs. With roughly 0.5–1.5B decisions, ×2 augmentation and 1–2 epochs, that is **1–6B presentations per teacher**. Measure the decision count in Phase 0 (`04a` §3).
- Compute is about 8–9 EFLOP per 1B presentations at b24c512, so roughly 30–40 h on a 4090 or 5–8 h on an H100 **per 1B**. Scale by the measured presentations.
- **Default:** 1 epoch with augmentation. A second epoch only if a quick-scale check shows a gain.
- The **binding constraint is CPU featurization**:
  - Features are about 137 KB per sample at FP32, so they can't be cached at 1B samples and are generated on the fly.
  - H100 instances come with only 16–26 vCPU.
  - Choose an H100 or a 4090 with ≥ 48 vCPU by the `03` §4.2 rule (D50).
- Use secure-tier hosts only (§1, D23).
- Expect **0.5–3 days per teacher run**, depending on measured presentations and featurization rate.

## 6. Student distillation

**Candidates:** b8c192, b12c256, b16c256 (see `03` §4.1 for speeds).

**Loss:**
- KL(π_ensemble ‖ π_student) at T = 1 over all training states, plus 0.1 × hard-label cross-entropy.
  - π_ensemble is the average of the 2-seed teacher ensemble's probabilities (§5). Distilling from an ensemble usually beats distilling from a single teacher.
- Plus MSE toward the teacher's placement, kyoku-Δ and auxiliary outputs, so the student carries every head needed for critic warm-start and the belief sampler.
- **Ensemble teacher outputs are cached per shard** (D40). This avoids two teacher forward passes per state per epoch. Count the caching pass in the budget.

**Qualification** (D31, D40): student candidates must pass both:
- **T-NI against the best single teacher** (δ = 0.01). This single reference is used both here and for L1;
- golden-set top-1 agreement within 1 pp of that teacher in every category.

**Selection is by RL learning curves, not by distillation alone.** The **two smallest** qualifying candidates go through RL on the leading engine arm, at 6 and 12 box-days (`05` §2).
- Choose by Δu per dollar at 12 box-days.
- If the larger net trails at 6 but leads at 12 (a crossover), choose the larger.
- Distillation non-inferiority shows a student reproduces the teacher. It doesn't show the student can learn stronger strategies.

The chosen student becomes **π_SL**: the RL initialization, the KL anchor, the permanent pool opponent and the deployed incumbent. The teachers stay available for offline work (belief labels, search evaluation, relabelling).

**Later option: kickstarting.** Distill the teacher on *self-play* states as an auxiliary loss early in Phase 2, if the KL anchor in package K2 (`05` §5.1) alone is insufficient.

## 7. Quick models (calibration, sweeps and the engine comparison)

- **Pilot quick students (Phase 0, D41):**
  - two b8c192 BC students with different seeds;
  - used for A/A′ (`02` §6), and as the policy initializations for engine arms P0/P1 (`05` §2).
- **Quick CQL model (Phase 1):** Mortal's offline value-based training (`mortal/train.py`, `min_q_weight` 5, next-rank auxiliary 0.2), on the **same subset and SL budget**. It is the initialization for arm V.
- **Net:** b8c192, trained on a 500K-game subset of the train window (default 2020-01 → 2025-06).
- **Time:** about 1 day per quick student on one box. Sweeps run in parallel on extra boxes (`00` Phase 1 budget).
- **Used for:**
  - the X sweep (§2);
  - AWR vs BC (§5);
  - the window ablation (§3);
  - decision weighting (§5);
  - the engine-comparison initializations (`05` §2).

Rankings found at quick scale are assumed to transfer to the teacher. Spot-check one of them at full scale.

## 8. Free-gain checks (Phase 1, one T-std each)

1. **Test-time m↔p symmetry averaging:** average the logits over obs and swap(obs).
   - First add a unit test: `swap(obs(events)) == obs(swap(events))` bit-for-bit, over 10K logs.
   - Some channels aren't purely tile-indexed, and the swap must only permute tile axis positions 0–8 ↔ 9–17.
   - Costs 2× inference, so it is for deployment and evaluation only.
2. **Checkpoint and EMA averaging.**
3. **A 2–3 model logit ensemble** (different seeds). Deployment only. BigCoach ships an ensemble as a product.

## 9. Exit (Phase 1)

- Teachers trained; two student candidates pass T-NI against the best single teacher.
- Golden-set per-category table logged.
- Quick-student sweep verdicts in the ablation table.
- BC human prior saved and hashed.
- L1 met, together with the riichi.dev smoke deploy (`07` §1).
