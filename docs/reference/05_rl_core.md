# 05: RL Core

**Phase 1 (learning-engine comparison) and Phase 2 (select and scale).** Revised 2026-10-01 per D28–D31, D35, D37, D39 and D41.

- **Start point:** Mortal-Policy is the code base, but its released AWR→PPO is a *starting implementation*, not a validated recipe. Nitasurin's strong results used withheld infrastructure and a value-based initialization (D28).
- **Method:**
  1. Pick the learning engine by an early comparison at matched compute (§2).
  2. Make the RL interfaces explicit (§3).
  3. Explore interdependent components as packages, then confirm in isolation (§5).
  4. Scale in tranches released by evidence (§6).
- **Main lever:** strength comes mostly from **scale** (`03`). This file keeps the recipe correct, and the scale-up conditional on measured learning.

---

## 1. Frozen for the whole of Phases 1–2 (hashed and tagged)

Changing any of these invalidates every comparison made before the change.

1. The pilot quick students, the quick CQL model, the π_SL student and the BC human prior (`04`).
2. The **GRP** checkpoint: train it with `train_grp.py` on `train` in Phase 0–1, then freeze it.
3. The utility u = [+0.90, +0.50, −0.10, −1.30] (`02` §2), and the **reward units** (§3.2).
4. The rule configuration and the observation schema (v4). Featurizer output hashes must match between the arena and training code paths.
5. The evaluation lab: seed families, test definitions, the population panel and the analysis code (`02`).
6. The software environment: pinned torch, CUDA and the Rust toolchain.

## 2. Learning-engine comparison (Phase 1 early research; D29)

### 2.1 Arms

| Arm | Engine | Initialization | Seeds |
|---|---|---|---|
| **P0** | The released policy recipe (dual-clip PPO on per-kyoku GRP advantage) **plus the §3 correctness fixes** (stored behaviour probabilities, fixed reward units, raw logits, consistent masks) | Pilot quick student | 1 (reference) |
| **P1** | P0 plus discovery package **K1**: critic V(h, i) + declared horizon + LR schedule (§4, §5) | Pilot quick student | 2 |
| **V** | **Value-based Mortal** (Equim's `mortal/train.py`): offline CQL on the same data (`min_q_weight` 5, next-rank auxiliary 0.2), then online value learning (no CQL) with the same GRP reward, `pts` = u, and the same client/server fleet | Quick CQL model, same SL budget | 2 |
| **H** | **Hybrid**, Nitasurin's actual path: V's champion generates self-play → behaviour-clone/distill a policy student on those samples → P1 online | Distilled from V | 1 (2 if it leads) |

### 2.2 Shared protocol

- **Same inputs for every arm:** data split, utility, GRP, observation, opponents (1v3 against the frozen initialization pool), and evaluation seeds.
- **Matched total compute:** 3 boxes × 6 days per arm-seed (1 learner + 2 actors), counting collection *and* learning. Evaluation runs on the shared eval box.
- **Initializations:** each gets the same SL budget (500K-game subset, same steps). Record any remaining difference as part of the recipe.
- **Tuning allowance:** at most 1 box-day of short sweeps per arm, written down before the runs start.

### 2.3 Pre-registered selection (at Gate G0)

1. **Improvement over initialization:** for each arm-seed, T-std against its own initialization on family B. An arm *passes* if both seeds pass, or one passes and the other is inconclusive with a positive point estimate.
2. **Head-to-head:** passing arms are compared final checkpoint against final checkpoint at matched box-days, by T-std.
3. **Tie-break** (inconclusive head-to-head), in order:
   1. stability (kill-rule firings, entropy and KL health);
   2. Δu per dollar;
   3. integration cost with search and the piKL dial. Policy-output arms (P1, H) integrate directly; V needs a policy view such as softmax(Q/τ).
4. **Confirm** the selected arm on fresh family-C nonces (L2; `02` §4).
5. **Scope:** this is quick-model evidence. In T1, the selected engine applied to the **final student** must also pass T-std against that student (D41).
6. **Failure:** a failing arm triggers diagnosis (§8) first. It is **never** taken as evidence that a whole algorithm family is ineffective (D28).

### 2.4 P0 config v0 (values are starting points; the repo's are placeholders)

| Key | Value | Why |
|---|---|---|
| `control.online` | true | Online phase |
| `env.pts` | [90, 50, −10, −130] | u ×100, used by the GRP reward |
| Reward standardization | **Off.** Raw GRP-Δu in u units (§3.2). | Fixed units shared by SL, critic and search |
| `policy.clip_ratio` / `dual_clip` | 0.2 / 3.0 | Repo defaults |
| `policy.entropy_weight` | 1e-3 | Repo default |
| `optim.scheduler.peak = final` | 2e-5 | 5–10× below SL; constant in P0 |
| `optim.weight_decay` | 0.0 | Weight decay pulls weights away from the SL solution |
| `optim.max_grad_norm` | 0.5 | The repo default of 0 disables clipping |
| `control.batch_size` | 512 | Repo default |
| Ratio reference | **Stored behaviour probability** (§3.4) | Replaces the `Old_mortal` snapshot |
| `train_play.default.explore_rate` | 1.0 | Sample actions during collection |
| `baseline.train.state_file` | The arm's frozen initialization | 1v3 against the frozen anchor |
| `enable_amp` / `enable_compile` | true / true | Throughput |

## 3. RL interfaces (D37; implemented in Phase 0)

### 3.1 Return horizon: declared, never implicit

| Horizon | Definition | Property |
|---|---|---|
| **K-surrogate** | Episode = kyoku; terminal reward = that kyoku's GRP-Δu | Upstream behaviour and P0's objective. A **surrogate**: it drops later GRP differences from each decision's return, so it is not policy-preserving for a fixed, imperfect GRP. |
| **H-hanchan** | GRP-Δu paid at each kyoku boundary; returns propagate to the end of the hanchan (γ = 1, GAE λ across kyoku) | Telescopes to u(final) − Φ(start), so it **is** policy-preserving. Higher variance; needs the critic. |

Package K1 compares the two. Every experiment records its horizon in the registry.

### 3.2 Reward units

- Use **raw GRP-Δu in u units**, with no running mean/std standardization of rewards. Upstream standardizes, which gives the same replay different labels as the statistics drift.
- Normalize only the **detached policy advantages**, per batch.
- The SL kyoku-Δ head (`04` §5), the critic and the search bootstrap (`06`) all use these units.

### 3.3 Ordered transition extraction

The upstream loader shuffles individual decisions and gives each one its kyoku's advantage. The new extraction works in three steps.

**1. Build per-seat ordered trajectories before shuffling.** Each decision record holds:

| Field | Content |
|---|---|
| `obs` | Observation |
| `action` | Action taken |
| `behaviour_prob` | Behaviour probability (stored at collection) |
| `mask` | Legal-action mask |
| `reward` | 0, or GRP-Δu at a kyoku boundary |
| `done` | Episode end under the declared horizon |
| `next_idx` | Index of the next trainee decision |
| Critic inputs | `obs` plus the privileged channels (§4) |

**2. Compute targets on the ordered sequence:**
- **detached** value targets (λ-returns);
- GAE advantages;
- the value loss target.

**3. Shuffle afterwards, and train.**

**Tests:**
- Recompute the targets from a fixed replay twice; the results must be bit-identical.
- `done` flags line up with kyoku or hanchan ends.
- No target crosses into another seat's trajectory.

### 3.4 Policy head and behaviour policy

- `CategoricalPolicy` currently returns softmax probabilities. Expose **masked raw logits** separately, and derive probabilities from them in exactly one place.
- Actors store the behaviour probability of the sampled action. Ratios use it, detached.
- Masks are identical in actor, learner, KL, entropy and ACH.

## 4. Critic specification (package K1)

- **Inputs:** the v4 observation h concatenated with **partial privileged information i**: the opponents' concealed hands, from the `Invisible` oracle channels.
  - **Exclude the wall order.**
  - V(h, i) is unbiased (Ebi et al., ICML 2026). A state-only V(s) is biased.
- **Data:** self-play logs record the arena seed (`StartGame.seed`), so `GameplayLoader(oracle=True, trust_seed=True)` reconstructs the hidden information exactly.
- **Network:** a separate b8c128 with input channels 1012 + the selected oracle channels. It has its own optimizer; the LR schedule is part of K1.
- **Targets:** per §3, under the declared horizon. Advantages are normalized per batch (never per episode).
- **Warm-up (B0)** before any policy gradient uses the critic:
  - Freeze the policy and collect about 0.3–0.5M hanchan.
  - Train the critic until its binned calibration (10 bins) is monotone and its explained variance has plateaued.
  - The student's kyoku-Δ head can initialize the observation pathway; it uses the same units.
- **Diagnostic:** the ratio of within-kyoku to across-kyoku advantage variance should rise above ~0.

## 5. Discovery and confirmation (D35)

### 5.1 Discovery packages

Each package is a bounded comparison of a coherent set of changes. Its box-day cap and numeric promotion criterion are written down **before** it runs.

| Package | Contents | Cap | Promotion criterion (exploratory) |
|---|---|---|---|
| **K1** | Critic V(h, i) + horizon (K vs H) + LR schedule | Inside arm P1 (§2) | Arm P1's selection result |
| **K2** | Update size (clip ε, LR) + entropy + **KL anchor schedule**, annealed together (Ataraxos-style power laws, e.g. c_KL(t) = c₀·t^−0.3, LR = clip(a·t^−b, lo, hi)) | 12 box-days | T-std vs the current champion, and fewer kill-rule firings |
| **K3** | **ACH vs PPO** inside the selected engine (§5.3) | 12 box-days | T-std vs the current champion |

**KL anchor (part of K2):**
- Reverse KL(π_θ ‖ π_SL), exact over the legal mask, computed on a frozen copy on the learner.
- Target 0.02 nats/decision at the start. Anchor **decay** is part of the schedule; LuckyJ reached its flexible style without human data (`01` §6).

### 5.2 Confirmation (needed before a result counts toward a rung)

1. Freeze the package.
2. Run **matched patched/unpatched training pairs** from the same checkpoint at a declared compute budget. Two pairs count as preliminary evidence (D41). Report the variation across pairs.
3. Evaluate on **fresh family-C nonces** and **reserved population-panel configurations** (`02` §4.2).
4. Run the ablations needed to attribute the gain. A package gain supports a claim about the package, not about each component.
5. Label the registry row *confirmed*. Everything else stays *exploratory*.

### 5.3 ACH specification (K3)

```
L_pi = − c(s,a) · y(a|s;θ) · A(s,a) / b(a|s)        # y = masked raw logits (§3.4); b = stored behaviour prob

c = 1[π/b < 1+ε] · 1[y_a − mean_legal(y) <  l_th]    if A ≥ 0
c = 1[π/b > 1−ε] · 1[y_a − mean_legal(y) > −l_th]    if A < 0
```

- Starting values: ε = 0.5, l_th = 8 (try 6), η(s) = 1.
- **Keep the champion's entropy coefficient** in the first comparison, so it is a policy-loss-only change. Tune β later, as its own step.
- One gradient step per sample.
- Theory covers 2-player zero-sum only (`01` §7).

## 6. Tranches and lanes (Phase 2; D30, D39)

**Tranches:**

| Tranche | Weeks | Fleet | Release rule |
|---|---|---|---|
| **T1** | 10–12 | Eval, learner, 3 actors | Released at G0 |
| **T2** | 13–15 | Eval, learner, **dedicated candidate learner**, 4 actors | End-of-T1 champion beats start-of-T1 champion (T-std), no severe population-panel regression, **and** the search pilot's belief stage has reported (D32) |
| **T3** | 16–18 | Same as T2 | Same rule, applied to T2 |

- **T1 must include the final-student check** (§2.3).
- **If a tranche gate fails:**
  - diagnose before extending (§8);
  - the remaining tranche money returns to the reserve;
  - consider reopening D29 if another arm had a positive point estimate at G0.
- **Lanes (T2–T3):**
  - The champion lane scales the current best.
  - The candidate lane trains the next package or patch from the same checkpoint, on its own learner box.
  - The eval box stays dedicated to evaluation.
- **Equal-compute rule:** anything that changes the cost per game (RL-6, RL-8) is compared at **equal GPU-hours**.
- **Accounting:** generated games, consumed samples and champion-lineage games are logged separately (`03` §1).

## 7. Single-variable patches (after the packages)

| ID | Patch | Test | Hypothesis |
|---|---|---|---|
| RL-5 | **Opponent pool + style variants** (§7.1) | T-std, plus the population panel | Robustness without a league |
| RL-6 | **4-seat self-play harvesting** for part of the collection | T-std at equal GPU-hours | About 2.5× the samples per simulated game |
| RL-7 | **Mixed reward**: add a small raw-score term to GRP-Δu, in fixed units | T-std | LuckyJ's "mixed reward" |
| RL-8 | **Sample reuse** (2 epochs, or the server's `sample_reuse_*`) | T-std at equal GPU-hours | Only if collection is the measured bottleneck |

### 7.1 Opponent pool and style variants

- **Pool:**
  - π_SL, permanent;
  - a snapshot ring of the last 6 champions (one added every ~2M games, FIFO);
  - style variants.
- **Draw per game:** π_SL 40%, ring 40%, variants 20%.
- **Style variants**, fine-tuned from the champion for 2–5M games with a different utility:
  - first-seeking u₁ = [+1.30, +0.30, −0.30, −1.30];
  - 4th-avoiding u₄ = [+0.60, +0.40, +0.20, −1.20].
  - Both sum to 0.
  - A call-heavy variant needs player-conditioned SL (`07` §4).
- The variants also populate the population panel (`02` §4.2) and serve as products (`07` §5).

## 8. Failure modes and playbook

Triage order for a dead run: units or interface bug → KL too low → LR too high → critic diverging → masking bug.

| # | Failure | Signature | Fix order |
|---|---|---|---|
| 8.1 | Entropy collapse | Discard entropy craters within days; tracking flat or falling | Verify per-batch advantage normalization → entropy coefficient ×3 (once) → forward-KL anchor |
| 8.2 | KL blow-up / drift | KL rising steadily; golden-set agreement falling fast | c_KL up or LR down → check mask consistency |
| 8.3 | Critic divergence | Explained variance falling; advantage std creeping up | Resume B0; lower λ to 0.9; check privileged channels; check `done` flags and horizon |
| 8.4 | Degenerate styles | Deal-in < 8% with win rate falling; calls > 38% with win value falling; deal-in > 15% | Pool mix → style variants → league trigger (§9) |
| 8.5 | Self-play drift | Beats the ring, loses to π_SL or reserved panel configurations | Shift the pool toward π_SL (60/30/10) |
| 8.6 | Silent implementation bugs | — | Unit tests below |
| 8.7 | Throughput or lag pathologies | GPU idle; replay queue growing; policy lag rising | `03` §3–6 |

**Unit tests (CI):**
- Illegal-action probability is exactly 0, in both the actor and learner paths.
- KL(π, π) = 0.
- First-minibatch ratio is 1.0 against the stored behaviour probability.
- Raw logits and probabilities are consistent.
- Aka flags are preserved.
- Featurizer hashes match.
- Ordered-target determinism and `done` alignment (§3.3).
- Fixed reward units: no running statistics in the reward path.
- The deterministic A/A passes (`02` §6).

**Permitted interventions during a long run**, pre-registered:
- an entropy bump (once);
- scheduled pool additions;
- the pool-mix shift under 8.5.

Anything else means stop, diagnose and restart from the last healthy checkpoint.

## 9. Deferred items and their triggers (full rationale in `08`)

| Item | Add only when… | Gate to keep it |
|---|---|---|
| League | A 2–3-day exploiter probe finds Δu ≥ +0.05, **or** 8.4/8.5 persists, **or** the population panel shows material intransitivity (D34) | The exploiter edge shrinks; T-std |
| Distributional or IQN value head | Critic calibration fails structurally after K1 | T-NI, plus better calibration |
| Belief features as policy input | Push/fold agreement stalls **and** S-5 isn't planned | T-std |
| Oracle *policy* distillation | Never by default (Suphx +0.06 dan; Mortal removed it) | T-std |
| V-trace / off-policy replay | RL-8 passes **and** lag pushes the ratio p99 above 2 | T-std at equal GPU-hours |
| Chronological history encoder | Diagnostics show event-ordering information is lost (e.g., a G1 category deficit search can't fix). Evaluate its effect on learning **and** on belief/search cost. | T-std under the identical recipe |
| Endgame curriculum from valid recorded states | Oorasu placement decisions remain a concentrated weakness. Requires complete legal states, with history and hidden information for the simulator and only permitted observations for the policy. | Full-hanchan T-std |
| Risk-sensitive quantile inference | **Cut** (`REVIEW.md` C5) | — |
