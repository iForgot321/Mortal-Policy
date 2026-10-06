# 06: Search

**Phase 1 (pilot, §0) and Phase 3 (expansion).** Revised 2026-10-01 per D32, D33 and D38.

Decision-time search is a proven way past the policy network in imperfect-information games:
- LuckyJ feeds search results into its policy as features.
- SPARTA and LBS improved Hanabi with one-ply search.
- Ataraxos (Stratego) uses belief-sampled, depth-limited rollouts with a regularized update.

riichi.dev allows **3 s per action plus a 15 s bank per kyoku on your own GPU**, so search can run at inference.

**Three investments, three separate gates** (D33):

| Investment | Gate |
|---|---|
| **Inference search** (S-3 at play time) | Must improve the frozen agent within the latency budget |
| **Expert iteration** (S-4) | **Unfunded** until it beats ordinary training at equal **total GPU-hours** (Ataraxos dropped search-based self-play because the data-generation slowdown outweighed the benefit) |
| **Search-as-features** (S-5) | Its own test of strength, inference cost and engineering burden |

---

## 0. Search pilot (Phase 1 early research; D32)

**Goal:** settle feasibility and the shared interfaces early, before large compute commitments.

**Decision class:** riichi vs dama at tenpai (closed hand, riichi legal).

**Three conditions**, on the same 20K sampled decisions from family-A self-play of the leading arm:

| # | Condition | What it measures | Use |
|---|---|---|---|
| 1 | **Policy only** | Baseline | — |
| 2 | **Oracle search**: S-3 with the *true* hidden state, taken from the trusted arena seed | Whether action evaluation and continuations are useful once hidden-state inference is removed as a source of error | **Lab-only diagnostic.** Isolated from deployment code paths; never cited as a strength result |
| 3 | **Belief search**: S-3 with uniform-proposal particles (§2) | The deployable version | Pilot verdict |

**Metric:** recovery ratio = (belief gain) ÷ (oracle gain), where gain = class-restricted paired GRP-Δu against the policy (common worlds, paired standard error).
- Oracle search is an **optimistic upper bound**: it can exploit hidden information, a form of strategy-fusion leakage. That is why the ratio is the metric.

**Go criteria** (pre-registered; required before tranche T2, `00`):
1. The oracle gain is significant: paired z > 2.5 on 20K decisions.
2. Recovery ≥ 0.5.
3. p99 request-to-response latency < 2.5 s on deploy-class hardware.

**Diagnosis:**
- **Oracle fails** → investigate the rollout policy, horizon or value bootstrap.
- **Oracle passes, belief fails** → investigate reconstruction validity, belief calibration or opponent-model mismatch.

**Then:** confirm full-game strength separately. Inference search on riichi-vs-dama alone gets a T-std against no-search; class-restricted gains are not full-game claims.

**Interfaces defined here and reused later:**
- the particle format (§2);
- the `branch()` helper (§3);
- the value-bootstrap units (fixed GRP-Δu, `05` §3.2);
- the latency harness.

**Schedule:**
- Oracle stage by G0 (week 9). It is nearly free, because self-play seeds are known.
- Belief stage before T2 (week 13).

## 1. Scope: where to search (Phase 3 expansion)

The classes are chosen at Gate G1 from the pilot and a scaled per-class analysis (§7).

| Class | Typical count per kyoku | Candidate |
|---|---|---|
| Riichi vs dama at tenpai | ~0.2–0.5 | Pilot class |
| Push/fold discard facing riichi or a threatening call | ~2–6 | Next |
| Call or pass (chi, pon, kan) | ~1–3 | When the policy margin is below a threshold |
| Oorasu (South-4) decisions with a placement at stake | Varies | Yes |
| Routine early discards | ~8–12 | Policy only |

**Margin gating:** skip search when the policy's top-1 probability is above p* (e.g. 0.9) and no threat flag is set.

## 2. S-1: belief particles (D38)

**A particle is a latent history, not a snapshot.** It must contain enough to reproduce every opponent's private observation at every past decision. Those observations are needed to compute action likelihoods.

**Particle contents:**
- Each opponent's **starting hand** (13 tiles).
- The **full draw sequence** to date: who drew what, in order, including rinshan draws after kan.
- The remaining **wall and dead-wall order**: future draws, rinshan, dora/ura indicators.
- **Red-five identities** throughout.
- The whole particle must be consistent with every public event: discards with tedashi/tsumogiri, calls, kans, riichi, dora reveals.

**Tile accounting:**
- Each physical tile appears exactly once.
- A **claimed discard is counted once**, even though it appears both in the discard history and in a meld.

**Transitions:** forward-simulate the particle's history through libriichi's rules. A particle is **accepted** only if it:
1. reproduces the root observation and the legal-action set exactly;
2. conserves the tile multiset and the aka identities;
3. satisfies the riichi, furiten, call and kan constraints, including tsumogiri after riichi and closed-hand tenpai at the riichi declaration.

**Weights:**

```
w ∝ prior(trajectory) × Π_opponent-actions π_blueprint(action | opponent's private observation in this trajectory)
      / proposal(trajectory | public history)
```

- **v1 proposal = uniform** over consistent deals and draws. Rejection-sample from the unseen-tile multiset, subject to the constraints. With a uniform proposal and a uniform prior, the weight is the likelihood alone.
- **A learned proposal** (LBS-style) is allowed **only** once a *joint* sampler with a computable density exists. For example, an autoregressive model over hands and draw slots, with the D38 correction. The summary heads (tenpai, waits, value) alone are **not** a joint sampler.

**Maintenance:**
- 64–256 particles, updated incrementally at each public event.
- Resample when the effective sample size (ESS) falls below 0.3 × N.
- **Collapse recovery:** rejuvenate with constraint-preserving swap moves (MCMC over unseen-tile assignments, Metropolis-accepted by likelihood). If ESS stays low, re-sample from scratch.

**Tests:**
- Every accepted particle replays to the root observation.
- Targeted scenario tests: riichi, furiten, chi/pon/daiminkan, ankan/kakan, rinshan, dead-wall dora, haitei.
- On held-out self-play, where the truth is known, measure:
  - tenpai-marginal calibration;
  - wait-set log-likelihood;
  - ESS per decision.

**Leaks are hard failures.** Inference search never reads the true hidden state. Only the oracle diagnostic (§0) may, and it is isolated in lab-only code.

## 3. S-2: world reconstruction from particles

- A particle **is** a full latent history. Build `libriichi::arena::Board` directly from it:
  - `haipai` from the starting hands;
  - `yama`, `rinshan` and dora/ura indicators from the wall order.
- Do **not** "add discarded or drawn tiles back to the current hand". Ownership transitions run forward from the latent history.
- Replay the kyoku's public events as forced actions up to the decision point, then branch.
- **New Rust helper**, batched across worlds and actions:

  ```
  fn branch(particle, history, candidate_action) -> RolloutState
  ```

- **Tests:**
  - For 10K recorded decisions, the *true* latent history must reproduce the logged continuation exactly under forced actions.
  - Every sampled particle must reproduce the root observation (§2).

## 4. S-3: one-ply search

At a searched decision:
1. **Candidates:** the policy's top-k legal actions (k = 3–4).
2. **Rollouts:** for each candidate a and particle w (common random numbers across candidates):
   - roll out with the blueprint;
   - **truncate** after T own turns (e.g. T = 6) and bootstrap with the critic, or with GRP at kyoku end.
   - The bootstrap uses fixed GRP-Δu units (`05` §3.2).
   - Truncation is consistent with the declared horizon (K or H, `05` §3.1).
3. **Opponent continuations**, Pluribus-style: blueprint, push-biased or fold-biased, with fixed mixture weights.
4. **Update.** Two variants, tested by T-std (D33):
   - **(a) Plain:** pick the best mean value, deviating from the policy's argmax only if the gain exceeds τ_dev and passes a paired standard-error check.
   - **(b) Regularized, the default candidate:** π_search ∝ π(a)·exp(Q̂(a)/λ). This is piKL/MMD-style, regularized toward the policy, as in Ataraxos's search update.
5. **Action selection** (an explicit deployment choice): **argmax** vs **sampling** from π_search. Ataraxos samples.
6. **Time management:**
   - Per-decision budget p99 < 2.5 s, request-to-response, **measured on the deploy box**.
   - Draw on the 15 s bank for riichi and oorasu decisions only.
   - Hard fallback: the policy's argmax.

**Budget** [calc/inf]:
- About 118 kyoku/s at batch 2000 is a *throughput* figure, not a latency figure.
- One search request's latency must be measured directly. The expectation is about 1–3K truncated rollouts per 3 s.

**Why one ply:**
- Deviating only at the root avoids most strategy fusion.
- Rollouts use the blueprint's imperfect-information policy.
- ReBeL, Student of Games and full information-set MCTS are infeasible here (`01` §7).

**Validation:**
- T-std on family B: the search-enabled agent vs the same network without search, then confirm (L4).
- Track per class: the deviation rate, the gain, and latency p50/p99.

## 5. S-4: expert iteration (separately gated; D33b)

- **Default: unfunded.**
- **Equal-cost test** (if G1 approves it): ordinary training vs training with searched targets, at **equal total GPU-hours**, including search generation.
- **Budget in GPU-hours.** Derive coverage from the measured cost ratio R of a searched decision to an ordinary one:

  ```
  search_share = f·R / (1 − f + f·R)
  ```

  For example, at R = 10, searching 5% of decisions consumes 34.5% of decision compute.
- **Targets:** π_search from the regularized update (§4).
- **Loss:** + c_ex · KL(π_search ‖ π_θ) on the stored states.
- **Precedents:**
  - JPS searched 1–5% of training games.
  - A policy cloned from SPARTA kept about 27% of the search gain.
  - Ataraxos found search-based data generation not worth its cost.
- **Funded only if** it wins T-std at equal total cost. Funding then comes from the Phase 3 actor budget, by decision-log entry.

## 6. S-5: search-as-features (separately gated; D33c)

- LuckyJ feeds search results into its policy network, in training and at play time (`01` §6).
- Mortal's v4 observation already has opponent-blind single-player EV and win-probability tables (`encode_sp_table`).
- **Extension:** per-candidate, opponent-aware features computed cheaply from S-1 particles and the waits and hand-value heads, without full rollouts:
  - estimated deal-in probability;
  - expected deal-in cost;
  - opponent-aware win probability and EV.
- This changes the observation schema, so it needs a student retrain or fine-tune.
- **Gate:** T-std against the champion at equal GPU-hours, with inference search on for both sides. Also report the inference-cost increase and the engineering time.

## 7. Gate G1 inputs

1. **Pilot results** (§0): recovery ratio, latency, the full-game T-std for riichi-vs-dama search.
2. **Scaled S-3 on the champion**, for each class in §1, on 20K sampled decisions from family-A self-play:
   - the deviation rate;
   - the paired gain;
   - latency on the deploy box.
3. **Decisions:**
   - classes for inference search (D33a);
   - whether S-4 gets its equal-cost test (D33b);
   - whether S-5 is attempted (D33c).

If a class shows large disagreement that search *can't* fix (high variance, poor belief calibration), that is the evidence that can reopen the architecture question (`05` §9).

## 8. Exit (Phase 3)

- Search-enabled agent passes T-std against no-search, then confirms (L4).
- p99 latency < 2.5 s on the deploy box; fewer than 0.5% of actions exceed the bank.
- Belief calibration and particle-validity reports logged.
- S-4 and S-5 verdicts in the ablation table, if they were run.
