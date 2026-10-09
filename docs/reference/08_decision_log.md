# 08: Decision Log

Every major decision, with its rationale, evidence and **what would reopen it**.

**Rules:**
- Add an entry *before* changing any goal, gate, utility, metric or phase order.
- Never delete an entry. Supersede it with a new one that references the old.

**Format:**
- ID, date, decision.
- Why: the rationale.
- Evidence: pointers into `01`, `REVIEW.md` or `PRO_APPROACH.md`.
- Reopen if: the trigger that would bring it back.

---

## Foundations

**D1 (2026-09-30): Target venue is riichi.dev (RiichiLab).**
- **Why:** mjai.app shut down on 2026-04-30. riichi.dev is its successor, with a persistent leaderboard, self-hosted bots, a GPU allowed, and a 3 s + 15 s-bank time budget.
- **Evidence:** `01` §3.
- **Reopen if:** riichi.dev closes, or changes its rating or time rules materially.

**D2 (2026-09-30): Utility u = [+0.90, +0.50, −0.10, −1.30] everywhere.** That covers the GRP pts, value targets and the primary metric.
- **Why:** it matches the OpenSkill Plackett–Luce update at equal strength, and it is zero-sum, so the null is 0.
  - Mortal-Policy's default [6,4,2,0] (linear rank) is the wrong objective for this ladder.
  - The v1 vector [0.75, 0.30, 0, −1.20] was tokujou 6-dan, mislabelled as houou 7-dan.
- **Evidence:** `01` §4–5.
- **Reopen if:** the agent is consistently top-3, since against weaker tables the effective weights flatten. Test a flatter u as a patch.

**D3 (2026-09-30): Primary metric is Δu in 1v3 duplicate, judged by GSPRT on seed-sets, with three seed families.**
- **Why:**
  - Mortal's version gaps are 0.001–0.02 avg rank, measured over 1M games.
  - v1's fixed 3,000-game gates and "≥50.8%" couldn't resolve real effects.
  - v1's A/A was degenerate under argmax.
- **Evidence:** `REVIEW.md` C1, C3; `02`.
- **Reopen if:** the measured σ_set makes T-std take more than 6 h. Then invest in control variates (VR-1/VR-2) before changing δ.
- **Amended 2026-10-01 by D34 and D36:** promotions also need the population panel (D34). The sequential test is calibrated by simulation, and VR-1 is replaced (D36).

**D4 (2026-09-30): Substrate is Mortal-Policy, keeping its GRP per-kyoku reward and 1v3-vs-frozen-pool collection.**
- **Why:**
  - It is a known-working recipe (the old mjai.app #1).
  - GRP is potential-based shaping, and it preserves the optimal policy.
  - v1's "terminal-only, no shaping" would have discarded working credit assignment.
- **Evidence:** `01` §1; `REVIEW.md` C2.
- **Reopen if:** RL-0 fails twice (G0 branch 4), or a value-based Mortal v4 path shows a clear advantage in a quick A/B.
- **Superseded in part 2026-10-01 by D28, D29 and D37:**
  - The "known-working recipe" claim is corrected (D28).
  - The learning engine is now chosen by an early comparison (D29). Mortal-Policy stays the code base.
  - GRP shaping preserves the optimum only over the full hanchan; the kyoku-terminal objective is a declared surrogate (D37).

**D5 (2026-09-30): Throughput is a top-level metric. Teacher → student distillation; the student is the self-play actor.**
- **Why:**
  - Strength scales with games played. The frontier trained at 10–100× the v1 scale (BigCoach: hundreds of millions of hanchan, hearsay).
  - Inference cost varies about 40× across the candidate nets.
- **Evidence:** `01` §8; `03`.
- **Reopen if:** never, as a principle. Specific choices (net size, BN vs GN) are made by T-NI.
- **Amended 2026-10-01 by D31:** distillation non-inferiority only qualifies student candidates. The actor size is chosen from RL learning curves.

**D6 (2026-09-30): Order is de-risk before investing.** Quick SL → RL-0 → teacher → student.
- **Why:** in the Mortal lineage, RL gains over SL are small and hard to get (Equim: none from months of PPO; Nitasurin: +0.6 pt). That is the largest project risk, and it is testable in about a week.
- **Evidence:** `01` §2.
- **Reopen if:** none. Fixed.
- **Superseded 2026-10-01 by D28, D29 and D41.** The released-recipe run (RL-0) is now a check of the released implementation, not a verdict on RL. RL feasibility is decided by the engine comparison (D29).

## Algorithms and components

**D7 (2026-09-30): Asymmetric critic V(h, i), using opponents' hands but not the wall order, plus GAE within the kyoku (RL-2).**
- **Why:**
  - Mortal-Policy has no critic, so every decision in a kyoku gets the same advantage.
  - V(h, i) is unbiased, and PerfectDou is the precedent.
  - libriichi's `Invisible` data makes it cheap.
- **Evidence:** `01` §1, §7.
- **Reopen if:** its GSPRT fails. Then try a symmetric critic (a plain V(h)) as the fallback patch.
- **Amended 2026-10-01 by D35 and D37:** the critic is tested as part of discovery package K1 (critic + horizon + LR schedule), with defined data and loss interfaces (D37).

**D8 (2026-09-30): KL anchor to π_SL with the target annealed upward (RL-3). It is a drift guard, not a source of strength.**
- **Why:**
  - Diplomacy showed that self-play started from human data still drifts.
  - LuckyJ reached flexible strong play without human data, so the anchor must decay.
  - NAGA's imitation style is the most rigid.
- **Evidence:** `01` §6–7.
- **Reopen if:** stability metrics show no benefit. Then drop it.

**D9 (2026-09-30): ACH is tested as a single-variable swap for the policy loss (RL-4).**
- **Why:** it is LuckyJ's named algorithm, and it is a few lines of code. Its theory is 2-player only, so for four players it is an empirical bet.
- **Evidence:** `01` §6–7.
- **Reopen if:** it fails T-std. Revert, and don't retry until after S-4.
- **Amended 2026-10-01 by D37:** ACH needs raw logits and stored behaviour probabilities, and keeps the champion's entropy coefficient in its first test.

**D10 (2026-09-30): Opponent diversity comes from a snapshot ring plus utility-vector style variants, not a league (RL-5).**
- **Why:**
  - A 4–6-learner league divides the throughput of one GPU. Even at the D21 budget it would take about half the Phase 2 fleet (D25).
  - There is no mahjong evidence of strong intransitivity.
  - BigCoach's style families are cheap and nearly equal in strength.
  - AlphaStar used PFSP, not the v1 plan's "Nash-weighted sampling".
- **Evidence:** `01` §6; `REVIEW.md` M2.
- **Reopen if:** the exploiter probe finds Δu ≥ +0.05, or degenerate-style / self-play-drift failure modes persist (`05` §9).

**D11 (2026-09-30): Search is a planned phase. One-ply SPARTA/LBS-style at high-leverage decisions, plus expert iteration, plus LuckyJ-style search-as-features.**
- **Why:**
  - It is the proven route past the policy net.
  - The riichi.dev budget allows it at inference.
  - LuckyJ uses search results as policy features, in training and at play time.
- **Evidence:** `01` §6–7; `06`.
- **Reopen if:** the S-3 GSPRT fails after belief calibration passes. Then fall back to S-4 expert iteration only.
- **Superseded 2026-10-01 by D32, D33 and D38:**
  - A narrow belief/search pilot moves into the early-research stage (D32).
  - Inference search, expert iteration and search-as-features become separate investments (D33).
  - Particles carry latent deal/draw history (D38).

**D12 (2026-09-30): Stay on libriichi; Mahjax gets a 2-day spike only.**
- **Why:** Mahjax is >10× libriichi on 8×A100 with a random policy. But it has no strength results, a provisional API, and porting the v4 observation encoder is a large job.
- **Evidence:** `01` §3; `03` §7.
- **Reopen if:** the spike passes parity, shows ≥ 3× end-to-end throughput, and integration is ≤ 2 weeks. Otherwise re-survey quarterly.
- **Amended 2026-10-01 (with D21):**
  - At fleet scale the payoff is larger. A third-party report puts real PPO training at 21–58K steps/s per GPU, which could mean 3–9× a libriichi box per dollar [inf].
  - The integration allowance is raised to **≤ 4 weeks**, run in parallel with Phases 1–2 on libriichi.
  - The ≥ 3× throughput threshold is now measured **per dollar**.
- **Spike result (2026-10-06): D12 stands.** Mahjax 0.1.6; `experiments/mahjax-spike.md`.
  - **Parity: fixable, not yet passing.** 30K replayed games (10K bot, 10K bot with random actions, 10K Tenhou human), 19.5M decisions, 99.99946% identical legal sets. Mahjax has three rule bugs relative to Tenhou: no furiten after passing a no-yaku winning tile, a honba ≥ 8 dealer-repeat cap, and riichi ankan of a waited tile. libriichi's arena has one deviation of its own: it pays a triple ron, which Tenhou aborts. Mahjax's payouts equal Tenhou's on the 10K Tenhou games.
  - **Throughput per dollar: passes.** Once an upstream memory bug is patched (`vmap` × `lax.cond` copied a 22 MiB table per environment), an RTX 3060 runs env + observation + b8c192 at 127K hanchan/h: $0.44–0.72 per million hanchan at Vast 3060 prices, 7.5–12× below libriichi's $5.4/M. The 5090 and H100 were not measured.
  - **Integration: fails, ≈ 6–9 weeks.** The blocker is the observation. v4's single-player-table channels (~12%) come from a recursive search that can't be ported to JAX hash-identically. Adopting Mahjax means a new observation schema in two languages, an SL retrain and a T-NI against v4.
  - **Reopen if** G0 finds actor throughput binding (fleet soak, `03` §3.7) **and** an SL A/B shows a JAX-portable observation (v4 without the single-player tables) is non-inferior to v4. Integration would then be ~4–5 weeks.

## Data

**D13 (2026-09-30): The NikkeTryHard dump is the sole data source. Snapshot and hash it; never redistribute; publish only the recipe.**
- **Why:**
  - Tenhou's archive can't be re-downloaded.
  - The data license is a disclaimer only, and Tenhou's terms prohibit redistribution.
- **Evidence:** `01` §3; `04` §1.
- **Reopen if:** an officially licensed source appears.

**D14 (2026-09-30; amended 2026-10-01): Player scoring = cross-fitted, multi-year, time-decayed empirical-Bayes skill, combined with Tenhou R at game start. Sweep arms A0–A5, with A0 (no filter) as the default.**
- **Why** (measured, `04a` §6.1):
  - The houou room gate already filters strongly. σ_b ≈ 0.02 u/game, k ≈ 1,100–1,700 games, one-season split-half reliability ≈ 0.11.
  - In-sample raw selection overstates the edge about 3×.
  - Even an ideal top-25% filter keeps players only about +0.02 u/game better than average, at the cost of 75% of the data.
  - The mjai files have no dan or rate, but the bundled source DBs do.
  - *Supersedes* the original "X ∈ {25, 40, 60, 100}% hard threshold" design.
- **Evidence:** `04a` §6; `REVIEW.md` M6.
- **Reopen if:** an arm beats A0 at T-std on two quick-student seeds. Then record the arm as D14a.
- **Amended 2026-10-01 by D40:** skill statistics, empirical-Bayes parameters and arm thresholds are fitted on the `train` split only, after the splits are frozen. Held-out outcomes never influence which seats are trained on.

**D27 (2026-10-01): Disconnects are masked at kyoku level for training, and BYE games are excluded from skill statistics.**
- **Rule:** drop all seats' decisions in every kyoku overlapping a disconnect window (`BYE` → reconnect); if the player never reconnects, drop through the end of the game. Compute GRP rewards on the full game before masking.
- **Why** (measured, `04a` §4.1):
  - Tenhou auto-play while disconnected is 100% tsumogiri, with no calls and no wins. The mjai files record these as if the human made them, which mislabels discards, passes, riichi skips and **declined wins**.
  - Opponents' decisions in those kyoku are likely conditioned on a disconnect the model can't see.
  - Disconnected seats finish 4th 33–35% of the time.
  - Disconnects affect 3.5% of games in 2009 and 8.6% in 2021. The kyoku-overlap rule costs only 0.6–1.2% of kyoku.
- **Evidence:** `04a` §4.1.
- **Reopen if:** the style panel or the Q4/Q5 counts show residual disconnect artifacts. Then escalate to truncate-from-first-BYE (5.5% of kyoku), or add a chronic-disconnector exclusion arm.

**D26 (2026-10-01): Exclude known unauthorized-bot accounts from all training and evaluation data.**
- **Which accounts:** for example 未払い, 秋野花, まだ垢 and ハラヤハ, which BigCoach/Gokujan's own site lists as accounts its bot played Houou games on (`01` §6).
- **How:** through the loader's `excludes` list (`04a` §4, Q8). Keep the list in `names/excludes_bots.txt`, with a source for every entry.
- **Why:**
  - The BC human prior, the golden set and the "human-like" measurements must be human.
  - Those accounts played Houou against Tenhou's rules.
- **Reopen if:** never for the human prior and golden set. A separate, labelled "bot-imitation" experiment would need its own decision.

**D15 (2026-09-30): Augmentation stays with the existing m↔p only.**
- **Why:** ryuuiisou is souzu-only. Mortal's m↔p-only swap is very likely deliberate.
- **Evidence:** `REVIEW.md` C6.
- **Reopen if:** an ablation of six-permutation augmentation, excluding ryuuiisou-live hands, passes T-std.

## Budget and infrastructure (added 2026-10-01)

**D21 (2026-10-01): Compute budget of $10–15K total, cloud only, no hardware purchase.** Plan ≈ $12.4K plus a $2.0K reserve (`00` budget table). *Reallocated by D42: ≈ $12.7K plus a $1.6K reserve.*
- **Why:**
  - The user chose this tier.
  - Purchase prices are inflated by the 2026 memory shortage: a 5090 is about $7K new, against an MSRP of $1,999.
  - Renting a 5090 for 12 months, 24/7, costs about $5K, which beats buying (`01` §10).
- **Evidence:** `01` §10.
- **Reopen if:**
  - spend runs above 120% of the phase plan for two consecutive months (D24);
  - or rental prices rise more than 50%;
  - or GPU purchase prices fall back near MSRP. Then re-run rent-vs-buy for the Phase 4 deploy box.

**D22 (2026-10-01): Scale out on consumer GPUs (4090/5090 with ≥ 32 vCPU), not up on datacenter GPUs.**
- **Fleet roles:** learner, eval and deploy on-demand; actors interruptible where that saves ≥ 25%.
- **H100 use:** burst only (SL teachers, if featurization can feed them), plus an optional learner upgrade from the reserve under the `03` §5 rule.
- **Why:**
  - Consumer GPUs give about 2.5× more CNN training throughput per dollar.
  - For small launch-bound nets an H100 is estimated at only 1.0–1.3× a 4090.
  - Mortal-Policy's `client.py`/`server.py` already implements the distributed actor–learner split.
- **Evidence:** `01` §1, §10; `03` §2.
- **Reopen if:**
  - the CPU-only inference benchmark (`03` §3.5) passes, which would add a monthly CPU-server actor tier;
  - or Mahjax passes D12, which would replace most actors with GPU-resident simulation.

**D23 (2026-10-01): Tenhou data is stored and processed only on secure-tier providers, on encrypted volumes, and deleted from burst hosts after use. Actors never receive it.**
- **Why:** Tenhou's terms prohibit redistribution. Community or peer hosts (such as Vast community listings) are third-party machines.
- **Evidence:** `01` §3; `04` §1.
- **Reopen if:** none.

**D24 (2026-10-01): Budget guardrails.**
- Monthly spend report against the `00` phase plan, with an alarm at 120%.
- Release idle boxes the same day.
- Any fleet expansion beyond the plan, or any draw on the reserve, needs a decision-log entry.
- Checkpoints go to object storage every ~200 iterations, so preempted or released boxes lose nothing.
- **Reopen if:** none.

**D25 (2026-10-01): The budget buys parallelism and replication first.**
- Two lanes (champion and candidate).
- Two training seeds for RL-0, RL-2 and RL-4.
- A dedicated eval box.
- Teachers trained under both AWR and BC at full scale, plus a second seed.
- A 5–10% expert-iteration share in Phase 3.
- **Why:** money converts into calendar time and trustworthy claims fastest this way. A bigger single model or a league buys less per dollar.
- **Evidence:** `02` §6.1; `05` §6 (lanes; formerly §3).
- **Reopen if:** a deferred item's trigger fires (`05` §9). The league remains trigger-only even at this budget, because 4–6 extra learners would take about half the Phase 2 fleet.
- **Amended 2026-10-01 by D30, D39 and D42:**
  - Phase 2 is released in tranches (D30).
  - The candidate learner is a dedicated, budgeted box (D39).
  - Expert iteration is unfunded until it beats plain training at equal total cost (D33). The 5–10% share is withdrawn.

## Research strategy and correctness (added 2026-10-01, from `REVIEW_V2_STRATEGY.md` and `REVIEW_V2.md`)

**D28 (2026-10-01): Fact correction about the Mortal-Policy evidence.** It supersedes the "known-working recipe" rationale in D4 and D6.
- **What #91 actually says** (`01` §2):
  - Nitasurin's **+0.6 pt** (500K hanchan) came from **value-based** online training over offline.
  - After moving to policy-based online training, "the performance improvement exceeded expectations". No number is given.
  - His top-ranked setup was **"Policy-based online + behavior cloning from value-based offline-to-online Mortal-V2 samples"**.
  - AWR "is a new addition when I decided to open-source, replacing Behavior Cloning".
  - Opponent pools, replay buffers and multi-GPU code were withheld.
  - Equim's policy-based attempts (before v1) failed to beat an SL baseline, while "simple value-based methods turned out to be effective".
- **Consequence:**
  - The released AWR→PPO code is a *starting implementation*, not a validated recipe.
  - A failure of RL-0 is evidence about the released code, not about RL.
- **Reopen if:** none. This is a factual correction.

**D29 (2026-10-01): Choose the learning engine by an early comparison at matched total compute.** Supersedes D4's fixed substrate; Mortal-Policy stays the code base.
- **Arms:**

  | Arm | What it is | Seeds |
  |---|---|---|
  | **P0** | Released policy recipe plus the D37 correctness fixes | 1 (reference) |
  | **P1** | P0 plus discovery package K1: critic + declared horizon + LR schedule | 2 |
  | **V** | Value-based Mortal online training: Equim's `train.py`, offline CQL init, then online DQN, same libriichi/GRP/fleet | 2 |
  | **H** | Hybrid, Nitasurin's actual path: V champion → BC/distill into a policy → policy online (P1 recipe) | 1, or 2 if leading |

- **Fairness:**
  - Same data split, utility, GRP, opponents and evaluation seeds.
  - Each arm-seed gets the same box-days, counting collection *and* learning.
  - Initializations are trained on the same SL budget. Record any difference.
- **Pre-registered selection:** see `05` §2.
- **Why:**
  - The lineage's evidence is split: Equim succeeded with value-based methods; Nitasurin with policy-based methods on top of value-based initialization.
  - The released code omits the infrastructure behind Nitasurin's result.
  - The engine is the largest unresolved research question, and it is cheap to test relative to Phase 2.
- **Reopen if:** the selected engine stalls at a tranche gate (D30) while another arm had a positive point estimate at G0.

**D30 (2026-10-01): Scale in evidence-released tranches.** Cumulative games become a forecast, not a criterion. Amends L3 and D25.
- **Structure:** Phase 2 is three tranches, T1–T3.
- **Release rule** (`00` Phase 2): T(k+1) is released only if both hold:
  - the champion at the end of T(k) beats the champion at the start of T(k) by T-std;
  - the population panel (D34) shows no severe regression.
- **On failure:** diagnose the limiting factor (`05` §8) before extending. Unspent tranche money returns to the reserve.
- **Why:**
  - 200M games doesn't show the recipe is still learning, or that the actor is large enough.
  - REVIEW_V2 #6 showed that even a healthy fleet could miss 200M.
  - Ataraxos: "quantity of games matters more than quality", but only while the curve still rises.
- **Reopen if:** none. The thresholds may be tuned by a new entry.

**D31 (2026-10-01): Actor size is chosen by RL learning curves.** Amends D5.
- **Method:**
  - The two smallest students that pass T-NI against the best single teacher both go through RL on the leading arm, at two budgets (6 and 12 box-days).
  - Pick by Δu per dollar at the larger budget.
  - If the larger net is behind at 6 box-days but ahead at 12 (a crossover), pick the larger.
- **Why:** distillation non-inferiority measures reproduction of the teacher, not capacity to learn beyond it. Jones (2021) found train/inference compute trade-offs in board games, which should be measured, not assumed.
- **Reopen if:** a tranche stalls with the policy at high entropy and no lag pathology. That suggests a capacity limit.

**D32 (2026-10-01): A narrow belief/search pilot runs in the early-research stage.** Amends D11.
- **Decision class:** riichi vs dama.
- **Three conditions:**
  1. Policy only.
  2. **Oracle search** given the true hidden state. Lab-only diagnostic, never deployed, never cited as strength.
  3. **Belief search** with uniform-proposal particles (D38).
- **Metric:** recovery ratio = (belief − policy) ÷ (oracle − policy), measured as class-restricted paired gain in GRP-Δu.
- **Go criteria for scaling search, required before tranche T2:**
  - the oracle gain is significant;
  - recovery is ≥ 0.5;
  - p99 latency is < 2.5 s on deploy-class hardware.
- **Diagnosis rule:**
  - Oracle fails → fix the rollout policy, horizon or value.
  - Oracle passes but belief fails → fix reconstruction, calibration or the opponent model.
- **Why:**
  - Search interfaces (latent history, joint sampling, likelihoods, rollout state, value calibration) are expensive to rework late.
  - REVIEW_V2 #10: G1 needed S-3 results from work scheduled after it.
- **Caveat:** oracle search is an optimistic upper bound. It can exploit hidden information, which is why the *ratio* is the metric.
- **Reopen if:** the pilot passes. Broader coverage then follows in Phase 3.

**D33 (2026-10-01): Inference search, expert iteration and search-as-features are separate investments.** Supersedes D11's bundle and the 5–10% expert-iteration share in D25.
- **D33a, inference search:** passes T-std against no-search on the frozen agent, within the latency budget.
- **D33b, expert iteration:**
  - **Unfunded by default.**
  - It earns funding only by beating ordinary training at equal **total GPU-hours**, counting search generation.
  - Its budget is set in GPU-hours. Coverage is derived from the measured cost ratio R, where search_share = fR / (1 − f + fR).
- **D33c, search-as-features:** its own test of strength, inference cost and engineering burden.
- **Also tested explicitly** (each as a T-std):
  - a regularized search update toward the policy (piKL/MMD-style), as the S-3 default variant;
  - argmax vs sampling from the search-updated policy, as a deployment choice.
- **Why:**
  - Ataraxos (superhuman Stratego for "a few thousand dollars") dropped search-based self-play because "the slowdown on data generation … outweighed potential benefits".
  - It plays by sampling from its search-updated policy, regularized with the same reverse KLs as training.
- **Reopen if:** expert iteration wins its equal-cost test. Then fund it from the Phase 3 actor budget.

**D34 (2026-10-01): Promotions and milestones also require a population panel.** Amends D3.
- **Panel:** a fixed set of opponent configurations: π_SL, older champions, style variants, shanten-greedy floors, and mixed tables. **Reserved configurations** never enter training or tuning.
- **Score:** a pre-registered population-weighted Δu, plus per-configuration severe-regression checks.
- **Scope:** used at tranche gates, G0–G2 and publication. 1v3 duplicate GSPRT stays the per-patch test.
- **Why:** the target arena has mixed tables. α-Rank-style evaluation motivates matchup visibility, though it is not evidence of intransitivity here.
- **Reopen if:** the panel shows a material intransitivity. That would also count toward the league trigger.

**D35 (2026-10-01): Two experiment modes, discovery and confirmation.** Amends the "change one variable at a time" rule in `00` and `05`.
- **Discovery:** bounded comparisons of coherent packages, each with a box-day cap and its numeric promotion criteria written down first:
  - **K1:** critic + horizon + LR schedule.
  - **K2:** update size + entropy + KL schedule, with Ataraxos-style coordinated annealing.
  - **K3:** ACH vs PPO inside the winning engine.
- **Confirmation:**
  - Freeze the package.
  - Run matched patched/unpatched training pairs at a declared compute budget (D41).
  - Use fresh seed-family C nonces and reserved panel configurations.
  - Run the ablations needed to attribute the gain.
- **Registry:** every row is labelled *exploratory* or *confirmed*. Only confirmed results count toward rungs.
- **Why:** components depend on each other. A package gain supports a claim about the package, not about each part.
- **Reopen if:** none.

**D36 (2026-10-01): Calibrate the sequential test and fix the control variates** (REVIEW_V2 #1, #2). Amends D3.
- **Minimum sample** before the first look: 5K sets.
- **Variance floor:** s² ≥ 0.25·σ²_set (σ_set from A/A′). If s² = 0, there is no verdict; keep sampling.
- **Calibration by simulation, before L0:**
  - Run the exact procedure (chunking, caps, floor) on streams resampled from empirical set-score distributions shifted to H0 and H1. Include sparse and skewed variants.
  - Accept only if the empirical error rates are ≤ 5%.
  - Otherwise widen the boundaries, or switch to a bounded-score sequential method (empirical-Bernstein or betting confidence sequence).
- **VR-1 (deal-luck sums) is withdrawn.** It is not zero-mean under a new policy.
  - Only **conditional chance residuals** are allowed: realized minus expected value at each chance event, under the true chance distribution. These are zero-mean by construction.
  - Raw utility remains the promotion metric until a correction passes validation under policies that change dealer repetition, game length and placement behaviour.
- **T-confirm:**
  - Fixed-N on fresh family-C nonces.
  - Pass = 95% lower bound > 0 (superiority) or > −δ (non-inferiority).
  - Used nonces are **retired**.
- **Synthetic hobbled patch:** random-action injection (ε = 0.1), not temperature, because temperature preserves argmax.

**D37 (2026-10-01): RL interfaces are explicit** (REVIEW_V2 #3, #4). Amends D4, D7 and D9.
- **Return horizon is a declared choice.**
  - **K-surrogate:** kyoku-terminal GRP-Δu. This is upstream behaviour and RL-0's objective. It is a *surrogate*, not policy-preserving.
  - **H-hanchan:** GRP rewards at kyoku boundaries, with returns propagated to the end of the hanchan. This one is policy-preserving.
  - Package K1 compares the two.
- **Fixed reward units:** raw GRP-Δu in u units.
  - No running standardization of rewards.
  - Only the detached policy advantages are normalized per batch.
  - The SL kyoku-Δ head, the critic and the search bootstrap all share these units.
- **Ordered transitions:** per-seat ordered trajectories are built before shuffling: obs, action, behaviour probability, mask, reward, done, next-trainee-decision index, critic inputs. Detached targets are computed, then shuffled.
- **Raw logits:** the policy head exposes masked raw logits separately from probabilities.
- **ACH:** uses stored, detached behaviour probabilities, and keeps the champion's entropy coefficient in its first test.

**D38 (2026-10-01): Belief particles carry latent deal/draw history** (REVIEW_V2 #5). Amends S-1 and S-2.
- **Particle contents:** each opponent's starting hand, the full draw sequence, the wall and dead-wall order, and red-five identities. All must be consistent with public events.
- **Weights:** prior × action likelihood ÷ proposal density. The proposal stays **uniform** until a learned *joint* sampler and its density exist.
- **Validity:** every accepted particle must:
  - reproduce the root observation and legal actions;
  - conserve tiles and aka;
  - count claimed discards once.
- **Tests:** riichi, furiten, calls, kan and the dead wall.
- **Collapse handling:** recover by rejuvenation when the effective sample size drops below a threshold.

**D39 (2026-10-01): Capacity gate and a dedicated candidate learner** (REVIEW_V2 #6, #7). Amends D25.
- **Fleet soak before T1** (at G0): ≥ 48 h end-to-end, which must show:
  - bounded replay-queue growth;
  - policy lag p95 ≤ 2 parameter versions;
  - sustained useful learner updates, including featurization, critic and KL costs.
- **Track separately:** generated games, consumed samples, and champion-lineage games.
- **Candidate learner:** a dedicated box in tranches T2 and T3. The eval box stays dedicated.
- **Calendar:** dates move when retries or optimization delay scaling.
- **Search budgets** are in GPU-hours. Latency is measured request-to-response on deploy hardware.

**D40 (2026-10-01): No held-out outcomes in data selection; SL accounting** (REVIEW_V2 #8, #11). Amends D14.
- **Order:** freeze the splits (`04a` §7) **before** player scoring (`04a` §6).
- **Scoring scope:** fit on `train` only, cross-fitted within it.
- **Golden set:** its population comes only from pre-`heldout_time` games, and is frozen before any of its outcomes are read.
- **SL budget:** counted in **sample presentations** (decisions × augmentation × epochs), measured.
- **Ensemble targets:** cached per shard.
- **One teacher reference:** the **best single teacher**, used for both student selection and L1.

**D41 (2026-10-01): Schedule prerequisites and recipe-claim controls** (REVIEW_V2 #9, #10).
- **Phase 0 exception:** two pilot quick students (different seeds), trained for the A/A′ calibration. They double as the early-research initializations.
- **Recipe claims** use matched patched/unpatched training pairs. Two pairs count as preliminary evidence.
- **Scope of early evidence:** results from quick-model initializations are labelled as such. The selected engine must also pass T-std against the **final student** in T1 before T2 is released.

**D42 (2026-10-01): Budget reallocation.** Amends D21 and D25.
- **Early-research cap:** $2.2K.
- **Phase 2:** tranched (T1 $1.4K; T2 and T3 $1.9K each, including the candidate learner).
- **Phase 3:** $2.2K, with expert iteration unfunded by default.
- **Reserve:** $1.6K.
- **Total:** about $12.7K planned + $1.6K reserve ≈ $14.3K (`00` budget table).
- **Why:** the strategy review's direction is to reallocate from the long RL and search phases, not to add on top.

**D43 (2026-10-03): The deterministic A/A runs in true FP32; verdict runs keep AMP.** Amends the A/A procedure (02 §6) behind L0.
- **Measured** (b40c192 checkpoint, 500 sets, the same family-A nonces each time; `experiments/lab-harness.md`):
  - AMP: 21/500 sets had divergent games.
  - FP32 with TF32 (the Ampere default for cuDNN convs): 2/500.
  - True FP32: 0/500, every set score exactly 0.
  - Rule-based A/A (shanten-greedy): exact.
- **Why:** every divergence was an argmax flip at a near-tie (top-2 probability gap ≤ 0.004 under AMP, ≤ 2e-4 under TF32). In 1v3 the challenger and champion engines run separate batches (1 vs 3 seats per game), and reduced-precision numerics depend on batch composition. That is inference noise, not a bookkeeping, seat-indexing or rule bug, which is what the A/A exists to catch.
- **Consequences:**
  - `python -m lab aa` uses `--no-amp` (FP32, no TF32, deterministic cuDNN). "Δu = 0 exactly" in L0 refers to this mode.
  - A/B verdict runs keep AMP, the deployment mode (02 §3: evaluate the mode you deploy). Near-tie flips are zero-mean noise absorbed by σ_set; they bias neither side.
  - Set scores are summed in integer hundredths of u, so an all-ranks set is exactly 0 (in binary floating point 0.9 + 0.5 − 0.1 − 1.3 ≠ 0).
- **Reopen if:** an FP32 A/A diverges at a decision that is not a near-tie (a real harness bug), or the deploy engine moves off AMP.

**D44 (2026-10-05): The sequential test is calibrated; σ_set = 0.328 and the nominal ±2.944 boundaries are kept for every test type.** Implements D36 / 02 §4.1.
- **A/A′** (02 §6): pilot quick students A (seed 1) vs B (seed 2), same recipe, 20K sets (80K games) on family A, AMP. Run `20261005-124255-measure`.
  - σ_set = 0.3277 against 0.415 for unpaired sets: duplicate play removes **37.7%** of the variance. Within-set game correlation −0.126.
  - Recipe noise: Δu(A − B) = +0.0015, 95% CI [−0.0031, +0.0061]. That is below δ = 0.01 (§6), so recipe-level claims do not yet require 2 training seeds. The D41 matched-pair rule for recipe claims still applies.
  - Per-seat table (challenger avg rank E/S/W/N): 2.494 / 2.512 / 2.498 / 2.488. Win and deal-in rates are flat across seats (`panel.json` in the run dir).
- **Calibration** (§4.1): the exact procedure (5K-set minimum, looks every 500, variance floor 0.25·σ²_set, caps) on streams resampled from the A/A′ distribution and its sparse, skew-left, skew-right and heavy-tailed variants, shifted to H0 and H1.

  | Test | Worst false pass | Worst false fail | E[sets] at H0/H1 | Simulated tests per variant |
  |---|---|---|---|---|
  | T-smoke | 0.40% | 0.40% | ≈ 5.1K | 10K |
  | T-NI | 3.03% | 3.28% | ≈ 7.8K | 10K |
  | T-std | 3.03% | 3.28% | ≈ 7.8K | 10K |
  | T-fine | 4.60% | 4.77% | ≈ 25K | 40K (10K gave 4.98%) |

- **Consequences:** `lab/lab.toml` now has `sigma_set = 0.3277` and `calibrated = true`. Registry rows from here on count toward rungs.
- **Caveat:** T-fine's margin is thin.
- **Reopen if:** σ_set or the set-score shape changes materially (e.g. control variates adopted, a much stronger agent, a different action-selection mode). Then rerun §4.1. For T-fine, also reopen if any recalibration exceeds 5%; widen its boundaries or switch it to a bounded-score method.

**D45 (2026-10-05): dataset-v1 is frozen.** Records the 04a §8 identity; the splits are a frozen item (05 §1).
- **Identity:** `7c6a9e60457e7eb9fe441e17a2a2db55d3d3f789176d6e7bf4c483c297959c6b` = sha256(MANIFEST.id `37d36e49…` ‖ pipeline commit `5cffb71` ‖ params.toml `e289e646…`). HF revision `6450bb58`.
- **Contents:** 2,512,433 games; 22 dropped by quarantine. Q6 masks: 215,206 games. Q8: 4 bot accounts.
- **Splits** (`SPLITS.sha256`, leak tests pass):

  | Split | Games |
  |---|---|
  | train | 1,805,376 (2015-01 → 2025-06) |
  | heldout_time | 102,054 (2025-07-01 → 2026-01-23) |
  | heldout_game | 48,388 |

- **Default held-out window kept:** no D14b. Its 7 months are complete, though the source DBs end mid-October 2025.
- **Golden set:** 164 players, 120K decisions.
- **Determinism test:** passed.
- **Card:** `~/mj-data/reports/DATASET_CARD.md` (private).
- **Reopen if:** a converter or quarantine defect is found that changes training labels. Then dataset-v2 with a new identity, and every comparison restarts on it.

**D46 (2026-10-05): Behaviour probabilities travel in a per-game sidecar** (option A of `docs/design/behaviour-probs.md`). Implements 05 §3.4 and the D37 ordered-transition field; decided by the user.
- **The gap:** mjai logs never show passes or claims overruled by a higher-priority claim, so their behaviour probabilities cannot be recovered from the log.
- **Recorded by the arena:** for every reaction of an agent that reports one, the board keeps `(event_idx, actor, kan_select, label, prob, applied)`. This includes passes and overruled claims. `applied` says whether the board executed the reaction, or for a pass, whether nobody claimed.
- **`prob`:** the probability that the engine's own sampler gave the emitted label: (1 − ε)·1[a = argmax] + ε·π(a). It is computed by the engine that samples, is 1 for forced moves, and is null when the emitted action was not drawn from the policy (the rule-based agari guard). A null keeps the entry out of ratio terms.
- **Storage:** one gzipped JSON file of columns per game, with the same file name as the mjai log, in a separate directory (`OneVsThree(behaviour_dir=…)`). Not next to the log: actors, the server and the trainer treat every file in a log directory as a game log.
- **The mjai logs are unchanged**, so the featurization and training paths stay hash-identical.
- **Join:** the learner matches loader entries on (event_idx, seat, kan_select) and requires equal labels. A label mismatch is allowed only for a reaction the board did not apply; that entry gets no behaviour probability.
- **Rejected:**
  - B, recomputing from the last K parameter versions: an extra forward per sample, and unrecoverable beyond K.
  - C, non-standard events in the mjai log: breaks every log consumer.
- **Reopen if:** actors stop shipping mjai logs, for example in favour of featurized trajectories that can carry the probability inline.

**D47 (2026-10-08): The CPU-only actor test is deferred** (`03` §3 item 5; D17b stays open). Decided by the user.
- **Why:**
  - On obs v4 the actor boxes are already CPU-bound: libriichi's arena loop takes ~89% of the main thread, and v4's single-player tables cost ~0.76 CPU-s per hanchan (`experiments/fleet-host-benchmarks.md`, `experiments/sp-ablation.md`). A CPU-only actor would put inference on those same cores.
  - v4 stays for Phase 1 (the SP ablation was inconclusive), so the case that would favour a CPU tier, encoding made nearly free by dropping the tables, does not apply.
  - `00`'s Phase 0 exit criteria (L0; σ_set, hanchan/h per box type, featurization samples/s) don't depend on it.
- **Consequence:** Phase 0 no longer owes a CPU-only verdict. Actors stay on GPU boxes (`03` §2).
- **Reopen if:** the observation drops the single-player tables (v5 or a cheaper variant), or actor cost dominates at G0. Then run the test as written (a few hours on a c3d-highcpu-180 spot, ~$5–10).

**D48 (2026-10-08): The fleet's private network is SSH tunnels over each box's published SSH port, not WireGuard or Tailscale.** Amends `03` §2. Decided by the user.
- **Why:**
  - Vast boxes are unprivileged Docker containers, which usually have no TUN device or `NET_ADMIN`. That rules out kernel WireGuard and the TUN modes of WireGuard and Tailscale.
  - Each userspace variant costs something the box workflow forbids:
    - Tailscale needs an account auth key on every box (a credential on a community host) and a SOCKS shim for `client.py`.
    - wireproxy needs a published UDP port and a third-party binary.
  - Every box already publishes SSH, the one port the workflow allows. Tunnels need no new software, port or account.
- **How** (`tools/fleet/fleet.py`):
  - `server.py` binds to loopback only, and refuses any other address unless the config sets `online.server.allow_non_loopback`. The trainer runs on the learner box.
  - Each actor generates its own key on its box. The learner installs it for an unprivileged `fleet` user, restricted by `authorized_keys` options and an `sshd_config` `Match` block:
    - forwarding only to `127.0.0.1:<port>`;
    - no remote forwarding, shell, PTY, agent or X11.
  - The actor pins the learner's host key, read by the workstation over its own SSH session. A loop on the actor holds `ssh -N -L 127.0.0.1:<port>:127.0.0.1:<port>`, and `client.py`'s backoff covers the gaps.
  - Replacing or tearing down an actor revokes its key.
- **Residual risk:** an actor host has root on its box, so it can still talk to `server.py`. Messages are now unpickled with `weights_only=True`, so a message can't run code, but a malicious host could still submit bad games. The tunnels keep out everyone else; they don't make actor hosts trusted. A WireGuard or Tailscale network would leave the same gap.
- **Reopen if:** tunnel throughput or reconnects limit the fleet soak (`03` §3.7), or the fleet moves to VM-based boxes, where WireGuard runs natively.

**D49 (2026-10-08): Observation v4 is Phase 1's observation, not the project's. It is reviewed at G0 against a measured value of one doubling of RL compute.** Amends `05` §1 item 4, which froze v4 through Phase 2. Decided by the user.
- **Why:**
  - The SP ablation (`experiments/sp-ablation.md`) left strength inconclusive: pooled −0.0024 u [−0.0056, +0.0009].
  - Its speed result is certain. v5 makes actors 2.4–2.8× and the learner 2.3× cheaper, about 1.2–1.5 extra doublings of RL compute at equal cost.
  - So v5 is the better RL observation if one doubling buys more than about 0.002–0.004 u. No public mahjong result gives that number.
- **What:**
  - All Phase 1 work (teachers, student candidates, every engine arm) uses v4, so Phase 1 results stay comparable with each other.
  - Every arm-seed freezes checkpoints at ¼ and ½ of its compute budget (billed box-hours; `fleet.py watch --budget-box-hours`), as well as the final one.
  - **The doubling estimate:** a fixed-N `measure` of the full-budget checkpoint against the ½-budget checkpoint of the same run, on family B.
    - P0's estimate is pre-registered in `experiments/p0.md` and gives the first number early in Phase 1.
    - D31's actor-size curves (6 vs 12 box-days) add more by G0.
  - **At G0**, the observation is reviewed with those estimates: keep v4 for Phase 2, or switch to v5 or the cheaper shanten-threshold-1 variant (`experiments/sp-ablation.md` §6).
- **Cost of a switch:**
  - Teachers and students are retrained on the new observation. Each teacher run costs under $100 (`03` §4.2) and runs about 3× faster under v5. That is about a week of calendar time before T1.
  - Phase 1's engine comparison stays labelled as v4 evidence.
- **Review:** at G0, scheduled by this entry.

**D50 (2026-10-08): Every fleet role runs on RTX 4090s through Phase 2, with the pinned torch 2.6 + CUDA 12.4.** Amends `03` §2, §4.2 and §5, `04` §5 and `07` §1, which named 5090s for the learner, eval, burst and deploy roles. Decided by the user.
- **Why:**
  - **5090s can't run the pinned environment.** torch 2.6 cu124 has no Blackwell kernels, and `05` §1 freezes the software environment for Phases 1–2. A torch upgrade changes numerics: the deterministic A/A and the featurizer hash checks would have to pass again, and comparisons across the upgrade would mix environments.
  - **No role needs a 5090** (`experiments/fleet-host-benchmarks.md`, `docs/design/cost-forecast-g1.md`):
    - actors are CPU-bound: the arena loop takes ~89% of the main thread, and the GPU sits at 11–17%;
    - the learner is featurization-bound at ~3.75K samples/s;
    - the eval box runs actor-sized nets, where a network up to b16c256 costs ≤ 6% on a 4090;
    - SL teachers are featurization-bound (`03` §4.2).
  - **4090s are cheaper and easier to rent.** On 2026-10-06 the ≥ 32-vCPU medians were $0.50 for a 4090 and $0.67 for a 5090, and Vast listed only 9 such 5090 offers.
- **Consequences:**
  - Learner, actor, eval and deploy boxes are 1× RTX 4090, with the vCPU and RAM specs unchanged.
  - Teachers (burst) run on a 4090 with ≥ 48 vCPU on the secure tier, or an H100 if the `03` §4.2 rule favours it. At the measured featurization rates it doesn't. D17a narrows to "4090 with ≥ 48 vCPU vs H100", pending a secure-tier featurization measurement.
  - The learner upgrade path (`03` §5) is 4090 → H100 SXM. Hopper runs on torch 2.6.
  - The $0.55 planning price stays.
- **Reopen if:** a role becomes GPU-bound on a 4090 (`03` §5's learner rule), or a torch upgrade is needed for another reason. The natural point is G1, when the Phase 1–2 freeze ends.

**D51 (2026-10-08): libriichi's arena aborts a triple ron (三家和了), as Tenhou and riichi.dev do.** Resolves D12a and changes the frozen rules config (`05` §1 item 4). Decided by the user.
- **Why:**
  - riichi.dev's server runs riichienv-core, whose default rule (the Tenhou preset) ends a triple ron in an abortive draw. That was verified by staging one (`01` §3, `tools/rules/riichienv_sanchaho.py`).
  - The training data, the GRP and Mahjax already follow Tenhou here. libriichi was the only engine that paid all three: an upstream deviation documented in `arena/board.rs`.
  - Matching the venue removes a known exception from the deploy parity tests (`07` §1).
- **What changes:** when three players declare ron on one discard (or kakan), the arena logs a zero-delta `ryukyoku` and ends the kyoku, the same shape as Tenhou's converted logs.
  - Nobody pays. The dealer repeats and the honba goes up by one; riichi deposits stay on the table.
  - A riichi declared on that discard is not accepted, as in riichienv.
  - The three ron reactions are recorded as not applied (D46).
- **Effect size:** 3–7 games per 10,000 between bots. Across a comparison that is at most ≈ 0.001 u, far below the lab's resolution. Phase 0 results (L0, σ_set, the D44 calibration, the SP ablation) were measured under the old rule and stand; none is re-run.
- **Known quirk, unchanged:** the loader labels the three declarers' decision at that discard as a pass (45), because no `hora` follows. The same holds for the 2 such games in the Tenhou corpus (6 labels in 1.19B). In RL those entries carry no behaviour probability, since their sidecar records are not applied, so they get no policy-gradient weight.
- **Timing:** before any Phase 1 run, so no Phase 1–2 comparison spans the change.
- **Reopen if:** riichi.dev's rule turns out to differ from riichienv's default.

**D52 (2026-10-08): Arm P1 runs both seeds on horizon H, and package K1's critic settings are fixed before the runs.** Fills in `05` §2.1 and §4–§5.1, which left both open. Decided by the user.
- **Horizon:** H (whole hanchan) on both P1 seeds.
  - P0 already covers K, so P1 tests the package "critic + H" against it, with a replicate. That is what the `05` §2.3 selection rule assumes: two runs of one recipe.
  - One seed on each horizon was rejected. Each horizon would get a single run, and run-to-run variation is as large as the K-vs-H difference we could hope to see (`experiments/sp-ablation.md` §6). It would also make "P1" two recipes under one label.
  - **Consequence:** if P1 passes, the gain belongs to the package. Attributing it to the critic or the horizon is the confirmation step's job (D35).
- **Critic (K1):**
  - b8c128, regressed onto λ-returns with MSE: the targets are bounded (u ∈ [−1.3, 0.9]), so the outliers Huber guards against can't occur;
  - γ = 1, λ = 0.95;
  - AdamW with weight decay 0, critic gradient clipping at 1.0;
  - a **constant learning rate**, 1e-4 or 3e-4, chosen by a pre-registered B0 check inside P1's tuning allowance (`experiments/k1-lr-check.md`).
- **Reopen if:** B0 never reaches a monotone calibration with a plateaued explained variance (`05` §4) under these settings.

## Cut list (v1 items removed, and what would bring each back)

| Item (v1) | Status | Reason | Re-entry trigger |
|---|---|---|---|
| Fixed 3,000-game gates; "≥50.8% win rate"; R1 "55% vs akochan" | **Cut** | Statistically unresolvable and miscalibrated | None |
| Risk-sensitive quantile inference (Stage 4.2) | **Cut** | Double-counts the utility; would need per-action return distributions Z(s,a) that the model doesn't have | None |
| IQN / distributional value head | Deferred | No evidence of a need; replaces the exact 4-class placement distribution | Structural calibration failure after RL-2 |
| League training as a fixed slot | Deferred | Compute-prohibitive on 1 GPU; at the D21 budget it would still take about half the Phase 2 fleet (D25) | Exploiter probe ≥ +0.05, or persistent drift |
| Belief-state loop (auxiliary outputs as policy input) | Deferred, and superseded by S-5 | Equim: auxiliary tasks gave no gain | Push/fold stagnation with S-5 not planned |
| Oracle *policy* distillation (v1 Stage 3a default) | Deferred indefinitely | Suphx +0.06 dan; Mortal removed it in v3 | New published evidence |
| V-trace + prioritized replay as "core" | Deferred | Economics decision; on-policy loop comes first | RL-8 passes and lag-induced ratio p99 > 2 |
| Hybrid transformer as the Stage 2 default | Deferred | Architectures converge in this lineage (Mortal 3.1 vs 1.0 Δ ≈ 0.002 avg rank); throughput cost | G1 names a representational bottleneck that search can't fix |
| 6-permutation suit augmentation | Deferred | ryuuiisou asymmetry | See D15 |
| Terminal-only reward with no shaping | **Cut** | GRP shaping already exists. It preserves the optimum only when returns run to the end of the hanchan; the horizon is now an explicit choice (D37). | None |
| 75% 4-seat self-play as default | Replaced by RL-6 patch | Departs from the released setup | RL-6 T-std at equal GPU-hours |
| Evaluation against "stock Mortal" | **Cut** | No public weights exist | Weights released |

## Open decisions (to be recorded when made)

- **D14a:** the winning player-scoring arm (A0–A5), from the `04a` §6.3 sweep.
- **D14b:** the held-out-by-time window, if the coverage audit (`04a` §5) moves it from 2025-07-01 → 2026-01-23.
- **D13a:** whether Mahjong Soul Throne data enters as a Phase 1 ablation (`04a` §9).
- **D16:** the student net size (`03` §4.1 + `04` §6).
- **D17:** AWR vs BC as the teacher objective (`04` §5). Both are trained at full scale; the T-std winner gets the second seed.
- **D17a:** H100 vs a 4090 with ≥ 48 vCPU for teacher runs (narrowed by D50), decided by the featurization rate (`03` §4.2) once it is measured on a secure-tier box.
- **D17b:** whether the CPU-only actor tier is adopted (`03` §3.5). The test is deferred (D47) until its reopen condition holds.
- **D18:** the Gate G0 outcome: the selected engine (D29) and actor size (D31), and the branch taken.
- **D30a/b:** the T2 and T3 release decisions.
- **D32a:** the search-pilot verdict (recovery ratio, latency) and the decision class to expand next.
- **D33b-1:** whether expert iteration earns funding.
- **D19:** the Gate G1 search classes for inference.
- **D20:** the decision on publishing weights, after re-reading Tenhou's terms (`07` §6).
- **D12a:** triple ron in libriichi's arena. **Resolved by D51** (2026-10-08): abort it, as Tenhou and riichi.dev do.
