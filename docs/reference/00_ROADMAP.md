# Riichi Mahjong AI v2: Master Roadmap

**Status:** active plan, revised 2026-10-01.
- It supersedes all six v1 documents in `mahjong/`. The reasons are in `mahjong/REVIEW.md` and `mahjong/PRO_APPROACH.md`.
- The 2026-10-01 revision adopts `mahjong/REVIEW_V2_STRATEGY.md` (research strategy) and `mahjong/REVIEW_V2.md` (correctness). The changes are logged as D28–D42 in `08_decision_log.md`.

**Budget basis:** **$10–15K total compute, cloud only**, with no hardware purchase (D21). After the D42 reallocation, the plan spends about $12.7K and holds about $1.6K in reserve.

**Purpose:** this is the entry point and the source of truth for **sequencing, gates, success criteria and budget**. Every other file supplies implementation detail. Where they conflict, this file wins on *what happens, when, and at what cost*, and the stage file wins on *how*. It is written so future sessions, human or Claude, can resume without re-deriving context.

---

## Document map

| File | Covers | Read when |
|---|---|---|
| `00_ROADMAP.md` | Goals, ladder, phases, gates, budget, risks | Always first |
| `01_fact_base.md` | Verified facts: substrate code, ecosystem, frontier agents, algorithms, cloud pricing, sources | Before relying on any fact |
| `02_evaluation_lab.md` | Metric, utility, duplicate format, calibrated GSPRT, population panel, style panel, kill rules, registry | Phase 0; before any experiment |
| `03_throughput.md` | Fleet architecture, benchmarks, fleet soak, inference optimization, simulator choice, cost model | Phase 0; before scaling RL |
| `04_data_and_sl.md` | Dataset, player scoring, splits, SL teachers, student candidates | Phases 0–1 |
| `04a_data_pipeline.md` | Step-by-step data runbook | Phase 0 (workstation) |
| `05_rl_core.md` | Learning-engine comparison, RL interfaces, discovery and confirmation, tranches, failure modes | Phases 1–2 |
| `06_search.md` | Search pilot, belief particles, one-ply search, separately gated expert iteration and search-as-features | Phase 1 (pilot), Phase 3 |
| `07_campaign_and_products.md` | riichi.dev deployment, blunder mining, human-like dial, styles, publishing, ethics | Phase 1 (smoke deploy) and Phase 4 |
| `08_decision_log.md` | Every major decision, its rationale, and what would reopen it; the cut list | When tempted to change course |

**Change control:**
- Every change to goals, gates, utility, metric, phase order or budget gets a dated entry in `08_decision_log.md` before the docs are edited.
- Experiment results go in the registry and ablation table (`02` §9), labelled *exploratory* or *confirmed*.

---

## Goal

Build the **strongest open, verifiable Riichi Mahjong AI** that a solo developer can build on a **$10–15K cloud budget**.

- **Venue:** the public, falsifiable arena is **riichi.dev** (RiichiLab). It is the successor to mjai.app, which shut down on 2026-04-30.
- **Human-like play** is a secondary product goal: a tunable, measurable dial for style, coaching and review. It is never used to evade bot detection (`07` §6).
- **Byproducts:** a review tool and published weights, code and ablations.

## Operating principles

1. **Measure first.** No experiment runs before the evaluation lab can deliver trustworthy, **calibrated** verdicts (`02`; D36).
2. **One objective everywhere.** The utility `u = [+0.90, +0.50, −0.10, −1.30]` is used for the reward, value targets and the primary metric. The return horizon is declared explicitly (D37).
3. **Throughput per dollar is the currency.**
   - Strength scales with games played and searched.
   - Consumer GPUs (4090/5090) deliver about 2.5× more CNN training throughput per dollar than H100/H200 (`01` §10).
   - So **scale out on consumer GPUs**.
4. **Start from the released implementation, but treat the learning engine as an open question.**
   - Mortal-Policy's released AWR→PPO is a starting implementation, not a validated recipe. Its author's strong results used withheld infrastructure and a value-based initialization (D28).
   - Policy-based, value-based and hybrid engines are compared early, at matched total compute (D29).
5. **Discover in packages, confirm in isolation** (D35).
   - Interdependent components are explored together under a box-day cap.
   - Gains are claimed only after confirmation against matched controls.
6. **Scale in tranches released by evidence** (D30). Cumulative games are a forecast; continued measured improvement is the criterion.
7. **Search is planned, but each search investment earns its own funding** (D32, D33). A narrow pilot comes early; expert iteration is unfunded until it wins at equal total cost.
8. **Keep a deployable incumbent at all times.** The π_SL student is deployed in Phase 1 and stays the incumbent until a confirmed successor replaces it.
9. **Evidence triggers additions.** Deferred techniques return only when their trigger fires (`08`).
10. **Stay inside ethical and legal lines.**
    - Bots play only where bots are allowed.
    - Keep AGPL compliance.
    - Never redistribute Tenhou logs, and never store them on community-tier hosts (D23).

---

## Strength ladder

Utility deltas (Δu) are per game against the named opponent in 1v3 duplicate (`02` §2–4). "Confirm" means a fixed-N run on fresh family-C nonces with the 95% lower bound > 0 (`02` §4, D36).

| Rung | Claim | Measurement | Expected |
|---|---|---|---|
| **L0** | The lab is trustworthy | Deterministic A/A gives Δu = 0 exactly; A/A′ gives σ_set; the sequential procedure is **calibrated by simulation** (error rates ≤ 5%); the synthetic hobbled patch fails | End of Phase 0 (week 3) |
| **L1** | The baseline is deployed | SL student non-inferior to the **best single teacher** (GSPRT, δ = 0.01); beats shanten-greedy; riichi.dev smoke-deploy rating stable after ≥ 200 games | Week 7 |
| **L2** | RL improves on its initialization | The selected engine (D29) passes T-std against its own initialization on **both** seeds, then confirms. Scope: quick-model initialization. The final-student check follows in T1 (D41). | Gate G0 (week 9) |
| **L3** | The scaled RL core | Tranches T1–T3 released by evidence (D30); confirmed packages; population-panel non-regression (D34); cumulative Δu vs π_SL confirmed; riichi.dev rating rising | End of Phase 2 (week 18) |
| **L4** | Inference search adds strength | Search-enabled agent beats the same network without search (T-std, then confirm) within the time budget, with fewer than 0.5% of actions timing out (D33a) | End of Phase 3 (week 24) |
| **L5** | Top of the public arena | Sustained riichi.dev top-10 over ≥ 500 ranked games, then top-3 | Phase 4 |
| **L6** | Frontier class (LuckyJ, NAGA, BigCoach ≈ 10-dan) | Aspirational; no direct measurement path | Not a criterion |

**riichi.dev 4-player reference points (2026-09-29):**
- #1 "zero-test2", 2091.
- #5 ≈ 2023.
- "Mortal-v4b" at #44, 1864.

Displayed ratings drift with SD ≈ 36–44 points (`07` §2).

---

## Phase map and budget

**Planning unit:** a **"box"** is one RTX 4090 (D50) with ≥ 32 vCPU, on-demand, at a planning price of **$0.55/h**. That is the Vast.ai median for 32+ vCPU hosts on 2026-10-01 (`01` §10). One box-day ≈ **$13.2**; one box-month (730 h) ≈ **$400**.

| Phase | Weeks | Fleet | Cost |
|---|---|---|---|
| **0** Foundations | 1–3 | 1 dev/eval box; benchmark rentals; **2 pilot quick students** for A/A′ (D41); Mahjax spike | ≈ $0.6K |
| **1** SL substrate + **early research** | 4–9 | SL substrate (eval box, teachers, student candidates) ≈ $1.0K. Early research **capped at $2.2K**: engine arms, actor-size curves, fleet soak, search pilot. | ≈ $3.2K |
| *Gate G0: Select* | 9 | — | — |
| **2** Select and scale (tranches) | 10–18 | **T1** (wk 10–12): 5 boxes (eval, learner, 3 actors), ≈ $1.4K. **T2** (wk 13–15) and **T3** (wk 16–18): 7 boxes (eval, learner, **candidate learner**, 4 actors), ≈ $1.9K each. **T2 and T3 are released only by D30.** | ≈ $5.2K |
| *Gate G1* | 18 | — | — |
| **3** Search expansion | 19–24 | 4 boxes (eval, learner, 2 actors/search). Expert iteration unfunded unless it wins its equal-cost test (D33b). | ≈ $2.2K |
| *Gate G2* | 25 | — | — |
| **4** Campaign + products (3 months budgeted) | 25+ | 1 deploy box, plus bursts for blunder mining | ≈ $1.5K |
| **Reserve** | — | Overruns, G0 retry, extra seeds, learner upgrade | ≈ $1.6K |
| **Total** | | | **≈ $12.7K planned + $1.6K reserve ≈ $14.3K** |

**Unreleased tranche money returns to the reserve.** If T2 or T3 is not released, that spend moves to diagnosis, inference search, or the reserve, by decision-log entry.

**Interruptible actors.** Actors hold no state, so interruptible bids should save roughly 25–35% of actor spend. Count any savings as extra reserve; don't count on them.

**Timeline:** campaign start at about **week 25 (month ~6)**; a mature system at **month 8–9**.

**Scale forecast** (not a criterion; D30): at about 30K hanchan/h per actor, measured in Phase 0:

| Stage | Generated hanchan | Of which champion-lineage |
|---|---|---|
| Early research | ≈ 50–60M | — |
| Phase 2 | ≈ 165M | ≈ 135M, plus adopted candidates |
| Phase 3 | ≈ 60M | — |
| **Project total** | **≈ 270–290M** | |

These counts are tracked separately as generated games, consumed samples and champion-lineage games (D39).

**Budget guardrails** (D24):
- Monthly spend report, with a burn-rate alarm at 120% of the phase plan.
- No fleet expansion beyond the plan without a decision-log entry.
- Release idle boxes the same day.

### Phase 0: Foundations (weeks 1–3)

**Deliverables:**
- **Evaluation lab** per `02`:
  - sequential test **calibrated by simulation** (D36);
  - A/A and A/A′ (using the 2 pilot quick students);
  - three seed families with nonce retirement;
  - the **population panel** definition (D34);
  - style panel and registry.
- **Data** per `04a`:
  - snapshot and hash on encrypted secure-tier storage;
  - **splits frozen before player scoring** (D40);
  - scoring fitted on `train` only.
- **Interfaces** per `05` §4 (D37): ordered trajectory extraction, fixed reward units, raw-logit policy head, declared horizons.
- **Fleet foundations** per `03` §2–3:
  - VPN; provisioning; monitoring;
  - per-box throughput;
  - featurization rate;
  - CPU-only inference benchmark (deferred 2026-10-08, D47);
  - Mahjax spike.

**Exit:** L0 met; σ_set, hanchan/h per box type and featurization samples/s logged with commit and config.

### Phase 1: SL substrate and early research (weeks 4–9)

**SL substrate** (weeks 4–7; `04` §5–6):
- Teachers: AWR and BC at b24c512, plus a second seed of the winner.
- Student candidates: the two smallest nets that pass T-NI against the best single teacher.
- riichi.dev smoke deploy of the best student. This is the **incumbent** (L1).

**Early research** (weeks 4–9, capped at $2.2K; `05` §2, `06` §0):
1. **Learning-engine comparison** (D29), at matched box-days (3 boxes × 6 days per arm-seed), from the pilot quick students and a quick CQL model:
   - P0 released + fixes ×1;
   - P1 + package K1 ×2;
   - V value-based ×2;
   - H hybrid ×1, started once V has a champion.
2. **Actor-size learning curves** (D31): the two student candidates on the leading arm at 6 and 12 box-days.
3. **Fleet soak** (D39): ≥ 48 h end-to-end at T1 scale, with the leading arm.
4. **Search pilot** (D32), riichi vs dama:
   - policy only, oracle search, belief search;
   - the oracle stage reports by G0;
   - the belief stage reports **before T2's release**.

### Gate G0: Select (week 9)

| Outcome | Decision |
|---|---|
| An arm passes T-std against its initialization on both seeds, the fleet soak passes, and L2 confirms | **Select** that arm (head-to-head T-std at matched compute decides between passing arms; ties go to the cheaper and more search-compatible arm). Select the actor size per D31. Start T1. |
| Arms pass, but the soak fails | Optimize (`03` §6) before T1; dates move (D39) |
| No arm passes, with healthy curves | Diagnose (`05` §8). One bounded retry of the best arm with package K2, 2 seeds, ≤ $0.6K from the reserve. |
| The retry fails | Re-scope: the SL incumbent plus inference search becomes the main path. Phase 2 money moves to Phase 3 (D33a), conditional on the pilot. |

**Observation review (D49):** keep v4 for Phase 2, or switch to v5 or a cheaper variant. The evidence is the doubling estimates (`05` §2.2) and the actor-size curves (D31), weighed against v5's break-even of about 0.002–0.004 u per doubling.

### Phase 2: Select and scale (weeks 10–18)

**T1 (weeks 10–12)**
- Scale the selected engine on the selected actor.
- **Final-student check:** the engine applied to the final student must pass T-std against that student (D41).
- Discovery packages K2 and K3 run within their box-day caps (`05` §5.1).

**Tranche gate (weeks 12–13)**
- **Release T2** if both hold:
  - end-of-T1 champion beats start-of-T1 champion by T-std, and no severe regression on the population panel (D30, D34);
  - **and** the search pilot's belief stage has reported (D32).

**T2 (weeks 13–15) and T3 (weeks 16–18)**
- Champion lane plus candidate lane, with the dedicated candidate learner (D39).
- Promising packages move to **confirmation**: matched patched/unpatched pairs and fresh seeds (D35, D41).
- T3 is released by the same rule at weeks 15–16.

**Exit:** L3.

### Gate G1 (week 18)

From the pilot plus a scaled S-3 run on the champion (`06` §7), decide:
- the search classes for inference (D33a);
- whether expert iteration gets its **equal-total-cost test** (D33b);
- whether to attempt search-as-features (D33c).

### Phase 3: Search expansion (weeks 19–24)

- S-1 (particle hardening) → S-3 at the selected classes → regularized vs plain search update, and argmax vs sampled moves (D33) → optional S-4 equal-cost test → optional S-5 (`06`).

**Exit:** L4.

### Gate G2 (week 25)

- Search latency p99 below the budget, measured request-to-response on the deploy box.
- Zero chombo or illegal actions across 100K lab games.
- Deploy supervisor, logging and blunder mining live (`07`).

### Phase 4: Campaign and products (week 25+)

- Continuous riichi.dev ranked play.
- Blunder-mining loop.
- Style variants and the human-like dial.
- Publish weights, code and the ablation table.

**Exit:** L5.

---

## Risks

1. **No learning engine improves on SL at our scale.**
   - *Mitigation:* the three-arm comparison with replicates (D29); diagnosis before any family-level conclusion; the G0 retry; the search-led fallback path.
2. **Wrong engine or actor size chosen from early curves.**
   - *Mitigation:* the crossover rule (D31); reopen D29 if a tranche stalls; tranche gates limit sunk cost.
3. **Search interfaces fail late.**
   - *Mitigation:* the early pilot with the oracle/belief decomposition (D32); the D38 particle specification; S-1–S-3 interfaces defined in Phase 1.
4. **Per-box throughput below plan.** Vast hosts are often older EPYC chips.
   - *Mitigation:* host allowlist; the fleet soak (D39); distillation; CUDA graphs; Mahjax.
5. **Cloud availability and interruptions.**
   - *Mitigation:* stateless actors that reconnect; on-demand learner and eval; checkpoints to object storage every ~200 iterations; scripted provisioning so any box is replaceable within 30 min.
6. **Cost overrun.**
   - *Mitigation:* D24 guardrails, tranche release (D30), the $1.6K reserve, interruptible actors.
7. **Data handling on third-party hosts.**
   - *Mitigation:* secure-tier providers only (D23); actors never receive Tenhou data.
8. **Miscalibrated verdicts or a broken harness.**
   - *Mitigation:* simulation-calibrated sequential test (D36); A/A and A/A′; kill rules (`02` §8); fixed-N confirmation with nonce retirement.
9. **Evaluation-population mismatch.**
   - *Mitigation:* the population panel with reserved configurations (D34); early smoke deploy; blunder mining.
10. **Network exposure.** Mortal-Policy's TCP server has no authentication.
    - *Mitigation:* bind it to loopback and reach it only through restricted SSH tunnels (D48).
11. **Ecosystem shift.**
    - *Mitigation:* quarterly re-survey of riichi.dev rules, Mahjax maturity and prices.
12. **Solo burnout.**
    - *Mitigation:* package-sized work with a verdict every few days; parallel lanes keep the queue moving.

---

## Milestones

| Month | Milestone | Rung |
|---|---|---|
| 1 | Lab calibrated, splits frozen, interfaces defined, throughput known | L0 |
| 2 | Student deployed (incumbent); engine arms and search-pilot oracle stage running | L1 |
| 2.25 | G0: engine and actor selected; RL improves on initialization | L2 |
| 2.5–4.5 | Tranches T1–T3; packages confirmed; search-pilot belief verdict | L3 |
| 4.5–6 | Inference search; expert iteration only if it wins at equal cost | L4 |
| 6 | Campaign start | L5 attempt |
| 6–9 | Iterate via blunder mining; publish weights, code, ablations | L5 |
