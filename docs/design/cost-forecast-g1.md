# Capacity and cost forecast through Gate G1 (`03` §10)

**Written:** 2026-10-08, v0. **Generator:** `python tools/fleet/forecast.py` prints every line item per
scenario; all inputs are named constants at the top of that script, with their sources. Re-run it
when prices or throughput are re-measured (`01` §10: re-verify prices before each phase).

**Scope:** Phases 0–2, up to G1 (week 18). Not included: Phase 3 ($2.2K), the reserve ($1.6K), the
G0 retry (≤ $0.6K from the reserve), extra seeds. **This forecast changes no budget;** any change it
suggests needs an `08` entry first.

## Summary

| Scenario | Phase 1 | Phase 2 (T1–T3) | **Phase 0–2 total** (plan $9,000) | If only T1 is released | Hanchan generated / consumed (M) |
|---|---|---|---|---|---|
| **plan** ($0.55 for every box) | $2,843 | $5,367 | **$8,225** | $4,295 | 247 / 242 |
| **measured** (2026-10-06 medians) | $3,178 | $5,573 | **$8,767** | $4,675 | 247 / 242 |
| **levers** (measured + cost levers) | $2,344 | $4,132 | **$6,492** | $3,317 | 347 / 308 |
| **pessimistic** | $4,250 | $6,516 | **$10,781** | $6,004 | 247 / 242 |

Phase 0 actual spend so far is $5.33 (benchmark rentals), plus ~$10 assumed for fleet-script tests.

| Scenario | Prices ($/box-hour) | SL featurization | Other assumptions |
|---|---|---|---|
| plan | all $0.55 | 8K samples/s | 3 actor boxes in T1 |
| measured | actors 4090 on-demand $0.50; learner/eval 5090 $0.67; secure-tier 5090 $0.90 | 8K/s | as plan |
| levers | interruptible actors $0.40; learner/eval on 4090s $0.50; secure-tier $0.67 | 8K/s | 2–3 actor processes per box (×1.5); T1 with 2 actor boxes; the 3 students distilled from one featurized stream |
| pessimistic | 5090 everywhere $0.67; secure-tier $1.30 | 5K/s | as plan |

**Since D50 (2026-10-08)** every role runs on RTX 4090s. The 5090 prices in *measured* and *pessimistic* no longer apply to the learner and eval box; *levers* already priced them as 4090s. The scenarios haven't been re-run yet.

## Findings

1. **The plan fits at measured prices, just.** Phase 1 comes to $3.18K against $3.2K; Phase 2 to
   $5.57K against $5.2K (+7%). The overrun comes from the learner and eval boxes being specified as
   5090s (`03` §2), whose ≥ 32-vCPU median ($0.67) is 22% above the planning price.
2. **The levers save ~25%** ($6.5K against $8.8K) and raise consumed hanchan by ~27%. In order of size:
   interruptible actors, 4090s for the learner and eval box (the eval box needs no 5090: actor sizes
   up to b16c256 cost ≤ 6% on a 4090), multi-process actors (fewer actor boxes for the same games),
   and one featurization pass for all three student candidates.
3. **The pessimistic case trips the D24 alarm in Phase 1** (133% of plan; +20% overall). The drivers are
   the SL runs on secure-tier boxes (up to $1.30/h, and slow if featurization doesn't scale) and 5090s
   everywhere.
4. **The learner is the binding constraint once the fleet scales.** One 4090 learner consumes about
   85K hanchan/h of trainee samples (3.75K samples/s, featurization-bound with 2 workers; its GPU
   alone could do several times that). T1's 3 actor boxes produce 95K/h single-process (112%). With
   2–3 actor processes per box, **even 2 actor boxes produce 112%**, so extra actors buy nothing
   until the learner is faster. The cheapest fix to try is more loader workers on a learner box with
   ≥ 64 GB RAM (`03` §5 lists the other alternatives); it is untested beyond 2 workers.
5. **SL is a small share of the money but a calendar risk.** On v4, each SL pass (a teacher run,
   the teacher-output cache, a student run) is featurization-bound: 2.34B presentations take ~81 h
   at 8K samples/s, ~130 h at 5K/s. The chain teachers (AWR ∥ BC) → second seed → cache → students is
   ~14 days at 8K/s and ~22 days at 5K/s, against weeks 4–7 for the whole SL substrate. Secure-tier
   supply is thin (9 Vast 5090 offers with ≥ 32 vCPU on 2026-10-06, fewer with ≥ 48).
6. **One unpriced item:** the teacher-output cache. Caching every head for 2.34B states in FP16 is
   up to ~0.8 TB (~$200/month at Vast's storage median). Caching the policy only, or top-k, makes it
   small; that is a design choice for the distillation code.

## Inputs (measured in Phase 0 unless noted)

| Input | Value | Source |
|---|---|---|
| P0 actor | 31.7K hanchan/h per box, 1 process | `experiments/fleet-host-benchmarks.md` (4090 + EPYC 9684X) |
| Multi-process actors | ×1.5 (eval mode: 1.48× with 2 processes, 1.62× with 3); P0 not measured | same |
| P0 learner | 3.75K samples/s (v4, 2 workers); 159 trainee samples per hanchan | same; workstation 3.6K/s (`experiments/sp-ablation.md`) |
| SL presentations | 2.34B = 1 epoch × 2 augmentation × 1.17B usable decisions (2015-01 → 2025-06) | `~/mj-data/reports/splits.md`, `04` §5 |
| SL featurization on a ≥ 48-vCPU secure-tier box | 8K samples/s base, 5K pessimistic [assumed] | Workstation loader 6.0K/s on 16 threads; the pipeline is untested beyond 2 workers |
| Prices | see the scenario table | `01` §10 (2026-10-06) |
| Schedule and box counts | Phase 1 weeks 4–9; T1–T3 3 weeks each; 6 arm-seeds × 3 boxes × 6 days; actor-size curves 2 × 12 box-days; soak 48 h | `00`, `05` §2, D31, D39 |

## Measurements that would tighten this (one rental session, ~$5)

1. **Learner with 4–8 loader workers** on a 32-vCPU, ≥ 64 GB box (any tier; self-play data only):
   settles finding 4 and the T1 actor count.
2. **SL featurization on a ≥ 48-vCPU secure-tier box** (Tenhou data, D23): settles finding 5 and the
   teacher hardware choice (D17a).
3. **P0 actor with 2–3 processes per box:** confirms the ×1.5 used for the levers scenario.

## Decisions this feeds (the user's)

- Approving Phase 1 rentals (`00`: ≈ $3.2K).
- Any lever adopted as plan: 4090s for the learner/eval roles, interruptible actors, T1's actor
  count. These change `03` §2's role table or the tranche layout, so each needs an `08` entry.
