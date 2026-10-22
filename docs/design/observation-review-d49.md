# Observation review (D49): v4 or v5, the evidence so far

Decision memo for the user, 2026-10-19. D49 schedules this review at G0. G0 isn't settled yet: the
retry `k2b` reports ≈ 2026-10-24. This memo collects the evidence so the call doesn't need the
experiment notes re-read. Nothing here changes the plan; a switch needs an `08` entry first.

## The question

- **v4** (Mortal's observation, 1012×34) includes the single-player (SP) EV tables. **v5** is v4
  with those 123 rows zeroed (`experiments/sp-ablation.md` §2).
- **v5's speed is large and certain** (§3): encoding 340× cheaper per decision, the arena 2.4×
  (2.8× per decision), the RL learner 2.3×, SL training 2.9× (8.6 h vs 25 h). In RL that is
  ≈ 1.2–1.5 extra doublings of compute at equal cost.
- **v5's strength cost is unresolved** (§5.1): SL pairs −0.0060 and +0.0012 u, pooled −0.0024
  [−0.0056, +0.0009]. The pairs disagree (z = 2.16), so the pre-registered verdict was
  inconclusive, and v5 wasn't adopted.
- **Break-even (§7):** v5 is the better RL observation if one doubling of RL compute buys more
  than ≈ 0.002–0.004 u.
- **A cheaper variant**, shanten threshold 1, would save ≈ 98% of v5's encoding cost and keep the
  tenpai and 1-shanten EV tables (§3.1, §6). It isn't built or measured.

## The doubling evidence

| Source | Δu per doubling | Status |
|---|---|---|
| P0, `p0-a` full vs p50, family B, 100K sets (`20261017-043916-measure`) | **+0.0071 [+0.0047, +0.0093]** | Pre-registered. The CI's lower bound is above 0.004, so by its rule "the G0 review weighs a switch" |
| P1, `p1-a` / `p1-b`: p50 (halfway check, family A) vs full (G0 T-std, family B) | ≈ +0.005 / +0.009 | Not a measurement: different nonce families, and the T-std means stopped early (optional-stopping bias). It points the same way as P0's |
| D31 actor-size curves (6 vs 12 box-days) | — | Not run: no arm reached G0 |
| `k2b` full vs p50 | — | Possible after 2026-10-24 (below) |

**Why P0's number shouldn't decide alone.** Both of its checkpoints are below the initialization:
p50 −0.033 (family A), full −0.027 (G0). By p25 the run had moved a quarter of its argmax
decisions away from the SL policy (`experiments/rl-diagnosis.md`). So the +0.007 measures, at
least in part, recovery from that drift, not what a doubling buys a run that improves. The
caveats stated before the result still hold: one run, a quick model, and "engine, not mahjong"
(`experiments/p0.md`). It meets the pre-registered bar, but what it measured isn't the quantity
the bar assumes.

## What G0's outcome changes

- **`k2b` passes:** RL scales in Phase 2, and the doubling value is the right question. `k2b` is
  the first run held near its initialization (KL ≈ 0.025 nats, ≈ 94% argmax agreement so far), so
  its full vs p50 would be the first clean estimate. Those KL and agreement figures are the
  trainer's batch readings; the p25 early check measures them properly. The p50 snapshot is taken
  automatically.
- **`k2b` fails:** the main path becomes the SL incumbent plus inference search, and Phase 2's RL
  money moves to Phase 3 (D33a). The doubling value then hardly matters. What matters is:
  - the SL strength cost, which is unresolved (above);
  - the search's rollout speed. Rollouts encode at every decision of four seats. The search
    pilot's latency on the 3060 is p99 2.21 s against 06 §0's 2.5 s limit. v5's arena speed-up
    (2.4–2.8× per decision) suggests rollouts would gain about as much, but that hasn't been
    measured.
- **Inconclusive:** a second seed needs your yes. Its p50 and full could give the doubling
  estimate too.

## v5's search speed, measured (2026-10-22, the user's go-ahead)

The fail branch's measurement, taken early so it's ready either way. Pilot A against its v5 timing
copy (`workdir/sp-ablation/pilot-a-as-v5.pth`: pilot A's weights with v5 input, so its moves mean
nothing; speed only), on the search pilot's decisions, the 3060, true FP32.
`experiments/search_v5_speed.py`; outputs in `workdir/search/v5-speed/`.

| | v4 | v5 | v5 / v4 |
|---|---|---|---|
| Batched rollouts (the oracle stage: 500 decisions, 65,000 rollouts, 2 passes agreeing within 0.3%) | 339 rollouts/s, 87.4 µs per logged event | 452 rollouts/s, 59.7 µs | **1.33×** per rollout, **1.46×** per event |
| One decision at a time (`python -m search latency`, 100 decisions, 64 particles), p50 | 0.85 s | 0.53 s | **1.6×** |
| Same, p99 (06 §0's limit: 2.5 s) | 2.00 s | 0.90 s | **2.2×** |

- **Below the arena's 2.4–2.8×.** Rollouts start at a riichi-legal decision, late in the kyoku,
  while v4's encoding cost sits in 3-shanten hands early in it (`experiments/sp-ablation.md` §3.1).
  In big batches the network's share of the time is larger too. One decision at a time, the batches
  are small and the encoder's share is larger, so the gain is bigger, most of all in the slow tail.
- The v5 copy's rollouts are 9% longer (36.9 vs 33.7 events), so the latency ratios, which aren't
  normalized per event, if anything understate v5's gain.
- **What it buys:** v4 already meets the latency limit on the 3060 (p99 2.0 s). v5 would leave room
  for ≈ 2.5× the particles or rollouts per decision, or search on more decision classes, within it.
  But the pilot found the belief, not the rollout count, limits riichi vs dama (28% of the
  disagreement with the oracle is sampling noise), so for now that's headroom, not quality.

## Timing

- **A switch costs least before the b24c512 teachers are trained.** D49 puts all Phase 1 SL work
  on v4. The teachers wait on the secure-tier data path (D23), which isn't built yet, so there's
  time.
- A switch made after the teachers are trained retrains them (< $100 each, ≈ 3× faster on v5,
  about a week of calendar time, per D49).

## Options

1. **Keep v4 for now, and decide when `k2b` reports (recommended).**
   - Pre-register now, before the result exists, a `k2b` full vs p50 measure: family B, 100K sets
     (≈ 15 h on the workstation by `experiments/p0.md`'s estimate, after G0's T-std), with the
     same reading as P0.
   - If `k2b` fails, measure rollout speed instead: the search pilot's rollouts with pilot A
     against its v5 timing copy (`workdir/sp-ablation/pilot-a-as-v5.pth`), on the workstation, a
     few minutes. Then decide on SL cost vs search speed.
   - Either way, decide before the teachers start.
2. **Switch to v5 now,** as the deliberate trade §6 describes: a possible cost of up to
   ≈ 0.5 pt per hanchan for ≈ 3× cheaper SL and actors. It needs an `08` entry saying the strength
   evidence was inconclusive. `k2b` and the pilot quick students stay v4 evidence.
3. **Build the shanten-threshold-1 variant first.** It probably keeps most of the SP information
   that matters for riichi vs dama (pair 1's dama shift, not replicated) at near-v5 speed. But a
   strength claim takes SL pairs, and at ±0.003 u of training luck per network that is ≈ 4 pairs
   to resolve ±0.005 (§6): ≈ 37 h each on the workstation, or cloud time.

## Decided (2026-10-21)

The user approved option 1's measure: `k2b` full vs p50, family B, 100K sets, after G0's T-std,
pre-registered in `experiments/k2-retry.md` → "Doubling estimate". The p50 vs p25 measure is
skipped. The v4/v5 decision itself waits for that result and G0's verdict.

## What I needed from you (answered above)

- Whether to pre-register the `k2b` full vs p50 measure now (option 1). It's an addition to
  `experiments/k2-retry.md`. It doesn't change D59's verdict, which stays the T-std against pilot A.
- Or a different option.
