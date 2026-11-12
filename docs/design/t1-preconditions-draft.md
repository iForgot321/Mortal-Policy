# Before T1 (D64): the critic check and the anchor-release pilot

**Status: decided 2026-11-12.** D64 says T1 and D31's curves don't start on K2's fixed anchor, and
names two things that come first. The critic check is done. The pilot's design was reviewed and
chosen by the user on 2026-11-12, and its pre-registration is `experiments/anchor-release.md`,
which is the text that counts. This file keeps the evidence and what the review changed.

## The evidence behind D64

| What | Number | Source |
|---|---|---|
| `k2b`'s anchor coefficient | 0.0759 from the run's first hour (t = 18,400 of 3.66M), never moved | the learner's `anchor.jsonl` |
| Reverse KL to pilot A / argmax agreement over the run | 0.025 → 0.019 / 94.1% → 95.3% | same |
| The same over the soak, at the same coefficient | 0.019 → 0.018 / 95.2% → 95.4% | the soak's `anchor.jsonl` |
| The run's second half (`full` vs `p50`, ≈ $130) | +0.0025 [+0.0004, +0.0046] | `experiments/k2-retry.md` |
| 47.8 h more at the same anchor (the soak, $125) | +0.0016 [−0.0050, +0.0082] | `experiments/fleet-soak.md` |
| Critic explained variance of the H return, all decisions / last kyoku | 1.2–1.8% / 6–10% | the learner's TensorBoard |
| Critic explained variance of the current kyoku's Δ | 13.7% | `experiments/critic-ceiling.md` |

The policy moved to where the PPO push and the anchor's pull balance, and stayed. The coefficient
was set by the start-up transient and never adapted after it: the KL stayed inside the
controller's band (0.013–0.03), so the anchor acted as a fixed penalty.

## 1. The critic check: done

`experiments/critic-ceiling.md` (run 2026-11-11): K1 explains 13.7% of the kyoku's GRP-Δu, above
every supervised arm on 100K games (8.4–9.0%). By its pre-registered reading the critic isn't the
lever, so the pilot runs with K1 as it is, and sample efficiency (RL-6, RL-8) moves up for T1. The
audit's "1.2–1.8%" was of the H return, which is mostly later-kyoku variance that nothing predicts.

## 2. The anchor-release pilot: pre-registered

`experiments/anchor-release.md`. Two candidates continue `k2b` full for 48 h on a learner and 2
actors: **A** keeps pilot A as the anchor with a rising target, **B** re-anchors to `k2b` full at
0.02 nats. Each is measured against the soak's last checkpoint at a table of pilot A and a table of
pilot B.

**What the review (2026-11-12) changed from the 2026-11-10 draft, and why:**

| The draft | Now | Why |
|---|---|---|
| A's target rises 0.02 → 0.04 | 0.02 → ≈ 0.06, most of it early | The controller acts only below target / 1.5, so 0.04 would have delivered ≈ 0.027, late. B can reach up to ≈ 0.08 from pilot A (KL isn't additive) |
| Pass against `k2b` full | Against `soak-a` full | D64 says the fixed anchor's continuation; the soak's own CI reaches +0.008 |
| "The soak's LR" | LR, clip and entropy weight all held at the soak's values | A new run clock restarts all three schedules (3×, 2× and 7× the soak's) |
| The larger gain at pilot B's table wins | Only if the candidates' paired difference excludes 0; otherwise B | One run each, and the lab can't separate differences under ≈ 0.004 |
| Pilot B's table: estimate above 0 | CI above 0, as at pilot A's table | An estimate above 0 passes half the time by chance |
| SE ≈ 0.0017; measures on the workstation | SE ≈ 0.0021; an eval box | 0.0017 is a direct 1v3's; 240K sets are ≈ 43 h and ≈ 40 GB here |
| Code: a stage-init tool | Also: A keeps its coefficient, the fleet ships B's anchor file, a smoke run of each config | Found in the review |

**The user's choices:** base `k2b` full on v4; a learner and 2 actors for 48 h per candidate
(≈ $175 for both); both candidates; the eval box for the measures.

**Not in this pilot:** who sits at the table (RL-5), and a two-penalty anchor (a strong one to the
current champion, a weak one to the supervised policy), which is worth weighing for T1.

## After these

D31's curves (two RL seeds per size, D65) and T1 run on the mechanism this pilot selects, with K1
as it is, from the students the distillation produces (`docs/design/distillation-draft.md`).
