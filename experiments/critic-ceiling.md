# D64's critic check: pre-registered 2026-11-11, before it runs

The user ran it on 2026-11-11 ("Do the critic check"). It is the first of D64's two conditions before
T1 (`docs/design/t1-preconditions-draft.md` §1). The workstation, $0; no lab nonces, no registry rows
(not a strength claim, like `experiments/k1-lr-check.md`).

## Why

`k2b`'s K1 critic explained 1.2–1.8% of the H return's variance (6–10% in the game's last kyoku), and
its loss never fell. Under H most of the return's variance is later kyoku that nothing predicts
(`k1_h_ceiling.py`: the current kyoku is 14% of it). So the question is about the part a critic can
reach: **how much of the current kyoku's GRP-Δu can a critic explain, with and without the opponents'
hands, and how far below that is K1?**

## Data (exists)

The K1 check's games: 100,000 hanchan of pilot A's B0 play for training (`workdir/k1-lr-check/pool`),
4,000 held out (`.../fresh`, nonces the check never trained on). The trainee's decisions only, v4,
with the exact hidden information (`GameplayLoader(oracle=True, trust_seed=True)`). Targets in u
(fixed units): the decision's own kyoku's GRP-Δu (kyoku-Δ), and the H return for K1's own figure.

The held-out games are split once, by file name: the first 1,000 select (early stopping, best
weights), the other 3,000 report. No choice sees the 3,000.

## Arms

| Arm | Input | Net | Training |
|---|---|---|---|
| `k1_k2b` | obs + opponents' hands | b8c128 (K1) | none: `k2b` full's critic as it is (trained online on H λ-returns) |
| `public` | obs | b8c128 | supervised on kyoku-Δ |
| `privileged` | obs + opponents' hands | b8c128 (K1's architecture) | supervised on kyoku-Δ |
| `privileged_b16c256` | obs + opponents' hands | b16c256 | supervised on kyoku-Δ |

The supervised arms train in one stream (`experiments/critic_ceiling.py`): AdamW, LR 1e-4, weight
decay 0 (D52), batch 512, MSE, AMP, gradient clip 1.0, seed 1. Every 7,500 steps (≈ a quarter pass)
each is scored on the 1,000 selection games; an arm stops after two evaluations in a row without
beating its best by 0.002, and keeps its best weights. At most 3 passes.

## Metrics, on the 3,000 report games

For each arm: explained variance (EV) of kyoku-Δ over all decisions, in the game's last kyoku, in
earlier kyoku, and by third of the kyoku's decisions; 10-bin calibration. For `k1_k2b` also EV of the
H return. And the mean of the later kyoku's return (H − kyoku-Δ) by kyoku index.

## Reading, fixed now

1. **Is the critic a lever?** If the better privileged arm's EV exceeds `k1_k2b`'s by **≥ 0.05**
   (all decisions): yes. T1's critic is then trained supervised on kyoku-Δ before any policy step
   (a supervised B0), with that arm's architecture. Below 0.05: K1 is near what its inputs allow at
   this data size, the critic isn't the lever, and sample efficiency (`05` §7 RL-6, RL-8) moves up
   the list for T1.
2. **Do the hidden hands matter?** `privileged` − `public` (both b8c128): below 0.02, they add
   little at this size; reported either way.
3. **Size:** `privileged_b16c256` − `privileged` ≥ 0.03: T1's critic takes the larger net.
4. **The later kyoku:** reported. If the mean later return depends on the kyoku index by more than
   0.05 u, a kyoku-Δ critic needs a per-kyoku offset to serve H's GAE.

D64's pilot (a rental) uses this reading; its own pre-registration follows.

## Smoke run (2026-11-11, 400 pool and 80 held-out games, not a result)

The pipeline runs end to end: 3 arms trained 121 steps, every metric computed, the report written.
It also showed the later kyoku's mean return isn't 0 for these games (−0.26 u from E1 down to ≈ −0.1):
pilot A sampling at T = 1 loses ground to argmax opponents over the game. That is why reading 4 is
there.

## Result (2026-11-11 10:00 → 12:37 UTC, the workstation)

`workdir/critic-ceiling/` (`report.json`, `log.jsonl`, the arms' best weights). Every supervised arm
stopped by the rule at step 37,500 (≈ 1.2 passes, 2.6 h): their selection EV peaked at step 22,500
(≈ 0.7 pass) and then fell, so they were overfitting the 100K games.

**On the 3,000 report games** (476,950 trainee decisions; Var(kyoku-Δ) 0.063 u², Var(H) 0.458 u²),
EV of kyoku-Δ:

| Arm | All | Last kyoku | Earlier kyoku | 1st / 2nd / 3rd third of the kyoku | Calibration |
|---|---|---|---|---|---|
| `public` (b8c128) | 0.0836 | 0.0607 | 0.0897 | 0.026 / 0.061 / 0.175 | monotone |
| `privileged` (b8c128) | 0.0872 | 0.0652 | 0.0933 | 0.027 / 0.063 / 0.182 | monotone |
| `privileged_b16c256` | 0.0902 | 0.0651 | 0.0968 | 0.028 / 0.065 / 0.189 | monotone |
| **`k1_k2b`** (as `k2b` left it) | **0.1371** | **0.1000** | 0.1469 | 0.040 / 0.094 / 0.294 | monotone |

`k1_k2b` on the H return: EV 0.0182 (0.1000 in the last kyoku, where H = kyoku-Δ), as the trainer
logged (1.2–1.8%).

**Reading, as fixed:**
1. **Is the critic a lever?** The better privileged arm (0.0902) is 0.047 *below* `k1_k2b`, not 0.05
   above it. **No:** by the rule, the critic isn't the lever, and sample efficiency (RL-6, RL-8) moves
   up the list for T1.
2. **The hidden hands:** `privileged` − `public` = +0.0036, below 0.02: they add little at this size.
3. **Size:** `privileged_b16c256` − `privileged` = +0.0030, below 0.03: T1's critic keeps b8c128.
4. **The later kyoku:** the mean later return runs from −0.081 u (E1) to −0.005 u (kyoku 11) over
   the kyoku with ≥ 5,000 decisions, a range of 0.076 > 0.05: **a kyoku-Δ critic would need a
   per-kyoku offset to serve H's GAE.** (Pilot A sampling at T = 1 against argmax pilots loses ground
   over the game; a policy's own critic absorbs that.)

**What this does and doesn't show (beyond the rule):**
- **K1 is not close to uninformative.** The audit (2026-11-09) read its 1.2–1.8% EV as a weak
  critic and compared it with the SL kyoku-Δ head's ≈ 10%. That compared two targets: the H return
  is 86% later-kyoku variance that nothing predicts. On the same target as the head, K1 explains
  13.7%, more than any arm here and more than the head on Tenhou games. The audit's finding 3
  overstated the problem.
- **The ceiling isn't bounded.** The supervised arms saw 100K games (≈ 19M decisions) and overfit;
  K1 saw ≈ 1.9B decisions online. So K1 beats them on data, not because its inputs are exhausted.
  Whether a larger or better-fed critic would help T1 is open; this check only rules out a
  supervised warm start on this pool as a quick win.
