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

## Result

(none yet)
