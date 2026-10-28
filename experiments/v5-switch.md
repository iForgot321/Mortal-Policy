# The switch to v5 (D61): its checks

D61 (the user, 2026-10-26) makes v5 the observation from G0 on. Before a v5 result counts it asks for
a v5 A/A with a frozen v5 hash, and a recipe check: does P1 + K2 work on v5 as it did on v4?

## Workstation checks: pre-registered 2026-10-26, before they run

`experiments/v5_checks.sh`, once the post-G0 queue has finished (≈ 15:00 UTC, before the soak's
tracking measures). Each passes or fails as stated; a failure reopens D61.

1. **v4 unchanged, v5 = v4 minus the single-player rows:** `tools/obs/hash_obs.py --version 4
   --check-v5 --expect workdir/sp-ablation/golden-v4.json` over the SP ablation's 36 logs (22,794
   decisions). Passes if it exits 0: every v4 hash matches, and at every decision v5's rows 0–888
   equal v4's, rows 889–1011 are zero, and the masks match.
2. **The frozen v5 hash:** `hash_obs.py --version 5 --out workdir/sp-ablation/golden-v5.json`,
   made read-only. Recorded here once made; later encoder changes check against it.
3. **Deterministic A/A with a v5 model:** `python -m lab aa --model No-SP A --no-amp --sets 500
   --block aa-harness-nn` (true FP32, D43; the earlier neural A/As' deals). Passes if every
   rotation of every set gives the same game (the lab's `PASS`).
4. **The drift tool on v5:** `experiments/rl_drift.py` now featurizes the states as the
   initialization's version (it was fixed at v4). No-SP A against itself on the diagnosis's states
   must give reverse KL 0 and argmax agreement 1.

### Results (2026-10-26 15:00–15:04 UTC): all four pass

1. `hash_obs.py --version 4 --check-v5`: every v4 hash matches `golden-v4.json` (`b9817059…`), and
   v5 = v4 with the single-player rows zeroed at all 22,794 decisions. **Pass.**
2. **Frozen v5 hash: `c0ec182577b80fcc4c03eda04a58c9e7e2ba6a9a29b0ab573c32519a75b82624`**
   (`workdir/sp-ablation/golden-v5.json`, read-only; 36 logs, 22,794 decisions).
3. Deterministic A/A with No-SP A (`20261026-080110-AA`, family A [500, +500), true FP32): 500 sets,
   every set score 0, **0 sets with a divergent game. Pass.**
4. `rl_drift.py` on v5 (`workdir/diag/v5-drift-selfcheck.txt`): No-SP A against itself on the
   diagnosis's states as v5, reverse KL 0.0000 and agreement 1.0000 for every kind of decision.
   **Pass.** (The per-kind counts differ from pilot A's by a few: they are grouped by the
   initialization's own argmax.)

D61's first check is met. The recipe check (below) remains before a v5 result counts.

## The recipe check: pre-registered 2026-10-26; launched on the user's word ("Launch")

Stage 1 (B0, `p1v5-b0`) was rented at 22:12 UTC on 2026-10-26: learner EPYC 7B13 (machine 11349),
actors Threadripper PRO 7975WX and EPYC 7713, ≈ $2.45/h; watcher `--budget-box-hours 12
--snapshot-at 0.5 --max-dollars 20`, watchdog 2026-10-27 06:12 UTC. Stage 2 follows if B0 passes
`experiments/b0_check.py`.

- **B0 result (2026-10-27 02:16 UTC): pass.** 12.16 box-hours, $9.95, `full` at step 382,800
  (`33fd344424704e92`). Over 568 windows: explained variance by third 0.0087 / 0.0115 / 0.0125,
  last-kyoku 0.0345 / 0.0476 / 0.0538, the final window monotone (100/190 in the last third). Both
  rising: a cleaner warm-up than P1's (`experiments/p1.md`).
- **Stage 2 (`k2v5`) rented at 02:17 UTC:** learner Threadripper PRO 7975WX (Zen 4, machine 146111,
  $0.905/h, a long contract; no Zen 4 EPYC had one), actors EPYC 7V13 and 7713; ≈ $1.98/h, ≈ $71 for
  108 box-hours; watcher `--budget-box-hours 108 --snapshot-at 0.5 --max-dollars 85 --anchor-alert
  0.10`, watchdog 2026-10-28 20:17 UTC. `full` ≈ 2026-10-28 14:30 UTC, then the check on the
  workstation.
- **Stage 2 p50 (2026-10-27 20:18 UTC):** 54.04 box-hours, $35.94, step 1,568,800 (`7d9aee69…`),
  ≈ 220K games/h. Already 1.5× `k2b`'s steps at its p25 (1,048,400 at 108 box-hours): the v5
  actors' speed shows up as learner steps, so `full` will hold ≈ 3× `k2b` p25's.
- **Stage 2 `full` (2026-10-28 14:22 UTC):** 108.25 box-hours, step 2,836,000 (`0510c68c…`; 2.7×
  `k2b` p25's 1,048,400), the fleet pulled and down, **$71.99** (cap $85). One actor host lost early
  (10-27 03:36), recovered in 1,075 s; no `anchor_alert`.

### Result (2026-10-28 14:38 UTC): pass, so the recipe transfers to v5

`k2_early_check.py` on `full` against No-SP A (`workdir/fleet/runs/k2v5/early-check-full.json`, 909 s):

| Measure | Threshold | `k2v5` full | `k2b` p25 (v4, vs pilot A) |
|---|---|---|---|
| Reverse KL to No-SP A (15,215 decisions) | ≤ 0.10 | **0.014** (median 0.004) | 0.025 |
| Argmax agreement | ≥ 90% | **95.7%** | 94.2% |
| Δu at argmax vs No-SP A (4,000 sets, diagnostic key) | CI not entirely below 0 | **+0.0132 ± 0.0052** (≈ [+0.003, +0.024]) | +0.012 ± 0.0055 |

Every threshold passes, and nothing fired. As with `k2b` at p25, the online network is above its
initialization on the diagnostic key, here with its 95% CI above 0. It's exploratory: the diagnostic
key is a fixed 4,000-set measure, not a lab verdict. With the v5 A/A and the frozen v5 hash already
passed, **D61's checks are met: v5 results count from here.** The recipe check cost $81.94 in all
(B0 $9.95 + stage 2 $71.99).

Over 2.7× `k2b` p25's learner steps, the policy drifted less (0.014 against 0.025). The anchor held it
nearer π_SL while it trained on more games.

D61 fixes its shape: a `k2`-style run from No-SP A to p25, judged by D59's early-check measures
against No-SP A, with the thresholds `k2b` cleared at p25.

- **Stage 1, B0 (`p1v5-b0`):** P1's critic warm-up, as `k2b`'s chain had (`p1a-b0`, D53). A critic
  doesn't transfer (v4's was trained with the single-player rows), so v5 warms up its own:
  `experiments/configs/p1-v5.toml` (p1.toml with `version = 5`, `workdir/p1v5/` paths, No-SP A as the
  frozen opponent), `--set critic.warmup=true`, from No-SP A, 2 actors × 2 processes,
  `--budget-box-hours 12`. Acceptance as P1's B0 (`experiments/p1.md`): over the last third, the
  10-bin calibration monotone in the final window and the explained variance (overall and last
  kyoku) not falling; one 4 box-hour extension if not. ≈ $7–9.
- **Stage 2, K2 (`k2v5`):** `experiments/configs/k2-v5.toml` (k2.toml with `version = 5`, `workdir/k2v5/`
  paths, No-SP A as the frozen opponent and π_SL), from B0's `full`, the learner check's settings
  (`dataset.num_workers=4`, `control.num_threads=4`), learner + 2 actors × 2 processes as `k2b`.
  `--budget-box-hours 108` (`k2b`'s p25), with a p50 snapshot at 54; `full` then is the checkpoint
  judged, and the fleet goes down. A Zen 4 learner if one with a long contract is offered
  (`--learner-machine`). ≈ $65–75 (≈ 36 h).
- **The check** (`experiments/k2_early_check.py --ckpt k2v5/snapshots/full.pth --init No-SP A --states
  workdir/diag/credit-a`, on the workstation, ≈ 1 h): **passes if** the reverse KL to No-SP A is
  ≤ 0.10, the argmax agreement ≥ 90%, and the Δu at argmax against No-SP A (4,000 sets, the
  diagnostic key) has a 95% CI not entirely below 0. `k2b` at p25: 0.025, 94.2%, +0.012 ± 0.0055.
  The states are the same games as `k2b`'s check, featurized as v5, so the numbers compare.
- **Differences from `k2b`, stated before the result:**
  - v5 actors are ≈ 2.4× faster, so 108 box-hours hold more games than `k2b`'s p25 (step
    1,048,400); with two v5 actors the learner probably binds. Steps and games are reported.
  - The starting network is No-SP A, a weaker or equal student to pilot A (SP ablation: pair 1
    −0.0060, pair 2 +0.0012), so Δu is measured against No-SP A, not pilot A.
- **Reading:** pass → the recipe transfers; with the A/A passed, v5 results count (D61). Fail → D61
  reopens and the user decides: back to v4, or a fix.
- **Cost and timing:** ≈ $75–85, cap $95. It can run alongside the soak (its own fleet and watcher);
  the check at the end needs the workstation for ≈ 1 h. A rental, so it waits for the user's yes.
