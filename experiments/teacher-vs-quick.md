# The teacher against a quick student: pre-registered 2026-11-09, the rental approved 2026-11-10

The audit of 2026-11-09 (`STATUS.md`) found no lab row that compares a teacher with a quick student.
BC s1 (b24c512, 42.7M parameters, 2.34B presentations, $54.72) is +0.9 pp validation top-1 over the
b8c192 heads-quick student (81.34% vs 80.46%). The distillation run (`docs/design/distillation-draft.md`,
≈ 128 h, ≈ $65) assumes the teachers are a stronger target than quick-scale BC. This test checks
that in play, before the run is rented.

- **Engines (frozen, read-only, both v5):**
  - Challenger: BC s1, EMA weights, `workdir/teacher/teacher-bc-s1/ema.pth` (`2f200d7c…`), D63's
    best single teacher.
  - Champion: No-SP A, `workdir/sp-ablation/nosp-a/nosp-a-final.pth` (`498aabc8…`), the v5 quick
    student that the heads check used as its control.
- **Where:** a Vast eval box (`tools/fleet/evalbox.py`), ≈ $0.50 if it ends near the minimum, as the
  AWR vs BC T-std did (6,500 sets, 43 min, $0.43). Weights only, no Tenhou data, so a community host
  is fine (D23). **The user approved the rental on 2026-11-10** ("Rent the 50 cent box"; quoted then
  at $0.542/h).
  - The box holds the lab lease, so it starts only once the `k2b` opponent chain on the workstation
    has ended. `experiments/teacher_vs_quick_chain.sh` waits for it, rents at ≤ $0.75/h, queues the
    test, and pulls and destroys the box when the queue drains.
  - Ceiling: 6 h of box time (≈ $3.3). A working box ends the test well before then (the GSPRT stops
    by ≈ 20K sets almost surely); past it, the box is treated as broken and forced down.
- **The test:**

  ```
  python3 tools/fleet/evalbox.py submit -- run --test T-std --family B \
      --challenger workdir/teacher/teacher-bc-s1/ema.pth --champion workdir/sp-ablation/nosp-a/nosp-a-final.pth \
      --description "teacher vs quick student: BC s1 (EMA) vs No-SP A"
  ```

  T-std as `lab/lab.toml` defines it: μ0 0, μ1 0.01, calibrated GSPRT, cap 100,000 sets; hitting the
  cap counts as a fail.
- **Reading, fixed 2026-11-10 before the run** (as proposed; the user's yes to the rental):
  - **Pass:** the teacher beats a quick student by the lab's δ. The distillation's premise holds.
  - **Fail, including the cap:** the teacher isn't shown stronger in play than a b8c192 trained for a
    day on the 3060. The distillation run is reviewed with the user before anything is rented.
  - **Either way, reported, exploratory:** mean Δu with its 95% CI, sets played, average rank and the
    panel. A pass at the minimum leaves the size within ≈ ±0.009.

## Result (2026-11-10 15:22 UTC): pass, so the distillation's premise holds

- **The test:** `20261110-145755-T-std` (job `20261110-145729-915`), on Vast instance 60082444 (Ryzen 9
  7950X3D, NL, $0.542/h), 14:57–15:22 UTC. The box was up 14:49–15:22 UTC, so ≈ $0.30. Family B
  nonces from 1,001,280,000.
- **Pass at the 5,000-set minimum,** LLR +7.20 (boundary +2.944). Δu (BC s1 − No-SP A) **+0.0219**,
  set SD 0.342, 95% CI (bootstrap over sets) **[+0.012, +0.031]**.
- **Reading, as fixed:** the teacher beats a quick student by more than the lab's δ in play, not only
  in top-1 (+0.9 pp). The distillation run's premise holds; renting it is still the user's call.
- **Exploratory:**
  - 20,000 games: BC s1's average rank 2.480, rank histogram 4,964 / 5,145 / 5,213 / 4,678.
  - Panel (`workdir/lab/20261110-145755-T-std/panel.json`, built afterwards with `python -m lab
    panel`, since `run` doesn't build one): win 21.9%, deal-in 12.8%, call 30.3%, riichi 17.8%, dama
    share of wins 14.6%.
  - Against the b8c192 models at the `k2b` tables (riichi 19.4–20.1%, dama 11.8–12.6%), BC s1
    riichis less and wins more by dama. Those tables have other opponents, so that's a loose
    comparison, not a result.
  - For scale: +0.022 is about the size of `k2b`'s RL gain over pilot A (+0.0195). Different pairs
    and observations, so not a comparison either.
