# AWR vs BC teacher T-std (04 §5, D17): pre-registered 2026-11-04, before AWR's weights exist

`04` §5 trains both objectives at b24c512 and gives the second seed to whichever wins T-std between
them. The second seed is on hold (the user, 2026-11-04); this test runs now.

- **Engines (the EMA weights, the user, 2026-11-04):** each run's `final.pth` → `ema.pth` via
  `tools/secure/export_ema.py`, read-only.
  - Champion: BC s1, `workdir/teacher/teacher-bc-s1/ema.pth` (`2f200d7c…`; from `final.pth`
    `6762492e…`, step 1,142,584).
  - Challenger: AWR s1, `workdir/teacher/teacher-awr-s1/ema.pth` (`381ce65f…`; from `final.pth`
    `bcb2987f…`, step 1,142,584), exported 2026-11-05 10:50 UTC.
- **Where (the user, 2026-11-04):** a Vast eval box (`tools/fleet/evalbox.py`). The test needs only
  weights, no Tenhou data, so a community host is fine (D23 covers the logs, not the weights). It is
  rented once AWR's weights are in, and downed when the job's rows are pulled.
- **The test:**

  ```
  python3 tools/fleet/evalbox.py submit -- run --test T-std --family B \
      --challenger workdir/teacher/teacher-awr-s1/ema.pth --champion workdir/teacher/teacher-bc-s1/ema.pth \
      --description "teacher objective (04 §5): AWR s1 vs BC s1, EMA weights"
  ```

  T-std as `lab/lab.toml` defines it: μ0 0, μ1 0.01, calibrated GSPRT, cap 100,000 sets; hitting the
  cap counts as a fail.
- **Reading, fixed now:**
  - **Pass:** AWR beats BC by the lab's δ, so AWR is the teacher objective and its second seed is
    the one to train.
  - **Fail, including the cap:** BC. It is the simpler objective, and it is kept as τ_human either
    way (`07` §4).
  - **Either way, reported, exploratory:** mean Δu with its 95% CI, sets played, average rank, and
    the panel.

## Result (2026-11-05 11:40 UTC): fail, so BC is the teacher objective

- **The test:** `20261105-105720-T-std` (job `20261105-105714-926`), on Vast instance 59250408 (Ryzen 9
  7945HX, $0.502/h), 10:57–11:40 UTC, ≈ $0.43. Family B nonces from 1,001,180,000.
- **Fail at 6,500 sets,** LLR −3.27 (boundary −2.944). Δu (AWR − BC) **−0.0007**, set SD 0.337, so
  ≈ [−0.009, +0.008]: the two objectives are even, within ≈ 0.009 u either way.
- **Exploratory:** 26,000 games, AWR's average rank 2.491, rank histogram 6,732 / 6,447 / 6,139 / 6,682.
  AWR took more firsts and more fourths than an even split; a more variable style, perhaps, but
  that's a reading of one histogram, not a result.
- **Reading (as pre-registered):** BC is the teacher objective, so the second seed is BC s2 when the
  user releases it (on hold, 2026-11-04). AWR's kyoku-level advantage, mostly luck, didn't buy
  measurable strength over plain imitation at this scale.
