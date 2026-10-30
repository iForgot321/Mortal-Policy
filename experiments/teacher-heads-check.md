# The teacher heads check (04 §5's guard): pre-registered 2026-10-29, before it runs

`docs/design/teacher-trainer-draft.md` → "Checked before any rental". It answers one question before
any teacher is rented: do the auxiliary heads (placement, kyoku-Δ, opponent tenpai and waits) hurt
the policy? It also runs the new trainer end to end.

## The run

- **What:** a b8c192 v5 BC quick student with the heads attached, trained by `mortal/train_teacher.py`
  with `experiments/configs/teacher-heads-quick.toml`.
- **The control is No-SP A** (`workdir/sp-ablation/nosp-a/nosp-a-final.pth`, `498aabc8…`): same data
  (quick500k), seed (1), schedule (peak 3e-4, warm-up 2,000, cosine to 1e-5, batch 2048, one epoch,
  no augmentation), Q6/Q8 masks and network. **What differs:**
  - the heads and their loss terms (the thing tested);
  - the new trainer's mechanics: fused AdamW, its own seeded file order (`TeacherDataset`),
    `drop_last` per worker, and no test play. None changes the objective; together they act like a
    different seed.
- **Where:** the workstation, ≈ 9 h. Nothing else runs on it meanwhile (RAM).
- **The checkpoint judged:** the raw final weights (`workdir/teacher/heads-quick/final.pth`, top-level
  `mortal` and `policy_net`), as No-SP A's are raw. The EMA weights are reported, not judged.

## The test

```
python -m lab run --test T-NI --family B \
    --challenger workdir/teacher/heads-quick/final.pth \
    --champion workdir/sp-ablation/nosp-a/nosp-a-final.pth \
    --description "teacher heads check (04 §5's guard): heads-on quick student vs No-SP A"
```

T-NI as `lab/lab.toml` defines it: δ = 0.01 (μ0 −0.01, μ1 0), calibrated GSPRT, cap 60,000 sets.
Hitting the cap counts as a fail.

## Reading, fixed now

- **Pass:** the teachers keep the heads attached. The new trainer is also shown non-inferior to
  the old one at quick scale.
- **Fail:** this run can't separate the heads from the trainer's mechanics. A second quick student
  then runs with `[teacher] detach = true` (heads on stop-gradient features, otherwise identical),
  with the same T-NI against No-SP A.
  - It passes: the teachers train with the heads detached.
  - It fails too: the trainer itself is the suspect, and it is investigated before any teacher
    rental.
- **Also recorded, exploratory (CLAUDE.md rule 1):**
  - validation losses per head and policy top-1 on 200 heldout_game logs, for the raw and the EMA
    weights;
  - the timing log (`timing.jsonl`): samples/s against No-SP A's 10.5K, and the time spent waiting
    for data. The opponent labels replay three more `PlayerState`s per event.

## Progress

- **Training started 2026-10-30 05:36 UTC** (`workdir/teacher/heads-quick/`, pid 73276). Steady state
  12.7–14.0K samples/s against No-SP A's 10.5K, with the GPU waiting 37–46% of the time for data:
  on the workstation's 2 workers, this small net is pipeline-bound, labels included.
  - The five losses fall from step 500 to 2,500: policy 1.36 → 0.69, placement 1.28 → 1.16,
    kyoku-Δ 0.032 → 0.031, tenpai 0.41 → 0.19, waits 0.34 → 0.19.
  - ≈ 6.8 h in all, so ≈ 12:30 UTC.
- `experiments/teacher_heads_chain.sh` (pid 74720) starts the pre-registered T-NI when the training
  ends.
