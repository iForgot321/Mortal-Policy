# Draft D62: the SL teachers train on Hyperstack RTX A6000s (for the user's review)

**Status: draft, 2026-10-29.** Nothing in `08` or the reference docs changes until the user approves.
Evidence: the measurement session (`docs/design/secure-data-path-draft.md` → "The measurement
session's result"). Follows the user's choices of Hyperstack and its object storage (2026-10-28/29).

---

**D62 (2026-10-__): the b24c512 teachers train on Hyperstack RTX A6000 VMs (secure tier, CANADA-1,
28 vCPU, on demand), not a 4090 with ≥ 48 vCPU or an H100. Resolves D17a.** Amends D50's teacher
consequence ("a 4090 with ≥ 48 vCPU on the secure tier, or an H100"), `03` §2's burst row and §4.2,
and `04` §5's hardware paragraph. Decided by the user.
- **Why** (the measurement session, 2026-10-29, $1.74; b24c512 v5, the real trainer with BC, batch
  2048, 4 workers, augmentation):

  | | RTX A6000 | L40 | A100 80GB spot |
  |---|---|---|---|
  | $/h | 0.507 | 1.007 | 1.087 |
  | Real trainer (samples/s) | 4,995 | 5,346 | ≈ 6.1K (estimated from its GPU-only 7.6K) |
  | One teacher (2.34B presentations) | 130 h, **$66** | 122 h, $122 | ≈ 107 h, ≈ $116, plus preemptions |

  - **The A6000 costs about half as much per teacher, at nearly the same speed.** b24c512 is small
    for these GPUs, so the faster cards barely pull ahead.
  - **The secure tier has no 4090 with ≥ 48 vCPU.** Vast Secure Cloud's 4090s have ≤ 32
    (2026-10-28), and Hyperstack offers no 4090.
  - **≥ 48 vCPU isn't needed on v5.** The floor came from v4's featurization (D50, `03` §4.2). On v5,
    28 vCPU give 12–17K samples/s with 4 workers against GPU rates of 6–7.6K, so `03` §4.2's 80% rule
    passes on every flavor measured.
  - **No H100 is needed.** None of the measured GPUs is fed short of data, and an H100 costs ≈ 2–4×
    as much per hour.
- **Consequences:**
  - Teacher runs (AWR s1 ∥ BC s1, then the winner's s2) use `n3-RTX-A6000x1` through
    `tools/secure/secure.py`, with the data from the bucket (`tools/secure/pack.py`, `bucket.py`).
    That's ≈ 130 h and ≈ $66 per teacher as measured, ≈ $200 for the three, inside the SL substrate's
    ≈ $1.0K (`00`).
  - **The environment matches the fleet's pin:** Python 3.12, torch 2.6.0+cu124, on Hyperstack's
    Ubuntu 22.04 R550 / CUDA 12.4 image (`tools/secure/bootstrap_vm.sh`). The GPU is Ampere (sm_86),
    which torch 2.6 has kernels for. SL training needs no bitwise determinism; the deterministic
    A/A and every lab verdict stay on the workstation and the 4090 fleet.
  - **Fallback if the A6000 is out of stock** (3 single-GPU VMs were listed on 2026-10-29; AWR ∥ BC
    needs 2): the L40 on demand, ≈ $122 per teacher, or waiting. An A100 spot only with the user's
    yes, since a preemption costs ≈ 0.5–1.5 h.
  - D50's other roles (learner, actors, eval, deploy on 4090s) are unchanged.
- **Not decided here:** the students' distillation and the teacher-output cache (`04` §6) get their
  own measurement or note when that draft exists; the same tools apply.
- **Reopen if:**
  - the A6000 is unavailable for more than a day when a teacher run is due;
  - Hyperstack's A6000 price moves by more than 30%;
  - the teacher network changes size (`04` §5's optional b40c512 would need its own measurement);
  - or a training-loop change moves the per-GPU ranking. The planned removal of per-step GPU syncs
    could lift every GPU's real rate toward its GPU-only rate; the A6000's lead per dollar holds unless
    the faster cards gain twice as much.

---

## Edits once approved (after the `08` entry)

- `03` §2's burst row: "1× RTX A6000 (Hyperstack, secure tier, 28 vCPU) for SL teachers (D62); an
  H100 or a 4090 with ≥ 48 vCPU are no longer needed"; `03` §4.2: the measured rates and D62.
- `04` §5's hardware paragraph: the A6000, its measured rate, and ≈ 5.4 days per teacher.
- `01` §10's secure-tier table: Hyperstack's live prices and stock (2026-10-29).
- `00`'s Open-items line for D17a: resolved by D62.
