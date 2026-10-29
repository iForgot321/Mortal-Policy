# Draft: the teacher trainer (`04` §5), for the user's review

**Status: draft, 2026-10-29.** The second piece of the SL substrate (G0's option 1), after the secure
data path (`docs/design/secure-data-path-draft.md`). It covers what the b24c512 teachers train on,
which heads they carry, the code that's new, how it's checked before any money is spent, and the
runs. It implements `04` §5 without changing it, so it needs no `08` entry. Nothing here rents.

## What `04` §5 asks for

- **Three teacher runs:** AWR and BC at b24c512 on v5, then a second seed of whichever wins T-std
  between them. The winning pair is the ensemble the students distil from (`04` §6). The BC teacher
  is kept either way as τ_human, the human prior for the piKL dial (`07` §4).
- **Data:** the train window 2015-01 → 2025-06, 1 epoch × 2 augmentation (m↔p), ≈ 2.34B
  presentations, ≈ 1.14M steps at batch 2048.
- **Optimizer:** AdamW with weight decay 0.1 on conv and linear weights only; peak LR 1e-4 or 3e-4;
  2–5K warm-up steps, then cosine; gradient clip 1.0; AMP; EMA of the weights at 0.999.
- **Heads, on a shared trunk:**

| Head | Target | Used by |
|---|---|---|
| Policy (46) | the human action | everything |
| Placement (4-class) | the seat's final placement | value warm-start, diagnostics |
| Kyoku-Δ (Huber) | this kyoku's GRP-Δu in fixed u units (D37) | the K1 critic's warm-start (`05` §4) |
| Opponent tenpai (3 × sigmoid) | is each opponent tenpai now? | the belief sampler (`06` S-1) |
| Opponent waits (3 × 34 sigmoid) | each tenpai opponent's winning tiles | the belief sampler, push/fold |
| Opponent hand value (3 × bucket) | expected win value if they win | search-as-features (`06` S-5) |

- **Loss:** L = L_policy + 0.5·L_placement + 0.5·L_kyokuΔ + 0.2·L_aux.
- **Guard:** if a T-NI check shows the auxiliary heads hurt the policy, detach them.

## What exists, and what is new

**Exists** (`mortal/train_offline_phase1.py` and the loader; Pilot A and B and No-SP A and B
trained with it):
- BC and AWR. AWR's advantages are the kyoku's GRP-Δ, standardized with running statistics as
  upstream does; β 1.0, clip 100.
- AMP and `torch.compile`, the weight-decay groups as `04` asks, `LinearWarmUpCosineAnnealingLR`,
  gradient clipping, resume from `state_file`, Q6/Q8 masks and m↔p augmentation.

**The labels are mostly there:**
- **Placement and kyoku-Δ:** `FileDatasetsIter(value_targets=True)`, built for arm V, already
  yields the kyoku's GRP-Δu in fixed u units and the seat's rank after each kyoku. The rank after
  the last kyoku is the final placement.
- **Opponent tenpai and waits:** the oracle encoder (`libriichi/src/dataset/invisible.rs`) already
  computes them for each opponent: a shanten one-hot row (tenpai = shanten 0) and a 34-wide waits
  row.
  - It also encodes the whole wall, which the teacher must not see and doesn't need, and it's slow
    (B0 drained only ≈ 45K hanchan/h because of it).
  - So the labels come from the loader directly: a new flag tracks the three opponents'
    `PlayerState`, as the oracle path does, and emits 3 tenpai bits and 3 × 34 wait bits per
    decision, with no wall and no oracle observation.
  - Labels are targets only, never inputs (CLAUDE.md rule 6). The policy's observation is
    untouched.

**New code:**
1. **Rust, the labels:** a `GameplayLoader` flag and `take_opponent_labels()`.
   - Tests: the labels equal the oracle encoder's rows on the same logs; an augmented game's labels
     are the swapped ones; v4 and v5 observation hashes are unchanged (`hash_obs.py --expect`, so
     arena and training stay hash-identical, rule 2).
2. **Python, the loader:** `FileDatasetsIter(teacher_labels=True)` yields, per decision: obs,
   action, mask, AWR advantage, final placement, kyoku-Δ, opponent tenpai and waits (bits, unpacked
   on the GPU).
3. **`mortal/model.py`, `TeacherHeads`:** small MLPs on the trunk's 1024-wide output (placement 4,
   kyoku-Δ 1, tenpai 3, waits 102), with a `detach` switch for the guard.
4. **`mortal/train_teacher.py`, a new script** rather than a rewrite of the upstream one (CLAUDE.md:
   new modules over rewrites). It is `train_offline_phase1.py`'s loop plus:
   - the heads and the weighted loss, with the waits loss masked to tenpai opponents;
   - EMA weights, saved beside the raw weights;
   - a fixed validation set: 2,000 `heldout_game` games, with every head's loss and the policy's
     top-1 every 10K steps;
   - a presentations counter;
   - checkpoints every 30 min into `workdir/secure-box/<run>/`, so `secure.py pull` gets them;
   - **resume that doesn't re-see data.** The file order is fixed by the seed, and each checkpoint
     stores how many files each worker has consumed, so a resumed run (after a preempted spot box)
     skips them and every presentation still happens once.
5. **Configs:** `experiments/configs/teacher-{bc,awr}-s{1,2}.toml` (b24c512, GN, v5, the box's data
   paths written by `secure.py ship`), and a quick-scale `teacher-heads-quick.toml`.
6. **Tests:** heads and loss shapes; the waits mask; EMA; the resume cursor (the same files in the
   same order with or without an interruption); the labels as in 1.

**The hand-value head is left out of the first version.** It is the one label the loader can't read
off a `PlayerState`: it needs a score for every wait of every tenpai opponent. Its only user is
search-as-features (`06` S-5), which is gated separately and unfunded. It can be added when S-5 is
funded, from the same trunk, without retraining the teachers' other heads.

## Checked before any rental (workstation, ≈ 9 h of the 3060, no money)

**The heads check (`04` §5's guard), at quick scale, against a control that already exists.**
- Train b8c192 v5 BC with the heads on, from the same data, seed and schedule as **No-SP A**
  (quick500k, `sp-nosp-a.toml`). No-SP A is the matched heads-off run (D41's matched pairs), so only
  one new run is needed: ≈ 8.6 h.
- Then T-NI against No-SP A (δ = 0.01, family B), pre-registered before it runs.
  - **Passes:** the heads stay attached.
  - **Fails:** they train detached, on stop-gradient trunk features.
- It also exercises the whole new code path end to end: labels, heads, EMA, validation, resume
  (killed once mid-run).

## The runs (rentals, each needing the user's yes)

On Hyperstack through `secure.py`, on the GPU the measurement session picks (D17a; its result
gives the hours and dollars):
1. **AWR s1 ∥ BC s1** on two boxes at once.
2. **T-std AWR vs BC** (teacher vs teacher in 1v3 duplicate). It needs only weights, so it runs on
   the workstation or a Vast eval box; it never touches Tenhou data.
3. **The winner's s2.**

Each run's outputs: the final and EMA weights, read-only and hashed; the validation curves; and the
golden-set per-category agreement table, computed on the workstation, where the golden set lives.

**Cost:** the estimate before the measurement is ≈ $55–160 per teacher, so ≈ $170–480 for the
three. The measurement replaces it. The quick check costs nothing.

## Choices: decided by the user (2026-10-29), all as recommended

1. **AWR's β:** upstream's 1.0 on standardized advantages. No quick-scale β check: the full-scale
   AWR vs BC T-std decides between them.
2. **Peak LR:** 1e-4 for b24c512.
3. **The quick-scale sweeps `04` lists:** skipped for now, with the defaults:
   - player scoring A0 (no filter; D14a unset);
   - the 2015-01 → 2025-06 window;
   - no decision weighting.

   Each is one quick student and one T-std on the workstation if wanted later.
4. **GN** in the trunk, as every network here uses.

**Hardware:** Hyperstack RTX A6000s, pending draft D62 (`docs/design/d62-draft.md`).

## Not in this draft

- Distillation into the student candidates, and the teacher-output cache (`04` §6). That's the
  next draft, after the teachers exist.
- The hand-value head (above).
- Publishing teacher weights, which needs a re-read of Tenhou's terms and an `08` entry (`04` §1).
