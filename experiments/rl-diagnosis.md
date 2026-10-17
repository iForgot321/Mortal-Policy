# Diagnosis of the online arms (05 §2.3.6, §8), started 2026-10-17

**Why:** every online checkpoint so far loses to its own initialization at argmax:
- G0: `p0-a` full −0.0268 and `p1-b` full −0.0131 (T-std, family B);
- the halfway checks: `p0-a`, `p1-a` and `p1-b` p50 −0.033 / −0.026 / −0.022 (family A);
- V's online test −0.043 (`v-tuning.md`; V has its own diagnosed mechanism, the Q collapse).

05 §8's triage order: units or interface bug → KL too low → LR too high → critic diverging →
masking bug. The user asked for the diagnosis on 2026-10-17, while the last evaluations run.

## 1. From the runs' own logs

### Training games improve, in every run

Each actor batch logs the trainee's rank histogram. The trainee samples every action
(`explore_rate = 1`, no agari guard) against three argmax copies of its initialization
(`mortal/player.py`). Its mean u under the lab utility, by tenth of the run:

| Run | First tenth | Last tenth | Games |
|---|---|---|---|
| `p0-a` | −0.111 | −0.065 | 15.2M |
| `p1-b` | −0.043 | −0.029 | 13.7M |
| `p1-a` | −0.046 | −0.034 | 13.7M (to 2026-10-17) |

So the sign of the learning signal is right. The lab plays the argmax action with the agari
guard on, and at argmax the same checkpoints lose.

### Entropy

| Run | Start | First tenth (mean) | End | Note |
|---|---|---|---|---|
| `p0-a` | 0.487 | 0.534 | 0.366 | rose to ≈ 0.58 in the first hours, then fell slowly |
| `p1-a` | 0.474 | 0.276 | 0.113 | 0.42 after 0.9 h, 0.29 after 3.4 h |
| `p1-b` | 0.482 | 0.306 | 0.109 | the same collapse (05 §8.1) |

- P0 and P1 have the same policy optimizer settings. The differences are P1's critic (GAE, λ
  0.95, γ 1: per-decision advantages) and horizon H. The policy LR is a constant 2e-5 in both.
- P0's advantage is the kyoku's GRP-Δu, not centered: `ppo/adv_mean` runs −0.004 to −0.010
  (std 0.25), because the sampling trainee does worse than its argmax opponents. A negative
  mean pushes probability away from the actions taken, which fits the early entropy rise.
- P1's critic explains little: `critic/explained_variance` 1.3% → 1.7%, and 5% → 10% in the
  last kyoku.

### Ruled out

- The initialization: P1's B0 snapshots are bit-identical to the pilots in `mortal` and
  `policy_net` (the warm-up trained only the critic).
- Policy lag: ≈ 7 versions, corrected by the stored behaviour probabilities
  (`p0.md` → "Policy lag").

## 2. Sampled vs argmax play (`experiments/rl_diag_sampling.py`, 2026-10-17)

Each run plays 4,000 duplicate sets against pilot A at argmax (agari guard on, AMP), on the
diagnostic key `0xd1a65a3b1e0fc0de`, nonces [0, 4000). Each takes ≈ 44 min on the workstation.
The temperature T samples from softmax(logits / T). "Entropy" is the mean entropy of the
distribution sampled from, over the decisions the engine was asked (single-option decisions skip
the engine). Logs and per-set arrays are in `workdir/diag/p0-*.{log,json,npz}`.

| Challenger | Play | Entropy | Δu vs pilot A (argmax) |
|---|---|---|---|
| pilot A | sampled, T = 1 (the start of every run) | 0.516 | −0.086 ± 0.006 |
| pilot A | sampled, T = 0.7 | 0.356 | −0.024 ± 0.006 |
| `p0-a` full | sampled, T = 1 (its training setting) | 0.368 | −0.064 ± 0.006 |
| `p0-a` full | argmax (the lab's setting) | — | −0.019 ± 0.006 |

- T = 0.7 was chosen beforehand: on `full`'s own sampled states, pilot A's entropy at 0.7 is
  0.377, against `full`'s 0.378 (a 20-set calibration).
- `full` sampled reproduces its last training games (−0.064 here, −0.065 in the logs).
- `full` at argmax agrees with the lab's T-std (−0.027 ± 0.0055 on family B).

Paired differences (same deals; sampling decorrelates the games, so the correlation is only
≈ 0.06):

| Comparison | Δu |
|---|---|
| RL's gain in its own setting: `full` sampled − pilot A sampled | +0.022 ± 0.008 |
| Sharpening alone: pilot A T 0.7 − pilot A T 1 | +0.062 ± 0.008 |
| **RL against sharpening at matched entropy:** `full` sampled − pilot A T 0.7 | **−0.040 ± 0.008** |

**Reading.** At matched entropy, P0's final policy is clearly worse than its initialization with
a lower temperature. Sharpening accounts for all of RL's gain in training games, and RL's other
changes to the policy cost ≈ 0.04 u. So P0 learned no improvement that shows at its own sampling
temperature, which rules out "learned caution that doesn't transfer to argmax". What's left is
a policy that drifted on a signal that is mostly noise or partly wrong.

This also explains P0's and P1's different starts in the logs: −0.122 for P0 against −0.062 for
P1 in the first hour, from the same −0.086. P1 sharpened within the hour, while P0 first got more
random.

## 3. Next

- **Interface check first** (05 §8's first item): are the advantages attached to the right
  decisions? A credit misalignment would look exactly like this (a right-signed but weak signal,
  plus drift). Check on fresh 1v3 logs, through the training data path (`rl_dataset.py`): a
  winning action and the discard that deals in should carry strongly positive and negative
  advantages respectively, in P0 (kyoku GRP-Δu) and in P1 (GAE).
- **The same sampled-vs-sharpened comparison for P1** (`p1-b` full against pilot B at the
  temperature that matches its entropy of ≈ 0.11).
- **When the damage happens:** `p25` and `p50` of `p0-a` at argmax on the diagnostic key.
