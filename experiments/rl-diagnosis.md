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
- Advantages are normalized per batch, mean included (`ppo.py`, `advantage_norm = true`); the
  logged `ppo/adv_mean` (−0.004 to −0.010) is the raw mean. P0's early entropy rise fits a
  weak advantage signal that the entropy bonus (1e-3) outweighs at first: P0's credit is the
  kyoku's outcome on every decision of the kyoku. P1's per-decision GAE credit sharpens instead.
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

### The same for P1 (`p1-b` against pilot B)

T = 0.22 matches `p1-b` full's entropy on its own states: 0.114 against pilot B's 0.129 at 0.25
and 0.103 at 0.20 (a 20-set calibration). Same key and nonces; `workdir/diag/p1b-*`.

| Challenger | Play | Entropy | Δu vs pilot B (argmax) |
|---|---|---|---|
| pilot B | sampled, T = 1 | 0.521 | −0.090 ± 0.006 |
| pilot B | sampled, T = 0.22 | 0.112 | −0.004 ± 0.005 |
| `p1-b` full | sampled, T = 1 | 0.114 | −0.022 ± 0.006 |
| `p1-b` full | argmax | — | −0.022 ± 0.006 |

| Comparison | Δu |
|---|---|
| RL's gain in its own setting | +0.068 ± 0.008 |
| Sharpening alone | +0.086 ± 0.008 |
| **RL against sharpening at matched entropy** | **−0.018 ± 0.008** (z ≈ −2.3) |

**The same reading, smaller:** P1's training-game gain is less than lowering pilot B's
temperature to the same entropy. At an entropy of 0.11, `full` plays nearly greedily: −0.022
sampled, −0.022 at argmax. The argmax result agrees with the lab's T-std (−0.0131 ± 0.0054 on
family B) within the noise.

## 3. Interface check (`experiments/rl_credit_check.py`, 2026-10-17)

200 fresh games per pilot, played as the actors play them, then run through the trainer's own
data path (`rl_dataset.OrderedDatasetsIter.trajectories`), checked against each log's events.

| Check | Pilot B games | Pilot A games |
|---|---|---|
| Kyoku GRP reward vs the trainee's point change, offset 0 / ±1 kyoku | r +0.834 / +0.018 | r +0.845 / ≤ +0.005 |
| Winning actions outside a won kyoku | 0 of 394 | 0 of 417 |
| Deal-in kyoku whose last decision isn't a discard | 0 of 310 | 0 of 282 |
| P0 advantage (K, MC) on the winning action / the deal-in discard | +1.26 / −1.16 sd | +1.23 / −1.15 sd |
| P1 advantage (H, GAE, `p1-b` full's critic) on the same | −0.03 / −0.67 sd | — |

- **The credit is where it should be.** Rewards sit on their own kyoku, and decisions on
  theirs. The behaviour join already checks labels against the sidecar.
- P1's ≈ 0 on the winning action is expected: the critic sees the hidden tiles, so V already
  counts the win once it's available.

## 4. Drift from the initialization (`experiments/rl_drift.py`, 2026-10-17)

On the trainee's decisions with more than one legal action in those games (every second one,
≈ 15K), each checkpoint against its initialization:

| Run | Checkpoint | KL(init ‖ ckpt) | Argmax agreement | Discard agreement | Entropy |
|---|---|---|---|---|---|
| `p0-a` | p25 | 0.64 | 78.0% | 73.1% | 0.449 |
| `p0-a` | p50 | 0.77 | 75.8% | 70.4% | 0.426 |
| `p0-a` | full | 0.92 | 75.6% | 70.3% | 0.366 |
| `p1-a` | p25 | 1.01 | 80.1% | 76.0% | 0.158 |
| `p1-a` | p50 | 1.22 | 79.4% | 75.2% | 0.134 |
| `p1-a` | latest (2026-10-17 06:55 pull) | 1.32 | 79.6% | 75.3% | 0.124 |
| `p1-b` | p25 | 0.94 | 81.1% | 77.3% | 0.160 |
| `p1-b` | p50 | 1.16 | 80.2% | 76.3% | 0.138 |
| `p1-b` | full | 1.38 | 79.5% | 75.4% | 0.113 |

(Initializations: entropy 0.522 for pilot A, 0.519 for pilot B, on these states.) Wins
(agreement 100%) and passes (≈ 95%) barely move. Discards move most.

- **Every run moved far, and early.** By p25, a fifth to a quarter of all argmax decisions, and
  a quarter to 30% of discards, differ from the initialization. After that, agreement levels
  off while the KL keeps rising: 05 §8.2's signature (KL rising steadily, agreement falling).
- P0 barely sharpened (0.52 → 0.37), yet it changed the most argmaxes. So the drift isn't just
  the entropy collapse.

## Diagnosis so far

1. **Interface:** clean (§3).
2. **Signal:** right-signed but weak. Training games improve, and in both arms sharpening
   explains more than all of that gain. At matched entropy, RL's other changes cost
   ≈ 0.040 u in P0 and ≈ 0.018 u in P1 (§2).
3. **Drift:** without an anchor, each run walks a quarter of its decisions away from the SL
   policy within the first quarter of its budget (§4). That drift is what loses at argmax.
4. **P1's entropy collapse** (05 §8.1) is a separate symptom of its per-decision credit. It
   doesn't make P1 learn more.

**Against the plan's anchor.** K2 (05 §5.1) anchors to the SL policy with the reverse
KL(π_θ ‖ π_SL), target 0.02 nats/decision at the start. The same measure on these states:

| Run | p25 | full |
|---|---|---|
| `p0-a` | 0.38 (median 0.16) | 0.44 (0.16) |
| `p1-b` | 0.35 (0.11) | 0.43 (0.14) |

That's ≈ 20× the anchor's target by the first quarter. P0 and P1 have no anchor. Their only
trust region is PPO's per-update clip, over 11K+ updates per run. 05 §8's fixes for 8.1/8.2
are a forward-KL anchor, a higher c_KL or a lower LR; K2 is the package that adds them. Running
it before G0's selection, or inside a rerun of the arms, is a plan change, so it needs an `08`
entry first.

## 5. Next

- **Interface check first** (05 §8's first item): are the advantages attached to the right
  decisions? A credit misalignment would look exactly like this (a right-signed but weak signal,
  plus drift). Check on fresh 1v3 logs, through the training data path (`rl_dataset.py`): a
  winning action and the discard that deals in should carry strongly positive and negative
  advantages respectively, in P0 (kyoku GRP-Δu) and in P1 (GAE).
- **The same sampled-vs-sharpened comparison for P1** (`p1-b` full against pilot B at the
  temperature that matches its entropy of ≈ 0.11).
- **When the damage happens:** `p25` and `p50` of `p0-a` at argmax on the diagnostic key.
