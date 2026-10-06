# Hand-off: wiring arms P0 and P1

**For:** a fresh agent picking up the RL wiring. **Written:** 2026-10-05, after commit `10c561f` (D46).
**Status:** M1–M6 implemented and smoke-tested 2026-10-06 (see `STATUS.md`). Open: the P1 horizon split and the K1 critic settings (§5), then B0 and the arm runs.

## 0. Read first, in this order

1. `CLAUDE.md`: rules, machine limits, build steps. Then `STATUS.md`.
2. `docs/reference/05_rl_core.md`:
   - §2.1 arms;
   - §2.4 P0 config v0;
   - §3 interfaces;
   - §4 critic;
   - §8 unit tests.
3. `docs/reference/08_decision_log.md`: **D37** (RL interfaces) and **D46** (behaviour sidecars).
4. `docs/design/behaviour-probs.md`: what D46 built, and the join semantics.
5. The code listed in §2 below.

## 1. Goal and definition of done

**Goal:** make arms **P0** and **P1** (05 §2.1) runnable end to end on the existing client → server → trainer loop:
1. 1v3 collection writes mjai logs plus behaviour sidecars.
2. The server buffers both.
3. The trainer builds per-seat ordered trajectories, attaches targets *before* shuffling, and trains with ratios against the **stored** behaviour probabilities.
4. P1 adds the K1 critic.

Running the comparison itself is **not** part of this. That takes 3 boxes × 6 days per arm-seed on cloud hardware.

**Done when:**
- The 05 §8 unit tests that cover these paths pass and run in CI (`pytest tests/`):
  - first-minibatch ratio = 1 against the stored behaviour probability, computed by the **trainer's own loss code**;
  - illegal-action probability is exactly 0 in the learner path;
  - ordered targets are deterministic (built twice, bit-identical);
  - `done` lines up with kyoku or hanchan ends;
  - no target crosses seats;
  - no running statistics in the reward path.
- `experiments/configs/p0.toml` exists, per 05 §2.4.
- A **local smoke run** (§4, M4) completes ≥ 2 parameter versions with no errors and with the expected diagnostics.
- P1: the critic trains, its diagnostics are logged, and a B0 warm-up mode exists. Running B0 for real needs the user's go-ahead (§5).

## 2. What already exists

| Piece | Where | Tests |
|---|---|---|
| Per-seat rewards, `done`, `next_idx` under horizon K or H | `trajectory.build` (`mortal/trajectory.py:38`) | `tests/test_trajectory.py` |
| GAE / λ-returns; Monte-Carlo returns | `trajectory.gae` (`:79`), `monte_carlo_returns` (`:103`) | same |
| Behaviour sidecar read + join to loader entries | `trajectory.load_behaviour` (`:109`), `join_behaviour` (`:124`) | `tests/test_behaviour.py` |
| Actors write sidecars | `TrainPlayer.behaviour_dir` (`mortal/player.py:113`), default `log_dir + '_behaviour'`; same file name as the log | same |
| Raw GRP-Δ rewards, no running statistics | `RewardCalculator(standardize=False)`, config `env.reward_standardization` (`mortal/dataloader.py:64`) | `tests/test_reward_units.py` |
| Masked raw logits | `CategoricalPolicy.logits` (`mortal/model.py`) | — |
| Index of the event each entry decides on | `Gameplay.take_event_idx()` | — |
| Exact hidden information for arena logs | `GameplayLoader(oracle=True, trust_seed=True)`; arena logs carry `StartGame.seed` | — |
| Frozen inputs | Pilot A `workdir/pilot/qs-a/pilot-qs-a-final.pth` (`15121432…`); pilot B `workdir/pilot/qs-b/pilot-qs-b-final.pth` (`b267c026…`); both b8c192, v4, `Norm="GN"`. GRP-v2 `workdir/grp-v2/grp-v2-frozen.pth` (`db28beac…`). | — |

**Measured facts you can rely on:**
- **Collection rate:** ≈ 22K hanchan/h on this workstation, b8c192 vs b8c192 with AMP (the A/A′ run: 80K games in 3 h 36 min).
- **Decisions:** ≈ 158 per seat per hanchan (D46 stress run: 633K entries over 1,000 games).
- **Memory:** a featurized fp32 decision is ~137 KB.

## 3. The gaps

1. **Transport.**
   - `client.py` submits only the log files (`mortal/client.py:85–95`).
   - `server.py` buffers and drains only logs (`handle_submit_replay` at `mortal/server.py:78`, `handle_drain` at `:96`).
   - `assert raw_count == S.buffer_size` (`:101`) counts every file in `buffer_dir`, so sidecars must not go there.
2. **Data.** `FileDatasetsIter.populate_buffer` (`mortal/dataloader.py:99`):
   - gives every decision its kyoku's reward (`:150`) and shuffles single decisions;
   - has no ordered trajectories, no behaviour probabilities, and no use of `event_idx`.

   SL training (`train_offline_phase1.py`) also uses this class, so leave its behaviour unchanged.
3. **Trainer** (`mortal/train_online.py`):
   - computes the ratio against an `Old_mortal` snapshot (`:74`, `:246`, refreshed at `:326`);
   - builds distributions from probabilities instead of logits (`:250`);
   - doesn't normalize advantages per batch;
   - has no notion of entries without a behaviour probability.
4. **No critic** (K1) and no B0 warm-up mode.
5. **Registry:** lab registry rows lack `arm`, `package` and `horizon` (02 §9; `lab/registry.py:93`). This is needed only when the first RL checkpoint gets a lab verdict.

## 4. Plan

Milestones in order. Each is a small, single-purpose commit, made only when the user asks.

### M1: Sidecar transport

- **Client:** for each log in `file_list`, read the sidecar with the same basename from `train_player.behaviour_dir`. Submit it as `'behaviour': {name: bytes}` next to `'logs'`.
- **Server:**
  - Write each sidecar under the same `{submission_id}_{filename}` name into a behaviour buffer dir.
  - Drain moves both directories. The drain reply also returns the behaviour drain dir.
  - Config: `online.server.behaviour_buffer_dir` / `behaviour_drain_dir`, defaulting to `buffer_dir + '_behaviour'` / `drain_dir + '_behaviour'`, mirroring `player.py:113`.
- **Invariants:**
  - `buffer_size` keeps counting logs only.
  - A submission without `'behaviour'` (an older client) still works: `msg.get('behaviour', {})`.
- **Acceptance:** a test drives the handler with temp dirs (submit → drain) and checks that every drained log has a sidecar of the same name.

### M2: Ordered dataset

Put this in a **new module** (e.g. `mortal/rl_dataset.py`), not as a rewrite of `FileDatasetsIter`.

For each drained log:
1. Load its sidecar with `tj.load_behaviour`.
2. For each **trainee** seat (`player_names=['trainee']`; champion seats are in the same log and are never trained on):
   - take `actions`, `masks`, `obs`, `at_kyoku`, `event_idx`, `grp`;
   - per-kyoku rewards from `RewardCalculator(standardize=False).calc_delta_pt(...)`, with `env.pts = [90, 50, -10, -130]`;
   - `tj.build(at_kyoku, rewards, horizon)`;
   - `b = tj.join_behaviour(event_idx, actions, player_id, sidecar)`.
3. **P0 advantage:** `tj.monte_carlo_returns` under horizon K. This equals each decision's kyoku reward, i.e. upstream's label without standardization.
4. Emit `(obs, action, mask, advantage, b)`, and **shuffle only after** targets are attached.

**Rules:**
- **A NaN `b`** means the entry gets weight 0 in the policy loss (agari guard, or a reaction the board voided). Keep a count. For the trainee (guard off) it should be ~0.
- **A log without a sidecar:** recommend skipping the file with a warning, plus a counter surfaced in tensorboard. Never train on it with a guessed `b`.
- **Memory:** follow the SL path. Use `obs_fp16` under AMP, keep `file_batch_size` small, and use 1–2 workers.
- **Units:** `env.pts` is u × 100, so rewards come out in u × 100, while 05 §3.2 says "u units". It's only a scale, but the critic, the SL kyoku-Δ head and search must share it. Pick one and write it down in the module docstring and the config.

**Acceptance:**
- Building twice from the same files gives bit-identical targets.
- `done` aligns with kyoku ends (K) or the game end (H).
- No seat's targets read another seat's data.
- `RewardCalculator` running statistics stay untouched (count stays 0).
- Every entry has a `b` or is flagged.

### M3: Trainer, P0 path

Make the changes in `train_online.py` behind **config keys whose defaults keep upstream behaviour**. This follows the `env.reward_standardization` precedent and keeps rebases easy.

Suggested keys:
- `policy.ratio_reference = 'snapshot' | 'stored'`;
- `dataset.ordered`;
- `policy.horizon = 'K' | 'H'`;
- `policy.advantage_norm` (per batch, on detached advantages).

With `'stored'`:
- ratio = exp(log π_θ(a) − log b), with `b` detached. The `Old_mortal` / `Old_policy_net` / `old_update_every` machinery is unused on this path.
- Log-probs and entropy come from `Categorical(logits=policy_net.logits(phi, masks))` (CLAUDE.md rule 4). `Categorical` clamps −inf logits, but check that the entropy stays finite under AMP.
- Normalize **advantages** per batch (detached), not rewards.
- Entries with NaN `b` contribute 0 to the policy loss.
- Log the ratio mean and p99, the clip fraction, entropy, the excluded fraction, and if possible the data's param version vs the current one (policy lag).
- Factor the PPO loss into a function so tests can call it directly.

**P0 config:** write `experiments/configs/p0.toml` from 05 §2.4:

| Setting | Value |
|---|---|
| `env.pts` | `[90, 50, -10, -130]` |
| Reward standardization | off |
| `clip_ratio` / `dual_clip` | 0.2 / 3.0 |
| `entropy_weight` | 1e-3 |
| LR (peak = final) | 2e-5 |
| `weight_decay` | 0 |
| `max_grad_norm` | 0.5 |
| `batch_size` | 512 |
| `explore_rate` | 1.0 |
| `baseline.train.state_file` | pilot A final |
| AMP and compile | on |
| GRP | GRP-v2 frozen |

Initialize by **copying** pilot A into a new run directory (e.g. `workdir/p0/`). Never write into `workdir/pilot/`.

**Gotcha:** `train_online.py` restores `state['steps']` (156,000 for pilot A) but loads the optimizer and scheduler only if the checkpoint's config was online. The run therefore starts with a fresh optimizer while the step counter reads 156K, and the `save_every` / `test_every` arithmetic uses that counter.

**Acceptance:**
- **05 §8 ratio test:** with learner weights equal to the actor's, the trainer's own loss code gives a first-minibatch ratio of 1 within 1e-4. Use a tiny network on CPU in FP32, with an arena run as in `tests/test_behaviour.py::arena_run`.
- Illegal actions have probability exactly 0 in the learner path.

### M4: Local smoke run

Server + 1 client + trainer on the workstation, at tiny scale:
- `train_play.games` 16–40;
- small `online.server.capacity`;
- **`force_sequential = true`**, so collection and training alternate;
- 1 dataloader worker;
- `test_every` set high (the online trainer exits after `test_play` at `train_online.py:400`, and `main()` respawns it).

**Check:**
- ≥ 2 parameter versions complete.
- Every drained log has a sidecar.
- The ratio mean ≈ 1 on each version's first minibatch. Expect AMP noise of ~1e-3, because actors compute `b` under AMP.
- The excluded fraction ≈ 0 for the trainee.
- Resident memory stays well under 15 GB (watch `free -g`).

This is plumbing, not an experiment: no registry rows, no strength claims.

### M5: K1 critic (P1)

**Spec:** 05 §4.
- A **separate b8c128** `Brain` whose input is the 1012 v4 channels plus the opponents' hand channels.
- Its own optimizer. The LR schedule is part of K1.

**Privileged channels:** the v4 `Invisible.encode` output (`libriichi/src/dataset/invisible.rs:152`) has 217 rows:

| Rows | Content |
|---|---|
| 0–50 | Opponents' hands: 3 opponents × 17 rows (tehai 4, aka 3, shanten 7 + 1, waits 1, furiten 1) |
| 51–216 | Wall order: yama 138, rinshan 8, dora 10, ura 10 |

Use rows 0–50. **Exclude rows 51 and above** (05 §4: exclude the wall order). This needs `GameplayLoader(oracle=True, trust_seed=True)`.

**Where GAE runs:** GAE needs critic values. The critic lives on the learner GPU, not in the dataloader workers. Recommended design:
1. Workers yield whole per-seat ordered trajectories.
2. The learner runs the critic over them with no grad.
3. It computes `tj.gae` targets (values detached).
4. Only then does it shuffle into minibatches.

Normalize advantages per batch, never per episode.

**B0 warm-up mode:**
- The policy is frozen and actors keep its parameters; only the critic trains.
- Train until the 10-bin calibration is monotone and explained variance has plateaued.
- 05 §4 asks for 0.3–0.5M hanchan, which is ~14–23 h at ≈ 22K hanchan/h here. **Ask before running it** (§5).

**Diagnostics:**
- explained variance;
- 10-bin calibration;
- the ratio of within-kyoku to across-kyoku advantage variance.

**Oracle observations never reach the policy or the actors** (CLAUDE.md rule 6). Add a test that the policy's input has exactly 1012 channels.

### M6: Registry fields

Add `arm`, `package` and `horizon` to `lab/registry.py` rows (02 §9). This can wait until the first RL checkpoint is evaluated.

## 5. Decisions

**Already made; don't reopen them:**
- D37 (interfaces) and D46 (sidecars and join semantics).
- The utility `[90, 50, -10, -130]`.
- GRP-v2 is frozen.
- The pilots are the P0/P1 initializations.
- The 05 §2.4 values are the P0 starting config.

**Ask the user before:**
- **Frozen items:** touching anything frozen (CLAUDE.md rule 2), or anything that contradicts 05.
- **P1 horizons:** how the two seeds split between K and H. 05 §3.1 / §5.1 say K1 compares them but don't say how P1's two seeds divide.
- **Runs:** starting any run beyond the M4 smoke test, including B0.
- **Spending:** any cloud spend.

## 6. Constraints that bite here

- **RAM (15 GB) is the binding limit.**
  - Outside the smoke run's sequential alternation, never run arena or lab jobs alongside training. An OOM kill takes out a dataloader worker and silently wedges the trainer.
  - When waiting on a background job, wait on its PID. A waiter that only watches for a success line hangs on a crash.
- **Rust changes:** probably none are needed. If you make any, rebuild with `PYO3_PYTHON` set and install the `.so` **by rename** (CLAUDE.md "Setup and build").
- **`Brain(..., Norm="GN")`** always.
- **Masks** come from the loader's `encode_obs`, which matches the actor's. Never recompute them differently in the learner.
- **Behaviour probability formula:** `explore_rate = 1.0` makes `b = π_actor(a)`. For other values see `MortalEngine.behaviour_dist`.
- **Kan selects:** kan-select entries carry a tile id as their label, at the same `event_idx` as the kan. `join_behaviour` handles this; keep it in mind when indexing.
- **`env.pts`:** never `[6, 4, 2, 0]`, which is the `config.example.toml` default (CLAUDE.md rule 3).

## 7. Out of scope

- Arms V and H, packages K2 and K3, and the KL anchor.
- Running the comparison, and any cloud work.
- Lab verdicts on RL checkpoints.

## 8. When you finish a session

- Update `STATUS.md`: the Interfaces row, next steps, and what was verified vs only attempted.
- Run the full checks in CLAUDE.md "Before committing".
- Commit only when asked.
