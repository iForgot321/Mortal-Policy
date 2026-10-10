# CLAUDE.md

Guidance for Claude Code (and humans) working in this repository.

## What this is

- A fork of [Nitasurin/Mortal-Policy](https://github.com/Nitasurin/Mortal-Policy) at `cf1f08e`, itself a fork of [Equim-chan/Mortal](https://github.com/Equim-chan/Mortal).
- **Goal:** the strongest open, verifiable Riichi Mahjong AI, evaluated on **riichi.dev**.
- **The plan lives in `docs/reference/`.** Read `00_ROADMAP.md` first (sequencing, gates, budget); facts are in `01_fact_base.md`, decisions in `08_decision_log.md`. If a change contradicts the plan, stop and ask. Changes to goals, gates, utility, metric, phase order or budget need a dated `08` entry *before* the docs are edited.
- `docs/outdated/` holds the superseded v1 plan and the previous CLAUDE.md. Don't follow them.

## Start of every session

1. **Read `STATUS.md`**: what's done, in progress and next. Update it at the end of the session (date, what changed, verified vs attempted, next step).
2. **Check the state** before changing anything:

   ```bash
   git status --short && git log --oneline -10
   tail -5 lab/registry/runs.jsonl                    # last lab runs
   ls ~/mj-data/splits/SPLITS.sha256                 # dataset frozen?
   find libriichi/src -name '*.rs' -newer mortal/libriichi.so | head   # output = stale .so
   nvidia-smi; free -g; ps aux | grep -E 'python|train' | grep -v grep # anything running?
   vastai show instances                             # live cloud boxes bill by the hour
   ps -eo pid,etime,args | awk '/[f]leet.py (watch|down)/'   # every live run needs its watcher (docs/design/handoff-*.md)
   ```

3. Report the state in a few lines before proposing work. If the state is ambiguous, ask. If a cloud run is
   live and its watcher is missing, restart the watcher first (the latest `docs/design/handoff-*.md` has the command).

## Machine

- **RTX 3060 12 GB, 16 cores, ~15 GB RAM, WSL2.** Not the 4090 fleet the plan budgets for (D50); that is cloud (D21).
- **RAM is the binding constraint.** Never run an arena/lab job concurrently with training: the OOM killer takes a dataloader worker and silently wedges the trainer. Featurized samples are ~137 KB each (fp32); keep worker buffers small, or use `dataset.obs_fp16` with AMP.
- Conda env `mortal` (Python 3.12, torch 2.6 cu124). **The base conda Python is 3.13**, so always build with `PYO3_PYTHON` set (below) or the `.so` won't import.

## Layout

| Path | What's there |
|---|---|
| `libriichi/` (Rust, PyO3) | Rules engine, mjai, obs encoding (`state/obs_repr.rs`, v4 = 1012×34), dataset loader (`dataset/gameplay.rs`), duplicate arena (`arena/`), agents (`agent/`, incl. our `shanten_greedy.rs`) |
| `mortal/` (Python, flat imports, run from here) | `model.py`, `train_offline_phase1.py` (AWR/BC), `train_online.py` (PPO), `train_grp.py`, `dataloader.py`, `reward_calculator.py`, `server.py`/`client.py`, `engine.py`, `one_vs_three.py` |
| `lab/` | **Evaluation lab** (`02`): GSPRT, calibration simulator, nonce ledger, registry. `python -m lab --help` from the repo root |
| `lab/registry/` | `runs.jsonl` (every lab run), `nonces.jsonl` (seed ledger), `ablation.md` (generated). Append-only, tracked in git |
| `tools/data/` | **Data pipeline** (`04a`): `download.py` → `convert.py` → `index.py` → `coverage.py` → `splits.py` → … ; `params.toml` is part of the dataset identity |
| `tools/fleet/` | **Rented GPU boxes** (`03`): Vast.ai CLI how-to (`README.md`), `fleet.py` (a run's learner + actors over SSH tunnels, D48), `setup_box.sh`, the benchmark driver `box_bench.sh`, benchmarks. Results in `experiments/fleet-host-benchmarks.md` |
| `tests/` | Python tests (`pytest tests/`) |
| `experiments/` | One markdown note per experiment; `experiments/configs/` holds the exact training configs (`MORTAL_CFG=...`) |
| `~/mj-data/` (outside the repo) | Tenhou data: `raw/` (immutable snapshot), `mjai/`, `db/`, `meta/`, `splits/`, `names/`, `reports/` |
| `workdir/` (ignored) | Checkpoints, tensorboard, lab game logs |

## Setup and build

```bash
export PYO3_PYTHON=$HOME/miniconda3/envs/mortal/bin/python LD_LIBRARY_PATH=$HOME/miniconda3/envs/mortal/lib
cargo build -p libriichi --lib --release
cp target/release/libriichi.so mortal/libriichi.so.new && mv mortal/libriichi.so.new mortal/libriichi.so
```

- **Install the `.so` by rename, never `cp` over it**: a running process has it mapped, and overwriting in place can crash it.
- Re-install after every Rust change; a stale `.so` is the most common silent bug.
- Training scripts run from `mortal/` (`config.toml` there, or `MORTAL_CFG=/path/to.toml`).

## Before committing

```bash
cargo fmt --all --check
cargo clippy -p libriichi --lib --all-features -- -Dwarnings   # 28 errors pre-exist (newer toolchain lints); add none
cargo test --workspace --release --no-default-features --features flate2/zlib
~/miniconda3/envs/mortal/bin/python -m pytest tests/ -q
```

Commit only when asked. Small, single-purpose commits.

## Project rules

1. **No strength claim without a lab verdict.** Every behaviour change is an experiment with a registry row, labelled *exploratory* or *confirmed*. Loss curves and test_play numbers are not verdicts. Rows made while `lab.toml` says `calibrated = false` are provisional.
2. **Frozen items need a decision-log entry to change:** utility vector, GRP checkpoint, observation schema (v4 through Phase 1, reviewed at G0: D49), rules config, seed families (`lab/seeds.toml`), lab test definitions (`lab/lab.toml`), data splits (`~/mj-data/splits`). Any featurizer change must keep arena and training paths hash-identical.
3. **Utility:** `env.pts = [90, 50, -10, -130]` (u × 100, D2). Never the repo default `[6, 4, 2, 0]`. `[90, 45, 0, -135]` is display-only.
4. **Masks must be identical** in actor, learner, KL, entropy and any loss. `CategoricalPolicy.forward` returns probabilities; use `CategoricalPolicy.logits` (masked raw logits) for losses, KL and ACH.
5. **Reward units.** Upstream `RewardCalculator` standardizes with running statistics; the plan uses fixed units for RL (D37). Know which path you're touching.
6. **Oracle observations stay out of play.** `Invisible`, `oracle=True`, `trust_seed=True` are for critics, labels and lab diagnostics only.
7. **Tenhou data** (D13, D23): never committed, never on community/peer cloud hosts, never redistributed. Only the recipe (`tools/data/`, `params.toml`, aggregate reports) and hashes are publishable. Known bot accounts live in `~/mj-data/names/excludes_bots.txt`.
8. **`server.py` has no authentication.** It binds to loopback only; actors reach it through restricted SSH tunnels (D48, `tools/fleet/fleet.py`).
9. **AGPL-3.0.** Keep license headers; publish the source of any deployed bot.
10. **Ethics.** Bots play only on riichi.dev. No features aimed at evading bot detection.

## Gotchas

- `Brain(..., Norm=...)` defaults to `"BN"`, but every script builds `Norm="GN"`. Always pass it and match the checkpoint.
- **Reduced-precision inference is batch-composition dependent.** In 1v3, challenger and champion run separate batches, so AMP (and TF32, which Ampere uses for "FP32" cuDNN convs by default) flip argmax at near-ties: 21/500 sets diverged under AMP, 2/500 under TF32, 0/500 in true FP32. That's zero-mean noise for A/B tests, but the **deterministic A/A must run with `--no-amp`** (D43).
- Set scores are summed in integer hundredths of u, so an A/A set is exactly 0 (in floats, 0.9 + 0.5 − 0.1 − 1.3 ≠ 0).
- `train_online.py`: the PPO ratio uses an `Old_mortal` snapshot, not stored behaviour log-probs; online mode exits after `test_play` (known hang) and `main()` respawns it.
- `optim.max_grad_norm = 0` means no clipping.
- `dataloader.py` shuffles individual decisions; every decision in a kyoku gets the kyoku's advantage (no ordered trajectories or critic yet, D37).
- `Tile::augment` swaps manzu↔pinzu only, deliberately (ryuuiisou).
- `GameplayLoader.validate_gz_log_files` replays all seats without featurizing (~3K games/s) and checks label legality. Full loading featurizes and is ~1000× slower.
- The NikkeTryHard converter omits the 4th `reach_accepted` in four-riichi abortive draws (quarantine code `Q4s`, kept).
- Upstream weights, tuned hyperparameters, opponent pools, replay buffers and multi-GPU code were never released.

## Working style

- Match the surrounding style: `rustfmt` + clippy for Rust; flat-module Python in `mortal/`.
- Prefer new modules (`lab/`, `tools/`, `tests/`) over large rewrites of upstream files, so rebasing stays easy.
- Never overwrite a checkpoint, dataset snapshot, split or registry. Frozen artifacts are made read-only.
