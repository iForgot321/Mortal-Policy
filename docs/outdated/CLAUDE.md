# CLAUDE.md

Guidance for Claude Code (and humans) working in this repository.

## What this is

- A fork of [Nitasurin/Mortal-Policy](https://github.com/Nitasurin/Mortal-Policy), forked at `cf1f08e` (2025-09-24). That is itself a fork of [Equim-chan/Mortal](https://github.com/Equim-chan/Mortal).
- **Goal:** the strongest open, verifiable Riichi Mahjong AI, evaluated on **riichi.dev**.
- **The project plan lives outside this repo**, in `~/mahjong/v2/`:
  - Read `00_ROADMAP.md` first.
  - Facts are in `01_fact_base.md`.
  - Decisions are in `08_decision_log.md`.
  - If a change contradicts the plan, stop and ask. Don't silently diverge.

## Start of every session: check the current state first

Some setup and plan steps have already been done. **Don't assume a step is pending, or done, without checking.** Before changing anything:

1. **Read `STATUS.md`** at the repo root, if it exists. It is the running log of what's done, what's in progress, and what's next.
2. **Inspect the repo:**

   ```bash
   git status --short && git log --oneline -15   # uncommitted work? recent progress?
   git remote -v                                 # fork remote + `upstream` (Nitasurin/Mortal-Policy)?
   ls -d lab deploy tests 2>/dev/null            # which project dirs exist yet
   ls mortal/config.toml 2>/dev/null && grep -c '/path/to' mortal/config.toml   # config present? placeholders left?
   grep -nE 'config.toml|data|checkpoints|logs' .gitignore                      # data/config ignored yet?
   ```

3. **Check the environment and build:**

   ```bash
   conda env list | grep -w mortal
   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   ls -l mortal/libriichi.so target/release/libriichi.so 2>/dev/null
   find libriichi/src -name '*.rs' -newer mortal/libriichi.so | head   # any output means the .so is stale: rebuild and re-copy
   (cd mortal && python -c "import libriichi; print('libriichi ok')")
   ```

4. **Check plan progress:**
   - In `~/mahjong/v2/08_decision_log.md`, see which **open decisions** are still open.
   - In the experiment registry (`lab/`), see which phase and step was last recorded.
   - Map both onto the phase map in `00_ROADMAP.md`.
5. **Report the state** in a few lines (done / in progress / missing) **before** proposing work.

**Rules:**
- Skip steps that are already done.
- Never overwrite an existing `mortal/config.toml`, checkpoint, dataset snapshot or registry. Back them up or ask first.
- If the state is ambiguous, ask instead of guessing. For example: a `.so` older than the Rust sources, uncommitted changes you didn't make, or a registry that disagrees with `STATUS.md`.
- **At the end of a session, update `STATUS.md`:**
  - date;
  - what changed (with commits);
  - what's verified vs only attempted;
  - the next step.

## Layout

| Path | What's there |
|---|---|
| `libriichi/` (Rust, PyO3) | Rules engine and mjai protocol (`mjai/`), observation encoding (`state/obs_repr.rs`; v4 = 1012×34), dataset loader (`dataset/gameplay.rs`; oracle obs in `dataset/invisible.rs`), duplicate arena (`arena/`), stats (`stat.rs`) |
| `mortal/` (Python, flat imports) | `model.py` (`Brain` ResNet + `CategoricalPolicy`; `GRP`) |
| | `train_offline_phase1.py` (AWR), `train_online.py` (PPO), `train_grp.py` |
| | `reward_calculator.py` (GRP per-kyoku reward), `dataloader.py` |
| | `server.py` / `client.py` (actor–learner), `player.py`, `engine.py`, `one_vs_three.py` (1v3 duplicate eval) |
| `exe-wrapper/`, `log-viewer/`, `docs/` (mdbook) | Upstream, rarely touched |
| `lab/` | **Added by us:** evaluation lab (GSPRT, calibration, panels) |
| `deploy/` | **Added by us:** riichi.dev adapter |
| `tests/` | **Added by us:** Python tests |

## Setup and build

Run only the steps the state check shows are missing.

```bash
conda env create -f environment.yml && conda activate mortal   # Python 3.12
pip install "torch>=2.4"                                        # not in environment.yml on purpose
cargo build -p libriichi --lib --release
cp target/release/libriichi.so mortal/libriichi.so              # Python imports this copy
cp mortal/config.example.toml mortal/config.toml                # or set MORTAL_CFG=/path/to/config.toml
```

- **Re-copy `libriichi.so` after every Rust change.** A stale `.so` is the most common silent bug.
- Run Python scripts **from `mortal/`**. Imports are flat, and scripts `import prelude` first.
- `config.example.toml` values are **placeholders**, not recommendations. `TRAIN_PLAY_PROFILE` selects a `[train_play.*]` profile.

## Before committing (mirrors upstream CI)

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -Dwarnings
cargo test --workspace --no-default-features --features flate2/zlib -- --nocapture
typos                          # config in typos.toml
pytest tests/                  # our Python tests
```

## Common entry points (from `mortal/`)

| Task | Command |
|---|---|
| GRP reward model | `python train_grp.py` |
| Offline SL/AWR | `python train_offline_phase1.py` |
| Online RL | `python server.py` on the learner, `python client.py` on each actor, `python train_online.py` on the learner |
| 1v3 duplicate eval | `python one_vs_three.py` (`[1v3]` config; supports akochan) |

## Project rules

1. **No strength claim without a lab verdict.**
   - Every behaviour change is an experiment and gets a registry row (`lab/`), labelled *exploratory* or *confirmed*.
   - "The loss went down" or "the test_play avg rank improved" is not a verdict.
2. **Frozen items need a decision-log entry to change:**
   - the utility vector;
   - the GRP checkpoint;
   - the observation schema (v4);
   - the rules config;
   - seed families;
   - data splits.

   Any featurizer change must keep the arena and training paths hash-identical. Add or update the parity test.
3. **Utility:** `env.pts = [90, 50, -10, -130]`.
   - Don't use the repo default `[6, 4, 2, 0]`.
   - The `[90, 45, 0, -135]` in `train_online.py` and `one_vs_three.py` is display-only.
4. **Masks must be identical** in actor, learner, KL, entropy and any custom loss.
   - `CategoricalPolicy.forward` returns **probabilities**, not logits.
   - Use the raw-logit accessor for ACH and KL work, and don't take `log` of masked probabilities.
5. **Reward units.** Upstream `RewardCalculator` standardizes rewards with running statistics. The plan uses fixed units (D37). Know which path you're touching, and never mix the two.
6. **Oracle observations stay out of play.** `Invisible`, `oracle=True` and `trust_seed=True` are for critics, labels and lab diagnostics only. They must never reach the inference path of an evaluated or deployed agent.
7. **Tenhou data:**
   - never committed;
   - never uploaded to community or peer cloud hosts;
   - never redistributed.
   - Keep data paths, `mortal/config.toml`, checkpoints and logs out of git (extend `.gitignore`).
   - Known bot accounts go in the `excludes` list.
8. **`server.py` has no authentication.** Bind it only to a private or VPN interface, never a public port.
9. **AGPL-3.0.**
   - Keep the license headers.
   - Any deployed bot's source must be published.
10. **Ethics.**
    - Bots play only on riichi.dev; no Tenhou or Mahjong Soul play.
    - No features aimed at evading bot detection.

## Upstream gotchas

- `Brain(..., Norm=...)` defaults to `"BN"`, but every script builds it with `Norm="GN"`. **Always pass `Norm` explicitly**, and match the checkpoint.
- In `train_online.py`:
  - The PPO ratio is computed against an `Old_mortal` snapshot, refreshed every `old_update_every` steps, not against a stored behaviour log-prob.
  - Online mode exits after `test_play` because of a known hang (see the `BUG` comment), and `main()` respawns the trainer. Don't remove the respawn without fixing the hang.
- `optim.max_grad_norm = 0` means **no gradient clipping**.
- `dataloader.py` shuffles individual decisions, and every decision in a kyoku gets the same advantage. There are no ordered trajectories and no critic yet.
- `Tile::augment` swaps **manzu↔pinzu only**. This is deliberate, because ryuuiisou is souzu-only. Don't extend it to all suit permutations casually.
- Action selection:
  - Training play samples (`explore_rate = 1.0`).
  - Test play and 1v3 use argmax.
  - With identical models and seeds, all four duplicate rotations are the same game, so the deterministic A/A must give Δ = 0 exactly.
- The upstream weights, tuned hyperparameters and parts of the online infrastructure (opponent pools, replay buffers, multi-GPU) were never released. Don't assume they exist.

## Working style

- **Make small, single-purpose commits.** Every experiment config should be reproducible from its commit plus its config hash.
- **Match the surrounding code style:**
  - Rust: `rustfmt`, plus `clippy` clean.
  - Python: the existing flat-module style. Don't add a package refactor without asking.
- **Prefer adding new modules** (`lab/`, `deploy/`, `tests/`) over large rewrites of upstream files, so rebasing on upstream stays easy.
