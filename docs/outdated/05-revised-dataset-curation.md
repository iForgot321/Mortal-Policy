I'll dig into this properly — first by inspecting your actual substrate (Mortal-Policy's ingestion pipeline) and the concrete data sources, since the curation plan's viability hinges on details like whether the mjai conversion preserves player identity. Let me start by cloning and inspecting the repos.The release CDN is blocked in my sandbox (fine on your machine — just a note). I'll verify name preservation from the converter source instead.One final check — the build instructions, so Stage 0 steps are concrete:# Investigation results: Stage 0/1, grounded in the actual code and data

I cloned and inspected both Mortal-Policy and the tenhou-to-mjai toolchain (release archives themselves are CDN-blocked in my sandbox, but the converter source settles the important questions). Several findings materially revise your curation plan — mostly in your favor.

## Finding 1: The player-filter infrastructure already exists end-to-end

This is the big one. Your curation doc treats player filtering as something to build; it's already plumbed through the whole stack:

- `config.example.toml` has `player_names_files = []` — plain text files, one username per line.
- `train_offline_phase1.py` (lines ~148–179) pre-filters the file list to games whose `start_game.names` intersect the whitelist, and caches it to `file_index.pth`.
- `libriichi/src/dataset/gameplay.rs` then generates training samples **only from whitelisted seats** — the full game is still simulated for context, but only the strong player's decisions become targets. This is strictly better than whole-game filtering: a game with one qualifying player among three weaker ones still contributes that player's decisions.

So Stage 1's biggest operation reduces to *producing the whitelist file*, plus a custom `file_index` builder if you want game-level predicates (more below).

## Finding 2: Suit augmentation is already built in — but it's only 2 of the 6 permutations

`Tile::augment` (tile.rs:154) swaps manzu↔pinzu and leaves souzu and honors fixed. `Event::augment` applies it across all event types including `dora_marker` and `ura_markers` — so the dora-consistency correctness concern in your curation doc is already handled, in Rust, behind a single config flag (`enable_augmentation`). The dataloader's epoch loop replays the file list once original, once swapped.

This changes the plan: don't build the mjai-JSON-level 6-permutation pipeline first. Flip the existing flag as your augmentation ablation (zero implementation cost), and only generalize `Tile::augment` to a parameterized permutation if the 2× version validates at Layer 2 and you want more. Given your own doc's estimate that the effective multiplier is ~1.2–2× regardless, the marginal value of 6 perms over 2 is probably small.

## Finding 3: Stage 0 has an undocumented prerequisite — the GRP

None of your planning docs mention this: offline Phase 1 AWR **requires a trained GRP** (global reward predictor) before it can run. `dataloader.py` loads `config['grp']['state_file']` and the `RewardCalculator` computes per-kyoku rewards as deltas of GRP-predicted expected placement pts (rank-probability matrix × the `pts` vector), normalized by running statistics. The GRP itself is small (GRU, hidden 64, 2 layers, trained by `train_grp.py` on the same mjai globs) — but it's a distinct training run with its own train/val globs in config.

Two consequences. First, the real Stage 0 order is: build libriichi → download dataset → **train GRP** → train offline Phase 1 → run 1v3 harness. Second, a curation-design decision: I'd train the GRP once on the *full unfiltered* archive and freeze it as part of your frozen assets. It predicts rank outcomes from score trajectories, not actions, so player skill barely matters to it, more data strictly helps it, and freezing it keeps every curation ablation single-variable (otherwise each dataset change perturbs the reward function too).

## Finding 4: The dataset is verified good, with two specific losses

The NikkeTryHard v2.0.0 release covers 2009–2026 (through Jan 2026), ~2.5M+ games, ~12GB, houou 4-player **hanchan only** — so the tonpuu-removal half of your hygiene operation is already done. Files are per-game gzip'd mjai, named by Tenhou log ID (`2024010100gm-...`), which embeds the datetime — your held-out-by-time split is trivial to construct from filenames alone.

Crucially, I verified in `convlog/src/conv.rs` that `start_game` carries the four **real Tenhou usernames** — the whitelist mechanism will work directly on this archive.

What the mjai conversion *drops* (confirmed in source: the intermediate tenhou-json model carries `connections`, `dan`, and `rate` fields, but convlog never emits them):

1. **Dan/rate at game time.** Fine — your plan computes skill from placements anyway, which is the right call since the mjai archive is self-sufficient for that.
2. **Disconnect (BYE) markers.** Your curation doc says "Tenhou flags these" — true in raw XML, false in the mjai archive. A disconnected player auto-plays tsumogiri, and if that seat is whitelisted, those junk decisions get attributed to a strong player. The practical fix is a heuristic during your stats pass: flag seats with long all-tsumogiri no-call stretches (say ≥10 draws) and drop those games from the file index. Since Mortal-Policy's loader can't exclude a seat per-game, dropping the whole file is the clean option, and it's rare enough to be cheap.

Two config gotchas: the example globs expect `*.json.gz` but the archive uses `.mjson` (gzip content, misleading extension) — the Rust loader uses `GzDecoder` regardless of extension, so just set the glob to `**/*.mjson`. And note `file_index.pth` caches the filtered list; delete it whenever the whitelist changes or you'll silently train on a stale index.

## Finding 5: The filter thresholds, made computable

Houou room pt structure is +90/+45/0/−15×(dan+2), and [90, 45, 0, −135] is the pt distribution for a 7-dan player in Tenhou houou hanchan (the room-dependent 1st/2nd rewards are 90/45 for houou, with the 4th-place penalty depending on current rank). Setting expected pt to zero gives the standard stable-rank closed form from placement rates: **stable dan = (6·p₁ + 3·p₂)/p₄ − 2**, so your "stable rank ≥ 7" threshold is simply (6·p₁ + 3·p₂)/p₄ ≥ 9. One contextual point worth internalizing: houou entry already requires 7-dan + R2000, so *everyone* in the archive was ≥7-dan when they played — your filter is selecting sustained above-entry performance, which is exactly what the Mortal author's private cleaning aimed at.

The stats pass itself is easy: names come from line 1 of each file, and final placements are computable per game (libriichi's `Grp` loader exposes `rank_by_player`, or a standalone one-pass Python/Rust scanner over score deltas — one pass over 2.5M files, embarrassingly parallel, well within your 1–2 day estimate). `Stat::from_dir` also gives you the full behavior battery (deal-in/agari/riichi/call rates) per player, which doubles as your Panel-3 reference bands later.

## Finding 6: The eval harness mostly exists

`libriichi::arena::OneVsThree` is seed-keyed and plays "sets" — 4 hanchan per wall seed with the challenger rotated through the seats, i.e., the duplicate protocol from your Stage 1.5 doc maps onto existing machinery via `one_vs_three.py` and the `[1v3]` config block (akochan support included). Your Stage 0 harness work is mostly the *analysis* side: wall-block bootstrap CIs and the A/A wrapper, not the arena itself.

## Revised execution order

**Stage 0 (this week):** conda env (Python 3.12) + torch, `cargo build -p libriichi --lib --release`, copy `libriichi.so` into `mortal/`. Download the 2020–2024 releases (~5.4GB, ~1.38M games). Fill in real config paths, set globs to `**/*.mjson`, train the GRP first, then Phase 1 AWR unfiltered as your baseline. Get one 1v3 run vs. akochan or a tsumogiri bot working end-to-end, even at small scale — the pipeline proof matters more than the numbers this week.

**Stage 1 (next):** (1) One-pass stats scan over the full 2009–2025 archive → per-player placement counts + tsumogiri-run disconnect flags; (2) apply stable-dan ≥ 7 ∧ avg placement ≤ 2.45 ∧ n ≥ 500 → whitelist file; (3) freeze held-out-by-player, held-out-by-time (filename dates make this trivial), and golden sets *before* retraining; (4) retrain on the whitelist (GRP stays frozen), run your three-layer validation; (5) flip `enable_augmentation` as the next single-variable ablation; (6) decision-level weighting last — that one genuinely requires new code (a loss-weight hook in `train_offline_phase1.py` keyed on decision context), and it's the only Stage 1 operation that does.

The net effect of this investigation: Stage 1 shrank. Two of the four operations (player filtering, augmentation) are config-plus-a-script rather than builds, hygiene is half done by the archive itself, and the one hidden addition is the GRP prerequisite in Stage 0. Want me to draft the one-pass player-stats/whitelist script next, or work through the GRP + Phase 1 config concretely?
