# 01: Fact Base

**Verified 2026-09-29/30.** This is the single place for facts the plan depends on.

**How facts were verified:**
- Code facts: checked against `Nitasurin/Mortal-Policy` HEAD `cf1f08e` (2025-09-24) and `Equim-chan/Mortal`.
- Everything else: web research, with sources in §9.

**Tags:**
- **[V]** verified from a primary source.
- **[calc]** computed.
- **[hearsay]** community report only.
- **[inf]** inference.

Re-verify quarterly. Update this file, not the stage docs, when a fact changes.

---

## 1. The substrate: Mortal-Policy, as it actually is [V]

| Area | Fact | File |
|---|---|---|
| Network | `Brain` is a Conv1d ResNet: kernel 3, channel attention, pre-activation, GroupNorm or BatchNorm, Mish. The tail is Conv1d → 32 channels → flatten 32×34 → Linear(1024). | `mortal/model.py` |
| Heads | `CategoricalPolicy` only: 1024 → 256 → tanh → 46 logits, masked. **No value or critic head.** A `DQN` class exists but online training doesn't use it. | `model.py`, `train_online.py` |
| Obs | v4 obs is 1012×34. Oracle obs (`Invisible`) is 217×34 and holds opponents' hands plus the wall, dora and ura. The wall is exact when the arena seed is trusted. | `libriichi/src/consts.rs`, `dataset/invisible.rs` |
| Obs content | Each player's kawa (discard pile) is encoded in order: first 6 plus up to 18 slots, tedashi flags, riichi markers, recency decay `exp(−0.2·turns_ago)`, last tedashis. Also single-player EV / win-probability / tenpai tables per discard (`encode_sp_table`). **Cross-player interleaving is not encoded explicitly.** | `state/obs_repr.rs` |
| Action space | 46 = 37 (discard or kan choice, including aka) + riichi + chi×3 + pon + kan + agari + ryukyoku + pass | `consts.rs` |
| Params [calc] | b24c512 ≈ 42.1M; b40c192 ≈ 10.9M | from `model.py` |
| SL (Phase 1) | **AWR**: weight = clamp(exp(A/β), max = `awr_clip`), where A = standardized per-kyoku GRP reward | `train_offline_phase1.py` |
| Reward | Per-kyoku Δ of GRP-expected pts (`rank_prob @ pts`). The last row is the one-hot final ranking, so **summed over the whole hanchan** the rewards telescope to u(final) − E_GRP[u]. That is potential-based shaping with a frozen potential. **Policy preservation holds only for full-hanchan returns.** Upstream treats each kyoku as terminal, which is a surrogate objective (`05` §3.1). Rewards are standardized by a running mean and std, so the units drift; v2 uses fixed units (D37). Every decision in a kyoku gets that kyoku's reward, and decisions are shuffled individually. | `reward_calculator.py`, `dataloader.py` |
| GRP | 2-layer GRU (hidden 64) over [grand_kyoku, honba, kyotaku, 4 scores]; a 24-way softmax over rank permutations → per-player rank probabilities | `model.py`, `train_grp.py` |
| Online RL | PPO with dual clip (clip 0.2, dual 3.0), plus entropy. The ratio is taken against an `Old_mortal` snapshot refreshed every `old_update_every` steps, not against a stored actor log-prob. **No KL-to-prior.** | `train_online.py` |
| Online data | `OneVsThree.py_vs_py(trainee, 3× frozen baseline)`; only `trainee` seats are trained on; exploration via `explore_rate` (1.0 = sample) | `player.py`, `server.py` |
| Replay server | Buffer with `capacity`, `sample_reuse_rate`, `sample_reuse_threshold`, `force_sequential` | `server.py`, config |
| Player filter | `[dataset] player_names_files` → `GameplayLoader(player_names=…)` keeps only named seats. `excludes` is also supported. | `dataset/gameplay.rs` |
| Augmentation | `enable_augmentation` / `augmented_first`. `Tile::augment` swaps **m↔p only**, leaving souzu and honours fixed (2× data). This preserves ryuuiisou (souzu-only). | `tile.rs`, `mjai/event.rs` |
| Arena | `OneVsThree` duplicate: 4 seat rotations per seed; walls deterministic per (seed, kyoku, honba); `ako_vs_py` runs akochan; `Stat` gives a battery of behaviour statistics | `arena/one_vs_three.rs`, `one_vs_three.py` |
| Board | `arena::Board` has public `haipai`, `yama`, `rinshan`, `dora_indicators`, `ura_indicators`, `scores`, `kyoku`, `honba`, `kyotaku`, "so the caller will be able to set" them. This makes world reconstruction for search possible. | `arena/board.rs` |
| Eval action | Test play is argmax (`explore_rate=0`) | `engine.py` |
| Config defaults | "Dummy placeholders". `pts = [6,4,2,0]` (linear rank). Test display uses [90, 45, 0, −135]. `[1v3]` default is 2000 × 500 = 1M games. | `config.example.toml` |
| Withheld | Weights, hyperparameters, "some online training features" (opponent pools, replay buffers, multi-GPU). BPPO is "coming soon". | README; #91 |
| Author's net | Per #91, Nitasurin used "a more lightweight model network" **instead of** a ResNet-1D-CNN. Size undisclosed. | discussion #91 |
| License | AGPL-3.0-or-later (code); CC BY-SA 4.0 (assets) | LICENSE |

## 2. Mortal reference results [V]

- **Evaluation scale.** Version-to-version 1v3 duplicate evaluations use **1,000,000 hanchan** (250K seeds × 4). Evaluations against akochan use 13–24K games, because akochan is slow.
- **Throughput.** "Up to 40K hanchans per hour" on an RTX 4090 + Ryzen 9 7950X at game batch 2000. That is about 118 kyoku/s [calc; ~10.6 kyoku per hanchan].
- **Version gaps are small.** Challenger (×1) avg rank vs champion (×3):

  | Comparison | Challenger avg rank | Δ |
  |---|---|---|
  | 4.1c vs 4.0 | 2.4852 | 0.015 |
  | 4.1a vs 4.0 | 2.4945 | 0.0055 |
  | 4.0 vs 3.1 | 2.4820 | 0.018 |
  | 3.1 vs 3.0 | 2.4768 | 0.023 |
  | 3.1 vs 1.0 | 2.4976 | 0.0024 |

  In pt/game on [90, 45, 0, −135]: 4.1a vs 4.0 = +0.42; 4.0 vs 3.1 = +1.23.
- **Against akochan:** Mortal 3.1 (×3) vs akochan (×1): akochan avg rank 2.5677 and −5.67 pt/game; Mortal +1.89 pt/game.
- **Oracle guiding:** Mortal used VLOG-style oracle guiding in v1/v2 and **removed it in v3** (commit `fee1778`, 2023-01-09). Equim, #102: "it didn't bring improvements in practice". Auxiliary tasks "didn't change anything". "Months of PPO" gave no gain over the SL baseline.
- **Weights:** not public (gist, 2022-08-19).
- **Nitasurin's evidence** (#91; re-verified 2026-10-01, D28):
  - "value-based online achieved only a very small performance improvement compared to offline": **≈ +0.6 PT in 500K hanchan (1v3). This is value-based, not policy-based.**
  - "after converting to Policy-based online, the performance improvement exceeded expectations". **No number given.**
  - His top-ranked setup: **"Policy-based online + behavior cloning from value-based offline-to-online Mortal-V2 samples"**.
  - "AWR … is a new addition when I decided to open-source, replacing Behavior Cloning." Withheld: "opponent pools, replay buffers, multi-GPU parallelism".
  - "conducting 100,000 games for a single weight version is unacceptable in terms of time cost". He used "a more lightweight model network"; "online training does not require as large a parameter scale as offline".
  - Offline-only V2 was about +1.8–2.0 pt vs Akagi community weights in 200K hanchan.
- **Equim** (#91): policy-based methods (pre-v1) "all failed to outperform the baseline", a supervised model, after "several months" of tuning. "Simple value-based methods turned out to be effective."
- **Implication:** the released AWR→PPO is a starting implementation, not the recipe that produced Nitasurin's results (D28).
- **Value-based path is public** [V, Mortal `mortal/train.py`]: offline training with CQL (`min_q_weight` 5) and an auxiliary next-rank loss (0.2). Online mode drops CQL. It uses the same client/server, GRP and libriichi arena as Mortal-Policy.
- **The "~50M policy vs ~10M value games" figure** is from commenter hyskylord, not the author. It is confounded, because the value run was stopped early.

## 3. Ecosystem [V]

### riichi.dev (RiichiLab), the target venue

- **History:** went live 2026-03-09. mjai.app shut down 2026-04-30, and its data was deleted.
- **Connection:** bots are **self-hosted over WebSocket** (`wss://game.riichi.dev/ws/ranked` with a bot token). Observations arrive base64-encoded and are decoded with the `riichienv` package. A validation game is required first.
- **Rules engine** [V, riichi.dev docs note; riichienv 0.4.10 source and a local run, 2026-10-08]:
  - The game server is built on **riichienv-core** (Rust). The `riichienv` Python package wraps the same core, and the docs say local simulation "matches the online server exactly".
  - riichienv's default rule is the Tenhou preset: Rust `GameRule::default()` = `default_tenhou()`, and `RiichiEnv(rule=None)` uses it.
  - Under that rule a **triple ron is an abortive draw**: `ryukyoku`, reason `sanchaho`, no payments. A ron on a riichi-declaration discard cancels the deposit, including in that case.
  - `default_mjsoul()` and the Python no-argument `GameRule()` pay all three instead.
  - Reproduce: `tools/rules/riichienv_sanchaho.py`. The server's exact rule object isn't published; this assumes it uses the default.
- **Bots per account:** at most **4 at a time** [user, 2026-10-12].
- **Time limit:** **3 s grace per action plus a 15 s bank per kyoku.** A timeout plays tsumogiri or pass, and "timeouts are never penalized". An illegal action is chombo.
- **Rating:** OpenSkill Weng-Lin Plackett–Luce, with μ₀ = 25, σ₀ = 8.333, τ = 0.3. Displayed as 1500 + 20·(μ − 3σ). Placement only; score is not used. "Stabilizes around ~50 games."
- **Board, 2026-09-29:**
  - 279 active bots.
  - #1 zero-test2 2091; #2 牛魔 2071; #3 Mootal 2062.
  - Mortal-v4b #44 at 1864.
  - No BigCoach or Gokujan bot in the top 50.
- **Old mjai.app:** 2 s per move, raised to 3 s in Mar 2025; CPU-only; Elo. Nitasurin's baseline "Niitsune" was #1 there, at 2213 over 7,848 games (Feb 2025).

### Data

- **NikkeTryHard/tenhou-to-mjai** (HF `nikketryhard/tenhou-houou-mjai`):
  - Houou 4-player **hanchan only**, 2009-02-01 → 2026-01-23, about 2.5–3.2M games.
  - Data license: **disclaimer only, not CC BY**. A stale NOTICE file says otherwise.
  - `start_game.names` is present. The mjai files carry **no dan or rate**.
  - Since 2026-09-29 the repo also bundles **`tenhou-original-xml.tar.zst`** (18.6 GB): SQLite DBs for 2009–2025 holding the raw mjlog, with the `UN` tag (**dan and rate at game start**) and `BYE` disconnect markers.
  - It also bundles **Mahjong Soul Jade/Throne shards** for 2019–2026 (≈ 30.8 GB).
  - **The README's per-year counts are unreliable.** The 2024 shard actually holds **184,424 games**, with all 12 months present (≈ 15K/month).
  - **Measured over the whole snapshot (rev `6450bb58`, 2026-10-03): 2,512,433 games**, 2009-02-20 → 2026-01-23, all months present, ≈ 1.66B decisions (all seats). 2025 holds 178,897 games. The source DBs end mid-October 2025. Details: `~/mj-data/reports/DATASET_CARD.md`.
  - **Disconnects** (measured from the source DBs): 3.5% of games in 2009 and **8.6% in 2021** have a `BYE`. While disconnected, Tenhou auto-plays tsumogiri only (100% of 77,886 discards in 2021), with no calls and no wins.
  - **Player skill spread in houou** (measured): σ_b ≈ 0.020–0.025 u/game; shrinkage k ≈ 1,100–1,700 games; one-season split-half reliability ≈ 0.11. R at game start in 2021: mean 2150, SD 59 (the room gate is 7-dan and R ≥ 2000).
  - Local audit of the 2009 shard: 6,897 games; 2 placeholder games; 8 of 67,331 kyoku transitions with a score-continuity break; ≈ 5.3 KB per game compressed; about 450 games/s/core to parse. Details in `04a` §0.
  - Contains placeholder-name games, e.g. `EXAMPLE`.
- **Tenhou terms** (tenhou.net/sc/raw) prohibit using logs to develop products that compete with Tenhou, and prohibit public log-download services. houou-logs: "Tenhou prohibits redistribution of game logs."
- **Re-download:** `scrawYYYY.zip` archives return 404 and `list.cgi` covers the current year only, so **2009–2025 can't be re-downloaded from Tenhou.**
- `hhim8826/tenhou-houou-mjai` (2015–2026) has names removed, so it can't be used for player filtering.
- `MahjongRepository/phoenix-logs` is archived; `Apricot-S/houou-logs` (MIT) is its successor.

### Simulators

- **libriichi**: Rust, CPU; the Mortal substrate.
- **Mahjax** (JAX, arXiv 2605.20577, May 2026, Apache-2.0):
  - Tenhou 4-player rules with aka, validated against Tenhou logs.
  - About 1M steps/s (with red fives) on 8×A100, with a random policy. The paper claims >10× libriichi.
  - The RL demo is single-kyoku, no-red rules. **No strength comparison with any external agent.** API "provisional".
  - [V, paper HTML, read 2026-10-08] **Observation:** a structured dictionary for Transformer agents, ego-centric: tokenized hand indices, action history, scalars such as shanten and scores. **No look-ahead or EV features** (nothing like Mortal's single-player tables).
  - [V, same] **RL demo:** behaviour cloning on 500K samples, then PPO with KL regularization for 100M env steps (≈ 5.8 h on one GH200); over 1,000 games against three BC opponents, average rank better than 2.5. Tabula rasa learning is left to future work.
- **pgx:** no riichi; only Sparrow Mahjong.
- **Mjx:** build broken.
- **RiichiEnv (smly):** Rust, active.

## 4. Tenhou scoring [V]

- **Hanchan pt:**
  - Houou: +90 / +45 / 0.
  - Tokujou: +75 / +30 / 0.
  - **4th-place penalty depends only on dan:** 1-dan −45, 2 −60, 3 −75, 4 −90, 5 −105, 6 −120, **7 −135**, 8 −150, 9 −165, 10 −180.
  - Houou 7-dan = [90, 45, 0, −135]. This is Mortal's display pt, and it is zero-sum.
- **Stable dan** (Tenhou manual):
  - Houou: ((60·n1 + 30·n2)/n4 − 20)/10.
  - Tokujou: ((50·n1 + 20·n2)/n4 − 20)/10.
  - For a uniform 25% placement split: houou 7.00, tokujou 5.00.
- **AIs are not allowed in Houou.** AI records (Suphx, NAGA, LuckyJ) are measured in **特上 (tokujou)**.

## 5. riichi.dev-matched utility [calc]

- The Plackett–Luce update is Δμ_i ∝ 1{placed} − Σ_{q ≤ rank_i} e^{μ_i/c}/C_q.
- At equal strength this gives per-rank weights **+0.750 / +0.417 / −0.083 / −1.083**. The gaps between adjacent ranks are 1 : 1.5 : 3, so 4th is heavily penalized.
- Scaled so 1st = 0.90, this is **u = [+0.90, +0.50, −0.10, −1.30]**. It sums to 0; per-game SD under a uniform placement split ≈ 0.831.
- Against a weaker table the weights flatten; at strength ratio 1.5 the gaps are 1 : 1.4 : 2.3. u is fixed at the equal-strength vector by decision D2 in `08_decision_log.md`; revisit if the agent becomes dominant.
- **Consequence:** linear average rank (`pts = [6,4,2,0]`) is the wrong objective for this ladder. Two agents with the same average rank can settle about 26 display points apart.

## 6. Frontier agents

| Agent | Record | Method | Style notes |
|---|---|---|---|
| **LuckyJ** (Tencent AI Lab) [V] | Tenhou 10-dan on 2023-05-30 after 1,321 games, zero demotions. Stable **10.68 in 特上** over 1,145 games; avg rank 2.262 ± 0.032; bootstrap CI ≈ 8.8–13.1 [calc]. Inactive since. | **No human data**; self-play with **ACH** (Actor-Critic Hedge); imperfect-information search with "optimistic value estimation" whose **results are fed as features into the policy network, in training and at play time**; "symmetry training"; "mixed reward". No paper. | Emergent "human-like" flexibility: gives up shanten for shape, value or safety when quiet 4.0% of the time (NAGA 2.1%, humans 3.5%); avg win 6,287 (Suphx 5,785); calls 35.5%; honitsu 8.1% of wins |
| **Suphx** (MSRA, 2019) [V] | 10-dan; stable 8.74 over 5,760 games | SL: discard 76.7%, riichi 85.7%, chi 95.0%, pon 91.9%, kan 94.0%. 50 blocks × 256 channels. RL on about 1.5M games per agent with 44 GPUs × 2 days. GRP reward. Oracle guiding. pMCPA (not used on Tenhou). **Look-ahead features:** 100+ 34-wide channels from a depth-first search over its own draws and discards, ignoring opponents (the same idea as Mortal's v4 single-player tables); no ablation reported. | Approx. ablation from Fig. 8: SL 7.66 → RL 8.06 → +GRP 8.24 → +oracle 8.30 |
| **NAGA** (Dwango Media Village) [V] | 10-dan on 2021-07-30 after 26,598 games; stable 7.23 | Supervised CNNs on top-table logs; later a single model. Styles ("types") are **imitation models of specific strong players**, e.g. Omega = 100% pro 渡辺太, Kagashi = one heavy-calling player. | Maximum-acceptance, efficiency-first |
| **BigCoach → 極雀 Gokujan** [V/hearsay] | Site claim: "Balance" played 1,747 Houou games as **unauthorized bots** (e.g. 未払い) for a stable dan of about 10.6 | Paid review site with an anonymous operator. [hearsay] Mortal code plus Houou imitation, then **hundreds of millions of self-play hanchan** with hand-tuned rewards. Families: RIGEL balanced, NOVA first-place-seeking, HALO 4th-avoiding (HALO 1.1 is a voting ensemble), ZENITH MoE, COMET call-heavy. | Its million-game head-to-heads all fall within 2.476–2.508 avg rank. "Human-like" is a community label meaning hard to detect. It is linked by model names to a Mahjong Soul cheating tool; shared operator unproven. |

### Style reference panel [calc, 特上 logs]

LuckyJ 1,080 games; Suphx 500; NAGA 500; humans = the other seats at LuckyJ's tables.

| Metric | LuckyJ | Suphx | NAGA | Humans |
|---|---|---|---|---|
| Win % | 22.94 | 22.79 | 24.11 | 21.02 |
| Deal-in % | 10.29 | 10.59 | 12.16 | 13.94 |
| Call rate % | 35.53 | 32.31 | 34.57 | 31.40 |
| Riichi rate % | 19.27 | 19.44 | 20.82 | 18.62 |
| Dama wins, % of wins | 9.7 | 12.0 | 10.6 | 13.3 |
| Avg win (pts) | 6,287 | 5,785 | 5,887 | 5,870 |
| Honitsu, % of wins | 8.1 | 6.0 | 5.6 | 5.1 |
| First discard is an honour, % | 45.5 | 52.4 | 61.4 | 57.8 |
| First discard is a 2–8 tile, % | 13.6 | 8.1 | 6.3 | 8.3 |
| Shanten-backward discards, quiet, % | 4.01 | 3.64 | 2.10 | 3.55 |
| Shanten-backward discards, facing riichi, % | 24.6 | 23.2 | 22.4 | 21.0 |

## 7. Algorithms the plan uses [V]

- **ACH (Fu et al., ICLR 2022).**
  - Logits y track cumulative advantage (Hedge ≈ neural weighted CFR).
  - Loss: −c·η·y_a·A/π_old + (α/2)(V−G)² + β·Σπ log π.
  - Gate c: when A ≥ 0, require ratio < 1+ε and y_a − ȳ < l_th; when A < 0, require ratio > 1−ε and y_a − ȳ > −l_th.
  - One gradient step per batch.
  - OLSS table values: ε = 0.5, λ = 0.95, lr = 2.5e-4, γ = 0.995, value coef 0.5, batch 8192, β = 5e-4, l_th = 8.
  - Results: beat PPO by 0.39 fan in 1v1 mahjong. JueJong vs champion: +0.82 ± 0.96 (p = 0.19, not significant).
  - Theory covers 2-player zero-sum only.
- **Asymmetric critics.**
  - A state-only V(s) is biased. V(h, s) is unbiased (Baisero & Amato, AAMAS 2022). V(h, i) with *partial* privileged information i is also unbiased and can match or beat full state (Ebi et al., ICML 2026).
  - PerfectDou (PPO + GAE with a perfect-information critic plus oracle reward shaping) beat DouZero with about 10× fewer steps. Ablation: perfect critic 0.524 win probability vs imperfect 0.486.
- **Sequential testing.**
  - Fishtest GSPRT: α = β = 0.05, stopping at LLR ±2.944.
  - LLR ≈ N/(2s²)·[(x̄−μ0)² − (x̄−μ1)²].
  - Expected N ≈ 8.67·σ²/δ² at the midpoint and 5.3·σ²/δ² at μ1, vs 10.8·σ²/δ² for a fixed-N test.
- **Variance reduction.**
  - AIVAT (heads-up no-limit: 85% SD reduction).
  - MIVAT (regression on post-chance features; smaller gains in 6-player).
  - **MJ-DLVAT** (CoG 2024): 45.5% fewer games for mahjong rank estimation.
  - Community riichi-eval control variates: variance ratio about 0.85.
- **Search.**
  - SPARTA (Hanabi, blueprint plus one-ply Monte Carlo, deviate above a threshold): 24.08 → 24.53.
  - LBS (learned belief): 4.6× faster, about 90% of the gain.
  - JPS (bridge): search on 1–5% of training games, distilled.
  - Pluribus: biased continuation strategies at the leaves.
  - ReBeL and Student of Games are infeasible here: information states can't be enumerated, and the guarantees are 2-player only.
  - pMCPA: 100K rollouts per round, which is infeasible online.
- **Human-regularized play.**
  - piKL: π ∝ τ·exp(Q/λ).
    - Chess: human accuracy 53.2 → 54.3%, and it wins 92% against the imitation model.
    - Go: 57.8 → 58.5%, winning 94.6%.
  - DiL-piKL (Diplomacy): pure self-play started from humans still drifted.
  - Maia-2: skill-aware; KataGo human-SL net has a piKL λ knob.
- **Distillation.** Policy distillation: a 4× smaller student reached 108% of its DQN teacher. Kickstarting: about 10× fewer steps.
- **Ataraxos** (Sokota, Vinitsky, Hu, Kolter, Farina, arXiv 2511.07312, Nov 2025): superhuman Stratego for "a few thousand dollars" (16 H100 × 1 week, plus 4 H100 × 4 days for the belief net).
  - **Search-based self-play was tried and dropped:** "the slowdown on data generation imposed by the search methods we tried outweighed potential benefits"; "quantity of games matters more than quality of data".
  - **Coordinated schedules:** "strong regularization and aggressive updates when the policy is weak", the reverse when strong. Magnet-KL and LR follow power-law annealing (e.g. KL 0.05/iter^0.3).
  - **Test-time search:** belief network → depth-limited rollouts → one tabular magnetic-mirror-descent step with the same reverse KLs as training. The move is **sampled** from the updated policy.
  - Unverified from the retrieved text: an ablation showing that removing regularization severely hurts search (the appendix was truncated).
- **Scaling in board games** (Jones, arXiv 2104.03113): trade-offs between model size, train compute and test-time compute, measured in AlphaZero/Hex. It motivates measuring learning curves early (D30, D31), but it is not a mahjong scaling law.
  - Commonly quoted from the paper body (secondary, a search summary; **not** in the abstract, not re-read): about +500 Elo per 10× train compute on every board size, and 10× train compute ≈ 15× test-time compute.
- **Scaling laws for AlphaZero** (Neumann & Gros, arXiv 2210.00849; Connect Four, Pentago) [V, read 2026-10-08]:
  - Playing strength (Bradley–Terry γ) ∝ N^0.87–0.88 in parameters and ∝ C^0.55 in training compute for compute-optimal agents; power law up to the perfect-play plateau. "Optimal training stops long before convergence."
  - That is ≈ +66 Elo per doubling of training compute [calc: 400 × 0.55 × log₁₀2].
- **Hand-crafted features vs compute** (2026-10-08; context: the SP ablation, `experiments/sp-ablation.md`):
  - **KataGo** (Wu, arXiv 1902.10565, Table 2) [V]: without its Go-specific inputs (liberties, komi parity, pass-alive regions, ladder features), reaching the same Elo at ≈ 2 days (2.5G self-play queries) took **1.55× the training time**. They "contribute noticeably to the learning speed, but account for only a small fraction of the total improvement". Other ablations: global pooling 1.60×, ownership/score targets 1.65×, playout-cap randomization 1.37×, auxiliary policy targets 1.30×, forced playouts + policy-target pruning 1.25×.
  - **AlphaGo** (Nature 2016): SL policy accuracy 57.0% with all hand-crafted inputs vs 55.7% with raw board + history (secondary, a search summary of the paper). AlphaGo Zero later used raw stones only.
  - **Stockfish NNUE** (2020): about half the nodes/s of classical Stockfish, +80–100 Elo (secondary: chessprogramming.org, chess.com; not checked against Fishtest).
  - **Search vs knowledge** (Junghanns & Schaeffer, IJCAI 1997) [V, abstract]: "neither extreme is feasible"; the trade-off point is game-specific, and extra search depth shows diminishing returns.
  - Mahjong: Suphx's look-ahead features and Mortal v4's SP tables were never ablated upstream (§6, §2); LuckyJ feeds opponent-aware search results into its policy (§6); Ataraxos dropped search from data generation (above).
- **α-Rank** (Omidshafiei et al., Sci. Rep. 2019): multi-agent evaluation that accounts for matchup structure. It motivates the population panel (D34).

## 8. Compute calibration

| System | Compute and scale | Result |
|---|---|---|
| Suphx | 44 GPUs × 2 days per agent; about 1.5M RL games | Stable 8.74 |
| JueJong / ACH | 800 CPUs, 8 M40 | Parity with champion (not significant) |
| OLSS blueprint | 8 V100 + 1,200 CPUs × 2 days | +0.14–0.22 fan from search |
| PerfectDou | 880 cores, 8 GPUs | Beat DouZero |
| BigCoach | Hundreds of millions of hanchan [hearsay] | ≈10.6 [site claim] |
| Ataraxos (Stratego) | 16 H100 × 1 week + 4 H100 × 4 days; "a few thousand dollars" | Superhuman; no search in data generation |
| Mahjax RL demo | 100M env steps ≈ 5.8 h on one GH200 (BC init, PPO + KL) | Beats its BC opponents (avg rank < 2.5, 1,000 games); no external comparison |
| This project | $10–15K cloud budget (D21, D42). Phase 2 in tranches: T1 = eval + learner + 3 actors; T2/T3 = eval + learner + candidate learner + 4 actors (4090/5090, ≥ 32 vCPU). Forecast ≈ 270–290M generated hanchan (D30). | Target L5 |

## 9. Sources

- **Code**
  - github.com/Nitasurin/Mortal-Policy
  - github.com/Equim-chan/Mortal
  - mortal.ekyu.moe/perf/strength.html
  - Mortal discussions #91 and #102; issues #50 and #112
  - gist.github.com/Equim-chan/cf3f01735d5d98f1e7be02e94b288c56
- **Arena**
  - riichi.dev/docs/ranked, riichi.dev/docs/rating, riichi.dev/leaderboard
  - github.com/smly/mjai.app (README; discussions #181, #198, #203)
- **Data**
  - github.com/NikkeTryHard/tenhou-to-mjai (LICENSE-DATA)
  - huggingface.co/datasets/nikketryhard/tenhou-houou-mjai
  - github.com/Apricot-S/houou-logs
  - tenhou.net/sc/raw, tenhou.net/man
- **Simulators:** github.com/nissymori/mahjax; arXiv 2605.20577
- **LuckyJ**
  - giinex.tencent.com/api/articles/23
  - news.qq.com/rain/a/20230711A03ZZ500
  - haobofu.github.io
  - nodocchi.moe
  - modern-jan.com/blog/luckyj_vs_naga_and_suphx/
- **ACH:** openreview.net/forum?id=DTXZqTNV5nW
- **OLSS:** proceedings.mlr.press/v202/liu23k.html
- **Suphx:** arXiv 2003.13590
- **NAGA:** dmv.nico/en/articles/mahjong_ai_naga/; note.com/naga_project/n/n61a012f27261
- **BigCoach / Gokujan**
  - gokujan.com (legal/terms, posts/models-overview, posts/model-stats-balance, data/model-head-to-head.v2.json)
  - note.com/qwert227/n/n09846f8ef1c9
- **Algorithms**
  - arXiv 1612.06915 (AIVAT); ijcai.org/Proceedings/09/Papers/326.pdf (MIVAT)
  - github.com/official-stockfish/fishtest (LLRcalc.py; wiki "Fishtest mathematics")
  - arXiv 2203.16406 (PerfectDou); arXiv 2105.11674 (Baisero & Amato); arXiv 2509.26000 (Ebi et al.)
  - arXiv 1912.02318 (SPARTA); arXiv 2106.09086 (LBS); arXiv 2008.06495 (JPS)
  - arXiv 2112.07544 (piKL); arXiv 2210.05492 (DiL-piKL); arXiv 2409.20553 (Maia-2)
  - arXiv 1511.06295 (policy distillation); arXiv 1803.03835 (kickstarting)
  - github.com/vivekjoshy/openskill.py; github.com/twu3202/riichi-eval
  - arXiv 2511.07312 (Ataraxos); arXiv 2104.03113 (Jones, scaling with board games); nature.com/articles/s41598-019-45619-9 (α-Rank)
- **Features vs compute, RL scaling** (2026-10-08)
  - ar5iv.labs.arxiv.org/html/1902.10565 (KataGo); ar5iv.labs.arxiv.org/html/2210.00849 (Neumann & Gros)
  - ar5iv.labs.arxiv.org/html/2003.13590 (Suphx look-ahead features); arxiv.org/html/2605.20577 (Mahjax observation and RL demo)
  - mlanthology.org/ijcai/1997/junghanns1997ijcai-search/ (search vs knowledge)
  - chessprogramming.org/Stockfish_NNUE; chess.com/news/view/stockfish-absorbs-nnue-100-elo
  - augmentingcognition.com/assets/Silver2016a.pdf (AlphaGo, Nature 2016)

## 10. Cloud pricing and hardware performance (prices fetched 2026-10-06; re-verify before each phase)

The previous snapshot (2026-10-01) is in git history. Rows not re-checked on 2026-10-06 say so.

**Sources:**
- getdeploying.com per-GPU summaries (19–57 providers each, data dated 2026-10-06);
- the Vast.ai `bundles` API: the **complete** listing of rentable single-GPU offers, collected by bisecting price bands under the API's 64-offer cap. The 2026-10-01 figures came from one ≤ 64-offer sample per query, so the two snapshots are not strictly comparable. Offer counts moved by up to ±30% between queries minutes apart. On-demand medians were stable to ±$0.01; interruptible medians moved by up to $0.06.
- Runpod, Lambda, Verda (formerly DataCrunch) and CloudRift pricing pages; Lium, TensorDock and Salad via getdeploying and gpus.io;
- Hetzner's GEX131 press release (2025-12-11); gcloud-compute.com (updated 2026-10-04); instances.vantage.sh (AWS us-east-1, updated 2026-10-06);
- B&H and bestvaluegpu.com (RTX 5090 purchase, October 2026);
- aime.info and lambdal/deeplearning-benchmark (ResNet-50 AMP; not re-checked).

### GPU rental, $/GPU-hour

"Cheapest" can come from a multi-GPU listing quoted per GPU.

| GPU | Cheapest | Median on-demand | Spot median | vCPU per GPU |
|---|---|---|---|---|
| RTX 4090 | $0.33 (Salad, stock unverified); $0.34 in stock (GPU.ai, Runpod Community) | $0.44 | $0.34 | Vast median 18; 30% of 162 offers have ≥ 32 (median $0.50, lower quartile $0.43). Runpod fixed at 6. |
| RTX 5090 | $0.35 (HyperAI) | $0.69 (Vast $0.63) | $0.48 | Vast median 24; 40% of 198 offers have ≥ 32 (median $0.67, lower quartile $0.60). Runpod fixed at 9. |
| L40S | $0.55 (Novita) | $1.57 | $0.98 | 4 (AWS) to 32 (Vast); mostly 8–16 |
| A100 80GB | $0.40 (Vast, 2× SXM listing) | $1.80 | $0.80 | 8–30 |
| H100 SXM/PCIe | $1.30 (Lium) | $3.49 (Vast single SXM median $2.27, 3 offers; Runpod SXM $3.49 Secure / $2.69 Community) | $2.00 | 16–28 (Lambda 26, Verda 30) |
| H200 | $2.09 (Beam, stock unverified); $3.29 in stock | $4.59 | $3.34 | 15–24 (Verda 44) |
| B200 | $3.75 (Packet·ai) | $6.79 | $4.78 | 15–28 |

**The planning price.** The `00` planning price ($0.55/h, the 2026-10-01 Vast median) now sits between today's ≥ 32-vCPU medians: 4090 $0.50, 5090 $0.67. A 5090 at the median is about 22% above plan. Changing the planning price is a budget change and needs an `08` entry first.

### Consumer-GPU providers against the ≥ 32 vCPU actor requirement (`03` §2)

| Provider | 1× 5090 $/h | vCPU per GPU | Tier | Notes |
|---|---|---|---|---|
| Vast.ai | Median $0.67 with ≥ 32 vCPU (4090: $0.50) | 32+ on 40% of offers | Marketplace | Interruptible ≥ 32-vCPU medians: 5090 $0.53, 4090 $0.40. Bandwidth priced per host (median $9/TB up, max $55/TB); storage median $0.27/GB-month, billed while stopped. Most offers are Docker containers; about 8% allow VMs. |
| Vast.ai Secure Cloud | $0.67–1.30 (9 offers with ≥ 32 vCPU); 4090 $0.76–0.83 (3 offers) | 32–48 (EPYC Rome to Genoa) | Vetted datacenters (ISO 27001 and/or Tier 3/4) | Thin supply |
| Lium | $0.58 (32 vCPU, 84 GB) | 32 | Decentralized marketplace (Bittensor) | |
| Theta EdgeCloud | $0.70 | 32 | Community network | 31 GB RAM |
| CloudRift | $0.65 ($0.55 on a 3-month reservation) | Not published | Datacenter | |
| TensorDock | $1.23 (spot $0.48), plus $0.003 per vCPU-hour | Configurable | Marketplace | |
| GPUhub; Nova Cloud | $0.46; $0.69 (spot $0.33) | 25; 22 | Datacenter | Below 32 |
| Runpod | $0.69 Community, $0.99 Secure | 9 (4090: 6) | Both tiers | Below 32 |
| Salad | $0.50 (spot $0.25) | 4 | Home machines | Below 32; not secure-tier |
| HyperAI, Novita, SwissGPU, Beam, Sesterce | $0.35–0.72 | 12–16 | Mixed | Below 32 |

### Secure-tier options for Tenhou-data work (D23)

| Option | Price | CPU | Notes |
|---|---|---|---|
| Vast Secure Cloud, 1× 5090 | $0.67–1.30/h | 32–48 vCPU | See above |
| Hetzner GEX131 (RTX PRO 6000 Blackwell Max-Q, 96 GB) | €889/mo or €1.42/h at launch (press release, 2025-12-11). A 2026-08 third-party review lists €1,199/mo + €599 setup; **current price not confirmed** | Xeon Gold 5412U, 24 cores / 48 threads; 256 GB | Hetzner-owned datacenters |
| Verda | RTX PRO 6000 $2.08 (spot $1.04); H100 SXM $3.77 (spot $1.89) | 30 | SOC 2, EU |
| Lambda | H100 PCIe $3.29; H100 SXM $4.29; GH200 $2.29 | 26; GH200 64 (ARM Grace) | GH200 would need an aarch64 build of libriichi |
| Runpod Secure | 5090 $0.99; H100 SXM $3.49 | 9; 20 | 5090 below the CPU requirement |

### CPU (per SMT thread-hour)

| Option | Price | $/thread-hour |
|---|---|---|
| OVH RISE-XL (EPYC Turin 9455, 96 threads) | $354/mo + $354 setup (2026-10-01; not re-checked: the page renders prices client-side) | **$0.0051** |
| Hetzner AX162-1 (EPYC 9454P, 96 threads) | $722/mo + $359 setup; hourly billing at $1.16 capped at the monthly price (2026-10-01; not re-checked: third-party listings conflict) | $0.0103 (LTD offer: $0.0053) |
| GCP c3d-highcpu-180 spot, us-central1 | $1.64/h (on-demand $6.72) | $0.0091 |
| GCP c3d-highcpu-180 spot, australia-southeast1 (cheapest region) | $0.76/h | **$0.0042** |
| AWS c7a.48xlarge spot | $3.42/h | $0.018 per *physical core* (192 cores, no SMT) |

### Training throughput relative to an RTX 4090 (ResNet-50 AMP, with torch.compile; not re-checked)

| GPU | Relative throughput |
|---|---|
| RTX 5090 | 1.58× |
| A100 | 1.55× |
| RTX PRO 6000 | 1.69× |
| H100 PCIe | 2.08× |
| H100 NVL | 2.73× |
| H200 | 3.85× |
| B200 | No CNN gain over H200 |

**Throughput per $/h** [calc]: a 4090 = 1,872 images/s scaled by the factors above, divided by today's median on-demand price (H100 NVL: Vast median $2.90).
- 4090 ≈ 4,250; 5090 ≈ 4,290.
- A100 ≈ 1,610; H100 NVL ≈ 1,760; H200 ≈ 1,570.
- So consumer GPUs still give about **2.5× more training throughput per dollar**.

**Small, launch-bound Conv1d inference:** unverified. The estimate is that an H100 is about 1.0–1.3× a 4090, because the 4090 has higher clocks and the host's single-thread speed dominates.

### Purchase prices (2026 memory shortage)

- RTX 5090: about **$5,000** new (B&H $4,999, October 2026; marketplace listings up to $9,500), about $3,800 used. MSRP $1,999. It was about $7,000 on 2026-10-01.
- Not re-checked since 2026-10-01: RTX 4090 about $4,400 new, $3,200–4,300 used; RTX PRO 6000 about $15,000; DDR5 128 GB $2,700–3,900.
- **Rent vs buy** [calc]: renting one 5090 box 24/7 for 12 months costs about $4.8K at the planning price or $5.9K at today's ≥ 32-vCPU median. Buying needs the card (about $5K) plus a host with ≥ 32 threads and RAM, plus power, offset by resale value. The 2026-10-01 version of this line assumed a $7K card, so the gap has narrowed. The D21 reopen trigger (purchase prices near MSRP) has not fired: $5K is 2.5× MSRP.
