# Mahjax spike (Phase 0, 03 §7, D12)

- **Plan item:** `03` §7. Three questions, then a verdict for `08` D12:
  1. rule parity on 10K libriichi logs;
  2. throughput on a 5090 and an H100, random policy then a b8c192-class net;
  3. integration cost (the v4 encoder in JAX, or retraining on Mahjax's own observation).
- **Adopt (D12)** only if parity passes (or the differences are documented and acceptable),
  end-to-end throughput is ≥ 3× a libriichi box **per dollar**, and integration is ≤ 4 weeks.
- **Not a lab run.** No registry rows, no strength claims. Tenhou logs were read on the workstation
  only (D23); this note gives aggregate counts only.
- **Code:** `tools/mahjax/` (`parity.py`, `bench.py`, `net.py`, `select_patch.py`,
  `box_mahjax.sh`), plus a Python binding for `PlayerState.action_mask` in libriichi
  (`state/getter.rs`; no behaviour change). Results: `workdir/mahjax-spike/`.
- **Versions:** Mahjax 0.1.6 (`773e9ef`, 2026-09-20, Apache-2.0), JAX 0.11.2 (CUDA 12 wheels) in a
  separate conda env `mahjax` (Python 3.12). The `mortal` env is untouched. libriichi imports
  into it from `mortal/`.

## Summary

| D12 criterion | Result |
|---|---|
| Rule parity | **Not 100%, but close and fixable.** 19.5M decisions over 30K games (10K bot, 10K bot with random actions, 10K Tenhou human): 99.99946% identical legal sets. Mahjax has **three rule bugs** relative to Tenhou: furiten after a passed no-yaku winning tile, a honba ≥ 8 dealer-repeat cap, and riichi ankan on a waited tile. libriichi has **one deviation** of its own: it pays a triple ron, which Tenhou aborts. Mahjax's payouts match Tenhou's on every win, draw and final score in 10K Tenhou games. |
| Throughput per dollar | **Passes, by 7.5–12×.** Upstream Mahjax copies a 22 MiB table per environment (a `vmap` × `lax.cond` artifact), so a 12 GB GPU runs out of memory at 256 environments. With that fixed (`select_patch.py`, same results), the workstation's RTX 3060 runs env + observation + b8c192 at **22.3K steps/s ≈ 127K hanchan/h**, 5.8× libriichi on the same machine. At a rented 3060's $0.056–0.092/h that is $0.44–0.72 per million hanchan, against libriichi's best $5.4/M. The 5090/H100 runs were not done (they need a rental and wouldn't change the verdict). |
| Integration ≤ 4 weeks | **No (estimate 6–9 weeks).** The v4 encoder can't be ported hash-identically: ~12% of its channels come from libriichi's single-player EV search. Adopting Mahjax means a new observation schema (a frozen item), implemented twice (JAX for actors, Rust for the lab, SL loader and deploy), an SL retrain, a T-NI against v4, and a new actor→learner data path. |

**Verdict: D12 stands (stay on libriichi).** Details and the conditions that would reopen it are at the end.

## 1. Rule parity

### Method

`parity.py` replays a libriichi mjai log through Mahjax's `red_mahjong` env (hanchan, no uma,
default `GameConfig`, i.e. Tenhou rules) and checks the two engines at every step.

- **Walls.** Mahjax deals each kyoku's wall from the key passed to `env.step`; its wall layout is
  fixed (hands at 84 + 13·seat, live draws from 83 down, dora/ura at 9/8, kan indicators at 9 − 2k /
  8 − 2k, rinshan at 10 + k). The harness rebuilds each wall from the log (haipai, live and rinshan
  draws, dora and ura markers; unseen slots get the remaining tiles) and passes it in place of the
  key. Every Mahjax draw is checked against the log's `tsumo`.
- **Decisions.** Mahjax asks one seat at a time about a discard, in claim priority; the harness
  answers each question from the log (the logged call or ron, else PASS). libriichi's side is a
  `PlayerState` per seat fed the same log; `action_mask` gives its legal set after every event.
  Mahjax's legal set (its full per-seat mask rows after a discard, so priority hides nothing) is
  projected onto Mortal's 46 actions (37 discards, riichi, chi low/mid/high, pon, kan, agari,
  ryukyoku, pass) plus the kan-select tile set, and compared **per seat per decision**.
- **Outcomes.** Agari deltas (each ron of a multi-ron separately), exhaustive-draw deltas, every
  kyoku start (round, honba, kyotaku, dealer, scores, dealt hands), whether and when the game ends,
  and final scores (sticks left on the table go to the top).
- A disagreement that leaves the replay in step is counted and the replay goes on; one that
  doesn't (the log's action is illegal in Mahjax, or one engine ends the kyoku or game and the
  other doesn't) stops that game.

**Negative controls** (200 games each; the harness must notice a changed rule):

| Change to Mahjax | Detected as |
|---|---|
| Kuikae allowed (mutant `_discard_tile_mask_after_chi`) | 213 mask mismatches, all after chi |
| `enable_special_abortive_draw = 0` | 12 games diverge (kyuushu illegal, kyoku continues) |
| `allow_double_ron = 0` | 15 games diverge (every multi-ron game) |
| `enable_pao = 0` | nothing (no pao in 200 games) |
| `allow_kuikae = 1` | nothing: **Mahjax never reads this flag** (nor `allow_open_tanyao`) |

### Corpora

| Corpus | Games | Kyoku | Decisions compared | Identical | Games replayed to the end |
|---|---|---|---|---|---|
| Pilot A vs B, argmax (A/A′ run `20261005-124255-measure`, first 10K logs) | 10,000 | 105,381 | 6,334,331 | 6,334,299 (32 differ) | 9,991 |
| Hobbled: pilot A with ε = 0.1 random actions on one seat (`20261005-163506-T-std`) | 10,000 | 106,100 | 6,576,314 | 6,576,277 (37 differ) | 9,993 |
| Tenhou houou, 2024-03 (first 10K, quarantined games excluded) | 10,000 | 106,406 | 6,564,609 | 6,564,573 (36 differ) | 9,996 |

In all: 317,887 kyoku, **19,475,254 decisions, 105 different (99.99946% identical)**. A game that
stops early still counts its decisions up to that point. Every difference and every early stop
falls into one of the cases below. Mahjax steps per hanchan under realistic play: 632.

What the logs exercised (counts from the logs; libriichi decisions are per seat):

| Exercised | Pilot | Hobbled | Tenhou |
|---|---|---|---|
| Ron wins | 56,414 | 54,522 | 53,876 |
| Tsumo wins | 35,981 | 35,444 | 36,399 |
| Multi-ron discards | 762 | 683 | 624 |
| Wins with ura revealed | 39,253 | 38,415 | 38,039 |
| Exhaustive draws | 12,396 | 15,475 | 15,535 |
| Nagashi mangan | 3 | 0 | 17 |
| Kyuushu kyuuhai | 1,239 | 1,188 | 1,152 |
| Four-riichi draws | 12 | 8 | 0 (all in quarantine code Q4s, excluded) |
| Four-wind, four-kan and triple-ron draws | 140 | 173 | 78 |
| Riichi / chi / pon | 82,887 / 65,253 / 113,397 | 80,272 / 78,730 / 114,803 | 79,392 / 84,917 / 118,891 |
| Daiminkan / ankan / kakan | 235 / 4,690 / 3,280 | 422 / 4,910 / 3,451 | 668 / 4,952 / 3,482 |
| Games ended by tobi / into the West round | 2,795 / 306 | 2,897 / 285 | 2,607 / 275 |
| libriichi decisions offering agari / kan / riichi | 92,777 / 70,346 / 204,533 | 91,226 / 71,567 / 203,000 | 90,871 / 72,376 / 198,562 |

Pao never came up in a way that could tell the engines apart (the pao control showed nothing in
200 games); its payment code was read, not tested.

### Differences found

| Difference | Pilot | Hobbled | Tenhou | Matches Tenhou |
|---|---|---|---|---|
| Mahjax offers a ron that libriichi forbids: furiten after passing a winning tile that has no yaku | 32 decisions | 36 | 35 | libriichi |
| Mahjax stops dealer repeats at honba ≥ 8 (game stops) | 2 games | 4 | 4 | libriichi |
| Mahjax allows a riichi ankan of a waited tile | 0 | 1 decision | 1 | libriichi |
| Triple ron: libriichi pays all three, Mahjax aborts (game stops) | 7 games | 3 | 2 in the logs, Mahjax agrees | Mahjax |
| libriichi's four-riichi draw comes one tsumo late (accepted) | 12 | 8 | — | same outcome |

**Mahjax vs Tenhou (Mahjax bugs):**

1. **No furiten after passing a winning tile that has no yaku.** A tenpai hand whose wait is
   discarded but which can't ron (no yaku on that tile) is furiten until its next discard on
   Tenhou; libriichi implements this ("no-yaku furiten … following the behavior of Tenhou's
   furiten display", `state/update.rs`). Mahjax sets same-turn furiten only when RON was legal
   and declined, so it later offers a ron in the same go-around (often houtei, which supplies the
   yaku, or the other side of a kuitan wait). About 3.4 per 1,000 games, in bot and human play
   alike. **Evidence:** libriichi's validator replayed all 2.51M Tenhou games of dataset-v1 against
   its masks and found 3 games with an illegal action (Q5), all discards or a converter glitch,
   none a ron. At the Tenhou corpus's rate the dataset holds ~8,800 such situations; if Tenhou
   offered the ron, players would have taken some of them.
2. **Dealer repeats stop at honba 8.** Mahjax ends renchan once honba ≥ 8
   (`is_eight_consecutive_deals`). Tenhou has no such rule (its rule page lists 八連荘 as not used),
   and the Tenhou corpus has 4 games where the dealer keeps the deal at honba 9–10. About 3–4 per
   10,000 games.
3. **Riichi ankan of a waited tile.** After riichi, Mahjax allows an ankan when the set of wait
   *types* is unchanged by shape; with the fourth copy kanned, that wait can no longer come. Tenhou's
   rule is "リーチ後の暗槓は待ちが変わらない場合のみ", which libriichi implements as "cannot kan a
   waited tile" (`algo/agari.rs`). Seen once in the hobbled corpus and once in the Tenhou corpus
   (each time the player had just drawn its winning tile and won). Not checkable against Tenhou logs,
   since nobody kans a tile they can win on.
4. Not a rule difference, but misleading: `GameConfig.allow_kuikae` and `allow_open_tanyao` are
   dead flags.

**libriichi vs Tenhou:**

1. **Triple ron.** libriichi's arena applies every ron (`arena/board.rs`, "multi-ron applies every
   hora"); Tenhou aborts a triple ron (三家和了 is an abortive draw; the Tenhou corpus has 2, logged
   as a zero-delta ryukyoku straight after the discard). Mahjax follows Tenhou. 3–7 per 10,000
   bot games. The rules config is frozen (CLAUDE.md rule 2), so this is reported, not changed.
2. **Four-riichi abortive draw one draw late.** libriichi deals the next tile, discards that
   player's decision, then logs the draw; Tenhou and Mahjax end the kyoku when the fourth riichi is
   accepted. Same outcome, different log shape; the harness accepts it.
3. **Not tested:** draw tenpai with all four copies of the only wait in the concealed hand. Tenhou
   counts it noten (形式聴牌あり, but not 純手牌で4枚使用); libriichi counts karaten as tenpai and
   neither engine special-cases the four-in-hand case, so parity would not show it.

**Agreement.** Outside these cases the engines agree everywhere the harness looks: every legal set,
every agari and exhaustive-draw delta (including double rons, nagashi mangan, ura and kan dora), every kyoku transition (renchan, honba, kyotaku, the all-last dealer stop, the West round,
tobi) and every final score. On the Tenhou corpus, Mahjax's payouts equal Tenhou's own on every
win, draw and final score.

**Harness cases, not rule differences** (handled in `parity.py`): a seat that could ron but chose
to chi is asked about the ron first in Mahjax (it passes, then gets the chi); Tenhou logs a triple
ron as a ryukyoku (every ron-capable seat declared it).

## 2. Throughput

### 2.1 The memory bug

Upstream Mahjax on the 3060 (random policy, `bench.py --upstream-cond`): 3.6K env steps/s at 16
environments, 5.3K at 64 (1.8 GB), 5.7K at 128 (3.2 GB), out of memory at 256. XLA's memory
analysis of one batched step: 22.4 MiB of temporaries **per environment**, almost all of it
`u32[B, 1953125, 3]`, a per-environment copy of the yaku table (`yaku_cache.npz`), plus the hand
table.

Cause: under `vmap`, a `lax.cond` whose predicate differs per environment becomes a select, and JAX
first broadcasts every operand of the cond to the batch size, closed-over constants included. The
step has conds around code that reads the tables (e.g. the next kyoku's deal inside `env.step`).
Mahjax's authors hit the same thing in `Shanten.discard` (their comment: "74GiB temp at batch 32")
and fixed it locally.

`select_patch.py` runs Mahjax's `lax.cond` / `lax.switch` as explicit selects, which is what the
batched cond computes anyway, keeping `lax.cond`'s output dtypes and weak types. JAX code is pure,
so results are identical; the parity runs use it. Temporaries fall from 22.4 to ~0.1 MiB per
environment (128 environments: 2,861 → 13 MiB).

### 2.2 RTX 3060 (workstation)

Random legal policy, hanchan, auto reset, 100 steps per jitted `lax.scan`, timed after a warm-up.
**Steps per hanchan under realistic play: 632** (6,320,341 Mahjax steps over the 10K pilot games,
PASS questions included); hanchan/h below = steps/s × 3600 / 632. Under `vmap` every step computes
every branch, so steps/s does not depend on the policy.

| Configuration | 256 envs | 1,024 envs | 2,048 envs | 4,096 envs |
|---|---|---|---|---|
| Upstream Mahjax (env only) | out of memory | — | — | — |
| Patched, env only | 34.5K steps/s | 39.4K (224K hanchan/h) | 39.1K | 39.1K |
| Patched, env + Mahjax observation | 30.7K | 34.4K (196K/h) | 34.2K | 34.4K |
| Patched, env + observation + **b8c192** (bf16) | 19.2K (109K/h) | **22.3K (127K/h)** | 22.1K | — |
| Patched, env + observation + b16c256 (bf16) | 12.0K (68K/h) | 13.8K (79K/h) | — | — |
| `no_red_mahjong`, env only (not Tenhou rules; reference) | — | 58.9K | 57.4K | — |

Peak GPU memory stays ≤ 1 GB in every patched row. JIT compile takes 15–30 s per configuration.

Reference on the same machine: libriichi 1v3 with pilot A (b8c192) on both sides, argmax, AMP,
16 threads: ~22K hanchan/h.

The observation used with the nets is Mahjax's own (`net.py`: hand, reds, discard shanten, waits,
per-discard waits, tiles seen, dora, scores, round state), tiled out to v4's 1012 channels. So the
observation cost is real and the network sees v4's input shape, but this is not v4.

### 2.3 Per dollar

The bar is libriichi's best measured cost, **$5.4 per million hanchan** (EPYC 9684X + RTX 4090 at
$0.464/h, 3 actor processes, eval mode; `experiments/fleet-host-benchmarks.md`). 3× that is
**≤ $1.8/M**. Mahjax needs almost no CPU, so cheaper low-vCPU offers qualify. Vast single-GPU
prices on 2026-10-06 (reliability > 0.98): RTX 4090 from $0.37/h (median $0.47), RTX 5090 from
$0.48/h (median $0.62), H100 SXM from $1.74/h (median $2.87, 4 offers). The bar is then
≥ ~205K hanchan/h on a $0.37 4090, ≥ ~270K/h on a $0.48 5090, ≥ ~970K/h on a $1.74 H100.

**A rented 3060 already clears the bar.** Vast lists RTX 3060s at $0.056–0.092/h (26 offers,
median $0.092). At the measured 127K hanchan/h with b8c192, that is **$0.44–0.72 per million
hanchan: 7.5–12× cheaper than libriichi's $5.4/M**, against the 3× required. Caveats:

- The measurement is the workstation's 3060 under WSL2, not a rented one.
- The observation is Mahjax's, not v4. A JAX port of v4 without its single-player tables is the same
  kind of array work and should cost about the same; the tables themselves are not portable (§3).
- At that rate one 3060 makes ~22K decisions/s (all four seats), about 6× what a 4090 learner
  consumed in the P0 test (3.75K samples/s). With Mahjax, the learner, not the actors, bounds
  the fleet; RL-6's extra harvested seats are free.
- On a GPU-only simulator the actor network is most of the cost: b16c256 runs at 0.62× b8c192 here,
  against ≤ 6% on libriichi boxes, where the CPU arena dominates. The actor-size choice (D31) would
  become a real throughput trade-off.

### 2.4 Not measured: 5090 and H100

The plan asks for both. They would refine the per-dollar figure but not change the verdict: the
cheapest rentable GPU already clears the bar, and integration, not throughput, is what fails.
`box_mahjax.sh` runs the same benchmarks on a rented box (no repo, no checkpoints, no Tenhou data:
only the three scripts and the pinned Mahjax source) and destroys it; a 5090 and an H100 for
~45 min each would cost about $2–3. Not run: renting needs the user's go-ahead.

## 3. Integration cost

What adopting Mahjax for actors would take, beyond the parity fixes:

| Work | Estimate | Notes |
|---|---|---|
| Fix the three rule bugs and the memory bug; rerun parity | ~1 week | Small, local changes; candidates for upstream PRs |
| **Observation** | 4–6 weeks + compute | See below |
| JAX actor network + per-version weight sync from the torch learner | ~0.5 week | `net.py` has the layers |
| Actor → learner data path (trajectories, behaviour probabilities, GAE/critic inputs) from JAX instead of mjai logs + `GameplayLoader` | 1.5–2 weeks | Replaces the P0/P1 transport built on 2026-10-06 (`rl_dataset.py`, sidecars) |
| **Total** | **≈ 6–9 weeks** | Against D12's ≤ 4 weeks |

**The observation is the blocker.** v4 (1012 × 34) has ~889 channels of table state that port to
JAX without trouble, and ~123 channels from the single-player tables (EV, win and tenpai
probabilities per discard over 17 turns, required tiles) computed by libriichi's recursive,
memoized search (`algo/sp/calc.rs`, ~1,000 lines). That search has data-dependent branching and
no fixed shape; a hash-identical JAX port (CLAUDE.md rule 2) is not realistic. The alternative is
a new schema without them (v4 minus the SP channels, or one built on Mahjax's observation). That is:

- a frozen-item change (decision-log entry);
- two implementations kept hash-identical: JAX for the actors, Rust for the lab arena, the SL loader
  and the riichi.dev bot, which sees only an mjai stream and can't build a Mahjax state;
- SL teachers and student retrained on it, and a T-NI against the v4 student before RL builds on it.

The strength cost of dropping the SP channels is unmeasured. Retraining the student on Mahjax's
own observation instead adds a sequence model (its history is a 200-action list) on top of all of
that.

## Verdict

D12 stands: **stay on libriichi.** Parity would pass after three small Mahjax fixes, and
throughput per dollar clears the bar several times over once the memory bug is patched, but
integration is well over the 4-week allowance because of the observation encoder.

The throughput result is strong enough that the observation question is worth keeping open:
the simulator would cut actor cost by roughly an order of magnitude and move the bottleneck to
the learner.

**Reopen if** (in addition to D12's quarterly re-survey):
- actor throughput becomes the binding constraint at G0 (the fleet soak, `03` §3.7), **and**
- an SL A/B shows a JAX-portable observation (v4 without the SP channels) is non-inferior to v4
  (T-NI). That A/B is the cheapest way to shrink the blocker and is worth running on its own if
  the observation is revisited for other reasons.

Integration would then be the JAX encoder, the rule fixes and the data path, about 4–5 weeks,
close to the allowance; the 5090/H100 per-dollar numbers (§2.4) would then decide.

## Artifacts

`workdir/mahjax-spike/`: `parity-*.json` (summaries with examples), `round1/` (first pass, before
the harness fixes), `bench-3060-*.json`, `parity-control-*.json`, `tenhou-exclude.txt`,
`run_parity.sh`, `run_round2.sh`.
