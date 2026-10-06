# 04a: Data Pipeline Runbook (Phase 0)

This is the step-by-step version of `04_data_and_sl.md` §1–4: download → snapshot/hash → convert → index → clean up → audit → score players → freeze splits → dataset card.

- **Where it runs:** the personal workstation, which counts as a machine you control under D23 (`00`, Phase 0).
- **Effort:** about **2–3 working days** including writing the code, about 3–6 hours of compute on 8–16 cores, and about **85 GB of disk** (250 GB free recommended).

The facts below were checked on 2026-10-01 against the HF repo and a local audit of the 2009 shard.

---

## 0. What the source actually contains [V, 2026-10-01]

**Repo:** HF `nikketryhard/tenhou-houou-mjai` (last modified 2026-09-29), license field "other", a disclaimer only (`01` §3).

| Asset | Size | Content |
|---|---|---|
| `tenhou-houou-2009.tar` … `tenhou-houou-2026.tar` (18 shards) | ≈ 18.1 GB total (**the README's per-year game counts are unreliable**: 2024 actually holds 184,424 games, not about 335K) | Uncompressed tars of per-game `<tenhou-log-id>.mjai.json.zst` files |
| `tenhou-original-xml.tar.zst` | 18.6 GB (about 21 GB unpacked) | **SQLite** DBs `2009.db` … `2025.db`, which are the upstream raw logs. These hold the mjlog XML, including the `UN` tag (names, **dan, rate**, sex) and `BYE` (disconnect) markers. Inspect the schema with `sqlite3 2024.db .schema`. |
| `majsoul-jade-*.tar`, `majsoul-throne-*.tar` (2019–2026) | ≈ 30.8 GB | Mahjong Soul ranked games in mjai. **Not used in v1 of the dataset** (§9). |

**Per-game files:**
- Named `YYYYMMDDHHgm-<type>-0000-<hash>`, for example `2024010100gm-00a9-0000-005a39ba`.
- `00a9` is the houou 4-player hanchan type code. All 6,897 games in 2009 have it.
- The date and hour come free from the file name.

**Game counts (README):**

| Year | Games | Year | Games | Year | Games |
|---|---|---|---|---|---|
| 2009 | 8k (6,897 measured) | 2015 | 170k | 2021 | 210k |
| 2010 | 80k | 2016 | 180k | 2022 | 280k |
| 2011 | 115k | 2017 | 190k | 2023 | 310k |
| 2012 | 135k | 2018 | 195k | 2024 | 335k |
| 2013 | 150k | 2019 | 190k | **2025** | **198k** (partial? audit in §5) |
| 2014 | 160k | 2020 | 245k | 2026 | 14k (January only) |

Total ≈ 3.16M.

**Format, from a local audit of the 2009 shard:**
- `start_game` = `{names[4], kyoku_first, aka_flag}`. No seed, dan or rate.
- Terminal events (`hora`, `ryukyoku`) carry `deltas`. `reach_accepted` carries only `actor`, so the 1,000-point deposit is implicit.
- About 1,086 events per game; 10.76 kyoku per game on average (max 34); about 5.3 KB per game compressed.
- 377 distinct players; 91 with ≥ 100 games.
- Python parse speed is about 450 games/s per core, so a full pass over 3.16M games is about 2 core-hours.

**Known defects (2009):**
- 2 placeholder games, with names `EXAMPLE`, `EXAMPLE2`…: 34 kyoku, all ryukyoku, no hora.
- **8 of 67,331** kyoku transitions where the start-of-kyoku scores don't equal the previous scores plus deltas minus riichi deposits. Classify these (§4, Q4).
- The tar entries carry epoch-zero timestamps; extract with `tar -xmf`.

**Loader compatibility:** Mortal-Policy's `GameplayLoader.load_gz_log_files` reads **gzip only** (`GzDecoder`). Files must be converted (§2).

---

## 1. D1: Download and snapshot (about 1–2 h, bandwidth-bound)

```bash
# encrypted volume, never synced to community/peer cloud hosts (D23)
export MJ=~/mj-data && mkdir -p $MJ/{raw/hf,db,mjai,meta,splits,names,reports}

# pin the exact repo revision
python -c "from huggingface_hub import dataset_info as d; print(d('nikketryhard/tenhou-houou-mjai').sha)" > $MJ/raw/HF_REVISION
hf download nikketryhard/tenhou-houou-mjai --repo-type dataset \
   --revision "$(cat $MJ/raw/HF_REVISION)" \
   --include "tenhou-houou-*.tar" "tenhou-original-xml.tar.zst" "README.md" \
   --local-dir $MJ/raw/hf

cd $MJ/raw && sha256sum hf/* > MANIFEST.sha256 && sha256sum MANIFEST.sha256 > MANIFEST.id
chmod -R a-w $MJ/raw          # raw is immutable from here on
```

- **Dataset identity:** `MANIFEST.id` plus the pipeline git commit plus the parameter file (§8). This identity is the only thing ever published (D13).
- **Backup:** copy `raw/` to an encrypted external drive, or to private, encrypted object storage you control. The data can't be re-downloaded from Tenhou.
- **Skip the Mahjong Soul shards** for now (§9).

## 2. D2: Extract and convert (about 1 h on 8–16 cores)

**Extract to scratch:**
```bash
tar -xmf hf/tenhou-houou-YYYY.tar -C $MJ/scratch/
```

**Convert each game:** `<logid>.mjai.json.zst` → `mjai/YYYY/MM/<logid>.json.gz` (gzip level 6), in parallel with `multiprocessing`. The directory layout matches Mortal's `globs`.
- Alternative: patch libriichi to read zstd. Rejected for now, because it creates fork drift for no gain.

**Count check:** per-year file counts must match the extracted counts, and roughly match the README table. Any decompression failure → quarantine (Q1).

**Unpack the SQLite DBs:**
```bash
tar --zstd -xf hf/tenhou-original-xml.tar.zst -C $MJ/db
```
Record `.schema` for each year in `reports/db_schema.txt`.

Delete `scratch/` afterwards. Expected disk: about 25 GB for the gzipped mjai, plus about 21 GB for the DBs.

## 3. D3: Index → per-game and per-seat tables (about 30–60 min)

One deterministic script, `tools/data/index.py`, which writes Parquet.

**`meta/games.parquet`** (one row per game):
- `log_id`, `ts` (from the ID), `year`, `month`, `type_code`, `aka_flag`
- `names[4]`, `n_kyoku`, `n_events`
- `final_scores[4]`, `final_ranks[4]`
- `content_sha256` (of the decompressed stream, for deduplication)
- `xml_found`, `dan[4]`, `rate[4]`, `bye_seats` (from the DB)

**`meta/seats.parquet`** (one row per game × seat):
- `log_id`, `seat`, `name`, `final_score`, `rank`, `u` (`02` §2)
- `dan_at_start`, `rate_at_start`, `disconnected`
- Optional counts for the style-panel reference bands: riichi, agari, houjuu (deal-in), calls.

**Implementation rules:**
- **Final ranks come from libriichi's own code path** (`GameplayLoader` → `take_grp()` → `rank_by_player`). Never compute them ad hoc. Tenhou tie-breaks by starting seat order, and leftover riichi sticks go to 1st. Using the training code path guarantees the GRP, the reward and the player scores all agree.
- **XML join:**
  - For each `log_id`, pull the mjlog from the DB and parse the `UN` tag. Names are URL-encoded. dan is an index: 16 = 七段, 17 = 八段, 18 = 九段, 19 = 十段, 20 = 天鳳位.
  - Assert that the decoded names equal the mjai `names`, after NFC normalization on both sides.
  - Record any `BYE`.
  - 2026 games have no DB rows, so set `xml_found = false`.
  - **Use a real XML parser, not regexes.** The 2026-10-01 regex prototype miscounted disconnects that happened before the first `INIT` (about 1.6% of 2021's disconnect games), and showed 11 unexplained riichi declarations "while disconnected".
  - Track disconnect state per seat: `BYE` sets it; a mid-game `UN` carrying only that seat's name clears it.
  - Represent "disconnected before round 1" explicitly.

## 4. D4: Hygiene and quarantine (reason-coded; nothing is dropped silently)

| Code | Rule | Action | 2009 count |
|---|---|---|---|
| Q1 | Decompression or JSON parse failure | Drop the game | 0 |
| Q2 | Placeholder or invalid names (`EXAMPLE*`, empty, fewer than 4 distinct) | Drop the game | 2 |
| Q3 | Structure: missing `start_game`/`end_game`, `type_code ≠ 00a9`, or `aka_flag` false | Drop the game. Upstream already kept only GO type 169 and excluded the no-aka (171), no-aka-no-kuitan (175) and fast (233) variants. | 0 |
| Q4 | Score continuity: start-of-kyoku scores ≠ previous scores + Σdeltas − 1000·reaches | Inspect 20 by hand and classify (double ron, chombo, a disconnect penalty, a converter bug). Drop if unexplained. | 8 transitions |
| Q5 | **libriichi replay failure**: `GameplayLoader` over all 4 seats raises an error | Drop the game. This also guarantees training-time compatibility. | measure |
| Q6 | **Disconnect window** (§4.1) | Drop **all seats' decisions in every kyoku that overlaps a disconnect window** (`BYE` → reconnect `UN`). If the player never reconnects, drop from the `BYE` kyoku to the end of the game. | 0.58% of kyoku (2021: 1.22%) |
| Q7 | Duplicates (same `log_id`, or same `content_sha256`) | Keep the first | measure |
| Q8 | **Unauthorized-bot accounts** (D26): known AI accounts that played houou against the rules, for example 未払い, 秋野花, まだ垢 and ハラヤハ | `excludes` list. Keeps the human prior and the golden set human. | 0 in 2009 |

**Outputs:**
- `meta/quarantine.parquet` (`log_id`, `seat` or `*`, `kyoku` or `*`, `code`, `detail`).
- `reports/quarantine.md`: counts per code per year.

**Expected:** under 0.5% of games dropped outright, plus about 1–1.5% of kyoku from Q6. Anything larger means stop and investigate the converter before going further.

**Mechanics:**
- Q6 and Q8 are applied through a **per-game seat × kyoku mask**, which the loader extension in §6 carries.
- Game-level drops are applied by leaving those files out of the split lists (§7).
- **GRP rewards must be computed on the full game *before* masking.** Rewards for kyoku that ended before the disconnect depend only on pre-disconnect scores, so they stay valid.

### 4.1 Disconnects: what the data shows [measured 2026-10-01 from the source DBs]

| | 2009 (6,897 games) | 2021 (187,313 games) |
|---|---|---|
| Games with ≥ 1 `BYE` | **3.5%** | **8.6%** |
| Games where the player never reconnects | 26 | 1,614 (0.9%) |
| Discards made while disconnected | 2,071, **100% tsumogiri** | 77,886, **100% tsumogiri** |
| Calls / agari while disconnected | 0 / 0 | 0 / 0 |
| Own discards per disconnect (median / mean / max) | 2 / 8.1 / 129 | 1 / 4.5 / 193 |
| Disconnected seat's final average rank (4th-place rate) | 2.71 (35%) | 2.68 (33%) |
| Kyoku dropped by the overlap rule (Q6) | 0.58% | 1.22% |
| Kyoku dropped by truncating from the first `BYE` | 2.2% | 5.5% |
| Data dropped by excluding the whole game | 3.5% | 8.6% |
| Most disconnect-games by one player in the year | 35 | 188 |

**Why it matters** (Tenhou's auto-play while a player is disconnected is tsumogiri-only, so the mjai files contain those moves as if the human made them):

1. **The disconnected seat gets wrong labels.**
   - Every discard is labelled tsumogiri.
   - Every chi/pon/kan opportunity is labelled *pass*.
   - Every riichi opportunity is labelled *no riichi*.
   - **Every ron and tsumo is labelled *decline the win*.**

   These are few in number (about 0.1% of discards) but systematically harmful. AWR then weights them by outcome.
2. **Opponents' decisions are contaminated.** Opponents can likely tell the player is gone (the Tenhou client shows it; unverified). Their pushes against a "riichi" or "dealer" who can't actually win look reckless to a model that has no disconnect feature.
3. **Outcomes are distorted.** The disconnected seat finishes 4th about 33–35% of the time, and opponents get a windfall. Both bias the player scores (§6) and the AWR advantages.

**Decision (D27):**
- Use the **kyoku-overlap rule** for training. It removes nearly all the harm for about 1.2% of kyoku. Whole-game exclusion would cost 8.6% of the modern data for little extra benefit.
- **Exclude every game that contains a `BYE` from player-skill statistics** (§6), for all four seats.
- **Chronic disconnectors are not excluded by default.** Their connected decisions are still human decisions. Log their per-player disconnect rate, and add "exclude players with a disconnect rate above 15%" as a sweep arm only if Q4/Q5 or the style panel show trouble.
- The 2026 games have no source-DB rows. They sit inside `heldout_time`, so build the **golden set only from games with DB coverage** (§7).

## 5. D5: Coverage and meta audit (about 1 h; decides the split dates)

Write `reports/coverage.md` with:
- **Games per month**, 2009-02 → 2026-01. Find out whether 2025's low count (198k vs 335k in 2024) is real (missing months, or a partial upstream scrape).
- **Held-out-by-time must be a contiguous, complete window** of at least about 6 months and at least 100K games. If 2025 H2 has gaps, move the window, and record the change in `08` as D14b.
- **Players per year** and the games-per-player distribution.
- **Per-year meta statistics** from `seats` (riichi, call, deal-in and win rates, average kyoku count). These show drift and inform the training-window ablation (2015+ default vs 2009+), and they give the reference bands for the style panel (`02` §7).

## 6. D6: Player scoring (about 1–2 h including checks)

> **Order (D40, 2026-10-01): run §7 (freeze the splits) *before* this section.**
> - All estimates below are fitted on **`train` games only**: skill, σ_b, k, the half-life choice, the (a, b) fit and the arm thresholds.
> - `heldout_time` and `heldout_game` outcomes never influence which training seats are admitted.
> - The §6.1 measurements were exploratory, made on whole years. Recompute them on `train` for the dataset card.

### 6.1 What the data says [measured 2026-10-01]

| Quantity | 2024 (mjai, 184,424 games) | 2021 (source DB, 187,313 games) |
|---|---|---|
| Players; median games per player | 2,425; 151 | — |
| Players with ≥ 100 games (their share of seats) | 1,431 (95%) | — |
| Per-game SD of u for a fixed player, σ_w | 0.83 | 0.83 (assumed) |
| **True-skill SD across players, σ_b** | **0.023–0.025** | **0.020** |
| **Shrinkage constant k = σ²_w/σ²_b** | **≈ 1,100–1,300 games** | **≈ 1,700 games** |
| Split-half reliability (players with ≥ 200 games per half) | **0.11** | — |
| Top 25% of seats, picked by **raw mean** on half E: mean u on E → on O | +0.066 → **+0.021** | — |
| Top 25% of seats, picked by **empirical Bayes** on E: mean u on O | **+0.026** | **+0.019** |
| Top 25% of seat-games, picked by **Tenhou R at game start**: mean u | — | +0.013 |
| Top 60% of seats: EB → O / R | +0.010 / — | +0.007 / +0.010 |

**Units.** All values are in **u**, the per-game utility u = [+0.90, +0.50, −0.10, −1.30] for 1st–4th (`02` §2).
- ×100 is close to Tenhou houou 7-dan pt ([90, 45, 0, −135]), so 0.01 u ≈ 1 pt/game.
- **Converting to avg rank depends on where the edge comes from.** 0.02 u is about 0.017 avg rank if it comes from turning 4ths into 3rds, and about 0.05 if it comes from turning 2nds into 1sts.
- k is dimensionless; read it as "prior weight, in games".

**Estimators** (one-way random-effects method of moments, computed per half):

For player p with n_p games and per-game utilities u_pg, let x̄_p = mean_g u_pg. Then:
- μ = Σ_p n_p·x̄_p / Σ_p n_p. This is exactly 0, because every game hands out one of each rank, so u sums to 0 across the four seats.
- **σ²_w = Σ_p Σ_g (u_pg − x̄_p)² / (N − P)**, where N is the number of seat-games and P the number of players. This is the pooled within-player variance, as in ANOVA.
- **σ²_b = Σ_{p: n_p ≥ 50} n_p·[(x̄_p − μ)² − σ²_w/n_p] / Σ_{p: n_p ≥ 50} n_p**, floored at 1e-6.
  - It uses Var(x̄_p) = σ²_b + σ²_w/n_p: the observed spread of player averages, minus the part explained by luck at each player's sample size.
- k = σ²_w / σ²_b.

**Cross-check:** the split-half correlation (0.11 at n ≈ 200–400 games per half) implies σ_b ≈ 0.014–0.020, consistent with the direct estimate.

**Caveats:**
- **Range restriction.** Players need 7-dan to stay in houou, and the n ≥ 50 cut excludes players with few games, including recently demoted ones. So σ_b is the spread among *regulars*, which is probably an underestimate of the whole room's spread.
- **2024 was computed from mjai.** Final ranks were reconstructed from the deltas (riichi deposits, leftover sticks to 1st, seat-order tie-breaks), and **disconnect games were not excluded**, because no source DB was used for 2024.
- **2021 used the XML `owari` final scores** and excluded BYE games, but σ²_w was **fixed at 0.69 rather than measured**.
- No opponent-strength adjustment. The houou pool is homogeneous, so this effect is minor.
- The values are within-year. Skill drift across years is handled by the time decay in §6.2.

**Three conclusions:**

1. **Houou is already a filtered population.**
   - Entry requires 7-dan **and** R ≥ 2000. R at game start in 2021 had mean 2150 and SD 59.
   - The true-skill spread among houou players is therefore small relative to per-game luck: σ_b ≈ 0.02 u ≈ **2 pt/game**, against σ_w ≈ 83 pt/game.
   - It is *not* negligible on a model-improvement scale. It is 2–4× Mortal's version-to-version gaps (0.4–1.2 pt/game), and about a quarter of Mortal's edge over akochan (≈ 7.6 pt/game).
   - Most of the "skill filter" was done by Tenhou's room gate.
2. **One season of results is mostly luck.**
   - With k ≈ 1,100–1,700, a player needs over 1,000 games before their own record outweighs the population average.
   - Split-half reliability is only 0.11 at 200+ games per half.
   - Picking on raw in-sample averages overstates the edge about 3× (+0.066 apparent vs +0.021 real).
3. **A filter trades a small, real quality gain against a large loss of volume.**
   - The best quarter of seats is about +0.02 u (≈ 2 pt) per game better than average out of sample.
   - A top-25% filter discards 75% of the data to get that.
   - **Whether a mild filter (A1 bottom trim, A2 soft weights) beats no filter is genuinely open.** The sweep decides it. Hard top-25% filtering is the least likely winner.

### 6.2 Method

**Inputs:** `seats` **restricted to the `train` split**, after quarantine, **excluding every game that contains a `BYE`** (§4.1), with each player's u per game.

1. **Split games** by `sha1(log_id) mod 2` into halves E and O. All selection is cross-fitted, so a seat in an O-game is judged only by E-half evidence, and vice versa.
2. **Pool all years within a half, with time decay.** Reliability grows with games:

   ρ(n) = σ²_b / (σ²_b + σ²_w/n) ≈ 0.12 at n = 200, ≈ 0.40 at n = 1,000, ≈ 0.67 at n = 3,000.

   Many strong regulars have thousands of games across 2015–2025, so pooling matters. Skill drifts over time, so weight each game by 0.5^(age / half-life). Choose the half-life from {1, 2, 4, ∞} years by out-of-sample fit on the other half.
3. **Empirical-Bayes skill:**
   - μ is the population mean;
   - σ²_w is the pooled within-player variance (≈ 0.69);
   - σ²_b = weighted mean of (x̄_p − μ)² − σ²_w/n_p, floored at a small ε;
   - k = σ²_w / σ²_b;
   - ŝ_p = (n_eff·x̄_p + k·μ) / (n_eff + k), where n_eff is the decayed game count.
4. **Combined score.** Fit, on half E,

   u_O ~ a·ŝ_E + b·(R_at_start − 2150)/60

   by least squares over seat-games, where R_at_start is Tenhou's own rating at game start (from the source DB). It is out-of-sample for that game and only weakly redundant with ŝ. Apply the fitted (a, b) symmetrically. This is score_p,g, which can vary per game because R does.
5. **Eligibility:** none. EB already handles players with few games. The minimum-games floor is dropped.

### 6.3 Sweep arms (quick students, `04` §7; judged by GSPRT)

| Arm | Seats used | Note |
|---|---|---|
| **A0: no filter** | All (after quarantine) | **Default.** Kept unless another arm wins its test. |
| A1: bottom trim | Drop seats with score in the bottom 20% | Keeps 80% of the volume; removes the weakest and most erratic play |
| A2: soft weight | All, with loss weight w = clip(1 + β·z_score, 0.5, 1.5), β tuned so the mean weight is 1 | No volume loss. Multiplies with the AWR weights, so test jointly with AWR β. |
| A3: top 60% | Seat-weighted cut | |
| A4: top 40% | Seat-weighted cut | |
| A5: R-only top 60% | By R_at_start | Checks whether the simple, leak-free rating is enough |

**Decision rule** (pre-registered; replaces an earlier ambiguous version):
1. **Screen offline (free).** Compute held-out-by-game log-likelihood and golden-set per-category agreement for every arm. Drop an arm only if it is clearly worse on both, beyond 2 SE. This is for pruning, never for picking the winner.
2. **Test the survivors against the control.** Run T-std of each surviving arm against A0 on family B, with a **Holm multiplicity correction**.
   - Comparing m arms with A0 at α = 0.05 each would give about a 23% chance of a false winner when m = 5.
   - So set the SPRT acceptance bound at ln(0.95 / (0.05/m)). That is 4.55 instead of 2.94 when m = 5, relaxing as arms are resolved.
3. **Winner:**
   - If **no arm passes** → keep **A0**.
   - If **one arm passes** → that arm.
   - If **several pass** → play them against each other in a short round-robin and take the highest rating. A tie goes to the arm keeping more data.
4. **Confirm before adopting.** Retrain the winner with a second training seed and run a fixed-size T-confirm on family C. If it doesn't hold, keep A0.

Record the result as D14a.

### 6.4 Outputs and checks

**Outputs:**
- `meta/player_scores.parquet` (per player per half: n, n_eff, x̄, ŝ, k, half-life);
- `meta/seat_scores.parquet` (`log_id`, `seat`, score);
- names and weight files per arm: `names/A{1,3,4,5}_fromE.txt` / `_fromO.txt`, `weights/A2_*.parquet`.

**Checks** (in `reports/player_scores.md`):
- k and σ_b per year;
- split-half reliability by games bucket (200 / 500 / 1,000 / 3,000);
- the out-of-sample mean u of every arm's kept seats (as in the table above);
- the fitted (a, b);
- 100% of listed names found in the data (NFC on both sides).

**Loader extension** (about 50–80 lines, `mortal/dataloader.py`):
- Accept a list of `(file_list, player_names | seat_weights, seat×kyoku masks)` groups and interleave them, so the E/O cross-fit runs in a single training job.
- The same mechanism carries the Q6 and Q8 masks and the A2 weights.

## 7. D7: Freeze the splits (about 1–2 h, including golden-set tagging)

**Splits:**

| Split | Rule (defaults; adjust per §5 and record in `08`) |
|---|---|
| `heldout_time` | 2025-07-01 → 2026-01-23 |
| `heldout_game` | Of the remaining games, `sha1(log_id) mod 100 < 2`. **All 4 seats** are excluded from training. |
| `train` | 2015-01-01 → the day before `heldout_time` starts, minus `heldout_game`, minus quarantine. Stored as `train_E.txt` and `train_O.txt`. |
| `train_ablate_2009` | `train` plus 2009–2014 |

**Golden set** (`splits/golden.parquet`):
- **Source:** `heldout_time` games.
- **Seats:** players in the top 5% of the §6 score, computed **only from games before** `heldout_time`, and n ≥ 500. Use only games **with source-DB coverage**, so disconnects can be masked (§4.1).
- **Freeze and hash the golden player list before reading any golden-set outcome** (D40).
- **Tagging:** replay each game with libriichi's `PlayerState` and tag every decision as one of:
  - routine discard;
  - riichi or dama;
  - call or pass;
  - push/fold (an opponent in riichi, and you not tenpai);
  - oorasu;
  - early (turns 1–6).
- **Sampling:** stratified, at least 10K per category, about 100K or more in total.
- **Stored per decision:** `log_id`, `seat`, `event_idx`, `category`, `mask_hash`, and the human action.

**Freeze:**
- Every list gets a `.sha256` entry in `splits/SPLITS.sha256`.
- Build symlink farms (`splits/train_E/…`) so Mortal's `globs` work unchanged.
- Build `file_index.pth` from the lists, never from a raw glob.

**Leak tests** (must pass; written to `reports/splits.md`):
- No `log_id` appears in two splits.
- No golden decision comes from a game in a training list.
- `max(train.ts) < min(heldout_time.ts)`.
- No `heldout_game` `log_id` appears in any symlink farm.
- The size of every split, in games and in estimated decisions.

## 8. D8: Dataset card and determinism

**`reports/DATASET_CARD.md`** (private, never published with the data):
- HF revision and `MANIFEST.id`;
- the pipeline commit;
- the parameter file (`params.toml`: thresholds, hash salts, dates);
- counts after every stage;
- the quarantine table;
- the coverage plot;
- player-score reliability;
- the split hashes.

**dataset-v1 identity** = sha256(`MANIFEST.id` ‖ pipeline commit ‖ `params.toml`). Log it with every run (`02` §9).

**Determinism test:** run D3–D7 twice from scratch. Every output hash must match. Hash-based splits make this easy; avoid unseeded randomness anywhere.

Then create the **`.gitignore` guard:** the repo contains only `tools/data/`, `params.toml` and the reports with aggregate numbers. It never contains data, names files or split lists.

## 9. Deferred: Mahjong Soul data

The Jade and Throne shards (about 31 GB, 2019–2026) are in the same repo. Keep them out of dataset-v1.
- **Rules:** they differ in places from Tenhou rules as libriichi implements them. Replay failures (Q5) would show it, but silent rule-dependent behaviour is the real risk.
- **Meta:** the Mahjong Soul meta differs from Tenhou's.
- **Provenance:** the logs were scraped from Mahjong Soul, so they carry their own terms-of-service exposure.

**Re-entry:** a Phase 1 ablation (quick student + Throne games, T-std), only if the X-sweep shows that data volume, not data quality, is binding.

## 10. Deliverables checklist (Phase 0 exit, data part)

- [ ] `raw/` immutable, with `MANIFEST.sha256`, `MANIFEST.id`, `HF_REVISION`; an encrypted backup exists.
- [ ] `mjai/YYYY/MM/*.json.gz`, with counts reconciled.
- [ ] `games.parquet`, `seats.parquet`, `quarantine.parquet`, plus the `reports/*.md` files.
- [ ] Coverage audit complete; held-out window confirmed (D14b if it moved).
- [ ] `splits/` lists, the golden player list (hashed before any outcome is read), `SPLITS.sha256`, and leak tests passing. **Done before scoring** (D40).
- [ ] `names/` files, the reliability report, and the R-arm lists, all fitted on `train` only.
- [ ] Golden-set decisions tagged and sampled.
- [ ] `DATASET_CARD.md`; determinism test passing; dataset-v1 identity recorded in `08` and the registry.
