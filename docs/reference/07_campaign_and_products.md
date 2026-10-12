# 07: Campaign and Products

**Phase 1** (smoke deploy) and **Phase 4** (campaign).

This file covers:
- deployment on riichi.dev and how to read its rating;
- the blunder-mining loop;
- the human-like dial and style products;
- publishing;
- ethics.

---

## 1. riichi.dev deployment

**Protocol** (`01` §3):
- The bot is **self-hosted** and connects over WebSocket to `wss://game.riichi.dev/ws/ranked` with a bot token.
- Each request carries the legal moves and a base64 observation. Decode it with the `riichienv` package.
- One validation game is required first.
- Time: **3 s grace per action plus a 15 s bank per kyoku.** A timeout auto-plays tsumogiri or pass, with no penalty.
- An illegal action is **chombo**.

**Adapter** (Phase 1): `deploy/riichidev_bot.py`.
1. Translate the riichienv observation into libriichi's event stream and player state. Mortal's observation v4 is built from mjai-style events.
2. Run the engine (student, then champion, then champion plus search).
3. Map the chosen action back to riichienv's legal-move format.

**Parity tests** (hard requirement before the first ranked game):
- Replay 1K lab games through the adapter and check the chosen actions are identical to the in-lab engine.
- Check every legal-move set returned by riichienv against libriichi's mask. Any mismatch is a stop.
- Illegal-action rate is 0 over 100K lab games.

**Deploy box** (`03` §2):
- 1× RTX 4090 (D50; revisit when search is enabled), ≥ 16 vCPU, **on-demand on a reliable or verified host**. Never interruptible, because a preempted bot times out mid-game.
- Pick the region with the lowest measured WebSocket RTT to `game.riichi.dev`. Measure from 3–4 regions in Phase 1.
- Budget: about $0.55/h, roughly $400/month (`00`).
- In Phase 1, the smoke deploy can share the eval box until search is enabled.
- Store the bot token in the host's secret store, never in the repo.

**Operations:**
- Process supervisor with auto-restart.
- Per-game mjai logs stored locally (for blunder mining).
- Latency metrics per decision class.
- Daily summary: games, placements, rating, timeouts, chombo count.
- **One deployed version at a time: the incumbent.** It is replaced only by a **confirmed** successor that passes:
  - a fixed-N family-C confirmation;
  - the population panel with no severe regression (`02` §4.2, D34).
- riichi.dev allows **4 bots per account at a time** (`01` §3). The incumbent takes one; anything
  else on the board at the same time shares the other three.
- **Action-selection mode** (argmax vs sampling from the policy or the search-updated policy) is part of the deployed version. It is chosen by its own T-std (D33), never switched ad hoc.

**Smoke deploy (Phase 1):**
- Deploy the π_SL student. It becomes the **incumbent** and stays deployable throughout early research and scaling (`00` principle 8).
- Get an early external anchor after ≥ 200 games, and prove the infrastructure.

## 2. Reading the rating

- **Scale:** OpenSkill Plackett–Luce, displayed as 1500 + 20·(μ − 3σ). Placement only. "Stabilizes around ~50 games."
- **Noise:** in simulation, the displayed rating drifts with SD ≈ 36–44 points and effectively reflects the last ~100 games [calc]. A single snapshot is noisy.
- **Rules of thumb:**
  - Compare 500-game rolling means, not snapshots.
  - Never use the rating to accept a patch. The lab decides; the arena confirms the trend.
- **Reference points (2026-09-29):** #1 2091; #5 ≈ 2023; Mortal-v4b #44 at 1864.
- **Utility match:** u (`02` §2) is derived from the same Plackett–Luce update at equal strength. If the agent becomes much stronger than its tables, the effective weights flatten (gaps 1 : 1.4 : 2.3 at strength ratio 1.5). Revisit u in `08` if we are consistently top-3.

## 3. Blunder mining (Phase 4 loop)

1. **Sources:** riichi.dev game logs (especially 4th places), lab self-play, and evaluation games.
2. **Detection:** run S-3 search (`06`) offline with a large budget at every decision in a sample. Flag decisions where the search's best action beats the played action by more than a threshold in GRP-Δu.
3. **Clustering:** by decision class × situation features (turn, threat level, placement pressure, hand shape). Rank clusters by total value lost.
4. **Feedback:**
   - Add flagged states as priority training states. If expert iteration has earned funding (D33b), they go to its buffer; otherwise they become targeted evaluation and curriculum cases.
   - If one cluster dominates across many games, run a **targeted exploiter probe** (`05` §9) against that weakness.
5. **Report:** a weekly top-10 clusters, with example positions. This doubles as the review tool's "common mistakes" view.

## 4. Human-like play: definitions and the dial

**"Human-like" means two different things.** Pick deliberately.

- **(a) Human judgment.** Flexible, value- and safety-aware choices that strong players recognize as "human".
  - LuckyJ got this **without human data**, as an emergent property of strong mixed-strategy play (`01` §6).
  - Measure it with the style panel (`02` §7): shanten-backward rate when quiet, average win value, honour retention.
- **(b) Human imitation.** Matching what a given human population actually plays.
  - Measure with golden-set agreement (`04` §4).
  - NAGA's types are imitations of specific players. Note that imitation-anchored agents come out *more* rigid on (a).

**The piKL dial** (implements (b), with strength as a controllable trade-off):

```
π_λ(a|s) ∝ τ_human(a|s) · exp( Q(s,a) / λ )
```

- τ_human = the BC human prior (`04` §5).
- Q = the S-3 search values, or critic-based action values where search isn't run.
- **Small λ:** strength-maximizing. **Large λ:** near-human.
- In chess and Go, piKL with a human prior was *both* more human (higher human-move accuracy) and far stronger than the prior alone (92–95% wins) [V].
- Expose λ as a product setting. Report the Δu-vs-agreement curve as a published result.

**Rank- or player-conditioned SL** (Maia-2 / NAGA style):
- Condition the SL policy on a skill bucket (cross-fit skill percentile) or a player-style embedding.
- This gives named styles (for example "top-5% houou", or "heavy caller") and supplies τ_human for any target population.
- Cost: one extra teacher run with a conditioning input. Phase 4 only.

## 5. Style products

- **Variants** from `05` §7: first-seeking u₁, 4th-avoiding u₄, plus the balanced champion. BigCoach's million-game head-to-heads show styles cost almost no strength at the frontier (all within 2.476–2.508 average rank).
- **Ensemble** (2–3 models, logit average) as a "robust" variant.
- **Review tool:**
  - Per-decision evaluation: policy probabilities, search values when available, and placement probabilities from GRP and the critic.
  - "Common mistakes" from blunder mining.
  - Compatible with the mjai-reviewer format.
  - Label it strength-first, and show which engine and style produced each recommendation.

## 6. Ethics and boundaries (non-negotiable)

- **Play only where bots are welcome: riichi.dev.**
  - No bots in Tenhou or Mahjong Soul ranked play. That breaks their terms and harms human players.
  - The BigCoach / 未払い episode is the cautionary example (`01` §6).
- **No detection evasion.** "Human-like" in this project means human-aligned play for style, coaching and review. No work goes into making bot play indistinguishable from humans on platforms that ban bots, and no features target agreement-rate thresholds with other engines.
- **AGPL-3.0:** Mortal-Policy is AGPL. Running a derived bot as a network service requires offering the source. Publish the source of every deployed version.
- **Data:** never redistribute Tenhou logs or derived per-game data. Publish only the recipe and the manifest hash (`04` §1). Re-read Tenhou's terms before publishing weights, and record the decision in `08`.

## 7. Publishing (Phase 4, from month ~9)

- **Weights:** the champion, the style variants and the student, with the SL teacher optional. Include a model card with training games, utility, and known weaknesses from blunder mining.
- **Code:** all patches (AGPL), the evaluation lab, the search stack.
- **The ablation table** (`02` §9), including failed and inconclusive rows. This is what makes the "strongest open agent" claim *verifiable*.
- **Style-panel comparison** against LuckyJ, Suphx, NAGA and humans (`01` §6).
