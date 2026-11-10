# `k2b` against opponents it didn't train against: pre-registered 2026-11-09, before any run

The audit of 2026-11-09 (`STATUS.md`) found that `k2b`'s gain is measured only against pilot A, the
frozen model it started from and trained against (+0.0195 [+0.0174, +0.0216],
`20261025-060641-T-confirm`). `02` §4.2's population panel was not run at G0 (the mixed-table arena
isn't built). This check plays `k2b` at two tables the 1v3 arena can already seat: three pilot Bs
and three shanten-greedy bots (P-FLR). The user asked for it on 2026-11-09. Exploratory, family A.

## The question

Does `k2b`'s gain over pilot A carry to a table of other opponents, or is it specific to pilot A?

A 1v3 score against another model mixes two things: how much better the challenger is, and how the
two models differ. So each table is read as a **pair on the same deals**: `k2b` full in the
challenger's seat, and pilot A in the same seat, against the same three opponents. The quantity is
the per-set difference

    D(table) = set score of k2b full − set score of pilot A      (same nonce, same opponents)

At a table of three pilot As, D is the confirmed +0.0195 (pilot A against itself scores exactly 0).

## Engines (all frozen, read-only)

| Name | File | sha256 | Obs |
|---|---|---|---|
| `k2b` full | `workdir/fleet/runs/k2b/snapshots/full.pth` | `f5a22c5c…` | v4 |
| pilot A | `workdir/pilot/qs-a/pilot-qs-a-final.pth` | `15121432…` | v4 |
| pilot B | `workdir/pilot/qs-b/pilot-qs-b-final.pth` | `b267c026…` | v4 |
| floor | `builtin:shanten-greedy` | (Rust, frozen) | n/a |

## The runs (workstation, one at a time, `experiments/k2b_opponents_chain.sh`)

Each is `python -m lab measure --sets 20000 --family A`, AMP on as in every measure so far.

| # | Challenger | Opponents | Block | Nonces | Note |
|---|---|---|---|---|---|
| 1 | `k2b` full | 3× shanten-greedy | `floor` (new) | next free family-A range | ≈ 2 h |
| 2 | pilot A | 3× shanten-greedy | `floor` | the same | the pair's baseline; ≈ 2 h |
| 3 | `k2b` full | 3× pilot B | `aa-prime` | [1000, +20000) | ≈ 3.5 h |
| (4) | pilot A | 3× pilot B | `aa-prime` | [1000, +20000) | **exists:** `20261005-124255-measure`, +0.0015 [−0.0031, +0.0061] |

- Run 2 is an addition to the audit's two runs: without pilot A at the same floor table, run 1 says
  only that `k2b` beats the floor, which any BC model does.
- Run 4 predates D51 (a triple ron became an abortive draw on 2026-10-08), the only change to play
  since. Its sets with a triple ron were played under the old rule; that is rare enough to leave the
  mean alone, and it is not rerun.
- The floor runs go first: they are the untried path (a neural challenger against the rule-based
  bot; a 100-set smoke run on a dev key played cleanly at ≈ 10K sets/h).

## Reading, fixed now

`experiments/k2b_opponents_paired.py` computes D per table with the lab's set-level bootstrap (95%).

- **CI entirely above 0:** the gain carries to that table. Reported with its size against +0.0195.
- **CI includes 0:** not shown to carry there; the upper bound says how much could.
- **CI entirely below 0:** `k2b` is worse than pilot A at that table. D < −0.02 with the upper
  bound below 0 is `02` §4.2's severe-regression condition.

Also reported: each run's Δu, CI, average rank and style panel, and the correlation of the paired
set scores.

Nothing follows automatically. G0's Select conditions were met on the 1v3 test against pilot A as
written; what a missing or negative carry-over means for T1 (the opponent pool, the panel before
the next gate) is the user's call.

## Results (2026-11-10 14:48 UTC): the gain carries to both tables

`workdir/lab/k2b-opponents-paired.json`. Exploratory; family A; AMP; 20,000 sets per run.

| Table | `k2b` full | pilot A | **D = `k2b` − pilot A** (95% CI) | Share of +0.0195 | Runs (`k2b`, pilot A) |
|---|---|---|---|---|---|
| 3× shanten-greedy | +0.4214 | +0.4130 | **+0.0084 [+0.0048, +0.0119]** | 43% | `20261109-221907-measure`, `20261110-030053-measure` |
| 3× pilot B | +0.0193 | +0.0015 | **+0.0177 [+0.0120, +0.0235]** | 91% | `20261110-041102-measure`, `20261005-124255-measure` |
| 3× pilot A (reference) | | | +0.0195 [+0.0174, +0.0216] | 100% | `20261025-060641-T-confirm` (100K, family C) |

- **Reading, as pre-registered:** both CIs lie entirely above 0, so `k2b`'s gain over pilot A carries
  to both tables.
  - **Against pilot B** it is about the size of the gain against pilot A; the CIs overlap. Pilot B is
    a twin of pilot A (same recipe, seed 2), so this says the gain isn't tied to pilot A's
    particular weights. It doesn't say much about opponents of another kind.
  - **Against the floor** it is smaller, ≈ 0.4×. My reading, not a result: both models win most
    sets there (average rank 1.93 vs 1.94, ≈ 40% firsts), so there is less to separate them.
- **Paired set scores:** correlation 0.59 at the floor table and 0.22 at the pilot B table; SD of D
  0.257 and 0.417.
- **Exploratory panel** (`k2b` vs pilot A at the same table):

  | Table | Average rank | Win % | Deal-in % | Call % | Riichi % |
  |---|---|---|---|---|---|
  | 3× shanten-greedy | 1.929 vs 1.943 | 28.7 vs 28.3 | 5.95 vs 6.02 | 31.5 vs 29.8 | 19.4 vs 19.4 |
  | 3× pilot B | 2.474 vs 2.498 | 22.0 vs 21.8 | 12.7 vs 13.1 | 29.6 vs 28.3 | 20.1 vs 20.0 |

  `k2b` calls ≈ 1.3–1.7 pp more than pilot A at both tables and deals in a little less.
- **Caveats:**
  - **The pilot B baseline predates D51.** Its logs hold 44 triple-ron kyoku in 28 of the 20,000
    sets (the new run: 0, since a triple ron is now an abortive draw). A set score spans at most 2.2,
    so the worst case moves D by ±0.0031 (28 × 2.2 / 20,000), and the lower bound stays above 0.
  - **The first run's registry row records commit `94b0c7cb`,** the state when it ended. It started at
    `dfa65a8`; in between only docs and `lab/diskguard.py` changed, which the running process hadn't
    loaded. It was paused 05:43–09:11 UTC (SIGSTOP); a set's games depend only on its seed.
  - **Not covered:** mixed tables (P-MIX, P-STY, P-RES; the 1v3 arena seats three identical
    opponents) and the panel's weighted score; no family-C confirmation.
- **Cost:** ≈ 6 h of workstation play, the 3.5 h pause aside (1.2 h + 1.2 h at the floor table,
  ≈ 17K sets/h; 3.6 h at the pilot B table, ≈ 5.5K sets/h); no cloud spend.
