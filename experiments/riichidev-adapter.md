# riichi.dev adapter and parity tests (07 §1)

- **Status:** built 2026-10-10; parity at the plan's scale passed 2026-10-11 (Results). The
  validation game passed on 2026-10-14. **The 100K-game test found two adapter bugs** (item 4).
  Both were fixed on 2026-10-15, and **the 100K test passed on 2026-10-18** (third run: 0 failures
  in 100,000 games). The user's
  decisions of 2026-10-12 are under "Before ranked play".
- **Code:** `deploy/riichidev_bot.py` (the adapter and its WebSocket runner),
  `deploy/riichidev_parity.py` (the parity harness), `tests/test_riichidev.py`.
- **Environment:** a venv on top of the frozen `mortal` env, adding riichienv 0.4.10 and
  websockets 17.2 (`deploy/__init__.py`). The `mortal` env is unchanged, and the new tests skip
  there.

## Protocol (riichi.dev/docs, read 2026-10-10)

- `wss://game.riichi.dev/ws/validate` (one East-only game in seat 0, against three tsumogiri
  bots) and `/ws/ranked` (hanchan). Auth: `Authorization: Bearer <token>`.
- The server streams mjai events (`start_game` with the seat `id`, `start_kyoku`, `tsumo`, …) and
  sends `request_action` with `request_id`, `possible_actions` (mjai) and a base64 riichienv
  observation. The bot answers one of the possible actions, echoing `request_id`; `action_ack`
  reports `accepted`, `rejected`, `unparseable`, `stale` or `defaulted` (timeout).
- An illegal or unparseable action is chombo. A timeout plays tsumogiri or pass, unpenalized.
  Mid-game reconnection isn't supported.
- `GET https://game.riichi.dev/status` answered `online` (5 active games). The host is in Japan
  (153.127.39.92). TCP connect from the workstation takes ≈ 140 ms, small against the 3 s grace.

## Design

- **Events in, libriichi decides.** For each request, the observation's `new_events()` (the mjai
  events new to this seat, which riichi.dev's Mortal guide recommends) feed a libriichi mjai
  `Bot` built on the lab's engine (`lab.arena.load_engine`, the same loader and settings as the
  lab; true FP32 by default, D43).
- **Every answer is one of `possible_actions`.** The reaction is matched to a legal move, and the
  server's own move dict is sent (plus `tsumogiri`, `actor`, `request_id`). With no match, or an
  exception, a legal fallback is sent: pass, else tsumogiri, as the server's timeout default.
- **Parity checked live, at every request:**
  - a shadow libriichi `PlayerState` gives libriichi's mask, which must equal the mask built from
    `possible_actions` (and the kan-select mask when a kan is legal);
  - libriichi must not want to act on an event the server didn't ask about;
  - the events in the observation must equal those streamed on the socket (type and actor).
    `--events stream` feeds the bot from the socket instead; the check then runs the other way.
- **Fail closed.** Any of these marks the game failed, and the runner then stops queueing games
  (`--keep-going` overrides).
- **Logs:** one JSONL per game in `workdir/riichidev/games/`: every server message except the
  base64 observation, and every decision (events fed, legal moves, the reaction with libriichi's
  q-values and mask, the response, time taken). `summaries.jsonl` has one line per game. The
  token is read from `$RIICHIDEV_TOKEN` or a file outside the repo, and never logged.

## Parity harness

Two modes, both checking at every request: riichienv's legal moves against libriichi's mask, and
the adapter's observation tensor against a libriichi `PlayerState` fed the full log, masked for
that seat as the arena masks it.

1. **Lab logs (`--logs`), 07's "replay 1K lab games":** each log is replayed through riichienv
   (`apply_event`), and every seat with legal moves is asked as the server would ask it. The
   adapter's moves must equal the log's. That needs logs played in every seat by the same engine,
   in true FP32: the deterministic A/A `20261008-150119-AA` (phase1-v0-final, b40c192, 2,000
   games, after D51).
2. **Self-play (`--selfplay N`):** riichienv in step mode, the server's flow, with the adapter in
   all four seats. A mirror env fed the log by `apply_event`, observed only for the seat asked,
   scopes `new_events()` as the server documents it.

### Findings while building it

- **Replay artifacts, not rule differences.** With `apply_event`, riichienv leaves a decision
  open across the next event. It asks again after `reach_accepted` (a riichi discard's calls),
  again after an open kan's `dora` (the rinshan discard), and it offers a kan's declarer a discard
  before the rinshan tsumo. Step mode does none of these. Calls on a riichi discard are offered at
  the discard, and an open kan's rinshan discard is asked at the rinshan tsumo, before the new
  dora shows, as in libriichi's arena. The replay skips these asks.
- **The logs agree on kan order** in both engines: open kan > tsumo > dora > dahai; ankan > dora
  > tsumo > dahai.
- **riichienv's `new_events()` cursor advances whenever an observation is made**, whether or not
  the seat has a decision. Python `env.step` makes one for every seat at every step, so a bot fed
  only at its requests would miss events locally. The server must observe a seat only when it
  asks it (otherwise riichi.dev's own Mortal guide would fail). That can't be confirmed offline,
  which is why the runner cross-checks the observation's events against the socket stream. The
  validation game will show which source is reliable.
- `MjaiReplay` (riichienv's replay reader) returns observations whose `new_events()` are wrong
  (another deal, then empty), so the harness doesn't use it.

## Results

- **Tests:** `tests/test_riichidev.py`, 6 pass in the riichidev venv (~1 min). They cover move
  mapping, replay and self-play on the fixture's small random network (kans are frequent), and the
  runner against a local mock of the validation game over a real WebSocket: the bearer header is
  sent, every answer is accepted, and the token never reaches the log. In the `mortal` env the file
  skips, and the suite is otherwise unchanged (83 pass).
- **Lab-log replay, 1,000 games of the FP32 A/A: pass** (`workdir/riichidev/parity-replay-aa-1000.json`,
  phase1-v0-final, 3.5 h on the workstation's 3060). 622,380 requests; observation tensors
  identical to the lab engine's at every one; 622,172 moves identical to the log's, 0 different,
  208 claims outranked by another seat's (a pass or a lower claim can't be checked against a log
  that shows the higher one); 0 adapter failures (mask, kan mask, unrequested action, legality).
- **Self-play, 300 riichienv games in step mode: pass** (`workdir/riichidev/parity-selfplay-300.json`,
  seeds 0–299, 68 min). 193,186 requests; observation tensors identical to a libriichi
  `PlayerState` fed the full log at every one; 0 fallbacks, 0 adapter failures; every seat fed the
  game's events in order.

## Before ranked play

1. **The validation game** (`--mode validate`): one unranked game against tsumogiri bots, and it
   activates the bot. Its log must show no failures, in particular no `event_source_mismatch`.
   - **Passed, 2026-10-14 07:34 UTC, on the eval box** (`validation_result: passed`). Pilot A,
     FP32, observation-fed, code `b7eecce` (pushed to the public fork first, at the user's word;
     `riichidev_bot.py` had changed since the 2026-10-12 push).
     - Seat 0, 1st (scores 56,300 / 23,000 / −2,300 / 23,000).
     - 102 requests; 0 failures (no `event_source_mismatch`), 0 timeouts.
     - Latency p50 ≈ 10 ms in every class. The maximum was 466 ms, one discard, probably the
       first call's warm-up.
     - The token appears 0 times in the game log, the summary and the runner's output.
     - Log `workdir/riichidev/games/20261014-073426-validate.jsonl`, summary
       `workdir/riichidev/summaries.jsonl` (pulled).
     - The token is in `~/.config/riichidev/token` on the workstation, and in
       `/root/.config/riichidev/token` on the box. Both are outside the repo, mode 600, with the
       same sha256.
     - The riichidev venv was rebuilt at `/workspace/venvs/riichidev`; its 10 tests pass there.
   - **Latency (TCP connect to `game.riichi.dev`):** ≈ 196 ms from the box (California),
     ≈ 134 ms from the workstation. Pick the ranked-play region by this measure before ranked
     play starts.
2. **Decided by the user, 2026-10-12:**
   - **Which checkpoint is deployed:** the π_SL student, as 07 says. Ranked play waits for it; no
     pilot goes on the board first.
   - **Where it runs:** the 4090 that 07 budgets (on-demand, verified host, ≈ $0.55/h), rented
     now and shared with the eval box until search is enabled (07 §1). Region: the lowest
     measured RTT to `game.riichi.dev`.
   - **AGPL:** push of `main` to the public fork approved. The source of each deployed version
     must be public before it plays (CLAUDE.md rule 9).
   - **riichi.dev allows 4 bots per account at a time** (`01` §3).
3. **Operations (07 §1), built 2026-10-12:**
   - **Latency per decision class:** each game log's `decision` lines carry the class (the move
     type sent; kans merged) and the ms; each game summary has n, p50, p99 and max per class.
   - **Supervisor:** `deploy/riichidev_supervise.sh` restarts the bot after a crash (backoff
     30 s → 10 min; it gives up after 5 crashes within an hour). It never restarts after a game
     with failures: the bot exits 3 and fails closed. Play stops between games when
     `workdir/riichidev/STOP` exists, never mid-game (no mid-game reconnection). `--games 0` plays
     until then.
   - **Daily summary:** `python3 -m deploy.riichidev_summary` gives, per UTC day: games,
     placements, mean rank and u, unfinished games, timeouts, chombo (rejected or unparseable
     actions) and other failures, plus latency per class. The rating isn't in the game messages;
     read it from the leaderboard.
   - Tests: `tests/test_riichidev_ops.py` (the restart policy with a fake bot, the summary, the
     latency classes); with `tests/test_riichidev.py`, 10 pass on the eval box.
4. **07's "0 illegal actions over 100K lab games": failed, with two adapter bugs in 25K games**
   (2026-10-14). The setup: 20 chunks of 5,000 riichienv self-play games (pilot A, FP32, seeds
   1000–100,999, `--light`: the legality checks without the observation comparison, which passed
   at scale above; 12 workers each, priority 9; reports `workdir/riichidev/parity-100k-c00…c19.json`).
   - First queued on the eval box on 2026-10-12, which closed before any chunk finished (chunk 0
     had reached 2,800 games with 0 failures).
   - Run again 06:28–11:30 UTC on 2026-10-14. Chunks 0–4 finished; the box was closed at the
     user's request with chunk 5 at 600 games and 14 chunks queued.

   | Chunk | Seeds | Requests | Fallbacks | Adapter failures | Result |
   |---|---|---|---|---|---|
   | c00 | 1000–5999 | 3,162,105 | 0 | 0 | pass |
   | c01 | 6000–10,999 | 3,151,731 | 0 | 0 | pass |
   | c02 | 11,000–15,999 | 3,161,816 | 0 | 1 | fail |
   | c03 | 16,000–20,999 | 3,154,260 | 2 | 6 | fail |
   | c04 | 21,000–25,999 | 3,198,549 | 0 | 0 | pass |

   No illegal move reached riichienv: every response was one of its legal moves, with the
   fallback where the adapter had none. The two bugs:
   - **Chankan is missed** (c03; seeds 20281 and 20282, seat 1, both after actor 0's kakan of 6p).
     riichienv asks seat 1 for `hora` or `none` (robbing the kan). libriichi offers the ron on the
     `kakan` event, which isn't the last event in the request, so the adapter logs an
     `unrequested_action`. After the remaining events, libriichi's mask is empty and it gives no
     reaction (`mask_mismatch`: libriichi [] vs riichienv [43, 45]; `no_reaction`). The fallback
     then passes. The network wanted the ron (p ≈ 0.88). This costs points, not legality. Likely
     cause: riichienv puts the events that follow a kakan before the chankan request, and
     libriichi's state moves past the chance.
   - **A chi riichienv doesn't offer** (c02; seed 14176, seat 1, after actor 0's tsumogiri 4m).
     libriichi's mask had chi-low and pass (bits 38 and 45); riichienv didn't ask seat 1 at all,
     so the bot answered nothing on the server. That's harmless here, but the two rule sets
     disagree on when a chi is allowed: kuikae, the last tile and a lack of a legal discard after
     the call are the candidates.

   **Both fixed on 2026-10-15, in the adapter.** libriichi is unchanged: it is the frozen rules
   engine, and the halfway measures were running on it.
   - **The chankan cause is an event-order convention.**
     - In seeds 20281 and 20282, actor 0 calls an open kan (8s), draws its rinshan, then declares
       a kakan (6p). riichienv reveals the open kan's dora right after the kakan, then asks the
       others about chankan: `kakan > dora > [chankan] > tsumo > dora`.
     - libriichi's arena reveals that dora later, at the rinshan tsumo. Its `PlayerState` closes
       the chankan window at the next event, and a dora counts.
     - A kakan's own dora comes after its rinshan tsumo in both engines (checked on six games'
       kans), so only the earlier kan's dora is ever moved.
     - Fix: `libriichi_order` feeds the doras that follow a kakan before it. libriichi then
       offers the chankan, with the indicators the server shows. The game logs keep the
       server's order.
   - **The chi cause is the four-kan abort (四槓散了), not a rule difference.**
     - In seed 14176 the 4m is the discard after a fourth kan by two players: kakan, daiminkan
       and kakan by actor 0, and an ankan by actor 2.
     - Both engines abort the kyoku unless that discard is ronned. libriichi's arena collects
       reactions, then aborts on anything but a ron (`check_four_kan`).
     - riichienv offers only the ron, and here nobody could ron. libriichi's `PlayerState`
       still offered seat 1 a chi that couldn't take effect.
     - Fix: on that discard (`four_kan_discard`), a libriichi call or pass means pass. The call
       bits leave the mask comparison. If the server asks (a ron is possible), the bot picks
       ron or pass by its q-values (`ron_or_pass`). A libriichi ron that the server didn't ask
       for still fails the game.
   - **The parity harness** feeds its reference `PlayerState` in the same order
     (`libriichi_order`). Fed the server's order, the reference has the chankan window closed,
     and its input rows for the candidate actions (the called tile, can-agari, the EV rows)
     differ from the adapter's.
   - **Tests:**
     - `tests/test_riichidev.py` gains unit tests for the three helpers.
     - Two regression tests replay seat 1's recorded requests from these games
       (`tests/data/riichidev/selfplay-{20281,14176}-seat1.json`, riichienv self-play). The
       pre-fix adapter fails both, with exactly the 100K test's failures.
   - **Verified:**
     - The three seeds pass with full checks.
     - 100 games, seeds 20,230–20,329 (both chankan seeds included), pass with full checks:
       61,463 requests, 0 failures, 0 observation mismatches, 0 fallbacks
       (`workdir/riichidev/parity-fix-20230-100.json`).
     - In 20281 and 20282 the bot now robs the kan, and the game ends there.
     - The adapter suite passes, 14 tests.

   Next: the 100K test again, all 20 chunks, on the next eval-box rental. That rental needs the
   user's go-ahead.
   - **Second run (r2), 2026-10-15/16: lost.** Its reports never left eval box 56069526, which
     Vast stopped at its host's contract end (2026-10-17 00:47 UTC). Its status lines had shown
     ≈ 15 chunks done with no failures, seeds 14,176 and 20,281 among them, but there's no
     report. **Third run (r3)** queued 2026-10-17 04:53 UTC on eval box 56337703 (the user's yes),
     with the same seeds and settings (`workdir/riichidev/submit-100k-r3.sh`).
   - **Third run (r3): pass** (finished 2026-10-18 by 13:55 UTC on eval box 56337703;
     `workdir/riichidev/parity-100k-r3-c00…c19.json`). All 20 chunks passed: 100,000 games,
     63,674,663 requests, **0 observation mismatches, 0 fallbacks, 0 adapter failures**. That meets
     07 §1's "illegal-action rate is 0 over 100K lab games". The last 5 reports were copied over a
     60%-loss home link, one file at a time. (The r2 reports were never recovered: Vast restarted
     the old box with an empty disk.)
   - Speed: ≈ 0.6–0.9 s per game with 20 workers, GPU-bound (each worker sends one decision at a
     time), so ≈ a day of box time. Fewer workers may be as fast; compare before queuing.
   - **Compared 2026-10-14 on the new box** (EPYC 9474F, 32 cores, 4090; 200 games, seeds
     200,000–200,199; all three passed with the same 127,879 requests): 150 s with 12 workers,
     149 s with 20, 148 s with 30. That's ≈ 0.75 s per game whatever the count, so the chunks
     run with 12 and leave CPU for the bot.
   - The deployed version will be the student. The adapter's legality doesn't depend on the
     network, but a short self-play run with the student goes before its first ranked game.
