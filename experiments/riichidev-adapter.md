# riichi.dev adapter and parity tests (07 §1)

- **Status:** built 2026-10-10; parity at the plan's scale passed 2026-10-11 (Results). Not yet
  connected to riichi.dev: the validation game needs the bot token. The user's decisions of
  2026-10-12 are under "Before ranked play".
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
4. **07's "0 illegal actions over 100K lab games": paused.** Queued on the eval box on 2026-10-12
   as 20 background chunks of 5,000 riichienv self-play games (pilot A, FP32, seeds 1000–100,999,
   `--light`: the legality checks without the observation comparison, which passed at scale
   above; 20 worker processes; reports `workdir/riichidev/parity-100k-c00…c19.json`). The box
   closed with chunk 0 at 2,800 games, 0 failures (`workdir/fleet/runs/eval/labq/` log); no chunk
   finished, so all 20 run when the box is back.
   - Speed: ≈ 0.6–0.9 s per game with 20 workers, GPU-bound (each worker sends one decision at a
     time), so ≈ a day of box time. Fewer workers may be as fast; compare before queuing.
   - The deployed version will be the student. The adapter's legality doesn't depend on the
     network, but a short self-play run with the student goes before its first ranked game.
