# riichi.dev adapter and parity tests (07 §1)

- **Status:** built 2026-10-10; parity at the plan's scale passed 2026-10-11 (Results). Not yet
  connected to riichi.dev: the validation game needs the bot token and the user's go-ahead.
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
2. **Decisions for the user:**
   - **Which checkpoint is deployed.** 07 deploys the π_SL student as the incumbent (L1); it
     isn't trained yet. Deploying a pilot first gives an earlier external anchor, but it would
     sit on the board as the bot's first rating.
   - **Where it runs.** 07 budgets an on-demand verified 4090 near the server (≈ $0.55/h). A
     b8c192 student is fast enough on the workstation's 3060, but the workstation must then stay
     on, and the token lives there.
   - **AGPL:** the deployed bot's source must be published (CLAUDE.md rule 9). Nothing has been
     pushed so far.
3. **Still to build** (07 operations): a process supervisor with auto-restart, the daily summary,
   and per-class latency metrics. 07's "0 illegal actions over 100K lab games" needs batched
   self-play: at one decision per inference call, 100K games take ≈ 8 days on the workstation.
