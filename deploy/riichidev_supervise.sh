#!/usr/bin/env bash
# Supervisor for play on riichi.dev (07 §1 operations): runs deploy.riichidev_bot with the given
# arguments until it stops for a reason.
#   exit 0  the games were played, or the stop file appeared: done;
#   exit 3  a game had failures and the bot failed closed: done, never restarted;
#   other   a crash: restart after a backoff (30 s, doubling to 10 min, reset after an hour of
#           play), and give up after 5 crashes within an hour.
# A crash mid-game forfeits the rest of that game (riichi.dev has no mid-game reconnection).
#
# Start it detached, from the repo root; the token file lives outside the repo:
#   setsid -f deploy/riichidev_supervise.sh --model PATH --mode ranked --games 0 \
#       --token-file ~/.config/riichidev/token >> workdir/riichidev/supervisor.log 2>&1 < /dev/null
# Stop it after the current game: touch workdir/riichidev/STOP (remove it before the next start).
# RIICHIDEV_PY picks the interpreter (default: the riichidev venv); RIICHIDEV_BACKOFF_S the first
# backoff.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
PY=${RIICHIDEV_PY:-$HOME/.venvs/riichidev/bin/python}
FIRST_BACKOFF=${RIICHIDEV_BACKOFF_S:-30}
say() { echo "$(date -u +%FT%TZ) supervisor: $*"; }

crashes=()
backoff=$FIRST_BACKOFF
while true; do
    say "starting the bot (pid $$ supervising)"
    started=$(date +%s)
    "$PY" -m deploy.riichidev_bot "$@"
    rc=$?
    case $rc in
        0) say "the bot exited 0 (games played, or the stop file): done"; exit 0 ;;
        3) say "the bot failed closed (a game had failures): not restarting; read its game log"; exit 3 ;;
    esac
    now=$(date +%s)
    (( now - started > 3600 )) && backoff=$FIRST_BACKOFF
    crashes+=("$now")
    recent=0
    for t in "${crashes[@]}"; do (( now - t < 3600 )) && recent=$((recent + 1)); done
    if (( recent >= 5 )); then
        say "crash (exit $rc), $recent within an hour: giving up"
        exit "$rc"
    fi
    say "crash (exit $rc); restarting in ${backoff} s"
    sleep "$backoff"
    backoff=$(( backoff * 2 > 600 ? 600 : backoff * 2 ))
done
