#!/usr/bin/env bash
# Keep one fleet process running on a box (03 §2, §3.7): run the command with its output appended
# to <log>, record every start and exit in supervise.log next to it, and restart it. The pause
# before a restart doubles from 10 s up to 5 min while the command keeps dying within 10 min.
# Run by fleet.py start, in a tmux session named after the process.
#
# Usage: box_supervise.sh <name> <log> <command> [args...]
set -u
NAME=${1:?usage: box_supervise.sh <name> <log> <command> [args...]}
LOG=${2:?usage: box_supervise.sh <name> <log> <command> [args...]}
shift 2
EVENTS=$(dirname "$LOG")/supervise.log
delay=5
while true; do
    t0=$(date +%s)
    echo "$(date -Is) $NAME start" >> "$EVENTS"
    "$@" >> "$LOG" 2>&1
    rc=$?
    up=$(($(date +%s) - t0))
    echo "$(date -Is) $NAME exit $rc after ${up}s" >> "$EVENTS"
    if ((up > 600)); then delay=5; else delay=$((delay * 2 > 300 ? 300 : delay * 2)); fi
    sleep "$delay"
done
