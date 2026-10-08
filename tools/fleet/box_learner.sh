#!/usr/bin/env bash
# The learner's two processes as one unit (03 §2): server.py, then train_online.py once the server
# listens. When either exits, the other is stopped and this script exits, so box_supervise.sh
# restarts both. They must restart together: a fresh server.py has no parameters and an empty
# buffer, and a running trainer would wait on it forever (and the actors on the trainer).
# Each runs in its own session, so stopping it reaches train_online.py's respawned child and the
# loader workers. Run by fleet.py start, from mortal/, under box_supervise.sh.
#
# Usage: box_learner.sh <python> <config> <port> <log dir>
set -u
PY=${1:?}; CFG=${2:?}; PORT=${3:?}; LOGS=${4:?}
export MORTAL_CFG=$CFG
SERVER=; TRAINER=
stop_both() {
    for pid in $SERVER $TRAINER; do kill -TERM -- "-$pid" 2>/dev/null; done
    sleep 10
    for pid in $SERVER $TRAINER; do kill -KILL -- "-$pid" 2>/dev/null; done
}
trap 'echo "$(date -Is) learner: signalled; stopping both"; stop_both; exit 143' HUP INT TERM

setsid "$PY" server.py >> "$LOGS/server.log" 2>&1 &
SERVER=$!
hex=$(printf '%04X' "$PORT")
for _ in $(seq 1 60); do
    grep -q "0100007F:$hex 00000000:0000 0A" /proc/net/tcp && break
    kill -0 "$SERVER" 2>/dev/null || break
    sleep 1
done
setsid "$PY" train_online.py >> "$LOGS/trainer.log" 2>&1 &
TRAINER=$!
echo "$(date -Is) learner: server $SERVER, trainer $TRAINER"
wait -n "$SERVER" "$TRAINER"
rc=$?
echo "$(date -Is) learner: a process exited ($rc); stopping both"
stop_both
exit "$rc"
