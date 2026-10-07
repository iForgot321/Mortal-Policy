#!/usr/bin/env bash
# Mahjax throughput on one rented Vast box (03 §7, spike item 2): wait for the box, install
# JAX + Mahjax (pinned) in a venv, run tools/mahjax/bench.py (env only, env + observation,
# env + observation + b8c192 / b16c256), copy the JSON back to workdir/mahjax-spike/<date>-vast-<label>/,
# and destroy the instance. Same safety rails as tools/fleet/box_bench.sh: destroy on EXIT, and a
# watchdog destroys the box after HARD_LIMIT seconds whatever state the driver is in.
#
# No repo, no checkpoints and no Tenhou data go to the box: only tools/mahjax/{bench,net,select_patch}.py
# and the Mahjax source at MAHJAX_DIR (a clean checkout of MAHJAX_COMMIT).
#
# Launch in its own session, then look up its PID (see box_bench.sh):
#     setsid -f tools/mahjax/box_mahjax.sh <instance_id> 5090 > /tmp/drv.out 2>&1 < /dev/null
#
# Usage: MAHJAX_DIR=workdir/mahjax-src tools/mahjax/box_mahjax.sh <instance_id> <label>
set -uo pipefail
ID=$1; LABEL=$2
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MAHJAX_DIR=${MAHJAX_DIR:?set MAHJAX_DIR to a mahjax checkout}
MAHJAX_COMMIT=773e9efaf8f6c3fad5ada1b98b567e28e3cfedee   # v0.1.6, the commit the spike tested
JAX_VERSION=0.11.2
OUT=$REPO/workdir/mahjax-spike/$(date -u +%Y%m%d)-vast-$LABEL
LOG=$OUT/driver.log
KEY=$HOME/.ssh/vast_ed25519
HARD_LIMIT=${HARD_LIMIT:-5400}
mkdir -p "$OUT"
log() { echo "$(date -u +%T) [$LABEL] $*" | tee -a "$LOG"; }

[[ $(git -C "$MAHJAX_DIR" rev-parse HEAD) == "$MAHJAX_COMMIT" && -z $(git -C "$MAHJAX_DIR" status --porcelain) ]] ||
    { log "MAHJAX_DIR is not a clean checkout of $MAHJAX_COMMIT"; exit 1; }

DESTROYED=0
destroy() {
    ((DESTROYED)) && return
    for i in 1 2 3 4 5; do
        if vastai destroy instance "$ID" -y >> "$LOG" 2>&1; then DESTROYED=1; log "destroyed instance $ID"; return; fi
        sleep 10
    done
    log "DESTROY FAILED for instance $ID: destroy it by hand"
}
( sleep $HARD_LIMIT; echo "$(date -u +%T) [$LABEL] watchdog: hard limit reached" >> "$LOG"; vastai destroy instance "$ID" -y >> "$LOG" 2>&1 ) &
WATCHDOG=$!

IP=; PORT=; SSHC=
copyback() {
    [[ -n $IP ]] || return
    for i in 1 2 3; do
        rsync -az -e "$SSHC" "root@$IP:/workspace/spike/out/" "$OUT/box/" >> "$LOG" 2>&1 && { log "results copied back"; return; }
        sleep 15
    done
    log "copy-back failed 3 times"
}
finish() { copyback; destroy; kill $WATCHDOG 2>/dev/null; log "DONE"; }
trap finish EXIT

log "waiting for instance $ID"
st=
for i in $(seq 1 100); do
    j=$(vastai show instance "$ID" --raw 2>/dev/null)
    st=$(printf '%s' "$j" | python3 -I -c 'import json,sys; print(json.load(sys.stdin).get("actual_status"))' 2>/dev/null)
    [[ $st == running ]] && break
    [[ $st == exited || $st == offline ]] && { log "instance is $st"; exit 1; }
    sleep 15
done
[[ $st == running ]] || { log "not running after 25 min (status $st)"; exit 1; }
read -r IP PORT < <(printf '%s' "$j" | python3 -I -c 'import json,sys; d=json.load(sys.stdin); print(d["public_ipaddr"].strip(), d["ports"]["22/tcp"][0]["HostPort"])')
SSHC="ssh -i $KEY -p $PORT -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$OUT/known_hosts -o ConnectTimeout=20 -o LogLevel=ERROR"
R() { $SSHC "root@$IP" "$@"; }
for i in $(seq 1 30); do R true 2>/dev/null && break; sleep 10; done
R true || { log "ssh never came up ($IP:$PORT)"; exit 1; }
log "running: $IP:$PORT"

R "mkdir -p /workspace/spike/out /workspace/spike/tools" &&
rsync -az -e "$SSHC" --exclude .git "$MAHJAX_DIR/" "root@$IP:/workspace/spike/mahjax/" >> "$LOG" 2>&1 &&
rsync -az -e "$SSHC" "$REPO/tools/mahjax/bench.py" "$REPO/tools/mahjax/net.py" "$REPO/tools/mahjax/select_patch.py" \
    "root@$IP:/workspace/spike/tools/" >> "$LOG" 2>&1 || { log "rsync failed"; exit 1; }
log "synced mahjax $MAHJAX_COMMIT and tools at $(git -C "$REPO" describe --always --dirty --abbrev=12)"

# write the run script, start it in tmux, poll for its exit file
R "cat > /workspace/spike/run.sh" <<EOF
set -x
cd /workspace/spike
PY=\$(command -v python3.12 || command -v python3)
[ -x /venv/main/bin/python ] && PY=/venv/main/bin/python
\$PY -m venv jx && . jx/bin/activate
pip install -q "jax[cuda12]==$JAX_VERSION" svgwrite typing-extensions fastapi uvicorn
pip install -q --no-deps -e mahjax
nvidia-smi > out/nvidia-smi.txt; lscpu > out/lscpu.txt; pip freeze > out/pip-freeze.txt
python -c "import jax; print(jax.devices())"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
S=\${STEPS_PER_HANCHAN:-618}
python tools/bench.py --batch 1024 4096 8192 16384 --seconds 60 --steps-per-hanchan \$S --out out/env.json
python tools/bench.py --net obs --batch 1024 4096 8192 --seconds 60 --steps-per-hanchan \$S --out out/obs.json
python tools/bench.py --net b8c192 --batch 1024 4096 8192 --seconds 60 --steps-per-hanchan \$S --out out/b8c192.json
python tools/bench.py --net b16c256 --batch 1024 4096 --seconds 60 --steps-per-hanchan \$S --out out/b16c256.json
python tools/bench.py --upstream-cond --batch 256 1024 --seconds 60 --steps-per-hanchan \$S --out out/env-upstream.json
EOF
R "tmux new-session -d -s bench 'bash /workspace/spike/run.sh > /workspace/spike/out/run.log 2>&1; echo \$? > /workspace/spike/out/run.exit'" || exit 1
log "bench started"
for i in $(seq 1 120); do
    sleep 30
    out=$(R "if [ -f /workspace/spike/out/run.exit ]; then echo EXIT \$(cat /workspace/spike/out/run.exit); elif tmux has-session -t bench 2>/dev/null; then echo RUNNING; else echo DEAD; fi" 2>/dev/null)
    case $out in
        EXIT*) log "bench exit ${out#EXIT }"; exit 0 ;;
        DEAD) log "bench session died"; exit 1 ;;
    esac
done
log "bench timeout"
exit 1
