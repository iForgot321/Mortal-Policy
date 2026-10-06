#!/usr/bin/env bash
# Drive one rented Vast box end to end (03 §3.1): wait for it, rsync the repo and the frozen
# inputs, run setup_box.sh (with tests) and the 15-min eval-mode benchmark, optionally a py-spy
# profile and P0 throughput at 800- and 2,000-game actor batches, copy the results back to
# workdir/fleet/<date>-vast-<label>/, and destroy the instance.
#
# Destroy always runs (EXIT trap), and a watchdog destroys the box after HARD_LIMIT seconds
# whatever state the driver is in. Launch it in its own session so it outlives the calling one,
# then look up its PID (backgrounding `a && b &` backgrounds the whole list, and `setsid` forks
# when it is already a session leader, so `$!` is easy to get wrong):
#     setsid -f tools/fleet/box_bench.sh <instance_id> 4090-7950x none > /tmp/drv.out 2>&1 < /dev/null
#     pgrep -o -f '[b]ox_bench.sh <instance_id>'
#
# Needs: the vastai CLI with an API key, ~/.ssh/vast_ed25519 registered with Vast, and a clean
# commit (the box gets the working tree as is).
#
# Usage: tools/fleet/box_bench.sh <instance_id> <label> <none|full>
set -uo pipefail
ID=$1; LABEL=$2; EXTRAS=${3:-none}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
OUT=$REPO/workdir/fleet/$(date -u +%Y%m%d)-vast-$LABEL
LOG=$OUT/driver.log
KEY=$HOME/.ssh/vast_ed25519
PY=/venv/main/bin/python
HARD_LIMIT=7200
mkdir -p "$OUT"
log() { echo "$(date -u +%T) [$LABEL] $*" | tee -a "$LOG"; }

DESTROYED=0
destroy() {
    ((DESTROYED)) && return
    for i in 1 2 3 4 5; do
        if vastai destroy instance "$ID" -y >> "$LOG" 2>&1; then DESTROYED=1; log "destroyed instance $ID"; return; fi
        sleep 10
    done
    log "DESTROY FAILED for instance $ID: destroy it by hand"
}
# watchdog: a separate process, so a hung ssh can't keep the box alive
( sleep $HARD_LIMIT; echo "$(date -u +%T) [$LABEL] watchdog: hard limit reached" >> "$LOG"; vastai destroy instance "$ID" -y >> "$LOG" 2>&1 ) &
WATCHDOG=$!

IP=; PORT=; SSHC=
copyback() {
    [[ -n $IP ]] || return
    for i in 1 2 3; do
        rsync -az -e "$SSHC" --exclude configs/ "root@$IP:/workspace/Mortal-Policy/workdir/box/" "$OUT/box/" >> "$LOG" 2>&1 &&
        rsync -az -e "$SSHC" --include='p0-tput-*/' --include='p0-tput-*/tb/***' --include='p0-tput-*/*.log' \
            --include='p0-tput-*/*.csv' --include='p0-tput-*/*.toml' --exclude='*' \
            "root@$IP:/workspace/Mortal-Policy/workdir/" "$OUT/" >> "$LOG" 2>&1 && { log "results copied back"; return; }
        sleep 15
    done
    log "copy-back failed 3 times"
}
finish() { copyback; destroy; kill $WATCHDOG 2>/dev/null; log "DONE"; }
trap finish EXIT

# 1. wait for the instance (max 25 min)
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

# run a step in a remote tmux session; echo its exit code (or dead/timeout/unreachable)
step() {  # name, max_minutes, command
    local name=$1 max=$2 cmd=$3 fails=0 out
    R "cd /workspace/Mortal-Policy && mkdir -p workdir/box && rm -f workdir/box/$name.exit && tmux new-session -d -s $name 'cd /workspace/Mortal-Policy && ($cmd) > workdir/box/$name.log 2>&1; echo \$? > workdir/box/$name.exit'" || { echo unreachable; return; }
    for i in $(seq 1 $((max * 2))); do
        sleep 30
        out=$(R "cd /workspace/Mortal-Policy; if [ -f workdir/box/$name.exit ]; then echo EXIT \$(cat workdir/box/$name.exit); elif tmux has-session -t $name 2>/dev/null; then echo RUNNING; else echo DEAD; fi" 2>/dev/null)
        case $out in
            EXIT*) echo "${out#EXIT }"; return ;;
            DEAD) echo dead; return ;;
            RUNNING) fails=0 ;;
            *) fails=$((fails + 1)); ((fails >= 10)) && { echo unreachable; return; } ;;
        esac
    done
    echo timeout
}

# 2. code and frozen inputs
cd "$REPO"
rsync -az -e "$SSHC" --filter=':- .gitignore' --exclude=.claude/ --exclude=.pytest_cache/ ./ "root@$IP:/workspace/Mortal-Policy/" >> "$LOG" 2>&1 || { log "code rsync failed"; exit 1; }
rsync -azR -e "$SSHC" workdir/pilot/qs-a/pilot-qs-a-final.pth workdir/grp-v2/grp-v2-frozen.pth "root@$IP:/workspace/Mortal-Policy/" >> "$LOG" 2>&1 || { log "frozen rsync failed"; exit 1; }
log "synced $(git describe --always --dirty --abbrev=12)"

# 3. setup (with tests)
rc=$(step setup 30 "tools/fleet/setup_box.sh"); log "setup: $rc"
[[ $rc == 0 ]] || exit 1

# 4. eval-mode benchmark, 15 min
rc=$(step bench 25 "$PY tools/fleet/bench_1v3.py --model workdir/pilot/qs-a/pilot-qs-a-final.pth --minutes 15 --out workdir/box/bench-eval.json"); log "bench: $rc"

# 5. extras: py-spy profile; P0 throughput at 800- and 2,000-game actor batches
if [[ $EXTRAS == full ]]; then
    rc=$(step pyspy 15 "$PY -m pip install -q py-spy && /venv/main/bin/py-spy record --native --rate 50 --format raw -o workdir/box/pyspy-raw.txt -- $PY tools/fleet/bench_1v3.py --model workdir/pilot/qs-a/pilot-qs-a-final.pth --minutes 2 --out workdir/box/bench-pyspy.json"); log "pyspy: $rc"
    for spec in 800:8 2000:14; do
        G=${spec%:*}; MIN=${spec#*:}
        R "bash -s" >> "$LOG" 2>&1 <<EOF
set -e
cd /workspace/Mortal-Policy
D=workdir/p0-tput-$G
mkdir -p \$D
sed -e "s#/workdir/p0/#/workdir/p0-tput-$G/#g" -e "s/^games = 800\$/games = $G/" \
    -e "s/^capacity = 1600\$/capacity = $((2 * G))/" -e "s/^port = 5000\$/port = $((5000 + G))/" \
    workdir/box/configs/p0.toml > \$D/cfg.toml
grep -E "^(games|capacity|port) = " \$D/cfg.toml
cp workdir/pilot/qs-a/pilot-qs-a-final.pth \$D/mortal.pth && chmod u+w \$D/mortal.pth
CFG=/workspace/Mortal-Policy/\$D/cfg.toml
cd mortal
tmux new-session -d -s p0-$G -n server "env MORTAL_CFG=\$CFG $PY server.py > ../\$D/server.log 2>&1"
sleep 4
tmux new-window -t p0-$G -n trainer "env MORTAL_CFG=\$CFG $PY train_online.py > ../\$D/trainer.log 2>&1"
tmux new-window -t p0-$G -n client "env MORTAL_CFG=\$CFG $PY client.py > ../\$D/client.log 2>&1"
tmux new-window -t p0-$G -n gpu "nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 5 > ../\$D/gpu.csv"
EOF
        log "p0-$G started for $MIN min"
        sleep $((MIN * 60))
        R "tmux list-windows -t p0-$G -F '#W' | tr '\n' ' '; tmux kill-session -t p0-$G" >> "$LOG" 2>&1
        log "p0-$G stopped"
    done
fi
exit 0
