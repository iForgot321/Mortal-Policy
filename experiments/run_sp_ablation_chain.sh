#!/usr/bin/env bash
# SP ablation, unattended on the workstation (docs/design/sp-ablation-handoff.md M3–M6):
#   train no-SP A → freeze → mixed-version smoke → T-NI vs pilot A + style panel
#   → only if that passes: train no-SP B → freeze → smoke → T-NI vs pilot B + style panel.
# A fail or an inconclusive verdict stops the chain (M5: one failed pair is enough to keep SP;
# inconclusive goes to the user). Arena work never overlaps training (RAM). Any error aborts.
#
# Usage: experiments/run_sp_ablation_chain.sh [a|b]   (the pair to start from; default a)
# Progress: workdir/sp-ablation/chain.log
set -euo pipefail

ROOT=/home/iforgot/projects/Mortal-Policy
PY=$HOME/miniconda3/envs/mortal/bin/python
WD=$ROOT/workdir/sp-ablation
LOG=$WD/chain.log
MIN_STEPS=150000   # max_steps is 158,000; the last save lands within 2K steps of the end
START=${1:-a}

log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

mem_used_gb() {  # MemTotal - MemAvailable
    awk '/^MemTotal:/{t=$2} /^MemAvailable:/{a=$2} END{printf "%.1f", (t-a)/2^20}' /proc/meminfo
}

train() {  # <a|b>
    local x=$1 dir=$WD/nosp-$1
    mkdir -p "$dir"
    log "training no-SP $x (sp-nosp-$x.toml); used RAM before start $(mem_used_gb) GB"
    (cd "$ROOT/mortal" && MORTAL_CFG=../experiments/configs/sp-nosp-$x.toml exec "$PY" train_offline_phase1.py) \
        > "$dir/train.log" 2>&1 &
    local pid=$! peak=0 now
    echo "$pid" > "$dir/train.pid"
    log "trainer pid $pid"
    while kill -0 "$pid" 2>/dev/null; do
        now=$(mem_used_gb)
        if awk -v n="$now" -v p="$peak" 'BEGIN{exit !(n > p)}'; then peak=$now; fi
        echo "$(date '+%F %T') used_gb $now" >> "$dir/ram.log"
        sleep 60
    done
    if ! wait "$pid"; then log "FAIL: no-SP $x trainer exited non-zero (peak used RAM $peak GB)"; exit 1; fi
    log "no-SP $x trainer exited; peak used RAM (system, 1-min samples) $peak GB"
}

check_and_freeze() {  # <a|b>
    local x=$1 dir=$WD/nosp-$1
    if tr '\r' '\n' < "$dir/train.log" | grep -a -q -E "Traceback|Error|Killed"; then
        log "FAIL: no-SP $x log shows an error"; exit 1
    fi
    local steps
    steps=$("$PY" -c "import torch; print(torch.load('$dir/mortal.pth', weights_only=False, map_location='cpu')['steps'])")
    if (( steps < MIN_STEPS )); then
        log "FAIL: no-SP $x stopped at $steps steps (< $MIN_STEPS)"; exit 1
    fi
    cp "$dir/mortal.pth" "$dir/nosp-$x-final.pth"
    chmod a-w "$dir/nosp-$x-final.pth"
    log "no-SP $x done at $steps steps; frozen $(sha256sum "$dir/nosp-$x-final.pth" | cut -c1-16)"
}

smoke() {  # <a|b>: v5 vs v4 through the lab's arena on the benchmark-only key (no nonces)
    local x=$1
    (cd "$ROOT" && "$PY" - "$WD/nosp-$x/nosp-$x-final.pth" "workdir/pilot/qs-$x/pilot-qs-$x-final.pth" \
        > "$WD/smoke-$x.log" 2>&1 <<'EOF'
import sys
import numpy as np
from lab.arena import load_engine, play_sets
chal = load_engine(sys.argv[1], 'cuda:0', 'challenger')
champ = load_engine(sys.argv[2], 'cuda:0', 'champion')
assert (chal.version, champ.version) == (5, 4), (chal.version, champ.version)
seeds, seats, ranks, scores = play_sets(chal, champ, 0x62656E6368, 0, 16)
assert len(ranks) == 64 and set(ranks.tolist()) <= {0, 1, 2, 3}
print(f'smoke: 16 sets, 64 games, v5 challenger avg rank {ranks.mean() + 1:.3f}')
EOF
    ) || { log "FAIL: smoke $x"; exit 1; }
    log "$(grep '^smoke:' "$WD/smoke-$x.log")"
}

tni() {  # <a|b> <pair number> <seed>
    local x=$1 n=$2 seed=$3 X
    X=$(echo "$x" | tr a-z A-Z)
    log "T-NI pair $n: no-SP $x vs pilot $x"
    (cd "$ROOT" && "$PY" -m lab run --test T-NI \
        --challenger "workdir/sp-ablation/nosp-$x/nosp-$x-final.pth" \
        --champion "workdir/pilot/qs-$x/pilot-qs-$x-final.pth" \
        --family B \
        --description "SP ablation pair $n: v5 (no SP) seed $seed vs pilot $X (v4) seed $seed, matched recipe") \
        > "$WD/tni-$x.log" 2>&1 || { log "FAIL: T-NI pair $n"; exit 1; }
    local run_id verdict
    run_id=$(grep -m1 -oE '^[0-9]{8}-[0-9]{6}-T-NI' "$WD/tni-$x.log")
    verdict=$(grep '^verdict:' "$WD/tni-$x.log" | awk '{print $2}')
    log "T-NI pair $n: run $run_id, verdict $verdict"
    (cd "$ROOT" && "$PY" -m lab panel --run "$run_id" > "$WD/panel-$x.log" 2>&1 \
        && "$PY" -m lab table >> "$WD/panel-$x.log" 2>&1) || log "WARN: panel/table for $run_id failed"
    VERDICT=$verdict
}

log "chain start at pair $START"
if [[ $START == a ]]; then
    train a
    check_and_freeze a
    smoke a
    tni a 1 1
    if [[ $VERDICT != pass ]]; then
        log "pair 1 verdict $VERDICT: stopping (M5); write up and report"
        exit 0
    fi
fi
train b
check_and_freeze b
smoke b
tni b 2 2
log "chain complete: pair 2 verdict $VERDICT"
