#!/usr/bin/env bash
# Unattended Phase 0 chain on the workstation (STATUS.md "Next steps" 1–3):
#   wait for pilot A → freeze A → train pilot B → freeze B → A/A' (20K sets, family A)
#   → §4.1 calibration report.
# Stops before anything that edits lab configuration (lab.toml calibration is a
# reviewed decision). Each step verifies the previous one; any failure aborts.
#
# Usage: experiments/run_pilot_chain.sh <PID of the running pilot-A trainer>
# Progress: workdir/pilot/chain.log
set -euo pipefail

ROOT=/home/iforgot/projects/Mortal-Policy
PY=$HOME/miniconda3/envs/mortal/bin/python
LOG=$ROOT/workdir/pilot/chain.log
A_PID=${1:?pid of the running pilot-A trainer}
MIN_STEPS=150000   # max_steps is 158,000; the last save lands within 2K steps of the end

log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

check_and_freeze() {  # <name>
    local dir=$ROOT/workdir/pilot/qs-$1
    if tr '\r' '\n' < "$dir/train.log" | grep -a -q -E "Traceback|Error|Killed"; then
        log "FAIL: pilot $1 log shows an error"; exit 1
    fi
    local steps
    steps=$("$PY" -c "import torch; print(torch.load('$dir/mortal.pth', weights_only=False, map_location='cpu')['steps'])")
    if (( steps < MIN_STEPS )); then
        log "FAIL: pilot $1 stopped at $steps steps (< $MIN_STEPS)"; exit 1
    fi
    cp "$dir/mortal.pth" "$dir/pilot-qs-$1-final.pth"
    chmod a-w "$dir/pilot-qs-$1-final.pth"
    log "pilot $1 done at $steps steps; frozen $(sha256sum "$dir/pilot-qs-$1-final.pth" | cut -c1-16)"
}

log "chain start; waiting for pilot A (pid $A_PID)"
while kill -0 "$A_PID" 2>/dev/null; do sleep 60; done
check_and_freeze a

log "training pilot B"
mkdir -p "$ROOT/workdir/pilot/qs-b"
(cd "$ROOT/mortal" && MORTAL_CFG=../experiments/configs/pilot-qs-b.toml "$PY" train_offline_phase1.py \
    > "$ROOT/workdir/pilot/qs-b/train.log" 2>&1) || { log "FAIL: pilot B trainer exited non-zero"; exit 1; }
check_and_freeze b

log "A/A' (20K sets, family A, block aa-prime)"
cd "$ROOT"
"$PY" -m lab measure \
    --challenger workdir/pilot/qs-a/pilot-qs-a-final.pth \
    --champion workdir/pilot/qs-b/pilot-qs-b-final.pth \
    --sets 20000 --family A --block aa-prime --purpose "A/A' noise floor (02 §6)" \
    --description "A/A': pilot quick students A (seed 1) vs B (seed 2), same recipe" \
    > "$ROOT/workdir/pilot/aa-prime.log" 2>&1 || { log "FAIL: A/A' run"; exit 1; }
RUN_DIR=$(tail -1 "$ROOT/lab/registry/runs.jsonl" | "$PY" -c "import json,sys; print(json.load(sys.stdin)['run_dir'])")
log "A/A' done: $RUN_DIR"

log "calibration (§4.1) from the A/A' set-score distribution"
"$PY" -m lab calibrate --from "$RUN_DIR/games.npz" > "$ROOT/workdir/pilot/calibration.log" 2>&1 \
    || { log "FAIL: calibration"; exit 1; }
log "chain complete; review workdir/pilot/calibration.log before setting lab.toml calibrated = true"
