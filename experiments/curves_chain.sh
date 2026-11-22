#!/bin/bash
# D31's curves end to end (experiments/size-curves.md), each step gated on the one before:
#   1. both B0s at once (curves_b0.sh b12, b16); wait until both are down;
#   2. curves_b0.sh SIZE --finish for both: the B0 rule, then the curves' inits. A failed rule
#      stops here (its one 4 box-hour extension is a rental of its own);
#   3. batch 1: curves_launch.sh for both sizes' seed 1, the second 3 min after the first; wait
#      until both are down. A run that ended without `full` (a stage check's stop, the dollar
#      stop, the deadline) stops here;
#   4. batch 2, the same, but only on the user's go (the user, 2026-11-22): without --batch2 the
#      chain ends after batch 1. A batch 2 already launched (a run of it live or done) goes on
#      without the flag, so a relaunch after a restart mid-batch doesn't need it;
#   5. the final measures on one eval box, then the reading (curves_measure_chain.sh).
# Batch 3, if the reading is close, needs the user's yes on a live quote: it isn't started here.
#
#   setsid -f bash experiments/curves_chain.sh >> workdir/fleet/curves-chain.log 2>&1 < /dev/null
#   setsid -f bash experiments/curves_chain.sh --batch2 >> workdir/fleet/curves-chain.log 2>&1 < /dev/null
#
# Safe to relaunch after a restart (first `bash experiments/curves_launch.sh RUN --resume` or
# `curves_b0.sh SIZE --resume` for any live run's watcher): a step whose result exists is skipped
# and a live run is waited on, not rented again. Limits: experiments/curves_common.sh.
BATCH2=
case ${1:-} in
--batch2) BATCH2=1 ;;
'') ;;
*) echo "usage: curves_chain.sh [--batch2]" >&2; exit 2 ;;
esac
GAP=${CHAIN_GAP:-180}  # between a batch's two launches; the tests shorten it
cd "$(dirname "$0")/.." || exit 2
say() { echo "$(date -u +%FT%TZ) [chain] $*"; }

state() {  # RUN: none | live | full | ended (down without `full`)
    python3 -I - "$1" <<'PYEOF'
import json, os, sys
p = f'workdir/fleet/runs/{sys.argv[1]}/fleet.json'
if not os.path.exists(p):
    print('none'); sys.exit()
d = json.load(open(p))
print('live' if d.get('boxes') else 'full' if 'full' in d.get('snapshots', {}) else 'ended')
PYEOF
}

wait_down() {  # RUN...: until none is live (a watcher downs its run at `full` or a stop)
    local r busy
    while :; do
        busy=
        for r in "$@"; do [[ $(state "$r") == live ]] && busy=1; done
        [[ -z $busy ]] && return 0
        sleep 600
    done
}

# STEP NAME RUN... : launch each run not yet launched (LAUNCH_<run> holds the command), wait for
# all to be down, and require `full` from each
step() {
    local name=$1 r pids=() rc=0 first=1
    shift
    for r in "$@"; do
        case $(state "$r") in
        none)
            [[ -z $first ]] && sleep "$GAP"  # two `up`s at once race for the same offers
            first=
            say "$name: launching $r"
            bash -c "${LAUNCH[$r]}" >> "workdir/fleet/$r-launch.out" 2>&1 < /dev/null &
            pids+=($!) ;;
        live) say "$name: $r is live; waiting for it" ;;
        *) say "$name: $r is done ($(state "$r"))" ;;
        esac
    done
    for p in "${pids[@]}"; do wait "$p" || rc=1; done
    if (( rc )); then
        say "$name: a launch failed (workdir/fleet/RUN-launch.out); stopping"
        wait_down "$@"
        exit 1
    fi
    wait_down "$@"
    for r in "$@"; do
        if [[ $(state "$r") != full ]]; then
            say "$name: $r ended without its full snapshot ($(state "$r")): stopping for the user"
            exit 1
        fi
    done
    say "$name: done"
}

declare -A LAUNCH=(
    [curves-b12-b0]="bash experiments/curves_b0.sh b12" [curves-b16-b0]="bash experiments/curves_b0.sh b16"
    [curves-b12-s1]="bash experiments/curves_launch.sh curves-b12-s1" [curves-b16-s1]="bash experiments/curves_launch.sh curves-b16-s1"
    [curves-b12-s2]="bash experiments/curves_launch.sh curves-b12-s2" [curves-b16-s2]="bash experiments/curves_launch.sh curves-b16-s2"
)

say "START${BATCH2:+ (--batch2)}"
if [[ ! -f workdir/curves/init-b12c256.pth || ! -f workdir/curves/init-b16c256.pth ]]; then
    step B0 curves-b12-b0 curves-b16-b0
    for s in b12 b16; do
        bash experiments/curves_b0.sh $s --finish || { say "B0 $s didn't pass its rule: stopping for the user"; exit 1; }
    done
fi
step "batch 1" curves-b12-s1 curves-b16-s1
if [[ -z $BATCH2 && $(state curves-b12-s2) == none && $(state curves-b16-s2) == none ]]; then
    say "batch 2 waits for the user's go: relaunch with --batch2 to rent it"
    exit 0
fi
step "batch 2" curves-b12-s2 curves-b16-s2
say "START measures"
bash experiments/curves_measure_chain.sh >> workdir/fleet/runs/eval/curves-measures.log 2>&1 < /dev/null
say "END measures rc=$? (workdir/fleet/runs/eval/curves-measures.log; the reading: workdir/lab/curves-reading.json)"
say "ALL DONE"
