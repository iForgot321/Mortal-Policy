#!/bin/bash
# D31's curves, the final measures (experiments/size-curves.md, "The measures"), on one Vast eval
# box once every run's p50 and `full` are in (after the last batch: while the box holds the lab
# registry, the workstation's stage checks can't play). In order:
#   1. wait for each run's p50 and `full` snapshots (sha256 as snapshots.jsonl records them);
#   2. wait for the workstation's lab jobs and stage checks to end (the box takes the lab lease);
#   3. rent at <= MAX_DPH $/h, retried every 30 min for up to 12 h (evalbox.py up --max-dph keeps
#      its provisioning retries within the limit too);
#   4. queue three measures per run, 40,000 sets each on family A block `curves` (the same deals
#      for all), AMP on: `full` and p50 against three copies of the run's starting student, then
#      `full` against three copies of BC s1 (EMA), the held-out table. A measure already in the
#      registry (same description) isn't queued again, so a relaunch resumes;
#   5. pull every hour, and pull + destroy the box once its queue drains (evalbox.py down refuses
#      while a job runs); after MAX_HOURS it is forced down;
#   6. the pre-registered reading (experiments/curves_reading.py).
#
#   setsid -f bash experiments/curves_measure_chain.sh [RUN ...] >> workdir/fleet/runs/eval/curves-measures.log 2>&1 < /dev/null
#
# RUNS default to the four runs of batches 1 and 2; name all six if batch 3 ran.
cd "$(dirname "$0")/.." || exit 2
E="python3 tools/fleet/evalbox.py"
PY=$HOME/miniconda3/envs/mortal/bin/python
MAX_DPH=${MAX_DPH:-1.00}
MAX_HOURS=${MAX_HOURS:-60}
WAIT_HOURS=${WAIT_HOURS:-240}
SETS=40000
BLOCK=curves
STUDENTS=workdir/secure/runs/students-s1/box/students-s1
BC=workdir/teacher/teacher-bc-s1/ema-from-state.pth
RUNS=("$@")
(( ${#RUNS[@]} )) || RUNS=(curves-b12-s1 curves-b16-s1 curves-b12-s2 curves-b16-s2)

say() { echo "$(date -u +%FT%TZ) $*"; }
has_box() {
    python3 -I -c 'import json, sys; sys.exit(0 if json.load(open(sys.argv[1])).get("boxes") else 1)' \
        workdir/fleet/runs/eval/fleet.json 2>/dev/null
}
stop_billing() {  # whatever state the box is in
    has_box || return 0
    say "downing the eval box ($1)"
    $E down || { $E pull; $E down --force; }
}
snap_ready() {  # run label: the snapshot is here and is the file the watcher recorded
    local snap=workdir/fleet/runs/$1/snapshots/$2.pth want
    [[ -f $snap ]] || return 1
    want=$(python3 -I -c 'import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
print(next((r["sha256"] for r in rows if r["label"] == sys.argv[2]), ""))' "workdir/fleet/runs/$1/snapshots.jsonl" "$2" 2>/dev/null)
    [[ -n $want && $(sha256sum "$snap" | cut -d' ' -f1) == "$want" ]]
}
all_ready() {
    local run
    for run in "${RUNS[@]}"; do snap_ready "$run" p50 && snap_ready "$run" full || return 1; done
}
measured() {  # a finished measure row with this description
    python3 -I - "$1" <<'PYEOF'
import json, sys
for line in open('lab/registry/runs.jsonl'):
    r = json.loads(line)
    if r.get('test') == 'measure' and r.get('verdict') == 'measured' and r.get('description') == sys.argv[1]:
        sys.exit(0)
sys.exit(1)
PYEOF
}
net_of() { case $1 in curves-b12-*) echo b12c256 ;; curves-b16-*) echo b16c256 ;; esac; }

say "runs: ${RUNS[*]}; waiting for their p50 and full snapshots"
for _ in $(seq 1 $((WAIT_HOURS * 6))); do
    all_ready && break
    sleep 600
done
all_ready || { say "not every p50 and full snapshot after $WAIT_HOURS h: no eval box"; exit 1; }
say "every snapshot is in"
while ps -eo args | grep -q -e "[p]ython -m lab" -e "[c]urves_stage_check.py"; do
    say "a lab job or stage check still runs here; waiting"
    sleep 300
done

jobs=()  # "run label champion-file champion-name"
for kind in full p50 heldout; do
    for run in "${RUNS[@]}"; do
        net=$(net_of "$run")
        case $kind in
        full) jobs+=("$run full $STUDENTS/ema-$net.pth student") ;;
        p50) jobs+=("$run p50 $STUDENTS/ema-$net.pth student") ;;
        heldout) jobs+=("$run full $BC BC-s1") ;;
        esac
    done
done
todo=()
for j in "${jobs[@]}"; do
    read -r run label champ name <<<"$j"
    measured "curves: $run $label at 3x $name" || todo+=("$j")
done
say "${#todo[@]} of ${#jobs[@]} measures to play"

drain() {  # wait for the box's queue, pulling every hour; down it when it drains
    local i
    for i in $(seq 1 $((MAX_HOURS * 6))); do
        sleep 600
        if $E down; then say "eval box down"; return 0; fi
        (( i % 6 == 0 )) && $E pull
    done
    say "jobs still run after $MAX_HOURS h: a broken box, not long measures"
    $E status
    stop_billing "time ceiling"
}

if has_box; then
    # a relaunch while the box works: its queue holds this chain's jobs already
    say "an eval box is up: waiting for its queue, not queueing again"
    drain
elif (( ${#todo[@]} )); then
    rented=
    for _ in $(seq 1 24); do
        say "START up (at most \$$MAX_DPH/h)"
        # a failed up destroys its box and releases the lease itself
        if $E up --max-dph "$MAX_DPH" --yes; then rented=1; say "END up"; break; fi
        say "END up failed; again in 30 min"
        sleep 1800
    done
    [[ -n $rented ]] || { say "no eval box at <= \$$MAX_DPH/h in 12 h: not rented"; exit 1; }
    for j in "${todo[@]}"; do
        read -r run label champ name <<<"$j"
        desc="curves: $run $label at 3x $name"
        say "START submit: $desc"
        if ! $E submit -- measure --sets $SETS --family A --block $BLOCK --arm P1 --package K2 --horizon H \
                --challenger "workdir/fleet/runs/$run/snapshots/$label.pth" --champion "$champ" \
                --description "$desc" --purpose "D31's curves (experiments/size-curves.md; exploratory)"; then
            say "END submit failed"
            stop_billing "submit failed"
            exit 1
        fi
    done
    say "END submit: ${#todo[@]} measures queued"
    drain
fi
missing=0
for j in "${jobs[@]}"; do
    read -r run label champ name <<<"$j"
    measured "curves: $run $label at 3x $name" || { say "MISSING: curves: $run $label at 3x $name"; missing=1; }
done
(( missing )) && say "some measures are missing: the reading is partial; relaunch this chain to play them"
say "START reading"
$PY experiments/curves_reading.py "${RUNS[@]}"
say "END reading rc=$?"
say "ALL DONE"
