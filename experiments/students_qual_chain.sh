#!/bin/bash
# The students' qualification on a Vast eval box (experiments/students-s1.md, fixed 2026-11-11; the
# user approved the box in advance on 2026-11-11 and started it on 2026-11-17, "Start it"). In order:
#   1. wait for the workstation's own lab runs to end (the box takes the lab lease);
#   2. rent at <= MAX_DPH $/h (re-quoted every 30 min, for up to 12 h);
#   3. queue the three T-NIs (each student's EMA weights against BC s1's), then the control's three
#      readings, in the note's order;
#   4. pull every hour (a box that vanishes keeps no results), and pull + destroy the box once its
#      queue drains (evalbox.py down refuses while a job runs). After MAX_HOURS it is forced down:
#      the worst case, every test to its cap, is 400K sets, ~40 h at ~10K sets/h; it should end well
#      before that, since GSPRTs usually stop early.
#   setsid -f bash experiments/students_qual_chain.sh >> workdir/fleet/runs/eval/students-qual.log 2>&1 < /dev/null
cd "$(dirname "$0")/.." || exit 2
E="python3 tools/fleet/evalbox.py"
MAX_DPH=0.75
MAX_HOURS=40
S=workdir/secure/runs/students-s1/box/students-s1
BC=workdir/teacher/teacher-bc-s1/ema-from-state.pth
NOSP_A=workdir/sp-ablation/nosp-a/nosp-a-final.pth

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
submit() {  # description, then the lab command and its flags
    local desc=$1; shift
    say "START submit: $desc"
    if ! $E submit -- "$@" --description "$desc"; then
        say "END submit failed: $desc"
        stop_billing "submit failed"
        exit 1
    fi
}

while ps -eo args | grep -q "[p]ython -m lab"; do say "a lab job still runs here; waiting"; sleep 300; done

for i in $(seq 1 24); do
    quote=$($E up 2>&1 | head -1)
    dph=$(echo "$quote" | sed -n 's/.*\$\([0-9.]*\)\/h.*/\1/p')
    if [[ -n $dph ]] && awk -v p="$dph" -v m="$MAX_DPH" 'BEGIN { exit !(p <= m) }'; then
        say "offer: $quote"
        break
    fi
    say "offer above \$$MAX_DPH/h or none ($quote); asking again in 30 min"
    dph=
    sleep 1800
done
[[ -n $dph ]] || { say "no offer at <= \$$MAX_DPH/h in 12 h: not rented"; exit 1; }

say "START up"
$E up --yes
rc=$?
say "END up rc=$rc"
if (( rc != 0 )); then
    stop_billing "up failed"
    exit 1
fi

for n in b8c192 b12c256 b16c256; do
    submit "students-s1 qualification: $n (EMA) vs BC s1 (EMA), T-NI" \
        run --test T-NI --family B --challenger $S/ema-$n.pth --champion $BC
done
submit "students-s1 control: distilled b8c192 vs the control (EMA both), T-std" \
    run --test T-std --family B --challenger $S/ema-b8c192.pth --champion $S/ema-control-b8c192.pth
submit "students-s1 control: the control (EMA) vs No-SP A, 20K measure" \
    measure --sets 20000 --family A --challenger $S/ema-control-b8c192.pth --champion $NOSP_A
submit "students-s1 control: BC s1 (EMA) vs No-SP A, T-std" \
    run --test T-std --family B --challenger $BC --champion $NOSP_A
say "END submit: six jobs queued"

for i in $(seq 1 $((MAX_HOURS * 6))); do
    sleep 600
    if $E down; then
        say "eval box down"
        say "ALL DONE"
        exit 0
    fi
    (( i % 6 == 0 )) && $E pull
done
say "jobs still run after $MAX_HOURS h: a broken box, not long tests"
$E status
stop_billing "time ceiling"
say "ALL DONE (forced)"
