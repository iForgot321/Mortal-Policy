#!/bin/bash
# The teacher vs quick student T-std on a Vast eval box (experiments/teacher-vs-quick.md; the user's
# yes to the rental, 2026-11-10). The box takes the lab lease, which would stop the workstation's
# next lab run, so this first waits for the k2b opponent chain (its PID, the first argument) to end
# with ALL DONE. Then: rent at <= MAX_DPH $/h (re-quoted every 30 min, for up to 6 h), queue the
# T-std, and pull + destroy the box once its queue drains (evalbox.py down refuses while a job runs,
# so it is retried every 10 min). After MAX_HOURS with the job still running, the box is a broken
# one, not a long test (the GSPRT stops by ~20K sets almost surely, ~2-3 h): pull once, force it down.
#   setsid -f bash experiments/teacher_vs_quick_chain.sh CHAIN_PID >> workdir/fleet/runs/eval/teacher-vs-quick.log 2>&1 < /dev/null
cd "$(dirname "$0")/.." || exit 2
CHAIN_PID=${1:?give the PID of the k2b opponent chain}
E="python3 tools/fleet/evalbox.py"
CHAIN_LOG=workdir/lab/k2b-opponents.log
MAX_DPH=0.75
MAX_HOURS=6

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

say "waiting for the k2b opponent chain (pid $CHAIN_PID)"
while kill -0 "$CHAIN_PID" 2>/dev/null; do sleep 60; done
if ! tail -1 $CHAIN_LOG | grep -q "ALL DONE"; then
    say "the k2b chain ended without ALL DONE (last line: $(tail -1 $CHAIN_LOG)): no eval box"
    exit 1
fi
while ps -eo args | grep -q "[p]ython -m lab"; do say "a lab job still runs here; waiting"; sleep 300; done

for i in $(seq 1 12); do
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
[[ -n $dph ]] || { say "no offer at <= \$$MAX_DPH/h in 6 h: not rented"; exit 1; }

say "START up"
$E up --yes
rc=$?
say "END up rc=$rc"
if (( rc != 0 )); then
    stop_billing "up failed"
    exit 1
fi
say "START submit"
if ! $E submit -- run --test T-std --family B \
        --challenger workdir/teacher/teacher-bc-s1/ema.pth \
        --champion workdir/sp-ablation/nosp-a/nosp-a-final.pth \
        --description "teacher vs quick student: BC s1 (EMA) vs No-SP A"; then
    say "END submit failed"
    stop_billing "submit failed"
    exit 1
fi
say "END submit: queued"

for i in $(seq 1 $((MAX_HOURS * 6))); do
    sleep 600
    if $E down; then
        say "eval box down"
        say "ALL DONE"
        exit 0
    fi
done
say "the job still runs after $MAX_HOURS h: a broken box, not a long test"
$E status
stop_billing "time ceiling"
say "ALL DONE (forced)"
