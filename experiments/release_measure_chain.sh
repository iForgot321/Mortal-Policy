#!/bin/bash
# The anchor-release pilot's candidate measures on a Vast eval box (experiments/anchor-release.md;
# the user's yes to the rental, 2026-11-12). In order:
#   1. wait for both runs' `full` snapshots (sha256 as snapshots.jsonl records it), up to WAIT_HOURS;
#   2. wait for the workstation's own lab runs to end (the box takes the lab lease) and check that
#      they made the block `release` (soak-a full's two runs, which fix the deals);
#   3. rent at <= MAX_DPH $/h (re-quoted every 30 min, for up to 12 h);
#   4. queue the four measures: A full and B full, each at a table of pilot A and of pilot B,
#      40,000 sets on block `release`;
#   5. pull every hour (a box that vanishes keeps no results), and pull + destroy the box once its
#      queue drains (evalbox.py down refuses while a job runs). After MAX_HOURS it is forced down:
#      160K sets take ~14 h at the ~11K sets/h b8c192 measures ran at.
#   setsid -f bash experiments/release_measure_chain.sh >> workdir/fleet/runs/eval/release-measures.log 2>&1 < /dev/null
cd "$(dirname "$0")/.." || exit 2
E="python3 tools/fleet/evalbox.py"
MAX_DPH=0.75
MAX_HOURS=22
WAIT_HOURS=96
SETS=40000
BLOCK=release
PILOT_A=workdir/pilot/qs-a/pilot-qs-a-final.pth
PILOT_B=workdir/pilot/qs-b/pilot-qs-b-final.pth

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
full_ready() {  # the run's `full` snapshot is here and is the file the watcher recorded
    local snap=workdir/fleet/runs/$1/snapshots/full.pth want
    [[ -f $snap ]] || return 1
    want=$(python3 -I -c 'import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
print(next((r["sha256"] for r in rows if r["label"] == "full"), ""))' workdir/fleet/runs/$1/snapshots.jsonl 2>/dev/null)
    [[ -n $want && $(sha256sum "$snap" | cut -d' ' -f1) == "$want" ]]
}

say "waiting for release-a's and release-b's full snapshots"
for i in $(seq 1 $((WAIT_HOURS * 6))); do
    full_ready release-a && full_ready release-b && break
    sleep 600
done
if ! { full_ready release-a && full_ready release-b; }; then
    say "no full snapshot for both runs after $WAIT_HOURS h: no eval box"
    exit 1
fi
say "both full snapshots are in"
while ps -eo args | grep -q "[p]ython -m lab"; do say "a lab job still runs here; waiting"; sleep 300; done
if ! grep -q "\"block\": \"$BLOCK\"" lab/registry/nonces.jsonl; then
    say "block $BLOCK isn't in the ledger: soak-a full's runs haven't fixed the deals. Not rented"
    exit 1
fi

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
for x in a b; do
    for table in A B; do
        champion=$PILOT_A; [[ $table == B ]] && champion=$PILOT_B
        say "START submit: $x full at 3x pilot $table"
        if ! $E submit -- measure --sets $SETS --family A --block $BLOCK --arm P1 --package K2 --horizon H \
                --challenger workdir/fleet/runs/release-$x/snapshots/full.pth --champion $champion \
                --description "anchor release: release-$x full at 3x pilot $table"; then
            say "END submit failed"
            stop_billing "submit failed"
            exit 1
        fi
    done
done
say "END submit: four measures queued"

for i in $(seq 1 $((MAX_HOURS * 6))); do
    sleep 600
    if $E down; then
        say "eval box down"
        say "ALL DONE"
        exit 0
    fi
    (( i % 6 == 0 )) && $E pull
done
say "jobs still run after $MAX_HOURS h: a broken box, not long measures"
$E status
stop_billing "time ceiling"
say "ALL DONE (forced)"
