# Shared by D68's curve launchers (experiments/curves_b0.sh, curves_launch.sh); sourced, not run.
# The caller sets RUN, UP (fleet.py up's arguments without --yes), HOURS (the run's wall-clock
# length at its full fleet, for the dollar stop) and WATCH (fleet.py watch's arguments after
# --run RUN, without --max-dollars).
#
# Limits, from the environment (the user, 2026-11-18: the budget is flexible, so a dearer box is
# fine when cheap ones are scarce):
#   MAX_DPH         $/h per box, for every box the run ever rents (fleet.py up --max-dph); 1.00
#   MAX_FLEET_DPH   the live quote for the whole fleet must be at or under this; 2.80
#   QUOTE_HOURS     how long to keep re-quoting (every 15 min) before giving up; 6
#   QUOTE_SLEEP     seconds between quotes; 900
#   MAX_DOLLARS     the watcher's stop; default 1.25 × the live quote × HOURS
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 2
F=${F:-python3 tools/fleet/fleet.py}
PY=$HOME/miniconda3/envs/mortal/bin/python
MAX_DPH=${MAX_DPH:-1.00}
MAX_FLEET_DPH=${MAX_FLEET_DPH:-2.80}
QUOTE_HOURS=${QUOTE_HOURS:-6}
QUOTE_SLEEP=${QUOTE_SLEEP:-900}
STUDENTS=workdir/secure/runs/students-s1/box/students-s1

say() { echo "$(date -u +%FT%TZ) [$RUN] $*"; }

has_boxes() {
    python3 -I -c 'import json, sys; sys.exit(0 if json.load(open(sys.argv[1])).get("boxes") else 1)' \
        "workdir/fleet/runs/$RUN/fleet.json" 2>/dev/null
}

# A failed launch leaves the run's state behind, and `up` refuses an existing run: move it aside.
set_aside() {
    local d=workdir/fleet/runs/$RUN n=1
    [[ -d $d ]] || return 0
    while [[ -e $d-failed$n ]]; do n=$((n + 1)); done
    mv "$d" "$d-failed$n" && say "moved the failed launch's state to $d-failed$n"
}

# Rent at a live quote within the limits, re-quoting every 15 min for up to QUOTE_HOURS.
rent() {
    local quote dph= rc
    for _ in $(seq 1 $((QUOTE_HOURS * 4))); do
        quote=$($F up $UP --max-dph "$MAX_DPH" 2>&1 | grep -o '≈ \$[0-9.]*/h for the fleet')
        dph=$(echo "$quote" | sed -n 's/.*\$\([0-9.]*\)\/h.*/\1/p')
        if [[ -n $dph ]] && awk -v p="$dph" -v m="$MAX_FLEET_DPH" 'BEGIN { exit !(p <= m) }'; then
            break
        fi
        say "quote '${quote:-none}': not within \$$MAX_DPH/h per box and \$$MAX_FLEET_DPH/h per fleet; again in 15 min"
        dph=
        sleep "$QUOTE_SLEEP"
    done
    if [[ -z $dph ]]; then
        say "no quote within the limits in $QUOTE_HOURS h: not rented"
        return 1
    fi
    MAX_DOLLARS=${MAX_DOLLARS:-$(awk -v p="$dph" -v h="$HOURS" 'BEGIN { x = 1.25 * p * h; print (x == int(x)) ? x : int(x) + 1 }')}
    say "quote: $quote (limits \$$MAX_DPH/h per box, \$$MAX_FLEET_DPH/h per fleet; stop at \$$MAX_DOLLARS)"
    say "START up"
    $F up $UP --max-dph "$MAX_DPH" --yes
    rc=$?
    say "END up rc=$rc"
    # rc 1 with boxes left means an actor didn't come up: the watcher replaces it
    if ! has_boxes; then
        say "no boxes are left: not launched"
        set_aside
        return 1
    fi
    echo "$MAX_DOLLARS" > "workdir/fleet/runs/$RUN/max-dollars"
}

watcher() {
    if ps -eo args | grep -q "[f]leet.py watch --run $RUN "; then say "the watcher already runs"; return; fi
    local stop=${MAX_DOLLARS:-$(cat "workdir/fleet/runs/$RUN/max-dollars" 2>/dev/null)}
    [[ -n $stop ]] || { say "no dollar stop recorded: set MAX_DOLLARS"; return 1; }
    setsid -f $F watch --run "$RUN" "${WATCH[@]}" --max-dollars "$stop" \
        >> "workdir/fleet/runs/$RUN/watch.out" 2>&1 < /dev/null
    say "watcher started (stop at \$$stop)"
}

# After a workstation or WSL restart: the watcher and the watchdog again, each only if missing.
resume() {
    watcher
    if ps -eo args | grep -q "[f]leet.py down --run $RUN "; then say "the watchdog already runs"
    else $F deadline --run "$RUN" --respawn-watchdog && say "watchdog respawned at the same deadline"; fi
    sleep 5
    ps -eo pid,etime,args | awk "/[f]leet.py (watch|down) --run $RUN /" | cut -c1-120
    $F status --run "$RUN" | tail -n 4
}
