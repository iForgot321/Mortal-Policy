#!/bin/bash
# One candidate of the anchor-release pilot (experiments/anchor-release.md): rent and start its
# fleet, then its watcher. The live quote is checked first: above MAX_FLEET_DPH nothing is rented.
#   setsid -f bash experiments/release_launch.sh a|b >> workdir/fleet/release-X-launch.out 2>&1 < /dev/null
# After a workstation restart, the watcher and the watchdog again, each only if missing:
#   bash experiments/release_launch.sh a|b --resume
# MAX_FLEET_DPH (default 1.90 $/h) and MAX_DOLLARS (default 95, the run's stop) are the approved limits.
cd "$(dirname "$0")/.." || exit 2
X=${1:?a or b}
RUN=release-$X
F="python3 tools/fleet/fleet.py"
MAX_FLEET_DPH=${MAX_FLEET_DPH:-1.90}
MAX_DOLLARS=${MAX_DOLLARS:-95}
UP="--run $RUN --config experiments/configs/release-$X.toml --init workdir/release/init-$X.pth --actors 2 --procs 2 --hard-hours 52 --verify"

say() { echo "$(date -u +%FT%TZ) $*"; }
watch() {
    if ps -eo args | grep -q "[f]leet.py watch --run $RUN "; then say "the watcher already runs"; return; fi
    setsid -f $F watch --run $RUN --replace --allow-zen2 --budget-box-hours 144 --snapshot-at 0.25,0.5 \
        --max-dollars "$MAX_DOLLARS" --anchor-alert 0.10 >> workdir/fleet/runs/$RUN/watch.out 2>&1 < /dev/null
    say "watcher started (stop at \$$MAX_DOLLARS)"
}

if [[ ${2:-} == --resume ]]; then
    watch
    if ps -eo args | grep -q "[f]leet.py down --run $RUN "; then say "the watchdog already runs"
    else $F deadline --run $RUN --respawn-watchdog && say "watchdog respawned at the same deadline"; fi
    sleep 5
    ps -eo pid,etime,args | awk "/[f]leet.py (watch|down) --run $RUN /" | cut -c1-110
    $F status --run $RUN | tail -n 3
    exit 0
fi

quote=$($F up $UP 2>&1 | grep -o '≈ \$[0-9.]*/h for the fleet')
dph=$(echo "$quote" | sed -n 's/.*\$\([0-9.]*\)\/h.*/\1/p')
if [[ -z $dph ]] || ! awk -v p="$dph" -v m="$MAX_FLEET_DPH" 'BEGIN { exit !(p <= m) }'; then
    say "quote '${quote:-none}' is above \$$MAX_FLEET_DPH/h or missing: not rented"
    exit 1
fi
say "quote: $quote (limit \$$MAX_FLEET_DPH/h)"
say "START up"
$F up $UP --yes
rc=$?
say "END up rc=$rc"
# rc 1 with boxes left means an actor didn't come up: the watcher replaces it. No boxes: torn down.
if ! python3 -I -c 'import json, sys; sys.exit(0 if json.load(open(sys.argv[1])).get("boxes") else 1)' \
        workdir/fleet/runs/$RUN/fleet.json 2>/dev/null; then
    say "no boxes are left: not launched"
    exit 1
fi
watch
say "LAUNCHED (up rc=$rc)"
