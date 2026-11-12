#!/bin/bash
# The anchor-release pilot's comparator (experiments/anchor-release.md, pre-registered 2026-11-12):
# soak-a full at a table of pilot A and at a table of pilot B, 40,000 sets each, on the workstation
# while the fleets train. The first run makes the family-A block `release`; every other measure of
# the pilot plays the same deals. A measure whose registry row exists (same description, verdict
# 'measured') is skipped, so a restart doesn't repeat one. A failed measure stops the chain.
#   setsid -f bash experiments/release_soak_chain.sh >> workdir/lab/release-soak.log 2>&1 < /dev/null
cd "$(dirname "$0")/.." || exit 2
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/mortal/lib
PY=$HOME/miniconda3/envs/mortal/bin/python
SOAK=workdir/fleet/runs/soak-a/snapshots/full.pth
A=workdir/pilot/qs-a/pilot-qs-a-final.pth
B=workdir/pilot/qs-b/pilot-qs-b-final.pth
PURPOSE="anchor-release pilot (experiments/anchor-release.md; exploratory)"

say() { echo "$(date -u +%FT%TZ) $*"; }
measured() {  # a finished measure row with this description
    python3 -I - "$1" <<'EOF'
import json, sys
for line in open('lab/registry/runs.jsonl'):
    r = json.loads(line)
    if r.get('test') == 'measure' and r.get('verdict') == 'measured' and r.get('description') == sys.argv[1]:
        sys.exit(0)
sys.exit(1)
EOF
}
measure() {  # champion, description
    if measured "$2"; then
        say "already measured: $2"; return 0
    fi
    say "START $2"
    $PY -m lab measure --challenger $SOAK --champion "$1" --sets 40000 --family A --block release \
        --arm P1 --package K2 --horizon H --description "$2" --purpose "$PURPOSE"
    local rc=$?
    say "END $2 rc=$rc"
    return $rc
}

measure $A "anchor release: soak-a full at 3x pilot A" || exit 1
measure $B "anchor release: soak-a full at 3x pilot B" || exit 1
say "ALL DONE"
