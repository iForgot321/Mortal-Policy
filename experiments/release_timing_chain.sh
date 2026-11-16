#!/bin/bash
# The stage-timing check (experiments/anchor-release.md, pre-registered 2026-11-16): B's p50 and p25
# at a table of pilot A, 40,000 sets each on block `release`, on the workstation, without game logs.
# A measure whose registry row exists (same description, verdict 'measured') is skipped, so a restart
# doesn't repeat one. A failed measure stops the chain. Then the paired reading.
#   setsid -f bash experiments/release_timing_chain.sh >> workdir/lab/release-timing.log 2>&1 < /dev/null
cd "$(dirname "$0")/.." || exit 2
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/mortal/lib
PY=$HOME/miniconda3/envs/mortal/bin/python
A=workdir/pilot/qs-a/pilot-qs-a-final.pth
PURPOSE="anchor-release stage-timing check (experiments/anchor-release.md; exploratory)"

say() { echo "$(date -u +%FT%TZ) $*"; }
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
measure() {  # snapshot label
    local desc="anchor release timing: release-b $1 at 3x pilot A"
    if measured "$desc"; then
        say "already measured: $desc"; return 0
    fi
    say "START $desc"
    $PY -m lab measure --challenger workdir/fleet/runs/release-b/snapshots/$1.pth --champion $A \
        --sets 40000 --family A --block release --arm P1 --package K2 --horizon H --no-logs \
        --description "$desc" --purpose "$PURPOSE"
    local rc=$?
    say "END $desc rc=$rc"
    return $rc
}

measure p50 || exit 1
say "START reading (p50)"; $PY experiments/release_timing.py; say "END reading rc=$?"
measure p25 || exit 1
say "START reading"; $PY experiments/release_timing.py; say "END reading rc=$?"
say "ALL DONE"
