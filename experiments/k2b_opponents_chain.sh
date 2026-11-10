#!/bin/bash
# The k2b opponent check (experiments/k2b-opponents.md, pre-registered 2026-11-09): three 20K-set
# measures on the workstation, one at a time (RAM), then the paired reading. A measure whose registry
# row exists (same description, verdict 'measured') is skipped, so a restart doesn't repeat one.
# A failed measure stops the chain.
#   setsid -f bash experiments/k2b_opponents_chain.sh >> workdir/lab/k2b-opponents.log 2>&1 < /dev/null
cd "$(dirname "$0")/.." || exit 2
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/mortal/lib
PY=$HOME/miniconda3/envs/mortal/bin/python
K2B=workdir/fleet/runs/k2b/snapshots/full.pth
A=workdir/pilot/qs-a/pilot-qs-a-final.pth
B=workdir/pilot/qs-b/pilot-qs-b-final.pth
FLOOR=builtin:shanten-greedy
PURPOSE="k2b opponent check (experiments/k2b-opponents.md; exploratory)"

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
measure() {  # challenger, champion, block, description
    if measured "$4"; then
        say "already measured: $4"; return 0
    fi
    say "START $4"
    $PY -m lab measure --challenger "$1" --champion "$2" --sets 20000 --family A --block "$3" \
        --description "$4" --purpose "$PURPOSE"
    local rc=$?
    say "END $4 rc=$rc"
    return $rc
}

measure $K2B $FLOOR floor "k2b opponent check: k2b full vs 3x shanten-greedy" || exit 1
measure $A $FLOOR floor "k2b opponent check: pilot A vs 3x shanten-greedy (the pair's baseline)" || exit 1
measure $K2B $B aa-prime "k2b opponent check: k2b full vs 3x pilot B" || exit 1
say "START paired reading"
$PY experiments/k2b_opponents_paired.py
say "END paired reading rc=$?"
say "ALL DONE"
