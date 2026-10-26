#!/bin/bash
# The fleet soak's tracking measures, on the workstation (experiments/fleet-soak.md; the user chose this
# over an eval box, 2026-10-26): as each of soak-a's snapshots lands (p25, p50, full), a 10K-set measure
# against k2b full on the family-A block `soak` (exploratory). It waits for the post-G0 queue
# (k2b_post_g0_chain.sh) to finish first, so two arena jobs never share the workstation's RAM.
# A measure whose registry row exists for these checkpoints is skipped, so a restart doesn't repeat one.
#   setsid -f bash experiments/soak_tracking.sh >> workdir/fleet/runs/soak-a/tracking.log 2>&1 < /dev/null
cd "$(dirname "$0")/.." || exit 2
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/mortal/lib
PY=$HOME/miniconda3/envs/mortal/bin/python
RUN=workdir/fleet/runs/soak-a
REF=workdir/fleet/runs/k2b/snapshots/full.pth

say() { echo "$(date -u +%FT%TZ) $*"; }
snap() { grep -q "\"label\": \"$1\"" $RUN/snapshots.jsonl 2>/dev/null; }
fleet_down() {
    python3 -I -c 'import json, sys; sys.exit(0 if not json.load(open(sys.argv[1]))["boxes"] else 1)' $RUN/fleet.json
}
measured() {  # a finished measure row for (challenger sha256, champion sha256)
    python3 -I - "$1" "$2" <<'EOF'
import json, sys
for line in open('lab/registry/runs.jsonl'):
    r = json.loads(line)
    if (r.get('test') == 'measure' and r.get('verdict') != 'aborted'
            and r['challenger'].get('sha256') == sys.argv[1] and r['champion'].get('sha256') == sys.argv[2]):
        sys.exit(0)
sys.exit(1)
EOF
}

ref_sha=$(sha256sum $REF | cut -d' ' -f1)
for label in p25 p50 full; do
    say "waiting for soak-a's $label"
    until snap $label || fleet_down; do sleep 300; done
    snap $label || { say "the fleet went down without $label"; exit 0; }
    while ps -eo args | grep -q "[k]2b_post_g0_chain.sh"; do sleep 300; done
    if measured "$(sha256sum $RUN/snapshots/$label.pth | cut -d' ' -f1)" "$ref_sha"; then
        say "$label already measured"; continue
    fi
    say "START $label vs k2b full"
    $PY -m lab measure --challenger $RUN/snapshots/$label.pth --champion $REF --sets 10000 --family A --block soak \
        --description "fleet soak tracking: soak-a $label vs k2b full" --purpose "fleet soak tracking (exploratory)"
    say "END $label rc=$?"
done
say "ALL DONE"
