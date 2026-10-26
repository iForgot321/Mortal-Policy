#!/bin/bash
# The fleet soak's eval box (experiments/fleet-soak.md, pre-registered 2026-10-26): up when soak-a's
# p25 snapshot lands; exploratory tracking measures of p25, then p50, against k2b full (10K sets each
# on the family-A block `soak`); pulled and downed once soak-a's fleet is down and the queue has
# drained (evalbox.py down refuses while jobs are queued or running, so it is retried).
#   setsid -f bash experiments/soak_eval.sh >> workdir/fleet/runs/soak-a/soak-eval.log 2>&1 < /dev/null
cd "$(dirname "$0")/.." || exit 2
E="python3 tools/fleet/evalbox.py"
RUN=workdir/fleet/runs/soak-a
REF=workdir/fleet/runs/k2b/snapshots/full.pth

say() { echo "$(date -u +%FT%TZ) $*"; }
snap() { grep -q "\"label\": \"$1\"" $RUN/snapshots.jsonl 2>/dev/null; }
fleet_down() {  # every box of the run gone (down, budget reached, watchdog or a stop)
    python3 -I -c 'import json, sys; sys.exit(0 if not json.load(open(sys.argv[1]))["boxes"] else 1)' $RUN/fleet.json
}
submit() {
    $E submit -- measure --challenger $RUN/snapshots/$1.pth --champion $REF --sets 10000 --family A --block soak \
        --description "fleet soak tracking: soak-a $1 vs k2b full" --purpose "fleet soak tracking (exploratory)"
}

say "waiting for soak-a's p25"
until snap p25 || fleet_down; do sleep 300; done
if ! snap p25; then say "the fleet went down before p25: no eval box"; exit 0; fi
$E up --yes || { say "eval box up failed"; exit 1; }
submit p25 && say "p25 queued"
until snap p50 || fleet_down; do sleep 300; done
snap p50 && submit p50 && say "p50 queued"
until fleet_down; do sleep 300; done
say "the fleet is down: pulling and downing the eval box once its queue drains"
for i in $(seq 1 24); do
    $E down && { say "eval box down"; exit 0; }
    sleep 600
done
# 4 h of refusals: a broken box, not a busy queue; pull what it can, then stop the billing
say "eval box still up after 4 h: pulling once, then forcing it down"
$E pull
$E down --force && say "eval box down (forced)"
