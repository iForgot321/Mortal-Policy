#!/bin/bash
# G0 for the K2 retry, started by itself (the user's go-ahead 2026-10-21; experiments/k2-retry.md).
# Waits until the watcher has pulled and verified k2b's `full` snapshot, then runs on this workstation:
#   1. G0's T-std, full vs pilot A: D59's verdict;
#   2. the pre-registered doubling estimate, full vs p50, 100K sets on family B (D49).
# A step whose registry row already exists for these checkpoints is skipped (aborted rows don't count),
# so a restart (workdir/fleet/runs/k2b/resume-supervision.sh) doesn't repeat a finished step. An
# interrupted step starts again on fresh nonces.
#   setsid -f bash experiments/k2b_g0_chain.sh >> workdir/fleet/runs/k2b/g0-chain.log 2>&1 < /dev/null
cd "$(dirname "$0")/.." || exit 2
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/mortal/lib
PY=$HOME/miniconda3/envs/mortal/bin/python
RUN=workdir/fleet/runs/k2b
FULL=$RUN/snapshots/full.pth
P50=$RUN/snapshots/p50.pth
INIT=workdir/pilot/qs-a/pilot-qs-a-final.pth

say() { echo "$(date -u +%FT%TZ) $*"; }

# the registry's run_id for (test, challenger sha256, champion sha256), if it has a finished row
done_row() {
    python3 -I - "$@" <<'EOF'
import json, sys
test, challenger, champion = sys.argv[1:4]
for line in open('lab/registry/runs.jsonl'):
    r = json.loads(line)
    if (r.get('test') == test and r.get('verdict') != 'aborted'
            and r.get('challenger', {}).get('sha256') == challenger and r.get('champion', {}).get('sha256') == champion):
        print(r['run_id'])
        sys.exit(0)
sys.exit(1)
EOF
}

say "waiting for k2b's full snapshot"
until [[ -f $RUN/snapshots.jsonl ]] && grep -q '"label": "full"' $RUN/snapshots.jsonl; do sleep 300; done
want=$(python3 -I -c 'import json, sys
for line in open(sys.argv[1]):
    r = json.loads(line)
    if r["label"] == "full":
        print(r["sha256"])' $RUN/snapshots.jsonl | tail -n 1)
full_sha=$(sha256sum $FULL | cut -d' ' -f1)
[[ $full_sha == "$want" ]] || { say "STOP: $FULL sha256 $full_sha, snapshots.jsonl says $want"; exit 1; }
init_sha=$(sha256sum $INIT | cut -d' ' -f1)
p50_sha=$(sha256sum $P50 | cut -d' ' -f1)
say "full is in: sha256 ${full_sha:0:16}"

if id=$(done_row T-std "$full_sha" "$init_sha"); then
    say "G0 T-std already in the registry: $id"
else
    say "START G0 T-std, full vs pilot A"
    $PY -m lab run --test T-std --family B --arm P1 --package K2 --horizon H \
        --challenger $FULL --champion $INIT --description "G0 retry: k2b full vs pilot A (D59)"
    rc=$?
    say "END G0 T-std rc=$rc"
    (( rc == 0 )) || exit $rc
fi

if id=$(done_row measure "$full_sha" "$p50_sha"); then
    say "doubling estimate already in the registry: $id"
else
    say "START doubling estimate, full vs p50"
    $PY -m lab measure --sets 100000 --family B --arm P1 --package K2 --horizon H \
        --challenger $FULL --champion $P50 --description "k2b full vs p50 (doubling estimate, D49)" \
        --purpose "doubling estimate (D49), pre-registered in experiments/k2-retry.md"
    rc=$?
    say "END doubling estimate rc=$rc"
    (( rc == 0 )) || exit $rc
fi
say "ALL DONE"
