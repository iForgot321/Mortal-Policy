#!/bin/bash
# After the G0 retry's pass (the user's go-ahead 2026-10-24; experiments/k2-retry.md): once the
# doubling estimate's registry row exists (experiments/k2b_g0_chain.sh), run on this workstation, in order:
#   1. L2's confirmation: T-confirm, k2b full vs pilot A, 100K sets on fresh family-C nonces (pre-registered);
#   2. the diagnosis's tools on k2b for the write-up (experiments/rl-diagnosis.md): drift at p25, p50
#      and full; full sampled and at argmax against pilot A, 4,000 sets on the diagnostic key (the
#      same deals as P0's runs in workdir/diag);
#   3. the search pilot again on k2b's full (experiments/search-pilot.md): a hand model on 4,000 games
#      of its self-play, then the pilot on the family-A block search-pilot.
# Steps already done are skipped (a registry row, an output file, or a stage's own resume), so
# workdir/fleet/runs/k2b/resume-supervision.sh can restart it after a reboot.
#   setsid -f bash experiments/k2b_post_g0_chain.sh >> workdir/fleet/runs/k2b/post-g0-chain.log 2>&1 < /dev/null
cd "$(dirname "$0")/.." || exit 2
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/mortal/lib
PY=$HOME/miniconda3/envs/mortal/bin/python
RUN=workdir/fleet/runs/k2b
FULL=$RUN/snapshots/full.pth
P25=$RUN/snapshots/p25.pth
P50=$RUN/snapshots/p50.pth
INIT=workdir/pilot/qs-a/pilot-qs-a-final.pth
DIAG=workdir/diag
HANDS=workdir/search/hands-k2b
HELD=workdir/search/dev-smoke-k2b
PILOT=workdir/search/pilot-k2b

say() { echo "$(date -u +%FT%TZ) $*"; }
step() {
    say "START $1"
    "${@:2}"
    local rc=$?
    if (( rc != 0 )); then say "FAILED $1 rc=$rc"; exit $rc; fi
    say "DONE $1"
}

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

full_sha=$(sha256sum $FULL | cut -d' ' -f1)
p50_sha=$(sha256sum $P50 | cut -d' ' -f1)
init_sha=$(sha256sum $INIT | cut -d' ' -f1)

say "waiting for the doubling estimate's registry row"
until done_row measure "$full_sha" "$p50_sha" > /dev/null; do sleep 300; done
while ps -eo args | grep -q "[k]2b_g0_chain.sh"; do sleep 60; done
say "doubling estimate in: $(done_row measure "$full_sha" "$p50_sha")"

# 1. L2's confirmation
if id=$(done_row T-confirm "$full_sha" "$init_sha"); then
    say "T-confirm already in the registry: $id"
else
    step T-confirm $PY -m lab confirm --challenger $FULL --champion $INIT --sets 100000 --kind superiority \
        --arm P1 --package K2 --horizon H --description "L2 confirmation: k2b full vs pilot A, 100K sets (D59)"
fi

# 2. the diagnosis's tools on k2b
[[ -s $DIAG/k2b-drift.txt ]] || step drift sh -c "$PY experiments/rl_drift.py --logs $DIAG/credit-a --init $INIT \
    $P25 $P50 $FULL > $DIAG/k2b-drift.txt 2>&1"
for mode in argmax sample; do
    [[ -s $DIAG/k2b-full-$mode.npz ]] || step full-$mode sh -c "$PY experiments/rl_diag_sampling.py play \
        --challenger $FULL --champion $INIT --mode $mode --sets 4000 --out $DIAG/k2b-full-$mode.npz > $DIAG/k2b-full-$mode.log 2>&1"
done

# 3. the search pilot on k2b's full
mkdir -p $HANDS $HELD $PILOT
step held-out-selfplay $PY -m search selfplay --run $HELD --engine $FULL --dev --nonce-start 0 --games 40
step hand-selfplay $PY -m search selfplay --run $HANDS --engine $FULL --dev --nonce-start 100000 --games 4000
[[ -s $HANDS/handmodel.pth ]] || step hand-train sh -c "$PY -m search.handmodel train --logs $HANDS/logs --val $HELD/logs \
    --init $FULL --out $HANDS/handmodel.pth --eval-every 2500 > $HANDS/train.log 2>&1"
if [[ ! -s $PILOT/decisions.jsonl ]]; then  # selfplay records a ledger row per call: once only
    step pilot-selfplay $PY -m search selfplay --run $PILOT --engine $FULL --block search-pilot --games 1200
    step pilot-decisions $PY -m search decisions --run $PILOT --engine $FULL --count 20000
fi
step pilot-oracle $PY -m search oracle --run $PILOT --engine $FULL --redeals 64
step pilot-belief $PY -m search belief --run $PILOT --engine $FULL --hand-model $HANDS/handmodel.pth --particles 64
step pilot-report $PY -m search report --run $PILOT
[[ -s $PILOT/latency.json ]] || step pilot-latency sh -c "$PY -m search latency --run $PILOT --engine $FULL \
    --hand-model $HANDS/handmodel.pth --particles 64 2>/dev/null > $PILOT/latency.json"
say "ALL DONE"
