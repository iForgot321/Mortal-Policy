#!/bin/bash
# D61's workstation checks for v5 (pre-registered in experiments/v5-switch.md; the user, 2026-10-26).
# Runs once the post-G0 queue (k2b_post_g0_chain.sh) has finished and no lab job is running:
#   1. v4's golden hash is unchanged, and v5 equals v4 with the single-player rows zeroed at every
#      decision (tools/obs/hash_obs.py --check-v5, 36 logs, 22,794 decisions);
#   2. the frozen v5 hash, written read-only to workdir/sp-ablation/golden-v5.json;
#   3. a deterministic A/A with a v5 model: No-SP A, true FP32 (D43), 500 sets on the family-A block
#      aa-harness-nn, the earlier neural A/As' deals;
#   4. rl_drift.py on the diagnosis's states with v5: No-SP A against itself (reverse KL 0, agreement 1).
#   setsid -f bash experiments/v5_checks.sh >> workdir/v5-checks.log 2>&1 < /dev/null
cd "$(dirname "$0")/.." || exit 2
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/mortal/lib
PY=$HOME/miniconda3/envs/mortal/bin/python
V5=workdir/sp-ablation/nosp-a/nosp-a-final.pth

say() { echo "$(date -u +%FT%TZ) $*"; }
step() {
    say "START $1"
    "${@:2}"
    local rc=$?
    say "END $1 rc=$rc"
    return $rc
}

say "waiting for the post-G0 queue and any lab job"
while ps -eo args | grep -qE "[k]2b_post_g0_chain.sh|[-]m lab (run|measure|confirm|aa)"; do sleep 120; done
step v4-unchanged-and-v5-is-v4-minus-sp $PY tools/obs/hash_obs.py --version 4 --check-v5 \
    --expect workdir/sp-ablation/golden-v4.json
if [[ ! -s workdir/sp-ablation/golden-v5.json ]]; then
    step frozen-v5-hash $PY tools/obs/hash_obs.py --version 5 --out workdir/sp-ablation/golden-v5.json &&
        chmod a-w workdir/sp-ablation/golden-v5.json
fi
step v5-aa $PY -m lab aa --model $V5 --no-amp --sets 500 --block aa-harness-nn \
    --description "D61: deterministic A/A with a v5 model (No-SP A, true FP32, D43)"
step v5-drift-selfcheck sh -c "$PY experiments/rl_drift.py --logs workdir/diag/credit-a --init $V5 $V5 \
    > workdir/diag/v5-drift-selfcheck.txt 2>&1; tail -n 5 workdir/diag/v5-drift-selfcheck.txt"
say "ALL DONE"
