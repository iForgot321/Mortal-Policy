#!/bin/bash
# One run of D68's curves (experiments/size-curves.md): rent its fleet at a live quote within the
# limits, then start its watcher. A learner and 2 actors × 2 processes from the size's init (its B0
# `full` without the run clock: experiments/curves_b0.sh --finish), 288 box-hours (≈ 4 days):
#   - snapshots p25, p50 and p75 every 72 box-hours, each the learner's new KL anchor (--reanchor);
#   - D68's check at each of them, on the workstation (experiments/curves_stage_check.py), which
#     downs the run on 'stop';
#   - then `full`, and the watcher downs the run.
#
#   setsid -f bash experiments/curves_launch.sh curves-b12-s1 >> workdir/fleet/curves-b12-s1-launch.out 2>&1 < /dev/null
#   bash experiments/curves_launch.sh curves-b12-s1 --resume   # after a restart: watcher and watchdog
#
# experiments/curves_batch.sh starts a batch's two runs together. Limits: experiments/curves_common.sh.
RUN=${1:?curves-b12-s1 ... curves-b16-s3}
[[ $RUN =~ ^curves-(b12|b16)-s([123])$ ]] || { echo "not a curves run: $RUN" >&2; exit 2; }
SIZE=${BASH_REMATCH[1]}
case $SIZE in b12) NET=b12c256 ;; b16) NET=b16c256 ;; esac
source "$(dirname "$0")/curves_common.sh"
INIT=workdir/curves/init-$NET.pth
HOURS=96  # 288 box-hours on 3 boxes
UP="--run $RUN --config experiments/configs/$RUN.toml --init $INIT --actors 2 --procs 2 --hard-hours 106 --verify"
CHECK="$PY experiments/curves_stage_check.py --run $RUN --start $STUDENTS/ema-$NET.pth --states workdir/diag/credit-$NET"
WATCH=(--replace --allow-zen2 --budget-box-hours 288 --snapshot-at 0.25,0.5,0.75 --reanchor
       --stage-check "$CHECK" --anchor-alert 0.10)

if [[ ${2:-} == --resume ]]; then
    resume
    exit 0
fi
[[ -f $INIT ]] || { say "no init $INIT: finish the size's B0 first (curves_b0.sh $SIZE --finish)"; exit 1; }
[[ -d workdir/diag/credit-$NET/logs ]] || { say "no monitoring states workdir/diag/credit-$NET"; exit 1; }
rent || exit 1
watcher || exit 1
say "LAUNCHED"
