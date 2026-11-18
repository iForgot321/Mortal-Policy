#!/bin/bash
# D68's critic warm-up (B0) for one student size (experiments/size-curves.md, "Before the first
# rental", item 4): the size's seed-1 config with critic.warmup = true (the policy stays frozen and
# only the critic trains), from the student's weights as a trainer's first state, on a learner and
# 2 actors × 2 processes, 12 box-hours (a p50 snapshot at 6, then `full` and down). On the first
# box it also makes server.py's token check (experiments/curves_token_check.py).
#
#   setsid -f bash experiments/curves_b0.sh b12|b16 >> workdir/fleet/curves-b0-b12.out 2>&1 < /dev/null
#   bash experiments/curves_b0.sh b12|b16 --resume    # after a restart: watcher and watchdog
#   bash experiments/curves_b0.sh b12|b16 --finish    # after `full`: b0_check.py, then the size's init
#
# --finish applies the B0 rule (experiments/b0_check.py: the calibration monotone in the final
# window, the explained variance not falling over the last third). On a pass it writes the curves'
# init, `full` without its run clock (tools/fleet/stage_init.py), so their schedules start at t = 0.
# On a fail it stops: the rule's one 4-box-hour extension is a rental of its own, from `full`.
# Limits: experiments/curves_common.sh (MAX_DPH, MAX_FLEET_DPH, MAX_DOLLARS, QUOTE_HOURS).
SIZE=${1:?b12 or b16}
case $SIZE in b12) NET=b12c256 ;; b16) NET=b16c256 ;; *) echo "b12 or b16, not $SIZE" >&2; exit 2 ;; esac
RUN=curves-$SIZE-b0
source "$(dirname "$0")/curves_common.sh"
STUDENT=$STUDENTS/ema-$NET.pth
B0_INIT=workdir/curves/b0-init-$NET.pth
INIT=workdir/curves/init-$NET.pth
HOURS=5  # 12 box-hours on 3 boxes, and setup
UP="--run $RUN --config experiments/configs/curves-$SIZE-s1.toml --init $B0_INIT --actors 2 --procs 2
    --hard-hours 8 --set critic.warmup=true --verify"
WATCH=(--replace --allow-zen2 --budget-box-hours 12 --snapshot-at 0.5)

case ${2:-} in
--resume)
    resume
    exit 0 ;;
--finish)
    full=workdir/fleet/runs/$RUN/snapshots/full.pth
    [[ -f $full ]] || { say "no full snapshot yet ($full)"; exit 1; }
    if [[ -f $INIT ]]; then say "$INIT exists"; exit 0; fi
    if ! $PY experiments/b0_check.py "workdir/fleet/runs/$RUN"; then
        say "B0 FAILED its rule: the one 4 box-hour extension, from full, needs its own rental"
        exit 1
    fi
    $PY tools/fleet/stage_init.py "$full" "$INIT" && say "the curves' init for $NET: $INIT"
    exit $? ;;
'') ;;
*) echo "unknown option $2" >&2; exit 2 ;;
esac

if [[ ! -f $B0_INIT ]]; then
    $PY tools/fleet/stage_init.py --from-policy "$STUDENT" "$B0_INIT" || exit 1
fi
rent || exit 1
watcher || exit 1
say "LAUNCHED"
$PY experiments/curves_token_check.py "$RUN" && say "token check passed" || say "TOKEN CHECK FAILED: see above"
