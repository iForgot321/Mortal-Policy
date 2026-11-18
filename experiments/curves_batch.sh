#!/bin/bash
# A batch of D68's curves: both sizes' runs for one seed, launched together so size isn't
# confounded with the period or the hosts (experiments/size-curves.md). Each run gets its own
# detached launcher (experiments/curves_launch.sh) and log, workdir/fleet/RUN-launch.out. The
# second starts 3 min after the first: two `up`s at once would try to rent the same cheapest
# offers, and a learner that loses the race tears its launch down.
#
#   bash experiments/curves_batch.sh 1|2|3
#   bash experiments/curves_batch.sh 1|2|3 --resume    # both runs' watchers and watchdogs
cd "$(dirname "$0")/.." || exit 2
K=${1:?batch 1, 2 or 3}
[[ $K =~ ^[123]$ ]] || { echo "batch 1, 2 or 3, not $K" >&2; exit 2; }
for run in curves-b12-s$K curves-b16-s$K; do
    if [[ ${2:-} == --resume ]]; then
        bash experiments/curves_launch.sh "$run" --resume
    else
        [[ $run == curves-b16-* ]] && sleep 180
        setsid -f bash experiments/curves_launch.sh "$run" >> "workdir/fleet/$run-launch.out" 2>&1 < /dev/null
        echo "$run: launcher started (workdir/fleet/$run-launch.out)"
    fi
done
