#!/usr/bin/env bash
# Start the teacher runs as A6000 stock allows (the user, 2026-10-30: both if available, otherwise
# staggered): for each run in order, launch experiments/secure_teacher.sh and, while Hyperstack
# refuses the rental (no state.json appears and the driver exits), retry every 10 min, up to 3 days.
# A run whose VM was rented is left to its driver; a failure after the rental stops the queue.
#   setsid -f bash experiments/secure_teacher_queue.sh teacher-awr-s1 teacher-bc-s1 >> workdir/secure/runs/teacher-queue.log 2>&1 < /dev/null
cd "$(dirname "$0")/.." || exit 2
say() { echo "$(date -u +%FT%TZ) $*"; }
driver_alive() { ps -eo args | grep -qE "^bash experiments/secure_teacher\.sh $1 "; }

for RUN in "$@"; do
    STATE=workdir/secure/runs/$RUN/state.json
    if [[ -f $STATE ]]; then
        say "$RUN: has a state already; skipped"
        continue
    fi
    deadline=$(($(date +%s) + 3 * 86400))
    until [[ -f $STATE ]]; do
        if (($(date +%s) > deadline)); then
            say "$RUN: no A6000 in 3 days; the queue stops"
            exit 1
        fi
        say "$RUN: trying to rent"
        setsid -f bash experiments/secure_teacher.sh "$RUN" "$RUN" n3-RTX-A6000x1 150 \
            >> "workdir/secure/runs/$RUN.session.log" 2>&1 < /dev/null
        sleep 15
        while driver_alive "$RUN" && [[ ! -f $STATE ]]; do sleep 15; done
        if [[ ! -f $STATE ]]; then
            say "$RUN: refused ($(grep -o '"message": "[^"]*"' "workdir/secure/runs/$RUN.session.log" | tail -1)); retry in 10 min"
            sleep 600
        fi
    done
    say "$RUN: rented; its driver supervises it"
    sleep 120
    if [[ -n $(python3 -c "import json; print(json.load(open('$STATE')).get('destroyed') or '')") ]]; then
        say "$RUN: destroyed right after the rental (see its session log); the queue stops"
        exit 1
    fi
done
say "queue done"
