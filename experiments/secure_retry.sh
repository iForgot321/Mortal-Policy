#!/usr/bin/env bash
# Rent a secure-tier run as stock allows: run experiments/secure_teacher.sh RUN CONFIG FLAVOR HOURS, and
# while it fails only because Hyperstack has no stock of the flavor, try again every 10 min, up to
# MAX_HOURS (default 72). Any other end (the run rented and finished, or failed after the rental)
# ends the retries with the driver's exit code. The driver's environment (TRAINER, INPUTS, MIN_RATE)
# passes through. secure_teacher_queue.sh does the same for runs that never had a state.
#   TRAINER=... setsid -f bash experiments/secure_retry.sh RUN CONFIG FLAVOR HOURS >> workdir/secure/runs/RUN.session.log 2>&1 < /dev/null
cd "$(dirname "$0")/.." || exit 2
RUN=$1 CONFIG=$2 FLAVOR=$3 HOURS=$4
LOG=workdir/secure/runs/$RUN.session.log
deadline=$(($(date +%s) + ${MAX_HOURS:-72} * 3600))
say() { echo "$(date -u +%FT%TZ) [retry] $*"; }
while true; do
    start=$(wc -l < "$LOG" 2>/dev/null || echo 0)
    say "renting $RUN ($FLAVOR, $HOURS h)"
    bash experiments/secure_teacher.sh "$RUN" "$CONFIG" "$FLAVOR" "$HOURS"
    rc=$?
    if ((rc == 0)); then
        say "the driver ended cleanly"
        exit 0
    fi
    # only a refusal for stock is retried: that attempt never rented, so it cost nothing
    if tail -n +"$((start + 1))" "$LOG" | grep -q -E 'Not Enough Stock|is out of stock' \
        && ! tail -n +"$((start + 1))" "$LOG" | grep -q 'rented VM'; then
        if (($(date +%s) > deadline)); then
            say "no stock in ${MAX_HOURS:-72} h: giving up"
            exit 1
        fi
        say "no stock; next try in 10 min"
        sleep 600
        continue
    fi
    say "the driver failed after renting (rc $rc): no retry"
    exit "$rc"
done
