#!/usr/bin/env bash
# One SL teacher run on a Hyperstack VM (D62; docs/design/teacher-trainer-draft.md): up → setup → ship
# → mortal/train_teacher.py, detached on the box → supervise → pull and down when it finishes. Runs on
# the workstation for days, so:
#   - it never downs the VM just because it exits: the trainer keeps going if this script dies. A
#     workstation reboot is recovered with --attach (it re-arms the watchdog, then supervises again);
#   - every 30 min it checks the trainer, and every 2 h it pulls workdir/secure-box (checkpoints,
#     timing, TensorBoard);
#   - a dead trainer is restarted (it resumes from its last checkpoint) up to 3 times; then the VM is
#     pulled and downed and the log says why;
#   - the watchdog (secure.py up --hours) destroys the VM at the deadline whatever happens here.
#   setsid -f bash experiments/secure_teacher.sh RUN CONFIG [FLAVOR] [HOURS] >> workdir/secure/runs/RUN.session.log 2>&1 < /dev/null
#   setsid -f bash experiments/secure_teacher.sh RUN CONFIG --attach >> workdir/secure/runs/RUN.session.log 2>&1 < /dev/null
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2
RUN=$1 CONFIG=$2
S="python3 tools/secure/secure.py"
STATE=workdir/secure/runs/$RUN/state.json

say() { echo "$(date -u +%FT%TZ) $*"; }
step() {
    say "START $1"
    "${@:2}"
    local rc=$?
    say "END $1 rc=$rc"
    return $rc
}
field() { python3 -c "import json; print(json.load(open('$STATE')).get('$1') or '')"; }
box_has() { $S exec "$RUN" "$1" >/dev/null 2>&1; }
start_trainer() {
    local d
    d=$(field data_dir)
    $S exec "$RUN" "mkdir -p workdir/secure-box && cd mortal && (MJ_DATA=$d MORTAL_CFG=../workdir/box/configs/$CONFIG.toml \
LD_LIBRARY_PATH=\$(/venv/main/bin/python -c 'import sysconfig; print(sysconfig.get_config_var(\"LIBDIR\"))') \
setsid nohup /venv/main/bin/python train_teacher.py >> ../workdir/secure-box/$CONFIG.log 2>&1 < /dev/null &)"
}

if [[ ${3:-} == --attach ]]; then
    say "attaching to $RUN"
    $S rearm "$RUN"
else
    FLAVOR=${3:-n3-RTX-A6000x1} HOURS=${4:-150}
    step up $S up "$RUN" --flavor "$FLAVOR" --hours "$HOURS" --yes || {
        [[ -f $STATE ]] && $S down "$RUN"
        exit 1
    }
    step setup $S setup "$RUN" || { $S down "$RUN"; exit 1; }
    step ship $S ship "$RUN" || { $S down "$RUN"; exit 1; }
    step start start_trainer || { $S down "$RUN"; exit 1; }
fi

restarts=0 n=0
while true; do
    sleep 1800
    n=$((n + 1))
    if [[ -n $(field destroyed) ]]; then
        say "the VM is destroyed (the watchdog, or by hand): supervision ends"
        exit 0
    fi
    if box_has "test -f workdir/secure-box/$CONFIG/final.pth"; then
        say "final.pth is in: pulling and downing"
        $S down "$RUN"
        say "ALL DONE"
        exit 0
    fi
    if ! box_has "pgrep -f '[t]rain_teacher.py' >/dev/null"; then
        if ! box_has true; then
            say "no SSH to the box this round"
            continue
        fi
        restarts=$((restarts + 1))
        if ((restarts > 3)); then
            say "the trainer died a 4th time: pulling and downing; see workdir/secure/runs/$RUN/box/$CONFIG.log"
            $S down "$RUN"
            exit 1
        fi
        say "the trainer is not running: restart $restarts (it resumes from its last checkpoint)"
        start_trainer
    fi
    ((n % 4 == 0)) && $S pull "$RUN" >/dev/null && say "pulled"
done
