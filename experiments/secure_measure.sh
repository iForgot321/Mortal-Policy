#!/usr/bin/env bash
# D17a's measurement session (docs/design/secure-data-path-draft.md; the user's yes, 2026-10-29): one
# Hyperstack VM per flavor, up → setup → ship → bench_teacher → pull → down. Once the VM exists,
# down always runs, whatever fails; the watchdog (secure.py up --hours) backs it up.
#   setsid -f bash experiments/secure_measure.sh RUN FLAVOR >> workdir/secure/runs/RUN.session.log 2>&1 < /dev/null
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2
RUN=$1 FLAVOR=$2
S="python3 tools/secure/secure.py"

say() { echo "$(date -u +%FT%TZ) $*"; }
step() {
    say "START $1"
    "${@:2}"
    local rc=$?
    say "END $1 rc=$rc"
    return $rc
}
finish() {
    step down $S down "$RUN"
    say "ALL DONE"
}

step up $S up "$RUN" --flavor "$FLAVOR" --hours 2.5 --yes || {
    [[ -f workdir/secure/runs/$RUN/state.json ]] && finish
    exit 1
}
trap finish EXIT
step setup $S setup "$RUN" || exit 1
step ship $S ship "$RUN" || exit 1
D=$(python3 -c "import json; print(json.load(open('workdir/secure/runs/$RUN/state.json'))['data_dir'])")
# libpython for the uv-made venv, as setup_box.sh sets it; detached on the box so a dropped SSH
# session can't kill the bench
LIB='LD_LIBRARY_PATH=$(/venv/main/bin/python -c "import sysconfig; print(sysconfig.get_config_var(\"LIBDIR\"))")'
step bench-start $S exec "$RUN" "mkdir -p workdir/secure-box && $LIB setsid nohup /venv/main/bin/python \
tools/secure/bench_teacher.py --data $D > workdir/secure-box/bench.log 2>&1 < /dev/null &" || exit 1
for _ in $(seq 1 75); do
    sleep 60
    if $S exec "$RUN" "grep -q '\"finished\"' workdir/secure-box/bench/teacher.json 2>/dev/null"; then
        say "bench finished"
        break
    fi
    if ! $S exec "$RUN" "pgrep -f '[b]ench_teacher.py' >/dev/null"; then
        say "bench process gone"
        break
    fi
done
step pull $S pull "$RUN"
