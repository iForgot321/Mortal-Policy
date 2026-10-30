#!/bin/bash
# The teacher heads check's chain (experiments/teacher-heads-check.md, pre-registered 2026-10-29): wait
# for the heads-on quick student's training to end, then run the pre-registered T-NI against No-SP A.
# Never alongside the training (RAM).
#   setsid -f bash experiments/teacher_heads_chain.sh TRAINER_PID >> workdir/teacher/heads-quick/chain.log 2>&1 < /dev/null
cd "$(dirname "$0")/.." || exit 2
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/mortal/lib
PY=$HOME/miniconda3/envs/mortal/bin/python
RUN=workdir/teacher/heads-quick
say() { echo "$(date -u +%FT%TZ) $*"; }

say "waiting for the trainer (pid $1)"
while kill -0 "$1" 2>/dev/null; do sleep 60; done
if [[ ! -f $RUN/final.pth ]]; then
    say "no $RUN/final.pth: the training did not finish; no T-NI"
    exit 1
fi
say "START T-NI"
$PY -m lab run --test T-NI --family B --challenger $RUN/final.pth \
    --champion workdir/sp-ablation/nosp-a/nosp-a-final.pth \
    --description "teacher heads check (04 §5's guard): heads-on quick student vs No-SP A"
say "END T-NI rc=$?"
