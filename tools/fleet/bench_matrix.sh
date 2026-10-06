#!/usr/bin/env bash
# Multi-process actors and the reduced inference matrix (03 §4.1) on one box, after setup_box.sh.
# Every run is tools/fleet/bench_1v3.py with pilot A deciding; results in workdir/box/matrix/.
#   1. one process (baseline), then 2 and 3 concurrent processes on different seeds;
#   2. the cost of bigger networks under AMP (--shadow b6c128 … b24c512);
#   3. torch.compile (CUDA graphs, batch buckets) for the two largest.
# Usage, from the repo root: PY=/venv/main/bin/python tools/fleet/bench_matrix.sh
set -uo pipefail
PY=${PY:-python}
MODEL=workdir/pilot/qs-a/pilot-qs-a-final.pth
OUT=workdir/box/matrix
mkdir -p "$OUT"

bench() {  # name, extra bench_1v3.py arguments...
    local name=$1
    shift
    "$PY" tools/fleet/bench_1v3.py --model "$MODEL" --minutes 0 --out "$OUT/$name.json" "$@" > "$OUT/$name.log" 2>&1
    echo "$(date -u +%T) $name: exit $?"
}

bench procs1 --min-chunks 3

for n in 2 3; do
    pids=()
    for k in $(seq 1 "$n"); do
        bench "procs$n-$k" --min-chunks 3 --nonce-offset $((1000 * k)) &
        pids+=($!)
    done
    wait "${pids[@]}"
done

for arch in b6c128 b8c192 b12c256 b16c256 b24c512; do
    bench "amp-$arch" --min-chunks 2 --shadow "$arch"
done

for arch in b16c256 b24c512; do
    bench "compile-$arch" --min-chunks 2 --shadow "$arch" --compile
done
