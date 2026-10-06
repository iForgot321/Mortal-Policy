#!/usr/bin/env bash
# Prepare a rented GPU box (03 §2–3) to run this repo: system packages, the image's Python,
# Rust, pinned Python packages, the libriichi build, frozen-input checks, box-local configs,
# tests. Idempotent: rerun it after every code sync.
#
# Expects:
#   - image vastai/pytorch:2.6.0-cuda-12.4.1-py312-22.04 (torch 2.6.0+cu124, Python 3.12) and a
#     GPU torch 2.6 cu124 has kernels for (Ampere/Ada, not Blackwell); root in the container;
#   - rsync on the box before the first sync: apt-get update && apt-get install -y rsync
#   - the repo rsynced from a clean commit, .git included, to /workspace/Mortal-Policy:
#       rsync -az -e "$BOX" --filter=':- .gitignore' --exclude=.claude/ --exclude=.pytest_cache/ \
#           ./ root@<ip>:/workspace/Mortal-Policy/
#   - the frozen inputs at the same relative paths (checked against tools/fleet/frozen.sha256):
#       rsync -azR -e "$BOX" workdir/pilot/qs-a/pilot-qs-a-final.pth \
#           workdir/grp-v2/grp-v2-frozen.pth root@<ip>:/workspace/Mortal-Policy/
#
# Never put credentials (GitHub, Claude, cloud keys) or Tenhou data (D23) on a community box:
# its host has root.
#
# Usage:  tools/fleet/setup_box.sh [--skip-tests]
#         PY_OVERRIDE=/path/to/python to pick the interpreter explicitly.
# Output: workdir/box/info.txt (host, allocation, versions, commit) and workdir/box/configs/
#         (experiments/configs with the workstation root rewritten to this checkout), e.g.
#         cd mortal && MORTAL_CFG=../workdir/box/configs/p0-smoke.toml python server.py
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
WS_ROOT=/home/iforgot/projects/Mortal-Policy   # the absolute root written in experiments/configs
TORCH_VERSION=2.6.0+cu124
RUST_VERSION=1.95.0                            # the workstation's toolchain
REQUIRED=(workdir/pilot/qs-a/pilot-qs-a-final.pth workdir/grp-v2/grp-v2-frozen.pth)

SKIP_TESTS=0
for arg in "$@"; do
    case $arg in
        --skip-tests) SKIP_TESTS=1 ;;
        *) echo "usage: $0 [--skip-tests]" >&2; exit 2 ;;
    esac
done

log() { echo "== $*"; }
die() { echo "FAIL: $*" >&2; exit 1; }
cd "$ROOT"

log "system packages"
# gcc for cargo (mimalloc) and Triton's launcher; zlib for cargo test's flate2/zlib
need=()
for p in build-essential pkg-config zlib1g-dev git rsync tmux curl ca-certificates procps; do
    dpkg -s "$p" >/dev/null 2>&1 || need+=("$p")
done
if ((${#need[@]})); then
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends "${need[@]}"
fi

log "python"
# the first interpreter that has the pinned torch: the image's venv is not always on PATH
# in a non-interactive ssh session
PY=
for c in ${PY_OVERRIDE:-} python python3 /venv/main/bin/python /opt/conda/bin/python; do
    p=$(command -v "$c" 2>/dev/null) || continue
    if [[ $("$p" -c 'import torch; print(torch.__version__)' 2>/dev/null) == "$TORCH_VERSION" ]]; then
        PY=$p
        break
    fi
done
[[ -n $PY ]] || die "no Python with torch $TORCH_VERSION; set PY_OVERRIDE=/path/to/python"
echo "$PY"
"$PY" - <<'EOF' || die "wrong Python, or its headers are missing"
import os, sys, sysconfig
assert sys.version_info[:2] == (3, 12), f'Python {sys.version} (want 3.12, as on the workstation)'
h = os.path.join(sysconfig.get_paths()['include'], 'Python.h')
assert os.path.exists(h), f'{h} missing: torch.compile (Triton) needs the Python headers'
EOF
# cargo test embeds Python, so libpython must be found at run time
LD_LIBRARY_PATH=$("$PY" -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export LD_LIBRARY_PATH

log "gpu"
"$PY" - <<'EOF' || die "this GPU can't run torch $TORCH_VERSION"
import torch
assert torch.cuda.is_available(), 'CUDA is not available'
major, minor = torch.cuda.get_device_capability(0)
# a cubin for sm_XY runs on devices with the same major version and minor >= Y
archs = [int(a[3:]) for a in torch.cuda.get_arch_list() if a.startswith('sm_')]
assert any(a // 10 == major and a % 10 <= minor for a in archs), (
    f'{torch.cuda.get_device_name(0)} is sm_{major}{minor}; this torch has {torch.cuda.get_arch_list()}. '
    'Blackwell (sm_120) needs torch >= 2.7 cu128, an infrastructure change (T-smoke).')
x = torch.ones(64, 64, device='cuda')
assert (x @ x).sum().item() == 64 ** 3
print(torch.cuda.get_device_name(0), f'sm_{major}{minor}')
EOF

log "rust $RUST_VERSION"
export PATH=$HOME/.cargo/bin:$PATH
if ! command -v rustup >/dev/null; then
    curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain "$RUST_VERSION"
fi
rustup toolchain install "$RUST_VERSION" --profile minimal
rustup default "$RUST_VERSION"

log "python packages"
# uv-made venvs have no pip
if "$PY" -m pip --version >/dev/null 2>&1; then
    "$PY" -m pip install -q -r tools/fleet/requirements-box.txt
elif command -v uv >/dev/null; then
    uv pip install -q --python "$PY" -r tools/fleet/requirements-box.txt
else
    "$PY" -m ensurepip -q
    "$PY" -m pip install -q -r tools/fleet/requirements-box.txt
fi
[[ $("$PY" -c 'import torch; print(torch.__version__)') == "$TORCH_VERSION" ]] \
    || die "pip moved torch off $TORCH_VERSION"

log "libriichi"
export PYO3_PYTHON=$PY
cargo build -p libriichi --lib --release
# install by rename, never cp over a .so a running process has mapped
cp target/release/libriichi.so mortal/libriichi.so.new
mv mortal/libriichi.so.new mortal/libriichi.so
(cd mortal && "$PY" -c 'import libriichi') || die "mortal/libriichi.so does not import"

log "frozen inputs"
while read -r sum path; do
    if [[ -f $path ]]; then
        echo "$sum  $path" | sha256sum -c --quiet - || die "$path does not match tools/fleet/frozen.sha256"
        chmod a-w "$path"
        echo "ok      $path"
    else
        echo "absent  $path"
    fi
done < tools/fleet/frozen.sha256
for path in "${REQUIRED[@]}"; do
    [[ -f $path ]] || die "$path is missing (every P0/P1 config needs it); rsync it from the workstation"
done

log "box configs"
mkdir -p workdir/box/configs
for f in experiments/configs/*.toml; do
    sed "s#$WS_ROOT#$ROOT#g" "$f" > "workdir/box/configs/$(basename "$f")"
done
ls workdir/box/configs

log "box info"
# rsync -a keeps the workstation's uid, which git refuses to read as root without this
git config --global --get-all safe.directory 2>/dev/null | grep -qxF "$ROOT" \
    || git config --global --add safe.directory "$ROOT"
# rsync changes inodes and mtimes; refresh the index so --dirty reflects content only
git update-index -q --refresh || true
{
    echo "date:      $(date -Is)"
    echo "commit:    $(git describe --always --dirty --abbrev=12)"
    echo "gpu:       $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader)"
    echo "cuda max:  $(nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | cut -d' ' -f3)"
    echo "cpu:       $(lscpu | sed -n 's/^Model name: *//p')"
    echo "cpus:      affinity $("$PY" -c 'import os; print(len(os.sched_getaffinity(0)))'), cgroup cpu.max $(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo n/a)"
    echo "ram:       $(free -g | awk '/^Mem:/ {print $2 " GB total, " $7 " GB available"}')"
    echo "disk:      $(df -h "$ROOT" | awk 'NR == 2 {print $4 " free of " $2}')"
    echo "python:    $("$PY" -c 'import sys; print(sys.version.split()[0])') ($PY)"
    echo "torch:     $("$PY" -c 'import torch; print(torch.__version__, "cuDNN", torch.backends.cudnn.version())')"
    echo "rustc:     $(rustc --version)"
} | tee workdir/box/info.txt

if ((SKIP_TESTS == 0)); then
    log "cargo test"
    cargo test --workspace --release --no-default-features --features flate2/zlib
    log "pytest"
    "$PY" -m pytest tests/ -q
fi
log "done: $ROOT is ready ($(git describe --always --dirty --abbrev=12))"
