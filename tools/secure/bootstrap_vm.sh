#!/usr/bin/env bash
# Prepare a Hyperstack VM (image "Ubuntu Server 22.04 LTS R550 CUDA 12.4") for this repo, the way
# Vast's vastai/pytorch image arrives prepared (docs/design/secure-data-path-draft.md). Two passes,
# run by tools/secure/secure.py setup:
#   sudo bash bootstrap_vm.sh root   apt packages (everything tools/fleet/setup_box.sh checks for, and
#                                    gnupg), /venv and /workspace owned by the login user
#   bash bootstrap_vm.sh user [ARGS] uv; a Python 3.12 venv /venv/main with torch 2.6.0+cu124 (the
#                                    image Vast boxes use has the same); then setup_box.sh ARGS
# Everything after the root pass runs as the login user, so later syncs never meet root-owned files.
set -euo pipefail

case ${1:-} in
root)
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
        build-essential pkg-config zlib1g-dev git rsync tmux curl ca-certificates procps gnupg
    u=${SUDO_USER:-ubuntu}
    install -d -o "$u" -g "$u" /venv /workspace
    ;;
user)
    export PATH=$HOME/.local/bin:$PATH
    command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    [[ -x /venv/main/bin/python ]] || uv venv -q --python 3.12 /venv/main
    uv pip install -q --python /venv/main/bin/python torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
    cd "$(dirname "${BASH_SOURCE[0]}")/../.."
    PY_OVERRIDE=/venv/main/bin/python tools/fleet/setup_box.sh "${@:2}"
    ;;
*)
    echo "usage: $0 root|user [setup_box.sh args]" >&2
    exit 2
    ;;
esac
