#!/usr/bin/env bash
set -eo pipefail
here=$(cd "$(dirname "$0")" && pwd -P)
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    echo 'Usage: bash run.sh [--panels 256,512,1280,2560] [--reuse-build gate-tile-nd-ZHIxuE] [--compile-only] [--layer0]'
    exit 0
fi
source "$here/../ascend310p_workspace/run_root.sh"
source "$here/../ascend310p_workspace/gate_conda.sh"
dflash_new_run gate-bandwidth
dflash_gate_conda
cd "$DFLASH_OUTPUT"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
exec "$DFLASH_PYTHON" -B -u "$here/run_gate.py" "$@"
