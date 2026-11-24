#!/usr/bin/env bash
set -eo pipefail
here=$(cd "$(dirname "$0")" && pwd -P)
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    echo 'Usage: bash run.sh [--gear 16|64|both] [--compile-only | --no-profile] [--resume full-all-w8-XXXXXX]'
    exit 0
fi
source "$here/../ascend310p_workspace/run_root.sh"
source "$here/../ascend310p_workspace/gate_conda.sh"
export DFLASH_RUN_ROOT=/home/w00949577/z50058744/qwen35-dflash-run
dflash_new_run full-all-w8
mkdir -p "$DFLASH_RUN_ROOT/t"
ipc_tmp=$(mktemp -d "$DFLASH_RUN_ROOT/t/i-XXXXXX")
ln -s "$ipc_tmp" "$DFLASH_OUTPUT/ipc-tmp"
export TMPDIR="$ipc_tmp" TMP="$ipc_tmp" TEMP="$ipc_tmp"
dflash_gate_conda
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
cd "$DFLASH_OUTPUT"
exec "$DFLASH_PYTHON" -B -u "$here/run_all.py" "$@"
