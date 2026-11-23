#!/usr/bin/env bash
set -eo pipefail
here=$(cd "$(dirname "$0")" && pwd -P)
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    echo 'Usage: bash run.sh [--gear 16|64|both] [--inputs-dir DIR] [--prepare-only|--compile-only|--no-profile] [--resume full-draft-panels-XXXXXX]'
    exit 0
fi
source "$here/../ascend310p_workspace/run_root.sh"
source "$here/../ascend310p_workspace/gate_conda.sh"
export DFLASH_RUN_ROOT=/home/w00949577/z50058744/qwen35-dflash-run
dflash_new_run full-draft-panels
# CPython appends /pymp-XXXXXXXX/listener-XXXXXXXX to TMPDIR. sun_path is 108 bytes on Linux.
mkdir -p "$DFLASH_RUN_ROOT/t"
ipc_tmp=$(mktemp -d "$DFLASH_RUN_ROOT/t/d-XXXXXX")
ln -s "$ipc_tmp" "$DFLASH_OUTPUT/ipc-tmp"
export TMPDIR="$ipc_tmp" TMP="$ipc_tmp" TEMP="$ipc_tmp"
dflash_gate_conda
export XDG_CACHE_HOME="$DFLASH_RUN_ROOT/cache" HF_HOME="$DFLASH_RUN_ROOT/cache/huggingface" TORCH_HOME="$DFLASH_RUN_ROOT/cache/torch"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2
export DFLASH_DRAFT_CHECKPOINT="${DRAFT_W8A16_DIR:-${DRAFT_DIR:-/home/w00949577/model_weights/qwen3.5-4b-dflash-w8a16}}"
export DFLASH_TARGET_CHECKPOINT="${TARGET_DIR:-/home/w00949577/model_weights/Qwen3.5-4B}"
for file in "$DFLASH_DRAFT_CHECKPOINT/model.safetensors" "$DFLASH_TARGET_CHECKPOINT/model.safetensors.index.json"; do
    [[ -s "$file" ]] || { echo "ERROR: missing checkpoint file $file" >&2; exit 2; }
done
cd "$DFLASH_OUTPUT"
export DFLASH_FRONTEND="$DFLASH_RUN_ROOT/envs/full-draft-frontend/bin/python"
(
    flock -x 9
    if [[ ! -x "$DFLASH_FRONTEND" ]]; then "$DFLASH_PYTHON" -m venv --system-site-packages "${DFLASH_FRONTEND%/bin/python}"; fi
    if ! "$DFLASH_FRONTEND" -I -c 'import torch, safetensors; from transformers import AutoTokenizer; assert torch.__version__.split("+")[0]=="2.8.0"' >/dev/null 2>&1; then
        args=()
        if [[ -n ${DFLASH_WHEELHOUSE:-} ]]; then args=(--no-index --find-links "$DFLASH_WHEELHOUSE"); fi
        "$DFLASH_FRONTEND" -I -m pip install --no-user --disable-pip-version-check --prefer-binary "${args[@]}" "torch==2.8.0" "transformers==4.57.6" "safetensors>=0.4.3,<1"
    fi
) 9>"$DFLASH_RUN_ROOT/envs/.draft-full-frontend.lock"
exec "$DFLASH_PYTHON" -B -u "$here/experiment.py" "$@"
