# Environment shared by the manually launched Target preparation and full generation.
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
source "$here/../ascend310p_workspace/run_root.sh"
source "$here/../ascend310p_workspace/gate_conda.sh"
export DFLASH_RUN_ROOT=/home/w00949577/z50058744/qwen35-dflash-run
dflash_new_run "$run_prefix"
mkdir -p "$DFLASH_RUN_ROOT/t"
ipc_tmp=$(mktemp -d "$DFLASH_RUN_ROOT/t/e-XXXXXX")
ln -s "$ipc_tmp" "$DFLASH_OUTPUT/ipc-tmp"
export TMPDIR="$ipc_tmp" TMP="$ipc_tmp" TEMP="$ipc_tmp"
dflash_gate_conda
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
export HF_HOME="$DFLASH_RUN_ROOT/cache/huggingface" XDG_CACHE_HOME="$DFLASH_RUN_ROOT/cache" TORCH_HOME="$DFLASH_RUN_ROOT/cache/torch"
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2
cd "$DFLASH_OUTPUT"
