#!/usr/bin/env bash
set -eo pipefail
usage() {
    echo "Usage: bash run.sh [--synthetic | --layer0 | --archive] [--compile-only | --env-only] [--reuse-build RUN]"
    echo "Default: a new Gate/Up A/B using deterministic shape-matched inputs. No old results required."
}
mode=synthetic
compile_only=()
env_only=false
selected=false
reuse=()
while (( $# )); do
    arg=$1; shift
    case "$arg" in
        -h|--help) usage; exit 0;;
        --env-only) env_only=true;;
        --reuse-build)
            if (( ! $# || ${#reuse[@]} )); then echo "ERROR: --reuse-build needs one previous run" >&2; exit 2; fi
            reuse=(--reuse-build "$1"); shift;;
        --compile-only) compile_only=(--compile-only);;
        --synthetic|--layer0|--archive)
            if $selected; then echo "ERROR: choose one input mode" >&2; exit 2; fi
            selected=true; mode=${arg#--};;
        *) usage >&2; exit 2;;
    esac
done
here=$(cd "$(dirname "$0")" && pwd -P)
source "$here/../ascend310p_workspace/run_root.sh"
dflash_new_run gate-tile-nd
if (( ${#reuse[@]} )); then
    if [[ "$mode" == archive ]]; then echo "ERROR: --reuse-build selects a fresh Gate build" >&2; exit 2; fi
    if [[ "${reuse[1]}" != /* ]]; then reuse[1]="$DFLASH_RUN_ROOT/runs/${reuse[1]}"; fi
fi
if [[ "$mode" == archive ]]; then
    dflash_load_cann
    tile_py=${DFLASH_PYTHON:-$DFLASH_RUN_ROOT/venv-hf513-probe/bin/python}
    if [[ ! -x "$tile_py" ]]; then echo "ERROR: missing archive interpreter: $tile_py" >&2; exit 1; fi
else
    source "$here/../ascend310p_workspace/gate_conda.sh"
    dflash_gate_conda
    tile_py=$DFLASH_PYTHON
fi
if $env_only; then exit 0; fi
cd "$DFLASH_OUTPUT"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
if [[ "$mode" == archive ]]; then
    if (( ${#compile_only[@]} )); then echo "ERROR: --compile-only is supported for fresh runs" >&2; exit 2; fi
    archive=${DFLASH_ARCHIVE_ROOT:-/home/w00949577/qwen35-dflash-run}
    mkdir ab
    exec "$tile_py" -B -u "$here/bench.py" "$archive" "$DFLASH_OUTPUT/ab"
fi
inputs=()
if [[ "$mode" == layer0 ]]; then inputs=(--layer0-inputs "$DFLASH_RUN_ROOT/inputs/c16-layer0-gate"); fi
exec "$tile_py" -B -u "$here/fresh_run.py" --out "$DFLASH_OUTPUT/ab" "${inputs[@]}" "${compile_only[@]}" "${reuse[@]}"
