#!/usr/bin/env bash
set -eo pipefail
if [[ ${1:-} == -h || ${1:-} == --help ]]; then
    echo "Usage: bash framework/custom_ops/draft_gate_tile_nd/run.sh   # C16 Layer0, device 0"
    exit 0
fi
if (( $# )); then echo "ERROR: unexpected argument: $1" >&2; exit 2; fi
tile_src=$(cd "$(dirname "$0")" && pwd)
tile_base=${DFLASH_RUN_ROOT:-/home/w00949577/qwen35-dflash-run}
tile_cann=${CANN_ROOT:-/usr/local/Ascend/cann-9.0.0}
tile_py=${DFLASH_PYTHON:-$tile_base/venv-hf513-probe/bin/python}
if [[ ! -x "$tile_py" || ! -r "$tile_cann/set_env.sh" ]]; then
    echo "ERROR: check DFLASH_PYTHON ($tile_py) and CANN_ROOT ($tile_cann)." >&2
    exit 1
fi
load_tile_env() { source "$tile_cann/set_env.sh"; }
if ! load_tile_env; then echo "ERROR: failed to initialize $tile_cann" >&2; exit 1; fi
set -u
tile_out=$(mktemp -d "$tile_base/gate-tile-nd-XXXXXX")
echo "RUN $tile_out"
exec env TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$tile_py" -B -u "$tile_src/bench.py" "$tile_base" "$tile_out"
