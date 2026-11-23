#!/usr/bin/env bash
set -eo pipefail
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    echo "Usage: bash run.sh [--device ID] [--blocks N] [--vector-only]"
    echo "Default: device 0, queried AI Core block count, UB vector + Gate/Down streaming metrics."
    exit 0
fi

src=$(cd "$(dirname "$0")" && pwd)
source "$src/../ascend310p_workspace/run_root.sh"
dflash_new_run vector
dflash_load_cann
cann=$CANN_ROOT
set -u
build="$DFLASH_OUTPUT/build"
mkdir "$build"
cd "$DFLASH_OUTPUT"
build_step() {
    local log=$1 rc
    shift
    if "$@" >"$log" 2>&1; then return 0; else rc=$?; fi
    cat "$log" >&2
    return "$rc"
}

echo "Building the Ascend310P3 vector probe..." >&2
build_step "$build/configure.log" cmake -S "$src" -B "$build" \
    -DASCEND_CANN_PACKAGE_PATH="$cann" -DCMAKE_BUILD_TYPE=Release
build_step "$build/build.log" cmake --build "$build" --target vector_bench --parallel 2
export LD_LIBRARY_PATH="$cann/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$build/vector_bench" "$@"
