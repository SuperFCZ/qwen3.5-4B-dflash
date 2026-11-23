#!/usr/bin/env bash
set -eo pipefail
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    echo "Usage: bash run.sh [--device ID] [--blocks N] [--vector-only]"
    echo "Default: device 0, queried AI Core block count, UB vector + Gate/Down streaming metrics."
    exit 0
fi

src=$(cd "$(dirname "$0")" && pwd)
cann=${CANN_ROOT:-/usr/local/Ascend/cann-9.0.0}
if [[ ! -r "$cann/set_env.sh" ]]; then
    echo "ERROR: $cann/set_env.sh is missing. Set CANN_ROOT to the installed CANN 9.0 directory." >&2
    exit 1
fi
load_cann() { source "$cann/set_env.sh"; }
if ! load_cann; then
    echo "ERROR: failed to initialize $cann/set_env.sh" >&2
    exit 1
fi
set -u
build=$(mktemp -d "${TMPDIR:-/tmp}/dflash-vector-XXXXXX")
cleanup() {
    local rc=$?
    if (( rc == 0 )); then rm -rf -- "$build"; else echo "Build/log directory: $build" >&2; fi
}
trap cleanup EXIT
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
