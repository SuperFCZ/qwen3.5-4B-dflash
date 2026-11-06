#!/usr/bin/env bash
set -eo pipefail
[[ $# == 1 && ! -e "$1" ]] || { echo 'usage: check.sh NEW_OUTPUT_DIR' >&2; exit 2; }
export CANN_ROOT=${CANN_ROOT:-/usr/local/Ascend/cann-9.0.0}
source "$CANN_ROOT/set_env.sh"
set -u
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$1"
cmake -S "$here/raw_host" -B "$1/host" -DCANN_ROOT="$CANN_ROOT" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON > "$1/configure.log" 2>&1
cmake --build "$1/host" --parallel > "$1/compile.log" 2>&1
ldd "$1/host/raw_int32_host" > "$1/ldd.log"
! grep -q 'not found' "$1/ldd.log"
b="$CANN_ROOT/aarch64-linux/asc"
bisheng -x cce --cce-aicore-only --cce-aicore-arch=dav-m200 -std=c++17 -O2 -D__NPU_ARCH__=2002 \
 -I"$CANN_ROOT/compiler/tikcpp/tikcfw" -I"$b" -I"$b/include" -I"$b/include/basic_api" -I"$b/include/utils" -I"$b/impl" -I"$b/impl/basic_api" \
 -fsyntax-only "$here/local_check/raw_interfaces.cpp" > "$1/interface.log" 2>&1
echo 'LOCAL_RAW_HOST_COMPILED_NOT_QUERIED; INTERFACES_SYNTAX_ACCEPTED; OPP_AND_NPU_NOT_RUN' | tee "$1/status.txt"
