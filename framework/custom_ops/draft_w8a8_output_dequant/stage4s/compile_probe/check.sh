#!/usr/bin/env bash
# Opt-in compile only; never execute. All output goes to a new private directory.
set -eo pipefail
[[ $# == 1 && ! -e "$1" ]] || { echo 'usage: check.sh NEW_OUTPUT_DIR' >&2; exit 2; }
export CANN_ROOT=${CANN_ROOT:-/usr/local/Ascend/cann-9.0.0}
source "$CANN_ROOT/set_env.sh"
set -u
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$1"
base="$CANN_ROOT/aarch64-linux/asc"
bisheng -x cce --cce-aicore-only --cce-aicore-arch=dav-m200 -std=c++17 -O2 \
 -D__NPU_ARCH__=2002 -I"$base" -I"$base/include" -I"$base/include/basic_api" \
 -I"$base/include/utils" -I"$base/impl" -I"$base/impl/basic_api" \
 -c "$here/copy.cpp" -o "$1/copy.o" > "$1/compile.log" 2>&1
sha256sum "$here/copy.cpp" "$1/copy.o" > "$1/identities.txt"
readelf -sW "$1/copy.o" > "$1/symbols.txt"
printf 'LOCAL_CALL_COMPILE_ONLY; SERVER_IDENTITY_AND_NPU_NOT_VERIFIED\n'
