#!/usr/bin/env bash
set -eo pipefail
[[ $# == 1 && ! -e "$1" ]] || { echo 'usage: check_l1.sh NEW_OUTPUT_DIR' >&2; exit 2; }
export CANN_ROOT=${CANN_ROOT:-/usr/local/Ascend/cann-9.0.0}
source "$CANN_ROOT/set_env.sh"
set -u
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$1"
b="$CANN_ROOT/aarch64-linux/asc"
bisheng -x cce --cce-aicore-only --cce-aicore-arch=dav-m200 -std=c++17 -O2 -D__NPU_ARCH__=2002 \
 -I"$CANN_ROOT/compiler/tikcpp/tikcfw" -I"$b" -I"$b/include" -I"$b/include/basic_api" \
 -I"$b/include/utils" -I"$b/impl" -I"$b/impl/basic_api" \
 -fsyntax-only "$here/l1_contract.cpp" > "$1/compile.log" 2>&1
sha256sum "$here/l1_contract.cpp" > "$1/source-identity.txt"
echo 'LOCAL_RESOURCE_TYPE_ASSERTIONS_ACCEPTED; NO_KERNEL_OR_PLATFORM_EXECUTION' | tee "$1/status.txt"
