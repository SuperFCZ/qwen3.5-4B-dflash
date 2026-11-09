#!/usr/bin/env bash
# Syntax-only SDK interface check. No compute Kernel, OPP, install or device call.
set -eo pipefail
[[ $# == 1 && ! -e "$1" ]] || { echo 'usage: check.sh NEW_DIR' >&2; exit 2; }
export CANN_ROOT=${CANN_ROOT:-/usr/local/Ascend/cann-9.0.0}
source "$CANN_ROOT/set_env.sh"
set -u
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
out=$1
python3 -B "$here/run_vdeq_review.py" review --cann-root "$CANN_ROOT" --output-dir "$out"
base="$CANN_ROOT/aarch64-linux/asc"
bisheng -x cce --cce-aicore-only --cce-aicore-arch=dav-m200 -std=c++17 -O2 -D__NPU_ARCH__=2002 \
 -I"$CANN_ROOT/compiler/tikcpp/tikcfw" -I"$base" -I"$base/include" -I"$base/include/basic_api" \
 -I"$base/include/utils" -I"$base/impl" -I"$base/impl/basic_api" \
 -fsyntax-only "$here/local_check/interfaces.cpp" > "$out/interface-syntax.log" 2>&1
printf '%s\n' 'LOCAL_INTERFACE_SYNTAX_CHECKED; L0C_INJECTION_RESOURCE_CONTRACT_UNRESOLVED; NO_KERNEL_NO_OPP_NPU_NOT_RUN' > "$out/local-status.txt"
cat "$out/local-status.txt"
