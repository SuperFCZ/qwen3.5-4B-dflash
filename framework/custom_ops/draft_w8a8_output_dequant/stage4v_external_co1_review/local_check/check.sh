#!/usr/bin/env bash
set -eo pipefail
[[ $# == 1 && ! -e "$1" ]] || { echo 'usage: check.sh NEW_DIR' >&2; exit 1; }
export CANN_ROOT=${CANN_ROOT:-/usr/local/Ascend/cann-9.0.0}
source "$CANN_ROOT/set_env.sh"
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
rc=0
python3 -B "$here/run_review.py" --output-dir "$1" || rc=$?
[[ $rc == 2 ]] || { echo 'audit failed or unexpected admission' >&2; exit 1; }
b="$CANN_ROOT/aarch64-linux/asc"
bisheng -x cce --cce-aicore-only --cce-aicore-arch=dav-m200 -std=c++17 -O2 -D__NPU_ARCH__=2002 \
 -I"$CANN_ROOT/compiler/tikcpp/tikcfw" -I"$b" -I"$b/include" -I"$b/include/basic_api" -I"$b/include/utils" -I"$b/impl" -I"$b/impl/basic_api" \
 -fsyntax-only "$here/local_check/interfaces.cpp" > "$1/interface-syntax.log" 2>&1
printf '%s\n' 'LOCAL_API_SYNTAX_CHECKED; FORMAL_HOST_BLOCKED; NO_COMPUTE_KERNEL_OPP_OR_NPU' > "$1/local-status.txt"
cat "$1/local-status.txt"
