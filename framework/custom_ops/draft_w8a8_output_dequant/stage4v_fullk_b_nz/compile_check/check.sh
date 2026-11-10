#!/usr/bin/env bash
set -eo pipefail
[[ $# == 1 && ! -e "$1" ]] || { echo 'usage: check.sh NEW_OUT' >&2; exit 2; }
export CANN_ROOT=${CANN_ROOT:-/usr/local/Ascend/cann-9.0.0}
source "$CANN_ROOT/set_env.sh"
set -u
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$1"
base="$CANN_ROOT/aarch64-linux/asc"
sha256sum "$here/interfaces.cpp" > "$1/source.sha256"
if bisheng -x cce --cce-aicore-only --cce-aicore-arch=dav-m200 -std=c++17 -O2 \
 -D__NPU_ARCH__=2002 -I"$CANN_ROOT/compiler/tikcpp/tikcfw" -I"$base" -I"$base/include" -I"$base/include/basic_api" \
 -I"$base/include/utils" -I"$base/impl" -I"$base/impl/basic_api" \
 -fsyntax-only "$here/interfaces.cpp" > "$1/compile.log" 2>&1; then
 printf '%s\n' 'INTERFACE_SYNTAX_ACCEPTED_LOCAL_ONLY; NPU_NOT_RUN' > "$1/status.txt"
else
 rc=$?;printf 'EXACT_INTERFACE_REJECTED returncode=%s; NPU_NOT_RUN\n' "$rc" > "$1/status.txt";exit "$rc"
fi
