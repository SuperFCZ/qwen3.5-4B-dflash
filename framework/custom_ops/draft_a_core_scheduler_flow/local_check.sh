#!/usr/bin/env bash
# Inside the Linux ARM64 SDK container. Only CPU Host targets and host executables.
set -euo pipefail
cd "$(dirname "$0")"
T=${1:?pass previously compiled scheduler A CustomOp template}
P=${2:?pass existing generated private API OPP}
CANN_ROOT=${CANN_ROOT:-/usr/local/Ascend/cann-9.0.0}
out=$(mktemp -d /work/build/scheduler-flow-XXXXXX)
echo "$out"
for v in A B; do
  d="$out/$v"; mkdir "$d"; cp -R "$T" "$d/project"
  cp op_host/* "$d/project/op_host/"
  case "$v" in A) policy=0;; B) policy=1;; esac
  printf '#pragma once\n#define DFLASH_SCHEDULER_POLICY %s\n' "$policy" > "$d/project/op_host/scheduler_variant.h"
  python3 -B - "$d/project" "$d/host" "$CANN_ROOT" <<'PY'
import subprocess,sys
from pathlib import Path
from flow_build import preset_options
src,dst,cann=map(Path,sys.argv[1:])
assert 'add_subdirectory(op_kernel)' not in (src/'CMakeLists.txt').read_text()
subprocess.run(['cmake','-S',src,'-B',dst,*preset_options(src,cann)],check=True)
PY
  cmake --build "$d/host" --target cust_optiling cust_op_proto --parallel 2
done
for t in runner om; do
  cmake -S "$t" -B "$out/$t" -DCANN_ROOT="$CANN_ROOT" -DPRIVATE_OPP="$P" -DCOMMON_INCLUDE="$(pwd)/../draft_quant/test"
  cmake --build "$out/$t"
done
python3 -B -m unittest discover -s test -v
