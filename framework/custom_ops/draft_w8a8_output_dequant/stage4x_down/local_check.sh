#!/usr/bin/env bash
# Real CANN private builds in new directories; never install or execute them.
set -eo pipefail
[[ $# == 1 && ! -e "$1" ]] || { echo 'usage: local_check.sh NEW_OUT' >&2; exit 2; }
export CANN_ROOT=${CANN_ROOT:-/usr/local/Ascend/cann-9.0.0}
source "$CANN_ROOT/set_env.sh"
set -u
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
out=$1;mkdir -p "$out"
for kind in candidate fault; do
  dest="$out/$kind";mkdir -p "$dest"
  python3 -B - "$here" "$dest/source" "$kind" <<'PYCODE'
import sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import x_common as c
from x_transform import materialize
materialize(Path(sys.argv[2]),sys.argv[3],c.identity,c.write)
PYCODE
  if [[ "$kind" == candidate ]]; then op=DFlashDownFullKBNz;else op=DFlashDownFullKBNzFault;fi
  msopgen gen -i "$dest/source/operator.json" -c ai_core-Ascend310P3 -lan cpp -out "$dest/CustomOp" -op "$op" > "$dest/msopgen.log" 2>&1
  cp "$dest/source/op_host/"* "$dest/CustomOp/op_host/"
  cp "$dest/source/op_kernel/"* "$dest/CustomOp/op_kernel/"
  python3 -B - "$here" "$dest" <<'PY'
import sys
from pathlib import Path
here,dest=map(Path,sys.argv[1:]);base=here.parent/'stage4v'
sys.path[:0]=[str(base),str(base/'test')]
from test_stage4v import fixture_tiling
from v_common import contract_header,p
platform,row=fixture_tiling()
row['fields'].update(singleCoreK=9728,Ka=9728,Kb=9728,depthA1=76,depthB1=8,stepKa=76,stepKb=4,shareL1Size=417792)
# Compile-only fixture, never accepted as Formal TilingContext evidence.
t1=dict(fields={k:row['fields'][k] for k in p.FIELDS},capacity=platform['capacity'],system_workspace_bytes=platform['system_workspace_bytes'],scratch=202240)
(dest/'CustomOp/op_host/t1_contract.h').write_text(contract_header(t1).replace('STAGE4V','STAGE4X').replace('S4V','S4X'))
PY
  (cd "$dest/CustomOp" && bash build.sh) > "$dest/build.log" 2>&1
  api="$dest/CustomOp/build_out/_CPack_Packages/Linux/External/custom_opp_ubuntu_aarch64.run/packages"
  cmake -S "$here/$kind/runner" -B "$dest/runner-build" -DCANN_ROOT="$CANN_ROOT" -DOPP_ROOT="$api" -DCOMMON_INCLUDE="$here/../../draft_quant/test" > "$dest/runner-configure.log" 2>&1
  cmake --build "$dest/runner-build" --parallel > "$dest/runner-compile.log" 2>&1
  LD_LIBRARY_PATH="$api/vendors/customize/op_api/lib:${LD_LIBRARY_PATH:-}" ldd "$dest/runner-build/guard_${kind}_runner" > "$dest/ldd.log"
  ! grep -q 'not found' "$dest/ldd.log"
  printf '%s\n' 'LOCAL_CANN_BUILD_ONLY; SYNTHETIC_T1_CONTRACT; INSTALL/HOST_QUERY/NPU/BENCHMARK_NOT_RUN' > "$dest/status.txt"
done
# Build actual Host-only query but never execute it in the NPU-less container.
cmake -S "$here/host" -B "$out/host-build" -DCANN_ROOT="$CANN_ROOT" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON > "$out/host-configure.log" 2>&1
cmake --build "$out/host-build" --parallel > "$out/host-compile.log" 2>&1
ldd "$out/host-build/stage4x_host" > "$out/host-ldd.log"
! grep -q 'not found' "$out/host-ldd.log"
