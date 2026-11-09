#!/usr/bin/env bash
# Real local compiler/package check, NO install/query/launch. Synthetic T1 fixture.
set -eo pipefail
[[ $# == 2 && -f "$1/host-src/t1_contract.h" && ! -e "$2" ]] || { echo 'usage: compile_opp.sh LOCAL_CHECK_DIR NEW_OUTPUT_DIR' >&2; exit 2; }
export CANN_ROOT=${CANN_ROOT:-/usr/local/Ascend/cann-9.0.0}
source "$CANN_ROOT/set_env.sh"
set -u
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
fixture=$(cd "$1" && pwd);out=$2
mkdir -p "$out"
msopgen gen -i "$here/operator.json" -c ai_core-Ascend310P3 -lan cpp -out "$out/CustomOp" -op DFlashFullKInt32Probe > "$out/msopgen.log" 2>&1
cp "$here"/op_host/* "$out/CustomOp/op_host/"
cp "$fixture/host-src/t1_contract.h" "$out/CustomOp/op_host/"
cp "$here"/op_kernel/* "$out/CustomOp/op_kernel/"
(cd "$out/CustomOp" && bash build.sh) > "$out/build.log" 2>&1
find "$out/CustomOp/build_out" -type f \( -name 'DFlashFullKInt32Probe_*.o' -o -name 'custom_opp_*.run' -o -name 'libcust_opmaster_rt2.0.so' \) -exec sha256sum '{}' \; > "$out/artifacts.txt"
test -s "$out/artifacts.txt"
package="$out/CustomOp/build_out/_CPack_Packages/Linux/External/custom_opp_ubuntu_aarch64.run/packages"
cmake -S "$here/runner" -B "$out/actual-runner" -DCANN_ROOT="$CANN_ROOT" -DOPP_ROOT="$package" -DCOMMON_INCLUDE="$here/../../draft_quant/test" > "$out/actual-runner-configure.log" 2>&1
cmake --build "$out/actual-runner" --parallel > "$out/actual-runner-build.log" 2>&1
ldd "$out/actual-runner/raw_probe_runner" > "$out/actual-runner-ldd.log"
! grep -q 'not found' "$out/actual-runner-ldd.log"
python3 -B - "$here" "$out" <<'PYCODE'
import sys
from pathlib import Path
here,out=map(Path,sys.argv[1:]);sys.path.insert(0,str(here))
import raw_probe_common as c
from raw_probe_dependencies import collect
header=Path('/usr/local/Ascend/cann-9.0.0/aarch64-linux/asc/include/utils/tiling/platform/platform_ascendc.h')
if c.identity(header)['sha256']!='44cf3f35292f15a8735b0d43b7e88198ca15d7b4a3bf44719eb3f099697f7f2a':raise ValueError('local platform identity mismatch')
c.write(out/'host-dependencies.json',collect(out/'CustomOp',Path('/usr/local/Ascend/cann-9.0.0'),c.identity(header)))
PYCODE
printf '%s\n' 'LOCAL_ONLY_NOT_SERVER_VERIFIED: isolated generated OPP and real generated-ABI Runner compiled/linked/packaged with LOCAL synthetic T1 fixture. Installation, Formal Host query, NPU NOT_RUN.' > "$out/status.txt"
cat "$out/status.txt"
