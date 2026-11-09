#!/usr/bin/env bash
set -eo pipefail
[[ $# == 1 && ! -e "$1" ]] || exit 2
export CANN_ROOT=${CANN_ROOT:-/usr/local/Ascend/cann-9.0.0}
source "$CANN_ROOT/set_env.sh"
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
out=$1;mkdir -p "$out"
msopgen gen -i "$here/operator.json" -c ai_core-Ascend310P3 -lan cpp -out "$out/CustomOp" -op DFlashExternalCo1EchoProbe > "$out/msopgen.log" 2>&1
cp "$here"/op_host/* "$out/CustomOp/op_host/"
cp "$here"/op_kernel/* "$out/CustomOp/op_kernel/"
python3 -B - "$here" "$out" <<'PY'
from pathlib import Path
import sys
here,out=map(Path,sys.argv[1:]);sys.path[:0]=[str(here),str(here/'test')]
import echo_common as c
from test_echo import fixture
sdk=c.old.vdeq_bind(Path('/usr/local/Ascend/cann-9.0.0'))
# Compilation fixture only, not an actual CO1 platform or admission report.
(out/'CustomOp/op_host/co1_fields.h').write_text(c.generated_header(c.old.schema_names(sdk),fixture()))
PY
(cd "$out/CustomOp" && bash build.sh) > "$out/build.log" 2>&1
package="$out/CustomOp/build_out/_CPack_Packages/Linux/External/custom_opp_ubuntu_aarch64.run/packages"
cmake -S "$here/runner" -B "$out/runner-build" -DCANN_ROOT="$CANN_ROOT" -DOPP_ROOT="$package" -DCONTRACT_INCLUDE="$out/CustomOp/op_host" -DCOMMON_INCLUDE="$here/../../draft_quant/test" > "$out/cmake.log" 2>&1
cmake --build "$out/runner-build" --parallel > "$out/runner.log" 2>&1
ldd "$out/runner-build/echo_runner" > "$out/ldd.log"
! grep -q 'not found' "$out/ldd.log"
printf '%s\n' 'LOCAL_COMPILE_ONLY: real Echo Kernel/Host/private API Runner linked; private install, Formal query and NPU NOT_RUN; VDEQ16 NOT_IMPLEMENTED' > "$out/status.txt"
cat "$out/status.txt"
