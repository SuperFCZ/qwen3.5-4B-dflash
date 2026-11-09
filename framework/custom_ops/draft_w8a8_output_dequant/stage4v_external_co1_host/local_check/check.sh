#!/usr/bin/env bash
# Local compile/package/link only. Never installs or calls the Host-only runner.
set -eo pipefail
[[ $# == 1 && ! -e "$1" ]] || { echo 'usage: check.sh NEW_DIR' >&2; exit 2; }
export CANN_ROOT=${CANN_ROOT:-/usr/local/Ascend/cann-9.0.0}
source "$CANN_ROOT/set_env.sh"
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
out=$1;mkdir -p "$out"
msopgen gen -i "$here/operator.json" -c ai_core-Ascend310P3 -lan cpp -out "$out/CustomOp" -op DFlashExternalCo1HostProbe > "$out/msopgen.log" 2>&1
cp "$here"/op_host/* "$out/CustomOp/op_host/"
cp "$here"/op_kernel/* "$out/CustomOp/op_kernel/"
python3 -B - "$here" "$out" <<'PY'
from pathlib import Path
import sys
here,out=map(Path,sys.argv[1:]);sys.path.insert(0,str(here))
import co1_host_common as c
sdk=c.vdeq_bind(Path('/usr/local/Ascend/cann-9.0.0'))
(out/'CustomOp/op_host/co1_fields.h').write_text(c.generated_header(c.schema_names(sdk)))
c.write(out/'sdk-binding.json',sdk)
PY
# CPU-only nested serializer reproduction, linked to the actual selected SDK.
c++ -std=c++11 -I"$out/CustomOp/op_host" -I"$CANN_ROOT/include" -I"$CANN_ROOT/include/experiment" \
 -I"$CANN_ROOT/aarch64-linux/asc/include" -I"$CANN_ROOT/aarch64-linux/asc/include/adv_api" -I"$CANN_ROOT/aarch64-linux/asc/include/utils" \
 "$here/local_check/serializer.cpp" "$CANN_ROOT/lib64/libregister.so" "$CANN_ROOT/lib64/libc_sec.so" \
 -o "$out/serializer" > "$out/serializer-build.log" 2>&1
ldd "$out/serializer" > "$out/serializer-ldd.log"
! grep -q 'not found' "$out/serializer-ldd.log"
"$out/serializer" "$out" > "$out/serializer-run.log" 2>&1
python3 -B - "$here" "$out" <<'PYCODE'
from pathlib import Path
import sys
here,out=map(Path,sys.argv[1:]);sys.path.insert(0,str(here))
import co1_host_common as c
import co1_serialization as s
schema=c.identity(here/'test/fixtures/tcube-schema.json')
p=dict(codec_and_schema_sdk=c.read(out/'sdk-binding.json'),schema_identity=schema,schema=c.read(c.verify(schema)))
p['serialization_binding']=s.bind_layout(p)
observation=c.read(out/'serializer-observation.json')
assert observation['status']=='LOCAL_SERIALIZER_SELF_CHECK_ONLY' and observation['cube_payload_source']=='SDK_PARENT_SERIALIZER_CPU_TEST'
# Exercise the pure wire validator with a synthetic format record, not a Context.
r=s.validate_serialization(dict(observation,cube_payload_source=s.SOURCE),(out/'serializer-parent.bin').read_bytes(),p)
r.update(status='LOCAL_SERIALIZER_SELF_CHECK_ONLY',cube_payload_source='SDK_PARENT_SERIALIZER_CPU_TEST',formal_context='NOT_RUN')
c.write(out/'serializer-binding.json',r)
PYCODE
(cd "$out/CustomOp" && bash build.sh) > "$out/build.log" 2>&1
package="$out/CustomOp/build_out/_CPack_Packages/Linux/External/custom_opp_ubuntu_aarch64.run/packages"
cmake -S "$here/runner" -B "$out/runner-build" -DCANN_ROOT="$CANN_ROOT" -DOPP_ROOT="$package" -DCOMMON_INCLUDE="$here/../../draft_quant/test" > "$out/runner-configure.log" 2>&1
cmake --build "$out/runner-build" --parallel > "$out/runner-build.log" 2>&1
ldd "$out/runner-build/co1_host_runner" > "$out/runner-ldd.log"
! grep -q 'not found' "$out/runner-ldd.log"
nm -u "$out/runner-build/co1_host_runner" > "$out/runner-symbols.log"
python3 -B - "$here" "$out" <<'PY'
from pathlib import Path
import sys,re
here,out=map(Path,sys.argv[1:]);sys.path.insert(0,str(here))
import co1_host_common as c
from co1_host_dependencies import collect
header=Path('/usr/local/Ascend/cann-9.0.0/aarch64-linux/asc/include/utils/tiling/platform/platform_ascendc.h')
if c.identity(header)['sha256']!='44cf3f35292f15a8735b0d43b7e88198ca15d7b4a3bf44719eb3f099697f7f2a':raise ValueError('platform mismatch')
c.write(out/'dependencies.json',collect(out/'CustomOp',Path('/usr/local/Ascend/cann-9.0.0'),c.identity(header)))
s=(out/'runner-symbols.log').read_text()
if 'aclnnDFlashExternalCo1HostProbeGetWorkspaceSize' not in s or re.search(r'\b(aclnnDFlashExternalCo1HostProbe|aclrtLaunch\w*|aclopExecute\w*)\b',s):raise ValueError('execution symbols imported')
c.write(out/'local-status.json',dict(status='LOCAL_COMPILE_ONLY',runner=c.identity(out/'runner-build/co1_host_runner'),kernel='EMPTY_COMPILE_ONLY_PLACEHOLDER',installation='NOT_RUN',formal_host='NOT_RUN',npu='NOT_RUN'))
PY
cat "$out/local-status.json"
