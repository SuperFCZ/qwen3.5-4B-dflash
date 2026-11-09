#!/usr/bin/env bash
# Offline Linux ARM64 checks only; no OPP install, platform query or Kernel launch.
set -eo pipefail
[[ $# == 1 && ! -e "$1" ]] || { echo 'usage: check.sh NEW_OUTPUT_DIR' >&2; exit 2; }
export CANN_ROOT=${CANN_ROOT:-/usr/local/Ascend/cann-9.0.0}
source "$CANN_ROOT/set_env.sh"
set -u
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
out=$1
mkdir -p "$out/api/vendors/customize/op_api/include" "$out/api/vendors/customize/op_api/lib" "$out/host-src"
api="$out/api/vendors/customize/op_api"
cat > "$api/include/aclnn_d_flash_full_k_int32_probe.h" <<'H'
#pragma once
#include <aclnn/aclnn_base.h>
extern "C" {
aclnnStatus aclnnDFlashFullKInt32ProbeGetWorkspaceSize(const aclTensor*,const aclTensor*,aclTensor*,uint64_t*,aclOpExecutor**);
aclnnStatus aclnnDFlashFullKInt32Probe(void*,uint64_t,aclOpExecutor*,aclrtStream);
}
H
cat > "$out/stub.cpp" <<'CPP'
#include "aclnn_d_flash_full_k_int32_probe.h"
aclnnStatus aclnnDFlashFullKInt32ProbeGetWorkspaceSize(const aclTensor*,const aclTensor*,aclTensor*,uint64_t*,aclOpExecutor**){return 1;}
aclnnStatus aclnnDFlashFullKInt32Probe(void*,uint64_t,aclOpExecutor*,aclrtStream){return 1;}
CPP
c++ -std=c++17 -shared -fPIC -I"$CANN_ROOT/include" -I"$api/include" "$out/stub.cpp" -o "$api/lib/libcust_opapi.so" > "$out/stub.log" 2>&1
cmake -S "$here/runner" -B "$out/runner" -DCANN_ROOT="$CANN_ROOT" -DOPP_ROOT="$out/api" -DCOMMON_INCLUDE="$here/../../draft_quant/test" > "$out/runner-configure.log" 2>&1
cmake --build "$out/runner" --parallel > "$out/runner-build.log" 2>&1
ldd "$out/runner/raw_probe_runner" > "$out/runner-ldd.log"
! grep -q 'not found' "$out/runner-ldd.log"
cp "$here"/op_host/* "$out/host-src/"
python3 -B - "$here" "$out" <<'PY'
import sys,re
from pathlib import Path
here,out=map(Path,sys.argv[1:]);sys.path.insert(0,str(here));sys.path.insert(0,str(here/'test'))
import raw_probe_common as c
from test_raw_probe import fixture
# Fixture is deliberately NOT a runtime TilingContext proof. Never run this .so.
(out/'host-src/t1_contract.h').write_text(c.contract_header(fixture()))
h=(here/'op_host/d_flash_full_k_int32_probe_tiling.h').read_text()
fields=re.findall(r'TILING_DATA_FIELD_DEF(?:_STRUCT)?\((\w+),\s*(\w+)\);',h)
assert len(fields)==8
text='#include "kernel_operator.h"\n#include "lib/matmul_intf.h"\nstruct LocalSyntaxTiling {\n'+''.join(t+' '+n+';\n' for t,n in fields)+'};\n'
text+='#undef GET_TILING_DATA\n#define GET_TILING_DATA(td,ptr) LocalSyntaxTiling td{}\n'
(out/'syntax_adapter.h').write_text(text)
c.write(out/'sdk-binding-local.json',c.bind(Path('/usr/local/Ascend/cann-9.0.0')))
PY
cmake -S "$here/local_check" -B "$out/host-build" -DCANN_ROOT="$CANN_ROOT" -DHOST_SOURCE="$out/host-src" > "$out/host-configure.log" 2>&1
cmake --build "$out/host-build" --parallel > "$out/host-build.log" 2>&1
ldd "$out/host-build/libraw_host_local.so" > "$out/host-ldd.log"
! grep -q 'not found' "$out/host-ldd.log"
base="$CANN_ROOT/aarch64-linux/asc"
bisheng -x cce --cce-aicore-only --cce-aicore-arch=dav-m200 -std=c++17 -O2 -D__NPU_ARCH__=2002 \
 -I"$CANN_ROOT/compiler/tikcpp/tikcfw" -I"$base" -I"$base/include" -I"$base/include/basic_api" \
 -I"$base/include/utils" -I"$base/impl" -I"$base/impl/basic_api" \
 -include "$out/syntax_adapter.h" -fsyntax-only "$here/op_kernel/d_flash_full_k_int32_probe.cpp" > "$out/kernel-syntax.log" 2>&1
printf '%s\n' 'LOCAL_ONLY_NOT_SERVER_VERIFIED: Host shared library linked, Runner linked with failing API stub, Kernel SDK syntax checked. Official OPP ABI/codegen, Formal Host, NPU NOT_RUN.' > "$out/status.txt"
find "$out" -type f \( -name raw_probe_runner -o -name libraw_host_local.so -o -name syntax_adapter.h -o -name status.txt \) -exec sha256sum '{}' \; > "$out/identities.txt"
cat "$out/status.txt"
