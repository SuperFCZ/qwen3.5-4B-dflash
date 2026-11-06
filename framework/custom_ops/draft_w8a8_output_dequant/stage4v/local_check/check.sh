#!/usr/bin/env bash
# Local checks only. Private API stub always fails; no execution or installation.
set -eo pipefail
[[ $# == 1 && ! -e "$1" ]] || { echo 'usage: check.sh NEW_OUTPUT_DIR' >&2; exit 2; }
export CANN_ROOT=${CANN_ROOT:-/usr/local/Ascend/cann-9.0.0}
source "$CANN_ROOT/set_env.sh"
set -u
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
out=$1
mkdir -p "$out/api/vendors/customize/op_api/include" "$out/api/vendors/customize/op_api/lib"
api="$out/api/vendors/customize/op_api"
cat > "$api/include/aclnn_d_flash_w8_a8_full_k_channel.h" <<'H'
#pragma once
#include <aclnn/aclnn_base.h>
extern "C" {
aclnnStatus aclnnDFlashW8A8FullKChannelGetWorkspaceSize(const aclTensor*,const aclTensor*,const aclTensor*,int64_t,int64_t,int64_t,aclTensor*,uint64_t*,aclOpExecutor**);
aclnnStatus aclnnDFlashW8A8FullKChannel(void*,uint64_t,aclOpExecutor*,aclrtStream);
}
H
cat > "$out/stub.cpp" <<'CPP'
#include "aclnn_d_flash_w8_a8_full_k_channel.h"
aclnnStatus aclnnDFlashW8A8FullKChannelGetWorkspaceSize(const aclTensor*,const aclTensor*,const aclTensor*,int64_t,int64_t,int64_t,aclTensor*,uint64_t*,aclOpExecutor**){return 1;}
aclnnStatus aclnnDFlashW8A8FullKChannel(void*,uint64_t,aclOpExecutor*,aclrtStream){return 1;}
CPP
c++ -std=c++17 -shared -fPIC -I"$CANN_ROOT/include" -I"$api/include" "$out/stub.cpp" -o "$api/lib/libcust_opapi.so" > "$out/stub.log" 2>&1
cmake -S "$here/runner" -B "$out/runner" -DCANN_ROOT="$CANN_ROOT" -DOPP_ROOT="$out/api" -DCOMMON_INCLUDE="$here/../../draft_quant/test" > "$out/runner-configure.log" 2>&1
cmake --build "$out/runner" --parallel > "$out/runner-build.log" 2>&1
ldd "$out/runner/stage4v_runner" > "$out/runner-ldd.log"
! grep -q 'not found' "$out/runner-ldd.log"
cmake -S "$here/host" -B "$out/host" -DCANN_ROOT="$CANN_ROOT" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON > "$out/host-configure.log" 2>&1
cmake --build "$out/host" --parallel > "$out/host-build.log" 2>&1
ldd "$out/host/stage4v_host" > "$out/host-ldd.log"
! grep -q 'not found' "$out/host-ldd.log"
base="$CANN_ROOT/aarch64-linux/asc"
# Parse actual Kernel against actual SDK, with a LOCAL ONLY tiling-decode adapter.
# This is NOT generated OPP ABI/link validation; SDK APIs and computation source are unchanged.
python3 - "$here" "$out" <<'PY'
import re,sys
from pathlib import Path
here,out=map(Path,sys.argv[1:])
h=(here/'op_host/d_flash_w8_a8_full_k_channel_tiling.h').read_text()
fields=re.findall(r'TILING_DATA_FIELD_DEF(?:_STRUCT)?\((\w+),\s*(\w+)\);',h)
if len(fields)!=8:raise ValueError('tiling schema changed')
text='#include "kernel_operator.h"\n#include "lib/matmul_intf.h"\n'
text+='struct LocalSyntaxTiling {\n'+''.join(t+' '+n+';\n' for t,n in fields)+'};\n'
text+='#undef GET_TILING_DATA\n#define GET_TILING_DATA(td,ptr) LocalSyntaxTiling td{}\n'
(out/'syntax_adapter.h').write_text(text)
PY
for fault in 0 1; do
if bisheng -x cce --cce-aicore-only --cce-aicore-arch=dav-m200 -std=c++17 -O2 -D__NPU_ARCH__=2002 -DSTAGE4V_FAULT="$fault" \
 -I"$CANN_ROOT/compiler/tikcpp/tikcfw" -I"$base" -I"$base/include" -I"$base/include/basic_api" \
 -I"$base/include/utils" -I"$base/impl" -I"$base/impl/basic_api" \
 -include "$out/syntax_adapter.h" -fsyntax-only "$here/op_kernel/d_flash_w8_a8_full_k_channel.cpp" > "$out/kernel-$fault-syntax.log" 2>&1; then
 echo 'LOCAL_SDK_SYNTAX_ACCEPTED; RUNNERS_LINKED_WITH_FAILING_API_STUB; ABI_OPP_NPU_NOT_RUN' | tee "$out/status.txt"
else
 rc=$?
 echo "LOCAL_KERNEL_SYNTAX_BLOCKED rc=$rc; RUNNERS_LINKED_WITH_FAILING_API_STUB; ABI_OPP_NPU_NOT_RUN" | tee "$out/status.txt"
 exit "$rc"
fi
done
find "$out" -type f \( -name stage4v_runner -o -name syntax_adapter.h -o -name status.txt \) -exec sha256sum '{}' \; > "$out/identities.txt"
