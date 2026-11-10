#!/usr/bin/env bash
# Rebuild only the unchanged historical Runner source, in a fresh local directory.
# This binary is not substituted for the accepted server Runner.
set -eo pipefail
[[ $# == 2 && ! -e "$1" ]] || { echo 'usage: local_build_check.sh NEW_OUT EXISTING_API_ROOT' >&2; exit 2; }
export CANN_ROOT=${CANN_ROOT:-/usr/local/Ascend/cann-9.0.0}
source "$CANN_ROOT/set_env.sh"
set -u
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
out=$1;api=$2
[[ -f "$api/vendors/customize/op_api/include/aclnn_d_flash_w8_a8_full_k_channel.h" ]] || exit 1
mkdir -p "$out"
cmake -S "$here/../stage4v_fullk_perf_explore/runner" -B "$out/build" -DCANN_ROOT="$CANN_ROOT" -DOPP_ROOT="$api" -DCOMMON_INCLUDE="$here/../../draft_quant/test" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON > "$out/configure.log" 2>&1
cmake --build "$out/build" --parallel > "$out/compile.log" 2>&1
LD_LIBRARY_PATH="$api/vendors/customize/op_api/lib:${LD_LIBRARY_PATH:-}" ldd "$out/build/fullk_explore_runner" > "$out/ldd.log"
! grep -q 'not found' "$out/ldd.log"
sha256sum "$out/build/fullk_explore_runner" > "$out/runner-identity.txt"
nm -u "$out/build/fullk_explore_runner" > "$out/symbols.log"
grep -q 'aclnnDFlashW8A8FullKChannelGetWorkspaceSize' "$out/symbols.log"
printf '%s\n' 'LOCAL_RUNNER_COMPILE_LINK_ONLY; KERNEL_BUILD/INSTALL/NPU/PROFILE_NOT_RUN' > "$out/status.txt"
