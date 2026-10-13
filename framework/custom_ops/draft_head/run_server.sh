#!/usr/bin/env bash
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python_bin=${MODEL_PYTHON:-python3}
cann_root=${CANN_ROOT:-${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}}
device=${DEVICE_ID:-0}
core_limit=${DFLASH_HEAD_CORE_LIMIT:-0}
native_only=${HEAD_NATIVE_ONLY:-0}
conversion=${HEAD_CONVERSION:-compare_select}
top1=${HEAD_TOP1:-scalar}
profile=${HEAD_PROFILE_STAGE:-none}
build_only=${HEAD_BUILD_ONLY:-0}
[[ "$top1" == scalar || "$top1" == vector ]] || { echo 'HEAD_TOP1 must be scalar or vector' >&2; exit 1; }
[[ "$profile" == none || "$profile" == matmul || "$profile" == convert ]] || { echo 'invalid HEAD_PROFILE_STAGE' >&2; exit 1; }
[[ "$build_only" == 0 || "$build_only" == 1 ]] || exit 1
if [[ "$profile" != none && ( "$build_only" != 1 || "$top1" != scalar ) ]]; then
  echo 'profiling observers require isolated build-only scalar configuration' >&2; exit 1
fi
if [[ "$build_only" == 1 ]]; then
  : "${HEAD_BUILD_STATE:?Set new build-state JSON path}" "${HEAD_CAPTURE:?Reuse capture}" "${HEAD_NATIVE_MANIFEST:?Reuse Native OM}"
  [[ ! -e "$HEAD_BUILD_STATE" && "$native_only" == 0 ]] || exit 1
fi
if [[ "$top1" == vector ]]; then
  : "${HEAD_CAPTURE:?Reuse existing Head capture}" "${HEAD_NATIVE_MANIFEST:?Reuse existing Native OM}"
fi
diagnostics=${HEAD_DIAGNOSTICS:-0}
[[ "$conversion" == cast || "$conversion" == minmax || "$conversion" == compare_select ]] || { echo 'HEAD_CONVERSION must be cast, minmax or compare_select' >&2; exit 1; }
[[ "$diagnostics" == 0 || "$diagnostics" == 1 ]] || { echo 'HEAD_DIAGNOSTICS must be 0 or 1' >&2; exit 1; }
if [[ "$conversion" == minmax && "$diagnostics" != 1 ]]; then echo 'unaccepted minmax requires HEAD_DIAGNOSTICS=1' >&2; exit 1; fi
if [[ "$diagnostics" == 1 ]]; then
  [[ "$native_only" == 0 ]] || { echo 'diagnostics require custom build' >&2; exit 1; }
  : "${HEAD_CAPTURE:?Reuse existing capture}" "${HEAD_NATIVE_MANIFEST:?Reuse existing Native OM}"
fi
if [[ ! "$core_limit" =~ ^(0|[1-9][0-9]?)$ ]] || (( core_limit>64 )); then
  echo 'DFLASH_HEAD_CORE_LIMIT must be 0..64' >&2; exit 1
fi
if [[ "$native_only" != 0 && "$native_only" != 1 ]]; then echo 'HEAD_NATIVE_ONLY must be 0 or 1' >&2; exit 1; fi
if [[ -f "$cann_root/set_env.sh" ]]; then source "$cann_root/set_env.sh"
elif [[ -f "$cann_root/bin/setenv.bash" ]]; then source "$cann_root/bin/setenv.bash"
else echo 'CANN environment missing' >&2; exit 1; fi
export ASCEND_HOME_PATH="$cann_root" ASCEND_INSTALL_PATH="$cann_root"
set -u
mkdir -p "$here/.build" "$here/.runs"
build_root=$(mktemp -d "$here/.build/head-XXXXXXXX")
run_root=$(mktemp -d "$here/.runs/head-XXXXXXXX")
exec > >(tee "$run_root/server.log") 2>&1
phase=prepare
trap 'rc=$?; if (( rc!=0 )); then echo "FAIL during $phase (exit $rc); evidence: $run_root" >&2; fi' EXIT
echo "Head full vocabulary correctness; build: $build_root; evidence: $run_root"
git -C "$here" rev-parse HEAD > "$run_root/source-commit.txt"
"$python_bin" "$here/../draft_quant/test/opp_preflight.py" --root "$build_root" --report "$run_root/path-preflight.json"

capture=${HEAD_CAPTURE:-}
if [[ -z "$capture" ]]; then
  phase=capture
  : "${AI_RUN_DIR:?Set AI_RUN_DIR outside the repository}"
  : "${DRAFT_DIR:?Set pinned W8 Draft checkpoint}" "${TARGET_DIR:?Set original Target checkpoint}"
  : "${C16_REPLAY_REPORT:?Set real C16 frozen replay report}" "${C64_REPLAY_REPORT:?Set real C64 frozen replay report}"
  artifact_root=$(mktemp -d "$AI_RUN_DIR/draft-head-XXXXXXXX")
  extra=()
  if [[ -n "${FEATURE_LAYERS:-}" ]]; then extra+=(--feature-layers "$FEATURE_LAYERS"); fi
  if [[ -n "${HEAD_WEIGHT_KEY:-}" ]]; then extra+=(--head-key "$HEAD_WEIGHT_KEY"); fi
  if [[ -n "${HEAD_EMBEDDING_KEY:-}" ]]; then extra+=(--embedding-key "$HEAD_EMBEDDING_KEY"); fi
  "$python_bin" "$here/test/capture.py" --draft-dir "$DRAFT_DIR" --target-dir "$TARGET_DIR" \
    --c16-replay-report "$C16_REPLAY_REPORT" --c64-replay-report "$C64_REPLAY_REPORT" \
    --output-dir "$artifact_root/capture" --device-id "$device" "${extra[@]}"
  capture="$artifact_root/capture/manifest.json"
fi
native=${HEAD_NATIVE_MANIFEST:-}
if [[ -z "$native" ]]; then
  phase=export_native
  : "${AI_RUN_DIR:?Set AI_RUN_DIR outside repository for native exports}"
  native_root=$(mktemp -d "$AI_RUN_DIR/head-native-XXXXXXXX")
  "$python_bin" "$here/test/export_native.py" --capture "$capture" --output-dir "$native_root/models" \
    --device-id "$device" --atc "$cann_root/bin/atc"
  native="$native_root/models/native.json"
fi
printf '%s\n' "$capture" > "$run_root/capture-path.txt"
printf '%s\n' "$native" > "$run_root/native-path.txt"
phase=build_native_runner
cmake -S "$here/test" -B "$build_root/native" -DCANN_ROOT="$cann_root"
cmake --build "$build_root/native" --parallel
extra=()
if [[ "$native_only" == 1 ]]; then
  extra+=(--native-only)
else
  phase=msopgen
  project="$build_root/CustomOp"; install="$build_root/opp"
  msopgen gen \
    -i "$here/DFlashDraftLmHeadTop1.json" \
    -c ai_core-Ascend310P3 -lan cpp \
    -out "$project" \
    -op DFlashDraftLmHeadTop1
  msopgen gen \
    -i "$here/DFlashDraftLmHeadTop1.json" \
    -c ai_core-Ascend310P3 -lan cpp \
    -out "$project" \
    -op DFlashDraftLmHeadTop1Audit \
    -m 1
  if [[ "$diagnostics" == 1 ]]; then
    for op in DFlashHeadProbe DFlashCastProbe; do
      msopgen gen -i "$here/test/probe/$op.json" -c ai_core-Ascend310P3 -lan cpp -out "$project" -op "$op" -m 1
    done
  fi
  if [[ "$top1" == vector ]]; then
    msopgen gen -i "$here/test/top1_probe/DFlashTop1Probe.json" -c ai_core-Ascend310P3 -lan cpp -out "$project" -op DFlashTop1Probe -m 1
  fi
  phase=check_generated_project
  for op in d_flash_draft_lm_head_top1 d_flash_draft_lm_head_top1_audit; do
    for part in op_host op_kernel; do
      if [[ ! -s "$project/$part/$op.cpp" ]]; then
        echo "msopgen did not generate $part/$op.cpp" >&2; exit 1
      fi
    done
  done
  for part in op_host op_kernel; do
    if [[ ! -s "$project/$part/CMakeLists.txt" ]]; then
      echo "Generated $part/CMakeLists.txt missing" >&2; exit 1
    fi
    cp "$project/$part/CMakeLists.txt" "$run_root/generated-$part-CMakeLists.txt"
  done
  cp "$project/CMakeLists.txt" "$run_root/generated-CMakeLists.txt"
  cp "$here"/op_host/*.cpp "$here"/op_host/*.h "$project/op_host/"
  cp "$here"/op_kernel/*.cpp "$here"/op_kernel/*.h "$project/op_kernel/"
  cp "$here/op_host/head_contract.h" "$project/op_kernel/head_contract.h"
  if [[ "$diagnostics" == 1 ]]; then
    for op in d_flash_head_probe d_flash_cast_probe; do
      cp "$here/test/probe/${op}_host.cpp" "$project/op_host/$op.cpp"
      cp "$here/test/probe/${op}_kernel.cpp" "$project/op_kernel/$op.cpp"
    done
    cp "$here/test/probe/"*_tiling.h "$project/op_host/"
    cp "$here/test/probe/vector_probe.h" "$project/op_kernel/"
    cat >> "$project/op_host/CMakeLists.txt" <<'PROBE_CMAKE_CHECK'
get_target_property(probe_sources cust_optiling SOURCES)
foreach(probe_name d_flash_head_probe d_flash_cast_probe)
  set(probe_found FALSE)
  foreach(probe_source IN LISTS probe_sources)
    get_filename_component(probe_basename "${probe_source}" NAME)
    if(probe_basename STREQUAL "${probe_name}.cpp")
      set(probe_found TRUE)
    endif()
  endforeach()
  if(NOT probe_found)
    message(FATAL_ERROR "Missing diagnostic Host source ${probe_name}")
  endif()
endforeach()
PROBE_CMAKE_CHECK
  fi
  if [[ "$top1" == vector ]]; then
    cp "$here/test/top1_probe/d_flash_top1_probe_host.cpp" "$project/op_host/d_flash_top1_probe.cpp"
    cp "$here/test/top1_probe/d_flash_top1_probe_tiling.h" "$project/op_host/"
    cp "$here/test/top1_probe/d_flash_top1_probe_kernel.cpp" "$project/op_kernel/d_flash_top1_probe.cpp"
  fi
  # Check resolved CMake target sources, including aux_source_directory/globs.
  # Only the generated build copy is extended; preserve the original above.
  cat >> "$project/op_host/CMakeLists.txt" <<'HEAD_CMAKE_CHECK'

if(NOT TARGET cust_optiling)
  message(FATAL_ERROR "Head project is missing the generated cust_optiling target")
endif()
get_target_property(head_tiling_sources cust_optiling SOURCES)
set(head_tiling_names)
foreach(head_source IN LISTS head_tiling_sources)
  get_filename_component(head_name "${head_source}" NAME)
  list(APPEND head_tiling_names "${head_name}")
endforeach()
foreach(head_op d_flash_draft_lm_head_top1 d_flash_draft_lm_head_top1_audit)
  if(NOT "${head_op}.cpp" IN_LIST head_tiling_names)
    message(FATAL_ERROR "Head CMake target does not include ${head_op}.cpp")
  endif()
  message(STATUS "Head CMake includes ${head_op}.cpp")
endforeach()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/d_flash_top1_probe.cpp" AND NOT "d_flash_top1_probe.cpp" IN_LIST head_tiling_names)
  message(FATAL_ERROR "Head CMake target does not include d_flash_top1_probe.cpp")
endif()
HEAD_CMAKE_CHECK
  diagnostic_args=()
  if [[ "$diagnostics" == 1 ]]; then diagnostic_args+=(--diagnostics); fi
  "$python_bin" "$here/test/build_config.py" --header "$project/op_host/head_config.h" \
    --output "$run_root/build-config.json" --core-limit "$core_limit" --conversion "$conversion" --top1 "$top1" --profile "$profile" "${diagnostic_args[@]}"
  cp "$project/op_host/head_config.h" "$project/op_kernel/head_config.h"
  phase=build
  (cd "$project"; bash build.sh)
  shopt -s nullglob
  packages=("$project"/build_out/custom_opp_*.run)
  if (( ${#packages[@]}!=1 )); then echo 'Expected one OPP package' >&2; exit 1; fi
  phase=install
  env -u ASCEND_CUSTOM_OPP_PATH bash "${packages[0]}" --install-path="$install"
  set +u
  source "$install/vendors/customize/bin/set_env.bash"
  set -u
  export LD_LIBRARY_PATH="$install/vendors/customize/op_api/lib:${LD_LIBRARY_PATH:-}"
  if [[ "$top1" == vector ]]; then diagnostic_args+=(--vector-top1); fi
  phase=installed_opp_preflight
  "$python_bin" "$here/test/head_opp_preflight.py" --install-root "$install" --report "$run_root/opp-preflight.json" "${diagnostic_args[@]}"
  phase=check_aclnn_headers
  for header in aclnn_d_flash_draft_lm_head_top1.h aclnn_d_flash_draft_lm_head_top1_audit.h; do
    if [[ ! -s "$install/vendors/customize/op_api/include/$header" ]]; then
      echo "Installed ACLNN header missing or empty: $header" >&2; exit 1
    fi
    echo "Installed ACLNN header verified: $header"
  done
  if [[ "$diagnostics" == 1 ]]; then
    for header in aclnn_d_flash_head_probe.h aclnn_d_flash_cast_probe.h; do
      [[ -s "$install/vendors/customize/op_api/include/$header" ]] || { echo "Missing $header" >&2; exit 1; }
    done
  fi
  if [[ "$top1" == vector ]]; then
    [[ -s "$install/vendors/customize/op_api/include/aclnn_d_flash_top1_probe.h" ]] || exit 1
  fi
  vector_build=OFF; profile_build=OFF
  [[ "$top1" != vector ]] || vector_build=ON
  [[ "$profile" == none ]] || profile_build=ON
  phase=build_custom_runner
  cmake -S "$here/test" -B "$build_root/custom" -DCANN_ROOT="$cann_root" -DASCEND_OPP_PATH="$install" -DHEAD_DIAGNOSTICS="$diagnostics" -DHEAD_VECTOR_TOP1="$vector_build" -DHEAD_PROFILE_ONLY="$profile_build"
  cmake --build "$build_root/custom" --parallel
  extra+=(--runner "$build_root/custom/head_custom_test" --build-config "$run_root/build-config.json")
fi
if [[ "$build_only" == 1 ]]; then
  "$python_bin" - "$HEAD_BUILD_STATE" "$run_root" "$build_root" "$install" <<'HEAD_STATE'
import json, os, sys
from pathlib import Path
out,run,build,install=map(Path,sys.argv[1:])
value=dict(status='BUILT_NOT_VALIDATED',run_root=str(run),build_root=str(build),install=str(install),
           config=str(run/'build-config.json'),runner=str(build/'custom/head_custom_test'),
           native_runner=str(build/'native/head_native_test'),top1_runner=str(build/'custom/head_top1_probe_test'),
           environment={k:os.environ[k] for k in ('ASCEND_OPP_PATH','ASCEND_CUSTOM_OPP_PATH','ASCEND_HOME_PATH','LD_LIBRARY_PATH','PATH','PYTHONPATH') if k in os.environ})
with out.open('x') as f: json.dump(value,f,indent=2)
HEAD_STATE
  echo "Built only; correctness NOT_RUN: $HEAD_BUILD_STATE"
  exit 0
fi
if [[ "$top1" == vector ]]; then
  phase=vector_top1_semantics
  "$python_bin" "$here/test/vector_top1_probe.py" --runner "$build_root/custom/head_top1_probe_test" \
    --output-dir "$run_root/top1-probe" --device-id "$device"
fi
if [[ "$diagnostics" == 1 ]]; then
  phase=conversion_diagnostics
  "$python_bin" "$here/test/conversion_probe.py" --runner "$build_root/custom/head_probe_test" \
    --output-dir "$run_root/diagnostics" --device-id "$device" --cann-root "$cann_root" \
    --build-config "$run_root/build-config.json" --timeout "${HEAD_TIMEOUT:-1800}"
  phase=sensitive_subset
  "$python_bin" "$here/test/run_suite.py" --capture "$capture" --native-manifest "$native" \
    --native-runner "$build_root/native/head_native_test" --output-dir "$run_root/sensitive" --device-id "$device" \
    --warmup "${HEAD_WARMUP:-5}" --repetitions "${HEAD_REPETITIONS:-30}" --timeout "${HEAD_TIMEOUT:-1800}" \
    --cases synthetic-overflow_inf,synthetic-nan_first,synthetic-signed_zero_underflow,synthetic-rounded_tie,synthetic-rounded_up "${extra[@]}"
fi
phase=full_vocabulary_suite
"$python_bin" "$here/test/run_suite.py" --capture "$capture" --native-manifest "$native" \
  --native-runner "$build_root/native/head_native_test" --output-dir "$run_root/data" --device-id "$device" \
  --warmup "${HEAD_WARMUP:-5}" --repetitions "${HEAD_REPETITIONS:-30}" --timeout "${HEAD_TIMEOUT:-1800}" "${extra[@]}"
echo "Summary: $run_root/data/suite.json; full Draft/Decode NOT_RUN"
