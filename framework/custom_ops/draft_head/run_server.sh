#!/usr/bin/env bash
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python_bin=${MODEL_PYTHON:-python3}
cann_root=${CANN_ROOT:-${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}}
device=${DEVICE_ID:-0}
core_limit=${DFLASH_HEAD_CORE_LIMIT:-0}
native_only=${HEAD_NATIVE_ONLY:-0}
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
  cp "$here"/op_kernel/*.cpp "$here/op_kernel/head_kernel.h" "$project/op_kernel/"
  cp "$here/op_host/head_contract.h" "$project/op_kernel/head_contract.h"
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
HEAD_CMAKE_CHECK
  "$python_bin" "$here/test/build_config.py" --header "$project/op_host/head_config.h" \
    --output "$run_root/build-config.json" --core-limit "$core_limit"
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
  phase=installed_opp_preflight
  "$python_bin" "$here/test/head_opp_preflight.py" --install-root "$install" --report "$run_root/opp-preflight.json"
  phase=check_aclnn_headers
  for header in aclnn_d_flash_draft_lm_head_top1.h aclnn_d_flash_draft_lm_head_top1_audit.h; do
    if [[ ! -s "$install/vendors/customize/op_api/include/$header" ]]; then
      echo "Installed ACLNN header missing or empty: $header" >&2; exit 1
    fi
    echo "Installed ACLNN header verified: $header"
  done
  phase=build_custom_runner
  cmake -S "$here/test" -B "$build_root/custom" -DCANN_ROOT="$cann_root" -DASCEND_OPP_PATH="$install"
  cmake --build "$build_root/custom" --parallel
  extra+=(--runner "$build_root/custom/head_custom_test" --build-config "$run_root/build-config.json")
fi
phase=full_vocabulary_suite
"$python_bin" "$here/test/run_suite.py" --capture "$capture" --native-manifest "$native" \
  --native-runner "$build_root/native/head_native_test" --output-dir "$run_root/data" --device-id "$device" \
  --warmup "${HEAD_WARMUP:-5}" --repetitions "${HEAD_REPETITIONS:-30}" --timeout "${HEAD_TIMEOUT:-1800}" "${extra[@]}"
echo "Summary: $run_root/data/suite.json; full Draft/Decode NOT_RUN"
