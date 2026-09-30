#!/usr/bin/env bash
set -eo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cann_root=${CANN_ROOT:-${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}}
python_bin=${MODEL_PYTHON:-python3}
device_id=${DEVICE_ID:-0}
suite=${DFLASH_SUITE:-a1}
core_limit=${DFLASH_CORE_LIMIT:-0}
dequant_mode=${DFLASH_DEQUANT_MODE:-batched}
pipeline_mode=${DFLASH_PIPELINE_MODE:-serial}
if [[ "$pipeline_mode" != serial && "$pipeline_mode" != prefetch ]] || \
   [[ "$pipeline_mode" == prefetch && "$dequant_mode" != batched ]]; then
    echo "DFLASH_PIPELINE_MODE must be serial or prefetch; prefetch requires batched dequantization" >&2
    exit 1
fi
if [[ "$dequant_mode" != legacy && "$dequant_mode" != batched ]]; then
    echo "DFLASH_DEQUANT_MODE must be legacy or batched" >&2
    exit 1
fi
if [[ ! "$core_limit" =~ ^(0|[1-9][0-9]{0,4})$ ]] || (( core_limit > 65535 )); then
    echo "DFLASH_CORE_LIMIT must be 0..65535 (0=auto, 1=single-core control)" >&2
    exit 1
fi
if [[ "$suite" != a1 && "$suite" != tiny && "$suite" != a2 && "$suite" != a3 && "$suite" != a31 && "$suite" != a32 && "$suite" != a4 ]]; then
    echo "DFLASH_SUITE must be a1, tiny, a2, a3, a31, a32 or a4" >&2
    exit 1
fi
if [[ ( "$suite" == a32 || "$suite" == a4 ) && "$dequant_mode" != batched ]]; then
    echo "$suite requires DFLASH_DEQUANT_MODE=batched" >&2
    exit 1
fi
if [[ "$suite" == a2 || "$suite" == a3 || "$suite" == a31 || "$suite" == a32 ]]; then
    if [[ ! -f "${A2_BUNDLE:-}" || ! -f "${A2_NATIVE_OM_MANIFEST:-}" ]]; then
        echo "$suite needs A2_BUNDLE=.../manifest.json and A2_NATIVE_OM_MANIFEST=.../native-om.json; see A2.md/A3.md" >&2
        exit 1
    fi
fi

if [[ "$suite" == a4 ]]; then
    for name in A4_C16_BUNDLE A4_C64_BUNDLE A4_C16_NATIVE_OM_MANIFEST A4_C64_NATIVE_OM_MANIFEST; do
        if [[ ! -f "${!name:-}" ]]; then
            echo "a4 requires $name pointing to the corresponding C16/C64 capture or native OM manifest; see A4.md" >&2
            exit 1
        fi
    done
fi

if [[ -f "$cann_root/set_env.sh" ]]; then
    cann_env="$cann_root/set_env.sh"
elif [[ -f "$cann_root/bin/setenv.bash" ]]; then
    cann_env="$cann_root/bin/setenv.bash"
else
    echo "CANN environment script not found under $cann_root" >&2
    exit 1
fi
export ASCEND_HOME_PATH="$cann_root"
export ASCEND_INSTALL_PATH="$cann_root"
source "$cann_env"
set -u
if ! command -v msopgen >/dev/null 2>&1; then
    echo "msopgen is unavailable after loading $cann_env" >&2
    exit 1
fi

# Each invocation owns a fresh build, isolated OPP and evidence directory.
# Prior packages, results and the model OPP installation are never overwritten.
mkdir -p "$here/.build" "$here/.runs"
# CANN 9.0 derives JSON from the FIRST '.o' in an object path. A directory
# such as a1.ovznlDEL truncates it to a1.json. Keep random suffixes after '-'.
build_root=$(mktemp -d "$here/.build/${suite}-XXXXXXXX")
run_root=$(mktemp -d "$here/.runs/${suite}-XXXXXXXX")
op_project="$build_root/CustomOp"
install_dir="$build_root/opp"
data_dir="$run_root/data"
exec > >(tee "$run_root/server.log") 2>&1
phase=prepare
trap 'status=$?; if (( status != 0 )); then echo "FAIL during $phase (exit $status); evidence: $run_root" >&2; fi' EXIT
echo "Target: Ascend310P3 / CANN 9.0.0; using $cann_root"
echo "Build: $build_root"
echo "Evidence: $run_root"
echo "A3 N-tile scheduling; build core cap: $core_limit (0=auto)"
echo "Dequantization build mode: $dequant_mode"
echo "Pipeline build mode: $pipeline_mode (prefetch targets down + streamed A1 only)"
git -C "$here" rev-parse HEAD > "$run_root/source-commit.txt"
phase=path_preflight
"$python_bin" "$here/test/opp_preflight.py" --root "$build_root" --report "$run_root/path-preflight.json"

phase=msopgen
msopgen gen -i "$here/DFlashGroupQuantLinear.json" -c ai_core-Ascend310P3 -lan cpp -out "$op_project"
if [[ ! -f "$op_project/op_kernel/d_flash_group_quant_linear.cpp" ]]; then
    echo "Unexpected msopgen kernel naming; inspect $op_project before building" >&2
    exit 1
fi
cp "$here/op_host/d_flash_group_quant_linear.cpp" \
   "$here/op_host/d_flash_group_quant_linear_tiling.h" \
   "$here/op_host/d_flash_group_quant_linear_contract.h" "$op_project/op_host/"
cp "$here/op_kernel/d_flash_group_quant_linear.cpp" "$op_project/op_kernel/"
phase=configure_core_limit
"$python_bin" "$here/test/a3_launch.py" --core-limit "$core_limit" --dequant-mode "$dequant_mode" \
    --pipeline-mode "$pipeline_mode" \
    --header "$op_project/op_host/d_flash_group_quant_linear_build_config.h" \
    --kernel-header "$op_project/op_kernel/d_flash_group_quant_linear_build_config.h" \
    --report "$run_root/build-config.json"
phase=build
(
    cd "$op_project"
    bash build.sh
)

shopt -s nullglob
packages=("$op_project"/build_out/custom_opp_*.run)
if (( ${#packages[@]} != 1 )); then
    echo "Expected one custom_opp_*.run in $op_project/build_out; found ${#packages[@]}" >&2
    exit 1
fi
phase=install
env -u ASCEND_CUSTOM_OPP_PATH bash "${packages[0]}" --install-path="$install_dir"
vendor_env="$install_dir/vendors/customize/bin/set_env.bash"
if [[ ! -f "$vendor_env" ]]; then
    echo "Installed custom OPP environment missing: $vendor_env" >&2
    exit 1
fi
set +u
source "$vendor_env"
set -u
vendor_api="$install_dir/vendors/customize/op_api"
if [[ ! -f "$vendor_api/include/aclnn_d_flash_group_quant_linear.h" ]]; then
    echo "Generated ACLNN header missing under $vendor_api/include" >&2
    exit 1
fi
phase=installed_opp_preflight
"$python_bin" "$here/test/opp_preflight.py" --install-root "$install_dir" --report "$run_root/opp-preflight.json"
export LD_LIBRARY_PATH="$vendor_api/lib:${LD_LIBRARY_PATH:-}"
phase=build_runner
cmake -S "$here/test" -B "$build_root/test" -DCANN_ROOT="$cann_root" -DASCEND_OPP_PATH="$install_dir"
cmake --build "$build_root/test" --parallel
phase=correctness_suite
if [[ "$suite" == a4 ]]; then
    "$python_bin" "$here/test/run_a4.py" --output-dir "$data_dir" \
        --build-config "$run_root/build-config.json" \
        --runner "$build_root/test/dflash_group_quant_linear_test" \
        --om-runner "$build_root/test/dflash_native_om_test" --device-id "$device_id" \
        --c16-bundle "$A4_C16_BUNDLE" --c64-bundle "$A4_C64_BUNDLE" \
        --c16-native-om-manifest "$A4_C16_NATIVE_OM_MANIFEST" \
        --c64-native-om-manifest "$A4_C64_NATIVE_OM_MANIFEST" \
        --warmup "${A2_WARMUP:-5}" --repetitions "${A2_REPETITIONS:-30}"
elif [[ "$suite" == a3 || "$suite" == a31 || "$suite" == a32 ]]; then
    timing_protocol=checked-v1
    stage=A3
    default_warmup=3
    default_repetitions=10
    if [[ "$suite" == a31 ]]; then timing_protocol=continuous-v1; stage=A3.1; fi
    if [[ "$suite" == a32 ]]; then
        timing_protocol=continuous-v1
        stage=A3.2
        default_warmup=5
        default_repetitions=30
    fi
    "$python_bin" "$here/test/run_a3.py" --output-dir "$data_dir" \
        --build-config "$run_root/build-config.json" \
        --timing-protocol "$timing_protocol" --stage "$stage" \
        --runner "$build_root/test/dflash_group_quant_linear_test" \
        --om-runner "$build_root/test/dflash_native_om_test" --device-id "$device_id" \
        --bundle "$A2_BUNDLE" --native-om-manifest "$A2_NATIVE_OM_MANIFEST" \
        --warmup "${A2_WARMUP:-$default_warmup}" --repetitions "${A2_REPETITIONS:-$default_repetitions}"
elif [[ "$suite" == a2 ]]; then
    "$python_bin" "$here/test/run_a2.py" --output-dir "$data_dir" \
        --runner "$build_root/test/dflash_group_quant_linear_test" \
        --om-runner "$build_root/test/dflash_native_om_test" --device-id "$device_id" \
        --bundle "$A2_BUNDLE" --native-om-manifest "$A2_NATIVE_OM_MANIFEST" \
        --warmup "${A2_WARMUP:-3}" --repetitions "${A2_REPETITIONS:-10}"
else
    "$python_bin" "$here/test/run_suite.py" --output-dir "$data_dir" \
        --runner "$build_root/test/dflash_group_quant_linear_test" --device-id "$device_id" --suite "$suite"
fi
echo "Summary: $data_dir/suite.json"
if [[ "$suite" == a2 || "$suite" == a3 || "$suite" == a31 || "$suite" == a32 || "$suite" == a4 ]]; then
    echo "Isolated native OM parity/timing recorded; full Draft and decode performance remain NOT_RUN."
else
    echo "Native OM / full Draft / performance remain NOT_RUN."
fi
