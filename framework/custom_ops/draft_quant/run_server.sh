#!/usr/bin/env bash
set -eo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cann_root=${CANN_ROOT:-${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}}
python_bin=${MODEL_PYTHON:-python3}
device_id=${DEVICE_ID:-0}

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
build_root=$(mktemp -d "$here/.build/tiny.XXXXXXXX")
run_root=$(mktemp -d "$here/.runs/tiny.XXXXXXXX")
op_project="$build_root/CustomOp"
install_dir="$build_root/opp"
data_dir="$run_root/data"
exec > >(tee "$run_root/server.log") 2>&1
phase=prepare
trap 'status=$?; if (( status != 0 )); then echo "FAIL during $phase (exit $status); evidence: $run_root" >&2; fi' EXIT
echo "Target: Ascend310P3 / CANN 9.0.0; using $cann_root"
echo "Build: $build_root"
echo "Evidence: $run_root"
git -C "$here" rev-parse HEAD > "$run_root/source-commit.txt"
"$python_bin" "$here/test/reference.py" prepare "$data_dir"

phase=msopgen
msopgen gen -i "$here/DFlashGroupQuantLinear.json" -c ai_core-Ascend310P3 -lan cpp -out "$op_project"
if [[ ! -f "$op_project/op_kernel/d_flash_group_quant_linear.cpp" ]]; then
    echo "Unexpected msopgen kernel naming; inspect $op_project before building" >&2
    exit 1
fi
cp "$here/op_host/d_flash_group_quant_linear.cpp" "$here/op_host/d_flash_group_quant_linear_tiling.h" "$op_project/op_host/"
cp "$here/op_kernel/d_flash_group_quant_linear.cpp" "$op_project/op_kernel/"
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
export LD_LIBRARY_PATH="$vendor_api/lib:${LD_LIBRARY_PATH:-}"
phase=build_runner
cmake -S "$here/test" -B "$build_root/test" -DCANN_ROOT="$cann_root" -DASCEND_OPP_PATH="$install_dir"
cmake --build "$build_root/test" --parallel
phase=custom_aclnn
while IFS= read -r case_name; do
    echo "Custom tiny: $case_name"
    "$build_root/test/dflash_group_quant_linear_test" "$device_id" "$data_dir/$case_name"
done < "$data_dir/cases.txt"
phase=native_eager
"$python_bin" "$here/test/reference.py" native "$data_dir" --device-id "$device_id"
phase=compare
"$python_bin" "$here/test/reference.py" check "$data_dir"
echo "Comparison: $data_dir/comparison.json"
echo "Native OM / full Draft / performance remain NOT_RUN."
