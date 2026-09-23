#!/usr/bin/env bash
set -eo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cann_root=${CANN_ROOT:-${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}}
soc_version=Ascend310P3
build_root="$here/.build"
op_project="$build_root/CustomOp"

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

mkdir -p "$build_root"
rm -rf -- "$op_project"
msopgen gen -i "$here/AddCustom.json" -c "ai_core-$soc_version" -lan cpp -out "$op_project"
cp "$here/op_host/add_custom.cpp" "$here/op_host/add_custom_tiling.h" "$op_project/op_host/"
cp "$here/op_kernel/add_custom.cpp" "$op_project/op_kernel/"
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
bash "${packages[0]}"

opp_path=${ASCEND_OPP_PATH:-"$cann_root/opp"}
vendor_api="$opp_path/vendors/customize/op_api"
if [[ ! -f "$vendor_api/include/aclnn_add_custom.h" ]]; then
    echo "Installed ACLNN header missing: $vendor_api/include/aclnn_add_custom.h" >&2
    exit 1
fi
export LD_LIBRARY_PATH="$vendor_api/lib:${LD_LIBRARY_PATH:-}"
cmake -S "$here/test" -B "$build_root/test" \
    -DCANN_ROOT="$cann_root" -DASCEND_OPP_PATH="$opp_path"
cmake --build "$build_root/test" --parallel
"$build_root/test/smoke_add_test" "${DEVICE_ID:-0}"
