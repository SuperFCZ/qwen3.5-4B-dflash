#!/usr/bin/env bash
# Separate processes/OPPs for the two builds; the production environment is not edited.
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python_bin=${MODEL_PYTHON:-python3}
cann_root=${CANN_ROOT:-${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}}
if [[ -f "$cann_root/set_env.sh" ]]; then
    source "$cann_root/set_env.sh"
elif [[ -f "$cann_root/bin/setenv.bash" ]]; then
    source "$cann_root/bin/setenv.bash"
else
    echo "CANN environment missing under $cann_root" >&2
    exit 1
fi
export CANN_ROOT="$cann_root"
exec "$python_bin" "$here/test/run_a5.py" "$@"
