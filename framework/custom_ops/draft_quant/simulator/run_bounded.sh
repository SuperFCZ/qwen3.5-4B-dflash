#!/usr/bin/env bash
set -eo pipefail
umask 077
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cann_root=${CANN_ROOT:-/usr/local/Ascend/cann-9.0.0}
if [[ -f "$cann_root/set_env.sh" ]]; then
    source "$cann_root/set_env.sh"
elif [[ -f "$cann_root/bin/setenv.bash" ]]; then
    source "$cann_root/bin/setenv.bash"
else
    echo "Missing CANN 9.0 environment: $cann_root" >&2; exit 1
fi
export CANN_ROOT="$cann_root"
exec "${MODEL_PYTHON:-python3}" "$here/bounded.py" "$@"
