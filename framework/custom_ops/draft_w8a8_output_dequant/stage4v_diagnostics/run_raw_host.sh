#!/usr/bin/env bash
set -eo pipefail
export CANN_ROOT=${CANN_ROOT:-/usr/local/Ascend/cann-9.0.0}
source "$CANN_ROOT/set_env.sh"
set -u
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "${MODEL_PYTHON:-python3}" -B "$here/raw_host.py" --cann-root "$CANN_ROOT" "$@"
