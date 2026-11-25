#!/usr/bin/env bash
set -eo pipefail
run_prefix=w8-e2e
source "$(dirname "$0")/launch_env.sh"
if [[ ${DFLASH_INSPECT_RECEIVER:-0} == 1 ]]; then
    "$DFLASH_PYTHON" -B -u "$here/inspect_receiver.py" | tee "$DFLASH_OUTPUT/receiver-inventory.txt"
fi
exec "$DFLASH_PYTHON" -B -u "$here/run_e2e.py" "$@"
