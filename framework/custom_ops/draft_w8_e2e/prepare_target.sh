#!/usr/bin/env bash
set -eo pipefail
run_prefix=e2e-target
source "$(dirname "$0")/launch_env.sh"
exec "$DFLASH_PYTHON" -B -u "$here/prepare_target.py" "$@"
