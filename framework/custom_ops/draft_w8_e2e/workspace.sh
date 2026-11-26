#!/usr/bin/env bash
set -eo pipefail
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    echo 'Usage: bash workspace.sh --from-run context-prefill-RUN [--resume workspace-e2e-RUN]'
    exit 0
fi
run_prefix=workspace-e2e
source "$(dirname "$0")/launch_env.sh"
exec "$DFLASH_PYTHON" -B -u "$here/run_workspace.py" "$@"
