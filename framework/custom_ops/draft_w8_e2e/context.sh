#!/usr/bin/env bash
set -eo pipefail
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    echo 'Usage: bash context.sh --from-run w8-e2e-RUN [--profile] [--resume context-prefill-RUN]'
    exit 0
fi
run_prefix=context-prefill
source "$(dirname "$0")/launch_env.sh"
exec "$DFLASH_PYTHON" -B -u "$here/run_context.py" "$@"
