#!/usr/bin/env bash
set -eo pipefail
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    echo 'Usage: bash prefill_head.sh --from-run workspace-e2e-RUN [--resume prefill-head-RUN]'
    exit 0
fi
run_prefix=prefill-head
source "$(dirname "$0")/launch_env.sh"
exec "$DFLASH_PYTHON" -B -u "$here/run_prefill_head.py" "$@"
