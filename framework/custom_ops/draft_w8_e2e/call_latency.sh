#!/usr/bin/env bash
set -eo pipefail
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    echo 'Usage: bash call_latency.sh --from-run context-prefill-RUN [--profile] [--resume draft-call-gap-RUN]'
    exit 0
fi
run_prefix=draft-call-gap
source "$(dirname "$0")/launch_env.sh"
exec "$DFLASH_PYTHON" -B -u "$here/diagnose_calls.py" "$@"
