#!/usr/bin/env bash
set -eo pipefail
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    echo 'Usage: bash probe_dynamic.sh --from-run RUN --provider DYNAMIC_VENDOR [--inspect-only]'
    exit 0
fi
run_prefix=target-opp-check
source "$(dirname "$0")/launch_env.sh"
exec "$DFLASH_PYTHON" -B -u "$here/probe_dynamic.py" "$@"
