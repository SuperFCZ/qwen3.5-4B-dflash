#!/usr/bin/env bash
set -eo pipefail
prefix_home=$(cd "$(dirname "$0")" && pwd -P)
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    echo 'Usage: bash state_only.sh --from-probe target-gdr-prefix-RUN [--resume target-gdr-state-RUN]'
    exit 0
fi
run_prefix=target-gdr-state
source "$prefix_home/../draft_w8_e2e/launch_env.sh"
exec "$DFLASH_PYTHON" -B -u "$prefix_home/state_only.py" "$@"
