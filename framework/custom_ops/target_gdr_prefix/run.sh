#!/usr/bin/env bash
set -eo pipefail
prefix_home=$(cd "$(dirname "$0")" && pwd -P)
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    echo 'Usage: bash run.sh --from-run target-conv-window-RUN [--resume target-gdr-prefix-RUN]'
    exit 0
fi
run_prefix=target-gdr-prefix
source "$prefix_home/../draft_w8_e2e/launch_env.sh"
exec "$DFLASH_PYTHON" -B -u "$prefix_home/run_prefix.py" "$@"
