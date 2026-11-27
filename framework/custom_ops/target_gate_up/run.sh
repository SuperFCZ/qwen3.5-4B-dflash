#!/usr/bin/env bash
set -eo pipefail
pair_home=$(cd "$(dirname "$0")" && pwd -P)
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    echo 'Usage: bash run.sh --from-run target-conv-window-RUN [--references-only] [--resume target-gate-up-RUN]'
    exit 0
fi
run_prefix=target-gate-up
source "$pair_home/../draft_w8_e2e/launch_env.sh"
exec "$DFLASH_PYTHON" -B -u "$pair_home/run_pair.py" "$@"
