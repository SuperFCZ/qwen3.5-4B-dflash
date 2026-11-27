#!/usr/bin/env bash
set -eo pipefail
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    echo 'Usage: bash conv_window.sh --from-run prefill-head-RUN --from-profile target-task-sample-RUN [--resume target-conv-window-RUN]'
    exit 0
fi
run_prefix=target-conv-window
source "$(dirname "$0")/launch_env.sh"
exec "$DFLASH_PYTHON" -B -u "$here/run_conv_window.py" "$@"
