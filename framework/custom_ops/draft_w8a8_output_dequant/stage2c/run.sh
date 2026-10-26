#!/usr/bin/env bash
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export CANN_ROOT=${CANN_ROOT:-/usr/local/Ascend/cann-9.0.0}
export ASCEND_HOME_PATH="$CANN_ROOT" ASCEND_INSTALL_PATH="$CANN_ROOT"
if [[ -f "$CANN_ROOT/set_env.sh" ]]; then
  source "$CANN_ROOT/set_env.sh"
elif [[ -f "$CANN_ROOT/bin/setenv.bash" ]]; then
  source "$CANN_ROOT/bin/setenv.bash"
else
  echo 'Selected CANN environment missing' >&2
  exit 1
fi
set -u
exec "${MODEL_PYTHON:-python3}" -B "$here/run_stage2c.py" "$@"
