#!/usr/bin/env bash
# Run in the same CANN / isolated model environment as the source evidence.
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
exec "${MODEL_PYTHON:-python}" -B -u "$here/test/single_graph.py" "$@"
