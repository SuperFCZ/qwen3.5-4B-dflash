#!/usr/bin/env bash
# Source the receiver's usual CANN + ADN/GDR environment first. No capture/export of golden.
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${MODEL_PYTHON:-python}"
: "${AC_OUTPUT_DIR:?Set a NEW AC_OUTPUT_DIR; old evidence/OPP must not be overwritten}"
if [[ -e "$AC_OUTPUT_DIR" ]]; then echo "Output already exists: $AC_OUTPUT_DIR" >&2; exit 1; fi
# Check before mkdir/tee, so even a mistaken output path cannot write Native Base.
"$python_bin" -B - "$AC_OUTPUT_DIR" "$@" <<'PY_CHECK'
import argparse
from pathlib import Path
import sys
p=argparse.ArgumentParser(add_help=False)
p.add_argument('--native-base',type=Path)
a,_=p.parse_known_args(sys.argv[2:])
if a.native_base and Path(sys.argv[1]).resolve().is_relative_to(a.native_base.resolve()):
    raise SystemExit('evidence output must be outside read-only Native Base')
PY_CHECK
mkdir -p -- "$(dirname -- "$AC_OUTPUT_DIR")"
export AI_RUN_DIR="$AC_OUTPUT_DIR"
export PYTHONDONTWRITEBYTECODE=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
# The Python entry creates the root atomically. Log stays beside it even on preflight failure.
"$python_bin" -B -u "$here/test/run.py" --output-dir "$AC_OUTPUT_DIR" "$@" 2>&1 | tee "$AC_OUTPUT_DIR.server.log"
