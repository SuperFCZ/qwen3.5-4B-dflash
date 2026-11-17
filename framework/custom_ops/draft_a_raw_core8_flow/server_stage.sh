#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 2 ]]; then echo 'usage: server_stage.sh admit|build|probes|export|smoke|benchmark NEW_ROOT' >&2; exit 2; fi
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
STAGE=$1
ROOT=$2
ARGS=("$STAGE" --output-dir "$ROOT/$STAGE")
FOUND=0
for PREV in admit build probes export smoke benchmark; do
  if [[ "$PREV" == "$STAGE" ]]; then FOUND=1; break; fi
  ARGS+=("--$PREV-report" "$ROOT/$PREV/report.json")
done
if [[ "$FOUND" != 1 ]]; then echo 'unknown stage; no automatic all stage' >&2; exit 2; fi
exec "${MODEL_PYTHON:-python3}" -B "$HERE/run_core8.py" "${ARGS[@]}"
