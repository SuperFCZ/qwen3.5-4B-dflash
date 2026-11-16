#!/usr/bin/env bash
# One explicitly selected stage per invocation. There is deliberately no "all".
set -euo pipefail
if [[ $# != 2 ]]; then
  echo "usage: bash server_stage.sh precheck|build|probes|export|smoke|benchmark NEW_EXPERIMENT_ROOT" >&2
  exit 2
fi
STAGE=$1
ROOT=$2
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ARGS=("$STAGE" --output-dir "$ROOT/$STAGE")
FOUND=0
for PREV in precheck build probes export smoke benchmark; do
  if [[ "$PREV" == "$STAGE" ]]; then FOUND=1; break; fi
  ARGS+=("--$PREV-report" "$ROOT/$PREV/report.json")
done
if [[ "$FOUND" != 1 ]]; then echo "unknown stage: $STAGE" >&2; exit 2; fi
# Existing output directories and failed/mixed predecessor reports are refused.
"${MODEL_PYTHON:-python3}" -B "$SCRIPT_DIR/run_experiment.py" "${ARGS[@]}"
"${MODEL_PYTHON:-python3}" -B "$SCRIPT_DIR/show_report.py" "$ROOT/$STAGE/report.json"
