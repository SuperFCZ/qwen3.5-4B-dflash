#!/usr/bin/env bash
# Explicitly requested stage only. No automatic repair, overwrite or downstream execution.
set -euo pipefail
[[ $# == 3 ]] || { echo 'usage: server_stage.sh PHASE NEW_EXPERIMENT_ROOT ACCEPTED_STAGE4W_ROOT' >&2; exit 2; }
phase=$1;root=$2;old=$3
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
args=(--output-dir "$root/$phase" --stage4w-root "$old" --cann-root "${CANN_ROOT:-/usr/local/Ascend/cann-9.0.0}" --device "${DEVICE_ID:-0}")
found=false
for p in source prepare hostplan build host tile0 smoke fault benchmark mlp; do
  if [[ "$p" == "$phase" ]]; then found=true;break;fi
  if [[ "$p" == benchmark && "$phase" == mlp ]]; then continue;fi
  [[ -f "$root/$p/report.json" ]] || { echo "missing predecessor: $p" >&2;exit 2; }
  args+=(--"$p"-report "$root/$p/report.json")
done
[[ "$found" == true ]] || { echo 'unknown phase' >&2;exit 2; }
exec python3 -B "$here/run_down.py" "$phase" "${args[@]}"
