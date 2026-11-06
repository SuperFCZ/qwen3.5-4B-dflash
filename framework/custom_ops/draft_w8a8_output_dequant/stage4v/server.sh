#!/usr/bin/env bash
# Explicit opt-in, no automatic retries, parameter changes or output overwrites.
set -euo pipefail
[[ $# -ge 1 && ! -e "$1" ]] || { echo 'usage: server.sh NEW_ROOT [consistent options]' >&2; exit 2; }
root=$1;shift
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
previous=
for item in prepare:STAGE4V_CPU_PREPARED_QUALITY_UNRESOLVED host:STAGE4V_FULL_K_HOST_FEASIBLE build:STAGE4V_BUILT_NPU_NOT_RUN bounded:STAGE4V_BOUNDED_MATCH full:STAGE4V_FULL_MATCH fault:STAGE4V_TILE_FAULT_COVERAGE_MATCH benchmark:STAGE4V_PAIRED_MEASURED profile:STAGE4V_PROFILE_COMPLETED; do
 phase=${item%%:*};status=${item#*:};args=("$phase" --output-dir "$root/$phase")
 [[ $phase == prepare ]] || args+=(--prepare-report "$root/prepare/report.json")
 [[ $phase == prepare || $phase == host ]] || args+=(--host-report "$root/host/report.json")
 [[ $phase == prepare || $phase == host || $phase == build ]] || args+=(--build-report "$root/build/report.json")
 if [[ $phase == full || $phase == fault || $phase == benchmark || $phase == profile ]]; then args+=(--prior-report "$root/$previous/report.json");fi
 bash "$here/run.sh" "${args[@]}" "$@"
 python3 - "$root/$phase/report.json" "$status" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
if r.get('status')!=sys.argv[2] or r.get('phase')!='complete':raise SystemExit('phase gate failed')
if r.get('quality_go')!='UNRESOLVED' or r.get('production')!='NOT_APPROVED':raise SystemExit('quality/production scope drift')
PY
 previous=$phase
done
