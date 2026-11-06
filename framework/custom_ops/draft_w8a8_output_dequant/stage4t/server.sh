#!/usr/bin/env bash
# Explicit user opt-in. Stop on any error or nonmatching status. Never overwrite.
set -euo pipefail
[[ $# -ge 1 ]] || { echo 'usage: server.sh NEW_ROOT [consistent run.sh options]' >&2; exit 2; }
root=$1;shift
[[ ! -e "$root" ]] || { echo 'output root already exists' >&2; exit 2; }
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
previous=
for item in build:STAGE4T_BUILT_NPU_NOT_RUN host-only:STAGE4T_HOST_ONLY bounded:STAGE4T_DIAGNOSTIC_BOUNDED_MATCH full:STAGE4T_DIAGNOSTIC_FULL_MATCH benchmark:STAGE4T_PAIRED_MEASURED profile:STAGE4T_PROFILE_COMPLETED; do
 phase=${item%%:*};status=${item#*:}
 args=("$phase" --output-dir "$root/$phase")
 [[ $phase == build ]] || args+=(--build-report "$root/build/report.json")
 [[ $phase == build || $phase == host-only ]] || args+=(--prior-report "$root/$previous/report.json")
 bash "$here/run.sh" "${args[@]}" "$@"
 python3 - "$root/$phase/report.json" "$status" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
if r.get('status')!=sys.argv[2] or r.get('phase')!='complete':raise SystemExit('phase gate failed')
if r.get('production_adoption')!='PROHIBITED':raise SystemExit('diagnostic scope missing')
PY
 previous=$phase
done
