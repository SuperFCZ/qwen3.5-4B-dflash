#!/usr/bin/env bash
# Explicit opt-in orchestration only. No retry, repair or directory reuse.
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then echo 'usage: all_stages.sh NEW_RUN_ROOT [DEVICE]' >&2; exit 2; fi
root=$1;device=${2:-0}
[[ ! -e "$root" ]] || { echo 'Run root already exists; refusing reuse' >&2; exit 1; }
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
check(){ python3 - "$1" "$2" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
if r.get('status')!=sys.argv[2] or r.get('phase')!='complete':raise SystemExit('Exact phase gate failed')
PY
}
bash "$here/run.sh" build --device "$device" --output-dir "$root/build"
check "$root/build/build-report.json" STAGE4Q_RUNNERS_BUILT
bash "$here/run.sh" smoke --device "$device" --runner-build-report "$root/build/build-report.json" --output-dir "$root/smoke"
check "$root/smoke/smoke-report.json" STAGE4Q_SMOKE_MATCH
bash "$here/run.sh" benchmark --device "$device" --runner-build-report "$root/build/build-report.json" --smoke-report "$root/smoke/smoke-report.json" --output-dir "$root/benchmark"
check "$root/benchmark/benchmark-report.json" STAGE4Q_PAIRED_MEASURED
bash "$here/run.sh" profile --device "$device" --runner-build-report "$root/build/build-report.json" --benchmark-report "$root/benchmark/benchmark-report.json" --output-dir "$root/profile"
check "$root/profile/profile-report.json" STAGE4Q_PROFILE_COMPLETED
