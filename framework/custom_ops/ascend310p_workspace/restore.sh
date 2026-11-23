#!/usr/bin/env bash
set -eo pipefail
here=$(cd "$(dirname "$0")" && pwd -P)
case "${1:-}" in
    --help|-h) echo "Usage: bash restore.sh [--inventory-only] [additional backup directories ...]"; exit 0;;
esac
mode=restore
if [[ ${1:-} == --inventory-only ]]; then mode=inventory; shift; fi
for directory in "$@"; do
    if [[ ! -d "$directory" ]]; then echo "ERROR: search directory does not exist: $directory" >&2; exit 2; fi
done
source "$here/run_root.sh"
dflash_new_run restore
base_py=${DFLASH_BASE_PYTHON:-/root/anaconda3/envs/ls_llm/bin/python3}
scanner=$base_py
if [[ ! -x "$scanner" ]]; then scanner=$(command -v python3); fi
"$scanner" -B "$here/recover_inputs.py" --run-root "$DFLASH_RUN_ROOT" --report "$DFLASH_OUTPUT/inventory.json" "$@"
if [[ "$mode" == inventory ]]; then exit 0; fi
if [[ ! -x "$base_py" ]]; then echo "ERROR: missing base interpreter: $base_py. Set DFLASH_BASE_PYTHON to ls_llm Python 3.11." >&2; exit 1; fi
dflash_load_cann
"$base_py" -B "$here/restore_python.py" "$DFLASH_RUN_ROOT" "$DFLASH_OUTPUT"
