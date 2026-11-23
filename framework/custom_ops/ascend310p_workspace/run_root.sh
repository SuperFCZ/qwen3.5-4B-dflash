# Shared by current 310P benchmark and recovery entry points; source in a child shell.
dflash_new_run() {
    DFLASH_RUN_ROOT=${DFLASH_RUN_ROOT:-/home/w00949577/z50058744/qwen35-dflash-run}
    case "$DFLASH_RUN_ROOT" in /*) ;; *) echo "ERROR: DFLASH_RUN_ROOT must be absolute" >&2; return 2;; esac
    mkdir -p "$DFLASH_RUN_ROOT/runs" || return
    DFLASH_RUN_ROOT=$(cd "$DFLASH_RUN_ROOT" && pwd -P) || return
    DFLASH_OUTPUT=$(mktemp -d "$DFLASH_RUN_ROOT/runs/$1-XXXXXX") || return
    export DFLASH_RUN_ROOT DFLASH_OUTPUT
    mkdir -p "$DFLASH_OUTPUT"/{tmp,cann-log,cache,work} || return
    export TMPDIR="$DFLASH_OUTPUT/tmp" ASCEND_PROCESS_LOG_PATH="$DFLASH_OUTPUT/cann-log"
    export ASCEND_CACHE_PATH="$DFLASH_OUTPUT/cache" ASCEND_WORK_PATH="$DFLASH_OUTPUT/work"
    export AI_RUN_DIR="$DFLASH_OUTPUT/work"
    exec > >(tee -a "$DFLASH_OUTPUT/terminal.log") 2>&1
    echo "RUN $DFLASH_OUTPUT"
}

dflash_load_cann() {
    CANN_ROOT=${CANN_ROOT:-/usr/local/Ascend/cann-9.0.0}
    export CANN_ROOT
    if [[ ! -r "$CANN_ROOT/set_env.sh" ]]; then
        echo "ERROR: missing CANN initialization script: $CANN_ROOT/set_env.sh" >&2
        return 1
    fi
    source "$CANN_ROOT/set_env.sh"
}
