# A dedicated prefix for new SDK-only Gate runs. This file is sourced inside the entry script.
dflash_gate_conda() {
    local env_dir="$DFLASH_RUN_ROOT/envs/gate310p" here manager recipe stamp
    here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
    recipe="$here/gate-requirements.txt"
    manager=${CONDA_EXE:-}
    if [[ ! -x "$manager" ]]; then manager=$(type -P conda || true); fi
    if [[ ! -x "$manager" && -x /root/anaconda3/bin/conda ]]; then manager=/root/anaconda3/bin/conda; fi

    # The caller's interactive environment stays untouched. Compilers also spawn 'python3' by name.
    unset PYTHONHOME PYTHONPATH PYTHONUSERBASE VIRTUAL_ENV LD_LIBRARY_PATH LD_PRELOAD
    unset CMAKE_PREFIX_PATH CPATH CPLUS_INCLUDE_PATH LIBRARY_PATH CC CXX CFLAGS CXXFLAGS LDFLAGS
    unset CONDA_PREFIX CONDA_PREFIX_1 CONDA_DEFAULT_ENV _CE_CONDA _CE_M PIP_USER PIP_PREFIX PIP_TARGET
    export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
    export CONDA_PKGS_DIRS="$DFLASH_RUN_ROOT/cache/conda" PIP_CACHE_DIR="$DFLASH_RUN_ROOT/cache/pip"
    mkdir -p "$DFLASH_RUN_ROOT/envs" "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR"
    (
        flock -x 9
        if [[ ! -f "$env_dir/conda-meta/history" ]]; then
            if [[ ! -x "$manager" ]]; then echo "ERROR: conda not found; set CONDA_EXE to the conda manager executable." >&2; exit 1; fi
            echo "Creating independent Conda environment: $env_dir"
            "$manager" create --yes --no-default-packages --prefix "$env_dir" 'python=3.11' pip || exit
        fi
        local py="$env_dir/bin/python"
        "$py" -I -c 'import sys;from pathlib import Path;assert sys.version_info[:2] == (3,11) and Path(sys.prefix).resolve() == Path(sys.argv[1]).resolve() == Path(sys.base_prefix).resolve(), "unexpected Conda prefix or Python version"' "$env_dir" || exit
        stamp=$(sha256sum "$recipe" | cut -d' ' -f1) || exit
        if [[ ! -f "$env_dir/.dflash-gate-requirements" || "$(cat "$env_dir/.dflash-gate-requirements")" != "$stamp" ]] ||
           ! "$py" -I -c 'import numpy,scipy,grpc_tools.protoc,google.protobuf' >/dev/null 2>&1; then
            local index=()
            if [[ -n ${DFLASH_WHEELHOUSE:-} ]]; then index=(--no-index --find-links "$DFLASH_WHEELHOUSE"); fi
            "$py" -I -m pip install --no-user --disable-pip-version-check --prefer-binary "${index[@]}" -r "$recipe" || exit
            "$py" -I -m pip check || exit
            printf '%s\n' "$stamp" > "$env_dir/.dflash-gate-requirements"
        fi
    ) 9>"$DFLASH_RUN_ROOT/envs/.gate310p.lock" || return
    export CONDA_PREFIX="$env_dir" CONDA_DEFAULT_ENV="$env_dir"
    export DFLASH_PYTHON="$env_dir/bin/python" PATH="$env_dir/bin:$PATH"
    dflash_load_cann || return
    export PATH="$env_dir/bin:$PATH"
    "$DFLASH_PYTHON" -B "$here/gate_env_check.py" "$env_dir" "$CANN_ROOT" "$DFLASH_OUTPUT" || return
}
