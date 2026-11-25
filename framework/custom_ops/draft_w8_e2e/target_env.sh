#!/usr/bin/env bash
# Called in a subprocess only when the two historical Target OMs must be rebuilt.
set -eo pipefail
root=${DFLASH_RUN_ROOT:?}
base=${DFLASH_TARGET_PYTHON:-/root/anaconda3/envs/torch_npu_py3.11/bin/python}
env_dir="$root/envs/target-export310p"
[[ -x "$base" ]] || { echo "Target export Python is missing: $base" >&2; exit 1; }
mkdir -p "$root/envs" "$root/cache/pip"
export PIP_CACHE_DIR="$root/cache/pip" PYTHONNOUSERSITE=1 PIP_DISABLE_PIP_VERSION_CHECK=1
(
    flock -x 9
    if [[ ! -x "$env_dir/bin/python" ]]; then "$base" -m venv --system-site-packages "$env_dir"; fi
    py="$env_dir/bin/python"
    "$py" -I -c 'import sys;from pathlib import Path;assert Path(sys.base_prefix).resolve() == Path(sys.argv[1]).parent.parent.resolve()' "$base"
    if ! "$py" -I -c 'import transformers;from transformers.initialization import copy_;assert transformers.__version__=="5.13.1"' >/dev/null 2>&1; then
        index=()
        if [[ -n ${DFLASH_WHEELHOUSE:-} ]]; then index=(--no-index --find-links "$DFLASH_WHEELHOUSE"); fi
        "$py" -I -m pip install --no-user --prefer-binary "${index[@]}" 'transformers==5.13.1'
    fi
) 9>"$root/envs/.target-export310p.lock"
export PATH="$env_dir/bin:$PATH"
exec "$env_dir/bin/python" -B -u "$@"
