#!/usr/bin/env bash
# Called by a user's copy of config/dflash_env.sh.example.
if [ -z "${BASH_VERSION:-}" ]; then
    echo "dflash-env: Bash source is required" >&2
    return 2 2>/dev/null || exit 2
fi
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "dflash-env: use source /path/dflash-env.sh" >&2
    exit 2
fi

_dflash_activate_env() {
    local dflash_name dflash_value dflash_run_real dflash_root_real
    for dflash_name in REPO_ROOT AI_RUN_DIR MODEL_PYTHON CANN_ROOT; do
        dflash_value="${!dflash_name:-}"
        if [[ "$dflash_value" != /* || "$dflash_value" == /absolute/path/* ]]; then
            echo "dflash-env: 请在配置文件中填写 $dflash_name 的绝对路径" >&2
            return 1
        fi
    done
    if [[ ! -f "$REPO_ROOT/models/dflash_v1/run_npu.py" ]]; then
        echo "dflash-env: REPO_ROOT 不是 DFlash 源码目录" >&2
        return 1
    fi
    if [[ ! -x "$MODEL_PYTHON" || ! -r "$CANN_ROOT/set_env.sh" ]]; then
        echo "dflash-env: 检查 MODEL_PYTHON 和 CANN_ROOT/set_env.sh" >&2
        return 1
    fi
    case "${VERIFY_GDR:-chunk}" in chunk|mtp) ;; *)
        echo "dflash-env: VERIFY_GDR 只能是 chunk 或 mtp" >&2; return 1 ;;
    esac
    case "${QUANT_MODE:-disable}" in enable|disable) ;; *)
        echo "dflash-env: QUANT_MODE 只能是 enable 或 disable" >&2; return 1 ;;
    esac
    case "${DRAFT_QUANTIZATION:-fp16}" in fp16|w4a16|w8a16) ;; *)
        echo "dflash-env: DRAFT_QUANTIZATION 只能是 fp16、w4a16 或 w8a16" >&2; return 1 ;;
    esac
    if [[ ! "${MAX_DRAFT_TOKENS:-15}" =~ ^([1-9]|1[0-5])$ ]]; then
        echo "dflash-env: MAX_DRAFT_TOKENS 必须为 1..15" >&2
        return 1
    fi
    dflash_run_real="$(realpath -m -- "$AI_RUN_DIR")" || return 1
    for dflash_name in REPO_ROOT TARGET_DIR DRAFT_DIR DRAFT_FP16_DIR DRAFT_W4A16_DIR DRAFT_W8A16_DIR RECEIVER_ROOT CANN_ROOT; do
        dflash_value="${!dflash_name:-}"
        [[ -n "$dflash_value" ]] || continue
        dflash_root_real="$(realpath -m -- "$dflash_value")" || return 1
        if [[ "$dflash_run_real" == "$dflash_root_real" || "$dflash_run_real" == "$dflash_root_real/"* ]]; then
            echo "dflash-env: AI_RUN_DIR 必须放在源码、模型和工具链目录之外" >&2
            return 1
        fi
    done

    # Keep CANN loading once per shell/root; do not activate another Python env.
    local dflash_cann_key="$BASHPID:$CANN_ROOT"
    if [[ "${_DFLASH_CANN_LOADED:-}" != "$dflash_cann_key" ]]; then
        source "$CANN_ROOT/set_env.sh" || return 1
        _DFLASH_CANN_LOADED="$dflash_cann_key"
    fi
    mkdir -p "$AI_RUN_DIR"/{reports,cache,tmp} || return 1
    export PYTHONDONTWRITEBYTECODE=1
    export TMPDIR="$AI_RUN_DIR/tmp"
    export HF_HOME="$AI_RUN_DIR/cache/huggingface"
    export TORCH_HOME="$AI_RUN_DIR/cache/torch"
    export XDG_CACHE_HOME="$AI_RUN_DIR/cache"
    export VERIFY_GDR="${VERIFY_GDR:-chunk}" DEVICE_ID="${DEVICE_ID:-0}"
    export KV_CAPACITY="${KV_CAPACITY:-2048}" MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
    export MAX_DRAFT_TOKENS="${MAX_DRAFT_TOKENS:-15}" QUANT_MODE="${QUANT_MODE:-disable}"
    export DRAFT_QUANTIZATION="${DRAFT_QUANTIZATION:-fp16}"
    export DRAFT_FP16_DIR="${DRAFT_FP16_DIR:-${DRAFT_DIR:-}}"
    dflash_name="DRAFT_${DRAFT_QUANTIZATION^^}_DIR"
    export DRAFT_SELECTED_DIR="${!dflash_name:-}"
    export DRAFT_VARIANTS_DIR="${DRAFT_VARIANTS_DIR:-$AI_RUN_DIR/artifacts-drafts}"
    export DRAFT_VARIANTS_MANIFEST="${DRAFT_VARIANTS_MANIFEST:-$DRAFT_VARIANTS_DIR/draft-variants.json}"
    export SELECTED_DRAFT_DEPLOYMENT_MANIFEST="$DRAFT_VARIANTS_DIR/$DRAFT_QUANTIZATION/$VERIFY_GDR/deployment-manifest.json"
    export MAX_SEQUENCE_LENGTH="$KV_CAPACITY" BLOCK_SIZE="$((MAX_DRAFT_TOKENS + 1))"
    export RECEIVER_MODELS_DIR=""
    if [[ -n "${RECEIVER_ROOT:-}" ]]; then
        export RECEIVER_MODELS_DIR="$RECEIVER_ROOT/models"
    fi
    if [[ "$VERIFY_GDR" == mtp ]]; then
        export DEPLOYMENT_MANIFEST="${MTP_DEPLOYMENT_MANIFEST:-}"
    else
        export DEPLOYMENT_MANIFEST="${CHUNK_DEPLOYMENT_MANIFEST:-}"
    fi

    # Remove this loader's old entries when re-sourcing a different config.
    local -a dflash_entries=("$REPO_ROOT/framework/python" "$REPO_ROOT")
    if [[ -n "${RECEIVER_ROOT:-}" ]]; then
        dflash_entries=("$REPO_ROOT/tools/python-bootstrap" "${dflash_entries[@]}" "$RECEIVER_ROOT")
    fi
    local -a dflash_existing dflash_paths=("${dflash_entries[@]}")
    local dflash_entry dflash_seen dflash_skip
    IFS=: read -r -a dflash_existing <<< "${PYTHONPATH:-}"
    for dflash_entry in "${dflash_existing[@]}"; do
        [[ -n "$dflash_entry" ]] || continue
        dflash_skip=0
        for dflash_seen in "${dflash_paths[@]}" "${_DFLASH_PYTHONPATH_ENTRIES[@]}"; do
            if [[ "$dflash_entry" == "$dflash_seen" ]]; then dflash_skip=1; break; fi
        done
        if [[ "$dflash_skip" == 0 ]]; then dflash_paths+=("$dflash_entry"); fi
    done
    _DFLASH_PYTHONPATH_ENTRIES=("${dflash_entries[@]}")
    printf -v PYTHONPATH '%s:' "${dflash_paths[@]}"
    export PYTHONPATH="${PYTHONPATH%:}"

    # Bash arrays used by the native examples; rebuild on every source.
    NPU_ARGS=(
        --target-dir "${TARGET_DIR:-}" --draft-dir "${DRAFT_DIR:-}"
        --verify-gdr "$VERIFY_GDR" --device "npu:$DEVICE_ID"
        --kv-cache-max-len "$KV_CAPACITY"
        --prompt "${PROMPT:-请用一句话解释什么是机器学习。}"
        --prompt-mode chat --enable-thinking
    )
    QUANT_ARGS=()
    if [[ "$QUANT_MODE" == enable ]]; then
        QUANT_ARGS=(--quant_mode enable --config "${QUANT_CONFIG:-}")
    fi
    printf '[dflash-env] verify=%s draft=%s target_quant=%s max_new_tokens=%s run=%s\n' \
        "$VERIFY_GDR" "$DRAFT_QUANTIZATION" "$QUANT_MODE" "$MAX_NEW_TOKENS" "$AI_RUN_DIR"
}

if _dflash_activate_env; then
    unset -f _dflash_activate_env
else
    unset -f _dflash_activate_env
    return 1
fi
