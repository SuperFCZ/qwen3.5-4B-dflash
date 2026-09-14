# Torch-NPU 使用

已有环境直接执行下列命令；第一次使用先展开文末的[首次准备](#首次准备)。
所有命令在同一 Bash 会话中执行。OM 测试见 [OM/C++ 使用](GDR_CHUNK_AIR_OM.md)。

## 运行 DFlash

```bash
export VERIFY_GDR=chunk       # chunk 或 mtp
export KV_CAPACITY=2048       # 64 的倍数，须容纳 prompt + 输出
export MAX_NEW_TOKENS=128      # 可改为 512、1024
QUANT_ARGS=()                 # FP16；W8A8 用下一行
# QUANT_ARGS=(--quant_mode enable --config "$QUANT_CONFIG")

NPU_ARGS=(
  --target-dir "$TARGET_DIR" --draft-dir "$DRAFT_DIR"
  --verify-gdr "$VERIFY_GDR" --device npu:0
  --kv-cache-max-len "$KV_CAPACITY"
  --prompt '请用通俗的中文解释什么是机器学习。'
  --prompt-mode chat --enable-thinking
)
cd "$AI_RUN_DIR"
"$MODEL_PYTHON" -B -m models.dflash_v1.run_npu \
  "${NPU_ARGS[@]}" "${QUANT_ARGS[@]}" \
  --execution-mode dflash --block-size 16 --max-new-tokens "$MAX_NEW_TOKENS" \
  --report "$AI_RUN_DIR/native-$VERIFY_GDR.json"
```

- 切换路线：修改 `VERIFY_GDR` 后重新执行此段；无需导出 OM，所选算子须已注册。
- `--block-size 16` = 1 个 anchor + 最多 15 个候选。
- 严格对照：把 `--execution-mode dflash` 改为 `validate`。
- 文字输出在报告的 `dflash.generated_text`。

## 测量普通 / DFlash 时延

```bash
for MODE in ordinary dflash; do
  "$MODEL_PYTHON" -B -m models.dflash_v1.benchmark_npu \
    "${NPU_ARGS[@]}" "${QUANT_ARGS[@]}" \
    --mode "$MODE" --block-size 16 --max-new-tokens "$MAX_NEW_TOKENS" \
    --warmup 3 --repetitions 10 \
    --report "$AI_RUN_DIR/benchmark-$VERIFY_GDR-$MODE.json"
done
```

比较报告中的整段生成时延、tok/s 和实际输出长度。此原生 benchmark 保留严格对照；
允许输出差异的多 prompt 实验使用 [OM 测试命令](GDR_CHUNK_AIR_OM.md#多-prompt-测试)。

## msprof

普通模型各采一次 prefill、decode：

```bash
export MSPROF_BIN=/absolute/path/msprof
# 只采 decode：设 PROFILE_STAGE=decode，并把下方 all 改为 "$PROFILE_STAGE"。
"$REPO_ROOT/tools/run_msprof.sh" \
  --label python-ordinary-all --output-dir "$AI_RUN_DIR/msprof/ordinary-all" \
  --python "$MODEL_PYTHON" --msprof-bin "$MSPROF_BIN" \
  --profile-backend python --profile-mode ordinary --profile-stage all \
  --profile-warmup 1 --aic-metrics PipeUtilization \
  -- "$MODEL_PYTHON" -B -m models.dflash_v1.run_npu \
    "${NPU_ARGS[@]}" "${QUANT_ARGS[@]}"
```

DFlash 采一轮 Draft → Verify → Commit：

```bash
"$REPO_ROOT/tools/run_msprof.sh" \
  --label python-dflash-round --output-dir "$AI_RUN_DIR/msprof/dflash-round" \
  --python "$MODEL_PYTHON" --msprof-bin "$MSPROF_BIN" \
  --profile-backend python --profile-mode dflash --profile-stage decode-round \
  --profile-warmup 1 --aic-metrics PipeUtilization \
  -- "$MODEL_PYTHON" -B -m models.dflash_v1.run_npu \
    "${NPU_ARGS[@]}" "${QUANT_ARGS[@]}" --block-size 16
```

单项采集将 `decode-round` 改为 `draft`、`verify` 或 `accept-commit`；
`all` 分别采全部阶段。Chunk 的第二遍 GDR 在 `accept-commit`，MTP 在此选择状态。
采集输出使用新目录；profiler 时延单独分析，不与无 profiler 的 benchmark 混算。

## 首次准备

<details>
<summary>环境、模型路径和外部 receiver（已有环境可跳过）</summary>

需要匹配设备的 CANN、Python 3.10 / PyTorch / torch_npu、自定义算子包、
Target 和官方 Draft checkpoint，以及外部 receiver 加载器；仓库不附带这些文件。
Torch-NPU 两条路线共用同一环境。Chunk 接口须支持 `INT16[B] effective_length`；
MTP 另需 `npu_gated_delta_rule_mtp`。缺少算子会报错。

```bash
export REPO_ROOT=/absolute/path/qwen3.5-4B-dflash
export AI_RUN_DIR=/absolute/path/new-run
export MODEL_PYTHON=/absolute/path/npu-env/bin/python
export CANN_ROOT=/absolute/path/ascend-toolkit
export TARGET_DIR=/absolute/path/Qwen3.5-4B
export DRAFT_DIR=/absolute/path/Qwen3.5-4B-DFlash
export RECEIVER_ROOT=/absolute/path/qwen35-receiver
export RECEIVER_MODELS_DIR="$RECEIVER_ROOT/models"

source "$CANN_ROOT/set_env.sh"
mkdir -p "$AI_RUN_DIR"/{reports,cache,tmp,python-bootstrap}
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$AI_RUN_DIR/tmp"
export HF_HOME="$AI_RUN_DIR/cache/huggingface"
export TORCH_HOME="$AI_RUN_DIR/cache/torch"
export XDG_CACHE_HOME="$AI_RUN_DIR/cache"

cat > "$AI_RUN_DIR/python-bootstrap/sitecustomize.py" <<'PY'
import os
import models
models.__path__.append(os.environ["RECEIVER_MODELS_DIR"])
PY
export PYTHONPATH="$AI_RUN_DIR/python-bootstrap:$REPO_ROOT/framework/python:$REPO_ROOT:$RECEIVER_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$AI_RUN_DIR"
"$MODEL_PYTHON" -m pip install numpy PyYAML safetensors huggingface-hub "transformers==5.14.1"
npu-smi info
"$MODEL_PYTHON" -B -c 'import torch, torch_npu; assert torch.npu.is_available()'
```

`AI_RUN_DIR` 使用源码之外的新目录；workspace 用户沿用 `ws session start` 分配的目录。
receiver 的 `models/export_model_wrapper_qwen3_5.py` 须提供
`Qwen3_5ForCausalLMWrapper` 和 `Qwen3_5ForCausalLM`。

</details>

<details>
<summary>W8A8 配置（FP16 可跳过）</summary>

准备与 Target 匹配的量化 Linear safetensors、INT8 embedding 和逐行 FP32 scale：

```bash
export QUANT_CONFIG="$AI_RUN_DIR/qwen35-w8a8.yaml"
cat > "$QUANT_CONFIG" <<'YAML'
quanted_pth: /absolute/path/w8a8/linear
embedding_weight_path: /absolute/path/w8a8/embedding_weight.bin
embedding_scale_path: /absolute/path/w8a8/embedding_scale.bin
YAML
```

修改上面三处路径，然后在运行命令中设置
`QUANT_ARGS=(--quant_mode enable --config "$QUANT_CONFIG")`。Draft 保持 FP16。

</details>
