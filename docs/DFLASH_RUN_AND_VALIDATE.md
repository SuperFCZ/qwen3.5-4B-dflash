# Torch-NPU 使用

先按 [README](../README.md#环境配置)准备环境配置文件。每个新终端执行：

```bash
source /absolute/path/dflash-env.sh
cd "$AI_RUN_DIR"
```

路径、提示词、输出长度和量化选项统一修改该文件。首次安装见文末[首次准备](#首次准备)，
OM 测试见 [OM/C++ 使用](GDR_CHUNK_AIR_OM.md)。

## 运行 DFlash

```bash
"$MODEL_PYTHON" -B -m models.dflash_v1.run_npu \
  "${NPU_ARGS[@]}" "${QUANT_ARGS[@]}" \
  --execution-mode dflash --block-size "$BLOCK_SIZE" --max-new-tokens "$MAX_NEW_TOKENS" \
  --report "$AI_RUN_DIR/native-$VERIFY_GDR.json"
```

- 配置中 `VERIFY_GDR=chunk` 或 `mtp` 选择路线，修改后重新 `source`；所选算子须已注册。
- `MAX_DRAFT_TOKENS=15` 对应 `BLOCK_SIZE=16`：1 个 anchor + 最多 15 个候选。
- 严格对照：把 `--execution-mode dflash` 改为 `validate`。
- 文字输出在报告的 `dflash.generated_text`。

## 测量普通 / DFlash 时延

```bash
for MODE in ordinary dflash; do
  "$MODEL_PYTHON" -B -m models.dflash_v1.benchmark_npu \
    "${NPU_ARGS[@]}" "${QUANT_ARGS[@]}" \
    --mode "$MODE" --block-size "$BLOCK_SIZE" --max-new-tokens "$MAX_NEW_TOKENS" \
    --warmup 3 --repetitions 10 \
    --report "$AI_RUN_DIR/benchmark-$VERIFY_GDR-$MODE.json"
done
```

比较报告中的整段生成时延、tok/s 和实际输出长度。此原生 benchmark 保留严格对照；
允许输出差异的多 prompt 实验使用 [OM 测试命令](GDR_CHUNK_AIR_OM.md#多-prompt-测试)。

## msprof

普通模型各采一次 prefill、decode：

```bash
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
    "${NPU_ARGS[@]}" "${QUANT_ARGS[@]}" --block-size "$BLOCK_SIZE"
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
"$MODEL_PYTHON" -m pip install numpy PyYAML safetensors huggingface-hub "transformers==5.14.1"
npu-smi info
"$MODEL_PYTHON" -B -c 'import torch, torch_npu; assert torch.npu.is_available()'
```

配置文件中的 `AI_RUN_DIR` 放源码之外；workspace 用户填写分配的运行目录。
加载配置会设置 CANN、缓存和 receiver 搜索路径。receiver 的 `models/export_model_wrapper_qwen3_5.py` 须提供
`Qwen3_5ForCausalLMWrapper` 和 `Qwen3_5ForCausalLM`。

</details>

<details>
<summary>W8A8 配置（FP16 可跳过）</summary>

准备与 Target 匹配的量化 Linear safetensors、INT8 embedding 和逐行 FP32 scale：

```bash
cat > "$QUANT_CONFIG" <<'YAML'
quanted_pth: /absolute/path/w8a8/linear
embedding_weight_path: /absolute/path/w8a8/embedding_weight.bin
embedding_scale_path: /absolute/path/w8a8/embedding_scale.bin
YAML
```

填写上面三处路径；在环境配置中设 `QUANT_MODE=enable` 后重新 `source`。
`QUANT_MODE=disable` 使用原生 FP16；Draft 始终保持 FP16。

</details>
