# 配置、使用与测试

正常流程只有三个入口：`export-air` → `compile-om` → `benchmark_gdr_lengths.py`。
FP16、W4A16、W8A16 共用命令和 runner，通过参数与 bundle 目录选择。

## 首次配置

需要匹配 Ascend310P 的 CANN、PyTorch/torch_npu、TorchAir、自定义算子包，以及 Target/Draft 权重。
外部 receiver 的 `models/export_model_wrapper_qwen3_5.py` 须提供 Target 加载器；
Chunk 算子须支持 INT16 effective_length，MTP 另需对应 GDR MTP 算子。

从仓库复制 [配置模板](../config/dflash_env.sh.example) 到源码外，填写后在 Bash 中加载：

```bash
cp -n config/dflash_env.sh.example /absolute/path/dflash-env.sh
# 编辑上述文件，填写本机路径和参数。
source /absolute/path/dflash-env.sh
```

| 配置 | 用途 |
|---|---|
| `REPO_ROOT`、`AI_RUN_DIR` | 源码与产物根目录；产物不放进源码 |
| `MODEL_PYTHON`、`CANN_ROOT`、`ATC_BIN`、`SOC_VERSION`、`DEVICE_ID` | 本机环境与设备 |
| `TARGET_DIR`、`RECEIVER_ROOT`、`QUANT_CONFIG` | Target、加载器与 W8A8 权重配置 |
| `DRAFT_FP16_DIR`、`DRAFT_W4A16_DIR`、`DRAFT_W8A16_DIR` | 所选 Draft checkpoint |
| `DRAFT_WEIGHT_PREPACK` | `runtime` 默认；`nz` 在 AIR 导出时自动离线打包 W8 |
| `OM_BUNDLE_DIR` | AIR/OM 目录；新导出使用空目录 |
| `CPP_RUNNER`、`RUNNER_CONFIG` | 所有 Draft 共用的 runner 与设备身份配置 |
| `VERIFY_GDR`、`KV_CAPACITY`、`MAX_DRAFT_TOKENS` | 默认 chunk、2048、15；容量须容纳输入和输出 |

仅填写要导出的 Draft 路径。新终端重新 source；更改环境配置后也需重新 source。
配置加载器设置 CANN、Python 搜索路径和运行目录缓存。

在 `QUANT_CONFIG` 指定的 YAML 中填写已有 Target W8A8 权重路径：

```yaml
quanted_pth: /absolute/path/w8a8/linear
embedding_weight_path: /absolute/path/w8a8/embedding_weight.bin
embedding_scale_path: /absolute/path/w8a8/embedding_scale.bin
```

在 `RUNNER_CONFIG` 指定的 JSON 中填写实际设备信息：

```json
{"device_model":"Ascend310P3","cann":"填写实际版本","driver":"填写实际版本",
 "firmware":"填写实际版本","runtime":"AscendCL C++","pad_token_id":0}
```

首次生成工厂配置并构建 runner；已有配置和同版本 runner 可复用：

```bash
"$MODEL_PYTHON" -B - <<'PY'
import json, os
from pathlib import Path
config = {
    "target_dir": os.environ["TARGET_DIR"],
    "draft_dir": os.environ.get("DRAFT_FP16_DIR", ""),
    "quant_config": os.environ["QUANT_CONFIG"],
    "receiver_models_dir": os.environ["RECEIVER_MODELS_DIR"],
    "max_sequence_length": int(os.environ["KV_CAPACITY"]),
    "include_ordinary_decode": True,
    "draft_attention_matmul_dtype": "float16",
    "dtype": "float16", "device": "npu:" + os.environ["DEVICE_ID"],
    "adn_rms_norm_ge_op_type": "AdnRmsNorm",
}
with (Path(os.environ["AI_RUN_DIR"]) / "factory.json").open("x") as f:
    json.dump(config, f, indent=2)
PY

"$MODEL_PYTHON" -B -m qwen35_dflash.ascend310p build-cpp \
  --build-dir "$AI_RUN_DIR/build/cpp" --ascendcl-root "$CANN_ROOT" \
  --output "$AI_RUN_DIR/reports/cpp-build.json"
```

## 1. 生成 AIR

下面在同一个 bundle 中导出三种 Draft，Target 图只导出一次。
使用 W8 离线 NZ 时，先在配置中设 `DRAFT_WEIGHT_PREPACK=nz`；FP16/W4 仍使用各自原路径。

```bash
"$MODEL_PYTHON" -B -m qwen35_dflash.ascend310p export-air \
  --factory qwen35_dflash.ascend310p.quant_factory:create_quant_incremental_graphs \
  --factory-config "$AI_RUN_DIR/factory.json" --bundle-dir "$OM_BUNDLE_DIR" \
  --draft-quantizations fp16 w4a16 w8a16 --verify-gdr "$VERIFY_GDR" \
  --draft-quant-matmul weight_quant --draft-weight-prepack "$DRAFT_WEIGHT_PREPACK"
```

- 只导出 W8：将 `--draft-quantizations` 改为 `w8a16`；只导出 FP16/W4 时使用 `runtime`。
- 两条验证路线一起导出：使用 `--verify-gdr both`。
- `nz` 直接从本次加载的 W8 checkpoint 生成载体，检查字节还原、padding 和哈希，再写入 AIR 常量；不需要先生成原始 W8 AIR，也不需要单独运行转换或探针脚本。
- `nz` 只支持 `weight_quant` 后端；载体保存在 bundle 的 `air/draft_w8a16/prepacked-weights/`。
- 对比 runtime/NZ 两种构建时，只需为 `OM_BUNDLE_DIR` 设置不同的空目录并重跑导出、编译。测试始终指定实际要测的 bundle。

## 2. 生成 OM

```bash
"$MODEL_PYTHON" -B -m qwen35_dflash.ascend310p compile-om \
  --air-manifest "$OM_BUNDLE_DIR/air-manifest.json" \
  --atc "$ATC_BIN" --soc-version "$SOC_VERSION" --resume
```

编译器从 AIR 清单读取权重模式。离线 NZ W8 自动编译 C16/C64 静态档位，无需额外档位参数。
`--resume` 校验并复用已完成产物；可加 `--draft-quantizations w8a16` 只编译所选 Draft。
成功组合写入 `draft-variants.json`；测试只加载所选的 PASS 部署。
编译日志位于 `$AI_RUN_DIR/log/dflash-atc/`。

## 3. 执行测试

默认内置 20 条输入：8 条短输入、12 条约 1K 输入。以下参数与已有 thinking-on 结果一致：

```bash
"$MODEL_PYTHON" -B "$REPO_ROOT/tools/benchmark_gdr_lengths.py" \
  --run-dir "$AI_RUN_DIR" --runner "$CPP_RUNNER" \
  --runner-config "$RUNNER_CONFIG" --model-dir "$TARGET_DIR" \
  --bundle-dir "$OM_BUNDLE_DIR" --draft-quantization fp16 w4a16 w8a16 \
  --verify-gdr "$VERIFY_GDR" --lengths 128 \
  --enable-thinking --warmup 1 --repetitions 3 \
  --max-draft-tokens "$MAX_DRAFT_TOKENS" --device-id "$DEVICE_ID" \
  --low-memory --allow-output-differences
```

`--draft-quantization` 选择 checkpoint 类型；runtime/NZ 和静态档位由 bundle 决定，测试不用再传预打包参数。
每个问题、输出上限的 ordinary 只测一次，各 Draft/Verify 共用基线；同一个 runner 支持全部组合。

| 需求 | 在同一测试命令中设置 |
|---|---|
| 只测 W8 离线 NZ | `--draft-quantization w8a16`，bundle 指向 NZ 构建目录 |
| 短、长各一条 | `--prompt-id zh_explain --prompt-id long_zh_summary` |
| 只测短 / 长输入 | `--prompt-group short` / `long` |
| 多个输出上限 | `--lengths 128 512` |
| 两条 Verify | `--verify-gdr both`，需已编译对应图 |
| 关闭 thinking | `--no-enable-thinking`；省略开关时默认关闭 |
| 离线数据集 | `--dataset-dir /absolute/path/datasets`，或 `--dataset-files /path/a.jsonl /path/b.jsonl` |
| 限制每个文件题数 | `--num-questions 10`；不设则读取全部 |
| 数据集与内置输入合测 | `--include-builtin-prompts` |
| 自定义 prompt 文件 | `--prompts /absolute/path/prompts.json` |
| 仅检查配置、容量 | `--plan-only` |
| 复用已有 ordinary | `--ordinary-baseline /absolute/path/previous-run/summary.json` |

数据集 JSONL 每行 `{"question":"完整问题"}`，也支持相同记录的 JSON 数组；
字段为 prompt 时加 `--dataset-field prompt`。数据不自动下载，不拼答案、不截短问题。
复用 ordinary 时检查模型、runner、设备、输入、thinking、输出上限和重复次数，不匹配即报错。

`--allow-output-differences` 允许保留完成的性能测量，一致性仍单独标记；
严格 token 对照时去掉此参数。指标定义见 [测试结果](RESULTS.md)。

输出在本次 `gdr-lengths-*/`：`summary.md/json` 为汇总，`cases.csv` 为逐题数据，
`datasets.csv` 与 `datasets/<ID>/` 为分文件结果，各 prompt-suite 的 `generations.txt` 为文字输出。

## Profiling 与原生对照

使用当前 bundle 与统一 runner 采集 W8 Draft；`SAVED_BATCH` 填已完成测试的 `runner-batch.json`：

```bash
"$MODEL_PYTHON" -B "$REPO_ROOT/tools/profile_om.py" \
  --run-dir "$AI_RUN_DIR" --runner "$CPP_RUNNER" --bundle-dir "$OM_BUNDLE_DIR" \
  --profile-om draft_w8a16 --prompt-report "${SAVED_BATCH}.cases/zh_explain.json" \
  --device-id "$DEVICE_ID" --max-new-tokens 16 --max-draft-tokens 15 --profile-warmup 1
```

可选 `prefill decode draft draft_w4a16 draft_w8a16 verify_chunk verify_mtp`；
`draft` 表示 FP16。各项顺序采集，profiler 时延与无 profiler 的生成时延分开报告。

原生 Torch-NPU 对照使用 `models.dflash_v1.run_npu`，环境中的 `QUANT_MODE` 控制原生 Target；
OM Target 始终按工厂配置使用 W8A8。普通模型 profiling 示例：

```bash
# 只采 decode：设 PROFILE_STAGE=decode，并将下方 all 替换为 "$PROFILE_STAGE"。
"$REPO_ROOT/tools/run_msprof.sh" \
  --label python-ordinary-all --output-dir "$AI_RUN_DIR/msprof/ordinary-all" \
  --python "$MODEL_PYTHON" --msprof-bin "$MSPROF_BIN" \
  --profile-backend python --profile-mode ordinary --profile-stage all \
  --profile-warmup 1 --aic-metrics PipeUtilization \
  -- "$MODEL_PYTHON" -B -m models.dflash_v1.run_npu \
    "${NPU_ARGS[@]}" "${QUANT_ARGS[@]}"
```
