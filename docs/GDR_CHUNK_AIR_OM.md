# Ascend310P OM 使用

流程：[架构](DFLASH_ARCHITECTURE.md)。测量结果：[结果汇总](DFLASH_CURRENT_USAGE_AND_RESULTS.md)。

## 环境配置

复制 [环境模板](../config/dflash_env.sh.example) 到源码外，填写实际路径，然后每个新终端执行：

```bash
source /absolute/path/dflash-env.sh
```

主要配置：

| 变量 | 内容 |
|---|---|
| `TARGET_DIR` | Qwen3.5-4B 权重 |
| `DRAFT_FP16_DIR` | FP16 Draft 权重 |
| `DRAFT_W4A16_DIR` | [W4A16 GPTQ 权重](https://huggingface.co/nota-ai/Qwen3.5-4B-DFlash-GPTQ-W4A16) |
| `DRAFT_W8A16_DIR` | [W8A16 权重](https://huggingface.co/naveenrajk/Qwen3.5-4B-DFlash-W8A16) |
| `OM_BUNDLE_DIR` | 统一 AIR/OM 目录；导出时使用空目录 |
| `DRAFT_QUANTIZATION` | 默认 Draft：`fp16` / `w4a16` / `w8a16` |
| `KV_CAPACITY` | 上下文容量，含输入和输出，默认 2048 |
| `MODEL_PYTHON`、`CANN_ROOT`、`ATC_BIN`、`SOC_VERSION` | 本机工具链 |

权重目录应包含 `config.json`、`model.safetensors`，加载时检查固定版本哈希。所有生成文件放在 `AI_RUN_DIR` 下。
原生 Torch-NPU 命令见 [原生推理](DFLASH_RUN_AND_VALIDATE.md)；本页选择参数控制 OM。

## 导出与编译

先完成 [首次环境准备](DFLASH_RUN_AND_VALIDATE.md#首次准备)和 Target W8A8 配置。
没有 `factory.json` 时创建：

```bash
"$MODEL_PYTHON" -B - <<'PY'
import json, os
from pathlib import Path
config = {
    "target_dir": os.environ["TARGET_DIR"],
    "draft_dir": os.environ["DRAFT_FP16_DIR"],
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
```

**1. 导出 AIR。** 自动锁定所选权重、共享 Target 特征，公共图只导出一次。

```bash
"$MODEL_PYTHON" -B -m qwen35_dflash.ascend310p export-air \
  --factory qwen35_dflash.ascend310p.quant_factory:create_quant_incremental_graphs \
  --factory-config "$AI_RUN_DIR/factory.json" --bundle-dir "$OM_BUNDLE_DIR" \
  --verify-gdr both --draft-quantizations fp16 w4a16 w8a16 \
  --draft-quant-matmul weight_quant
```

只需要 FP16 时，导出命令使用 `--draft-quantizations fp16`；不加载量化权重。
只需要一条验证路线时，导出命令使用 `--verify-gdr chunk` 或 `mtp`。

**2. 编译 OM。** 等上一步返回终端提示符后执行；优先完成 FP16，逐图打印 START/DONE。

```bash
"$MODEL_PYTHON" -B -m qwen35_dflash.ascend310p compile-om \
  --air-manifest "$OM_BUNDLE_DIR/air-manifest.json" \
  --atc "$ATC_BIN" --soc-version "$SOC_VERSION" --resume
```

`--resume` 校验 AIR、OM 哈希、SoC、编译器和参数后复用已完成的组合，继续编译缺失的图。
每个成功组合立即写入 `draft-variants.json`，可直接测试。量化 Draft 编译失败会记录错误并继续其余组合，
保留 FP16 及其他成功结果；同一失败图在两条 Verify 路线间不重复编译。

**只编译或复用 FP16：**

```bash
"$MODEL_PYTHON" -B -m qwen35_dflash.ascend310p compile-om \
  --air-manifest "$OM_BUNDLE_DIR/air-manifest.json" \
  --draft-quantizations fp16 \
  --atc "$ATC_BIN" --soc-version "$SOC_VERSION" --resume
```

已有 FP16 部署清单但缺少索引时，此命令会校验并补齐索引，无需重编。
所选组合全部成功时退出码为 0；有编译失败时为 1。整体索引可为 `PARTIAL`，
测试只要求选中的 Draft/Verify 条目为 `PASS`，不会加载失败或未编译的量化 Draft。
没有成功部署清单记录的残留 OM 不会被复用；若属于本次所选组合会报出路径，保留并移到别处后再重试。
未选中的残留 OM 保持原样。

全部选择时，`$OM_BUNDLE_DIR/om/` 下只有 **7 个 OM**：

```text
prefill.om       decode.om
verify_chunk.om verify_mtp.om
draft.om        draft_w4a16.om draft_w8a16.om
```

`draft-variants.json` 索引各组合；部署清单与它同目录。运行时只加载选定的一个 Draft 和一个 Verify。
AIR 导出成功不代表 OM 编译成功；以 `draft-variants.json` 中对应条目的 `PASS` 为准。
编译日志位于 `$AI_RUN_DIR/log/dflash-atc/`。未修改图时，重试编译无需重新导出 AIR。

**3. 构建 runner。**

```bash
"$MODEL_PYTHON" -B -m qwen35_dflash.ascend310p build-cpp \
  --build-dir "$AI_RUN_DIR/build/cpp" --ascendcl-root "$CANN_ROOT" \
  --output "$AI_RUN_DIR/reports/cpp-build.json"
```

在 `RUNNER_CONFIG` 指向的 JSON 中填写实际 `device_model`、`cann`、`driver`、`firmware`，
另设 `"runtime": "AscendCL C++"`、`"pad_token_id": 0`。

## 量化 Draft MatMul

W4/W8 使用 CANN [WeightQuantBatchMatmulV2](https://github.com/Ascend/op-plugin/blob/cdca1dbc8949cc32bab4a5b2291bf9d9cddcf052/docs/zh/custom_APIs/torch_npu/torch_npu-npu_weight_quant_batchmatmul.md)：
FP16 激活、group-128 scale、`inner_precise=0`。W8 使用 INT8 权重；W4 压缩存储，
图内无损展开为 INT8。Embedding 和 LM head 保持 FP16。

导出默认把逻辑 `[N,K]` 权重经 `TransData` 转为 INT8 FRACTAL_NZ
`[ceil(K/32),ceil(N/16),16,32]`，设置 `transpose_weight=true`，scale 为 `[K/128,N]`。
布局记录在 `weight-quant-layout.json`。修改算子或布局后，使用新目录重新导出 AIR、编译 OM。

**ATC 小图编译：**

```bash
"$MODEL_PYTHON" -B "$REPO_ROOT/tools/probe_draft_matmul_atc.py" \
  --atc "$ATC_BIN" --soc-version "$SOC_VERSION" --device-id "$DEVICE_ID" \
  --bits 4 8 --projection tiny --group-size 128 \
  --weight-format nz --weight-layout nk \
  --output-dir "$AI_RUN_DIR/matmul-atc-nz"
```

实际投影尺寸使用 `--projection gate_up q kv down fc`，包含 M=16/64 两档。
此探针只检查编译；OM 数值和时延用运行测试验证。ATC 日志在 `$AI_RUN_DIR/log/dflash-atc/`。

**原生 NPU MatMul 数值与时延：**

```bash
"$MODEL_PYTHON" -B "$REPO_ROOT/tools/benchmark_draft_matmul.py" \
  --device-id "$DEVICE_ID" --bits 4 8 --projection gate_up q kv down fc \
  --warmup 1 --repetitions 3 --output "$AI_RUN_DIR/draft-matmul.json"
```

输出同输入下的时延、数值差异和 PyTorch 分配器峰值。
显式解量化对照使用导出参数 `--draft-quant-matmul dequant`，仅影响 W4/W8。

### W8 固定权重离线 NZ 转换

预打包为实验选项。先从 W8 成员清单生成可复用的 NZ 文件，转换在 CPU 执行：

```bash
"$MODEL_PYTHON" -B "$REPO_ROOT/tools/pack_draft_weights_nz.py" \
  --manifest "$OM_BUNDLE_DIR/air-manifest-w8a16-chunk.json" \
  --output-dir "$AI_RUN_DIR/w8-nz-weights"
```

输出 `manifest.json` 和 `weight-*.nz.bin`；脚本检查哈希、零填充和逐字节还原。
在目标环境用 `probe_draft_matmul_atc.py --prepack-weights` 检查编译，并完成 OM 数值验证后，
可在正常 `export-air` 命令中增加：

```bash
--draft-weight-prepack-manifest "$AI_RUN_DIR/w8-nz-weights/manifest.json"
```

使用新的 bundle 目录导出、编译；不传该选项时使用默认运行时转换。
复测对比固定权重 `TransData` 数量、Draft ms/call、生成时延和接受率。

## 统一测试

同一入口测试短输入、约 1K 长输入、离线开源数据集、三种 Draft 和两条 Verify。
测试默认关闭 thinking；普通模型与所有 Draft 使用相同的非 thinking 输入模板。

**只跑 FP16 Draft 的离线数据集**，读取目录内全部题目：

```bash
"$MODEL_PYTHON" -B "$REPO_ROOT/tools/benchmark_gdr_lengths.py" \
  --run-dir "$AI_RUN_DIR" --runner "$CPP_RUNNER" \
  --runner-config "$RUNNER_CONFIG" --model-dir "$TARGET_DIR" \
  --bundle-dir "$OM_BUNDLE_DIR" \
  --draft-quantization fp16 --verify-gdr chunk --lengths 128 \
  --dataset-dir /absolute/path/datasets --no-enable-thinking \
  --warmup 0 --repetitions 1 \
  --max-draft-tokens "$MAX_DRAFT_TOKENS" --device-id "$DEVICE_ID" \
  --low-memory --allow-output-differences
```

上面为单次测量；重复测量可设 `--warmup 1 --repetitions 3`。
需要 thinking 时改为 `--enable-thinking`。每个文件的接受率、吞吐与加速比保存在 `datasets.csv` 和分文件报告。

下面命令合并 **8 条短 prompt + 12 条长 prompt + 每个离线文件前 10 题**：

```bash
"$MODEL_PYTHON" -B "$REPO_ROOT/tools/benchmark_gdr_lengths.py" \
  --run-dir "$AI_RUN_DIR" --runner "$CPP_RUNNER" \
  --runner-config "$RUNNER_CONFIG" --model-dir "$TARGET_DIR" \
  --bundle-dir "$OM_BUNDLE_DIR" \
  --draft-quantization fp16 w4a16 w8a16 --verify-gdr both --lengths 128 \
  --no-enable-thinking \
  --dataset-dir /absolute/path/datasets --num-questions 10 --include-builtin-prompts \
  --warmup 1 --repetitions 3 \
  --max-draft-tokens "$MAX_DRAFT_TOKENS" --device-id "$DEVICE_ID" \
  --low-memory --allow-output-differences
```

| 需求 | 参数 |
|---|---|
| 只测短 / 长输入 | 去掉三个数据集参数；设 `--prompt-group short` / `long` |
| 只测内置 20 条 | 去掉 `--dataset-dir`、`--num-questions`、`--include-builtin-prompts` |
| 只测离线数据集 | 去掉 `--include-builtin-prompts` |
| 长输入与离线数据一起测 | 保留数据集参数，增加 `--prompt-group long` |
| 只测一种 Draft | `--draft-quantization w4a16`；也可写 `"$DRAFT_QUANTIZATION"` |
| 只测一条 Verify | `--verify-gdr chunk` 或 `mtp` |
| 多个输出上限 | `--lengths 128 512 1024` |
| 指定离线文件 | 用 `--dataset-files /path/gsm8k.jsonl /path/humaneval.jsonl` 替换 `--dataset-dir` |
| 文件全部题目 | 去掉 `--num-questions` |
| 只检查配置与容量 | 增加 `--plan-only` |

离线格式：JSONL 每行 `{"question":"完整问题"}`，或相同记录的 JSON 数组；字段为 `prompt` 时加 `--dataset-field prompt`。
不会下载数据、拼接答案或截短问题。自定义短/长输入用 `--prompts /path/prompts.json`；筛选 ID 用 `--prompt-id zh_explain`。
`--include-builtin-prompts` 合并所选本地 prompt 与离线文件，`--num-questions` 只限制离线文件。

**每个问题、每个输出上限的普通模型只测一次**，所有 Draft/Verify 组合共用其输出和时延。

## 复用已有普通模型数据

在同一测试命令后增加：

```bash
--ordinary-baseline /absolute/path/previous-run/summary.json
```

支持测试目录、矩阵/单套测试的 `summary.json`，或 `runner-batch.json`；多长度测试按输出上限匹配。
脚本核对普通 OM、接口、runner/设备、thinking 设置、输入 token 和题目顺序、输出上限、EOS、预热及测量次数。
开启 thinking 或未记录该设置的结果，不能用于当前非 thinking 测试。
匹配后只执行 DFlash；缺失或不匹配会报明原因，**不会自动重跑普通模型**。原始报告保持不变。

## 查看结果

- `gdr-lengths-*/summary.md`：Draft × Verify × 输出长度的总表及短/长分组，含接受率、吞吐、加速比和阶段时延。
- `datasets.csv`、`datasets/<ID>/summary.md`：每个离线文件在各 Draft/Verify 下的结果。
- `cases.csv`：逐题数据，含 Draft 类型；各精度子目录的 `generations.txt` 保存文字输出。

加速比包含 Prefill + Decode；指标定义及已有结果见 [测试结果](DFLASH_CURRENT_USAGE_AND_RESULTS.md)。

## 单 OM profiling

将环境文件中的 `SAVED_BATCH` 设为已有测试的 `runner-batch.json` 路径。
下面使用 `zh_explain` 的输入，依次采集全部 7 个 OM，无需完整生成：

```bash
"$MODEL_PYTHON" -B "$REPO_ROOT/tools/profile_om.py" \
  --run-dir "$AI_RUN_DIR" --runner "$CPP_RUNNER" \
  --bundle-dir "$OM_BUNDLE_DIR" --profile-om all \
  --prompt-report "${SAVED_BATCH}.cases/zh_explain.json" \
  --device-id "$DEVICE_ID" \
  --max-new-tokens 16 --max-draft-tokens 15 --profile-warmup 0
```

`--profile-om` 可选一个或多个：`prefill decode draft draft_w4a16 draft_w8a16 verify_chunk verify_mtp`；
`draft` 表示 FP16。例如只测量化 Draft：`--profile-om draft_w4a16 draft_w8a16`。
部分编译目录也可使用；只选择已有 `PASS` 部署的 OM，例如 `--profile-om draft`。

共享 Prefill/Decode 各采集一次，两个 Verify 优先使用 FP16 Draft 准备输入。
各项依次加载、采集、卸载，不同时加载 7 个 OM；准备工作在窗口外，单项失败后继续其余项。
Prefill 窗口覆盖全部输入块，其余窗口各测一次调用。输入 token 直接沿用 `--prompt-report`，包括其 thinking 设置。
结果在 `msprof/oms-*/summary.csv`；各 OM 子目录保存独立的算子明细、热点和原始采集文件。
这些是带 profiling 开销的独立窗口时延，不能相加作为整段生成时延。

## 精度与执行口径

Recurrent state 存储/传输为 FP32，conv/KV 为 FP16。Draft 采用 16/64 双档，最多输出 15 个候选；
Draft 默认 `deterministic=0`，投机始终开启。

公开 W4/W8 Draft 为五层，当前 FP16 为六层，特征层也不同，因此是不同 checkpoint 的对比。
量化路径以压缩权重常驻，MatMul 选择见[量化 Draft MatMul](#量化-draft-matmul)。
