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

**2. 编译 OM。** 等上一步返回终端提示符后执行；逐图打印 START/DONE。

```bash
"$MODEL_PYTHON" -B -m qwen35_dflash.ascend310p compile-om \
  --air-manifest "$OM_BUNDLE_DIR/air-manifest.json" \
  --atc "$ATC_BIN" --soc-version "$SOC_VERSION" --resume
```

`--resume` 校验 AIR、OM 哈希、SoC、编译器和参数后复用已完成的组合，继续编译缺失的图。
没有成功部署清单记录的残留 OM 会报出路径；保留并移到别处后再重试。

全部选择时，`$OM_BUNDLE_DIR/om/` 下只有 **7 个 OM**：

```text
prefill.om       decode.om
verify_chunk.om verify_mtp.om
draft.om        draft_w4a16.om draft_w8a16.om
```

`draft-variants.json` 索引各组合；部署清单与它同目录。运行时只加载选定的一个 Draft 和一个 Verify。
只编译部分类型可改 `--draft-quantizations w8a16`；只要一条验证路线可改 `--verify-gdr chunk`。
AIR 导出成功不代表 OM 编译成功；以第二步成功及 `draft-variants.json` 的 PASS 为准。
编译日志位于 `$AI_RUN_DIR/log/dflash-atc/`。中断后直接重跑第二步，不需要重新导出 AIR。

**3. 构建 runner。**

```bash
"$MODEL_PYTHON" -B -m qwen35_dflash.ascend310p build-cpp \
  --build-dir "$AI_RUN_DIR/build/cpp" --ascendcl-root "$CANN_ROOT" \
  --output "$AI_RUN_DIR/reports/cpp-build.json"
```

在 `RUNNER_CONFIG` 指向的 JSON 中填写实际 `device_model`、`cann`、`driver`、`firmware`，
另设 `"runtime": "AscendCL C++"`、`"pad_token_id": 0`。

## 量化 Draft MatMul

W4/W8 默认采用 CANN [WeightQuantBatchMatmulV2](https://github.com/Ascend/op-plugin/blob/cdca1dbc8949cc32bab4a5b2291bf9d9cddcf052/docs/zh/custom_APIs/torch_npu/torch_npu-npu_weight_quant_batchmatmul.md)，保留 FP16 激活、group-128 scale 和
`inner_precise=0`。W8 直接传 INT8 权重；W4 压缩存储，临时无损展开为 INT8 后调用。
这是 A16 权重量化接口，不能按 W8A8 的纯整数矩阵乘理解；Embedding/LM Head 保持 FP16。

AIR 保存前将权重转置折叠为 `transpose_weight=true`，算子接收 `[N,K]` INT8 权重。
scale 仍须实际转置为 `[K/128,N]`；该属性不作用于 scale，不能用 reshape 替代。
外部压缩权重和 scale 接口不变，布局检查写入 `weight-quant-layout.json`。
中间节点的 shape 从 TorchAir 转换时的类型元数据校验，不要求 GE 输出描述已完成 shape 推导。
不自动关闭 ATC 融合；[同类转置融合规则](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/maintenref/graphubfusionref/atlasrr_30_0074.html)注明不可关闭。
此图适配已做主机验证，310P 编译及数值仍需实测。

**先测小图编译**，不加载模型权重，覆盖 W4/W8、16/64 动态档和 gate/up 的实际矩阵尺寸：

```bash
"$MODEL_PYTHON" -B "$REPO_ROOT/tools/probe_draft_matmul_atc.py" \
  --atc "$ATC_BIN" --soc-version "$SOC_VERSION" --device-id "$DEVICE_ID" \
  --bits 4 8 --projection tiny gate_up \
  --output-dir "$AI_RUN_DIR/matmul-atc-check"
```

只测最小图可设 `--projection tiny`。结果与 AIR/OM 位于指定目录；ATC 日志位于
`$AI_RUN_DIR/log/dflash-atc/`。这是编译检查，不能代替 OM 数值或性能验证。
重试时使用新的输出目录。失败报告会标出 `prepare/export/compile` 阶段并保存异常堆栈；
图检查失败时，`weight-quant-layout.json` 还会记录节点、shape 和 dtype；
ATC 失败摘要保留 tiling 属性及 traceback 中的具体约束。
该 A16W8 接口在不同芯片上的 group 支持有约束；必须确认本机 group-128 编译通过。
若出现 `Antiquant shape expect [G,N], but is [N,G]`，更新代码并重新导出 AIR；
仅重编已有 AIR 无法修正布局。复测先用 `--projection tiny`，通过后再测 `tiny gate_up`。

导出前还有原生 NPU 数值检查，导出后检查五层 Draft 的 26 个融合节点。
原生调用通过不代表 ATC 编译通过；单独测原生 MatMul 时延使用：

```bash
"$MODEL_PYTHON" -B "$REPO_ROOT/tools/benchmark_draft_matmul.py" \
  --device-id "$DEVICE_ID" --bits 4 8 --projection gate_up q kv down fc \
  --warmup 1 --repetitions 3 --output "$AI_RUN_DIR/draft-matmul.json"
```

输出相同输入下的时延、数值差异和 PyTorch 分配器峰值；这是合成输入的原生调用测试，
OM 峰值显存与完整模型接受率仍需统一测试和 profiling 验证。
压缩常驻权重字节数不增加，不缓存完整 FP16 权重；CANN 内部工作区不能据此推断。

修改 MatMul 或其 AIR 布局后，**需要重新导出 AIR，再编译 OM**，使用上面的正常命令和新目录。
仅重编已有 AIR 不会改变图内算子；本次不需要更新 C++ runner。
对照路径可在导出时设 `--draft-quant-matmul dequant`，或在 factory JSON 中设
`"draft_quant_matmul": "dequant"`；该设置仅影响 W4/W8，FP16 Draft 与 Target 计算不变。

## 统一测试

同一入口测试短输入、约 1K 长输入、离线开源数据集、三种 Draft 和两条 Verify。
测试默认关闭 thinking；普通模型与所有 Draft 使用相同的非 thinking 输入模板。
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

接受率为总接受数/总提议数；加速比为普通模型总耗时/DFlash 总耗时。
计时排除预热、加载和重置；图时延为同步 OM 调用时间。
DFlash Prefill 包含长输入建 Draft 缓存的调用，不能与图累计耗时重复相加。
允许输出差异仍保留差异记录；多轮漂移报告为 `DRIFT_OBSERVED`。这些指标不代表任务正确率。

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
压缩权重以一维输入传输，图内恢复形状；五层 Draft 的动态档位共 99 维，低于 ACL 的 128 维上限。
若加载时报 `aclmdlGetInputDynamicDims failed: 500001`，请更新 runner，并在空目录重新导出、编译量化 Draft。
真实 TorchAir/ATC 编译、峰值显存、接受率和加速效果须在 310P 上验证。
