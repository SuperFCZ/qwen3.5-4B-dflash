# OM/C++ 使用

先按 [README](../README.md#环境配置)保存环境配置。新终端执行：

```bash
source /absolute/path/dflash-env.sh
cd "$AI_RUN_DIR"
```

配置里的 `VERIFY_GDR=chunk|mtp` 自动选择对应的 `DEPLOYMENT_MANIFEST`。
两个 manifest 填实际已编译路径；新版共用同一目录里的 3 个公共 OM，各自引用自己的 Verify。
首次部署见文末[导出与编译](#导出与编译)。`QUANT_MODE` 只控制原生推理，OM 精度由编译产物决定。

## 多 prompt 测试

```bash
"$MODEL_PYTHON" -B "$REPO_ROOT/tools/benchmark_prompts.py" \
  --run-dir "$AI_RUN_DIR" --runner "$CPP_RUNNER" \
  --deployment-manifest "$DEPLOYMENT_MANIFEST" \
  --runner-config "$RUNNER_CONFIG" --model-dir "$TARGET_DIR" \
  --max-new-tokens "$MAX_NEW_TOKENS" --max-draft-tokens "$MAX_DRAFT_TOKENS" --device-id "$DEVICE_ID" \
  --low-memory --allow-output-differences
```

默认 8 条 prompt，每模式 3 次预热 + 10 次测量，投机始终开启。
`--allow-output-differences` 用于当前允许输出差异的实验；去掉则严格比较 token/EOS。
`--low-memory` 先测全部普通模式，再测全部 DFlash。

常用调整：`--prompt-id math` 只测一条；配置文件中 `MAX_NEW_TOKENS=512` 改输出上限；
`--prompts /path/prompts.json` 自定义输入，格式：
`[{"id":"my_case","prompt":"你的问题","category":"自定义"}]`。

结果在终端打印的 `prompt-suite-*`：`summary.md` 看速度/接受率，
`generations.txt` 看文字，`runner-batch.json.cases/` 保存每轮和分项计时。

## 多长度与双路线测试

默认同时测 Chunk、MTP；下列参数可以组合：

| 选择 | 参数 |
|---|---|
| 只测 Chunk / MTP | `--verify-gdr chunk` / `--verify-gdr mtp`；两条都测用 `both`（默认） |
| 只测一个输出长度 | `--lengths 512` |
| 只测一条 prompt | `--prompt-id math`；可重复参数选择多条 |

```bash
"$MODEL_PYTHON" -B "$REPO_ROOT/tools/benchmark_gdr_lengths.py" \
  --run-dir "$AI_RUN_DIR" --runner "$CPP_RUNNER" \
  --runner-config "$RUNNER_CONFIG" --model-dir "$TARGET_DIR" \
  --chunk-deployment-manifest "$CHUNK_DEPLOYMENT_MANIFEST" \
  --mtp-deployment-manifest "$MTP_DEPLOYMENT_MANIFEST" \
  --lengths 32 64 128 256 512 1024 --max-draft-tokens "$MAX_DRAFT_TOKENS" --device-id "$DEVICE_ID" \
  --low-memory --allow-output-differences
```

- 只测一条路线时，只需提供该路线的 `--*-deployment-manifest`，另一份清单不读取、不加载。
- 双路线模型须有相同容量；所选路线均须容纳 `prompt tokens + max_new_tokens`，短 prompt 套件建议 2048。
- 加 `--plan-only` 只检查清单、哈希和容量，不执行设备模型。旧 512 容量 OM 需扩容重导出。
- 6 个长度 × 2 条路线 × 8 条 prompt，共 96 个组合，按组串行运行；提前 EOS 按实际长度统计。
- `gdr-lengths-*/summary.md` 汇总接受率、加速、普通 Decode / Draft / Verify 的 ms/call；
  `cases.csv` 每个组合一行。分项缺失显示 N/A。

### 约 1K 输入上下文

[内置长 prompt](../config/prompts_long_1k.json)共 4 条：中文摘要、跨段检索、约束规划、英文分析。
使用锁定的 Qwen tokenizer 和默认聊天模板，输入分别为 **983、998、1001、1008 token**；
运行报告的 `Input tokens` / `input_tokens` 会记录当前 tokenizer 的实际长度。
对应 ID：`long_zh_summary`、`long_zh_qa`、`long_zh_plan`、`long_en_analysis`。

```bash
"$MODEL_PYTHON" -B "$REPO_ROOT/tools/benchmark_gdr_lengths.py" \
  --run-dir "$AI_RUN_DIR" --runner "$CPP_RUNNER" \
  --runner-config "$RUNNER_CONFIG" --model-dir "$TARGET_DIR" \
  --chunk-deployment-manifest "$CHUNK_DEPLOYMENT_MANIFEST" \
  --mtp-deployment-manifest "$MTP_DEPLOYMENT_MANIFEST" \
  --prompts "$REPO_ROOT/config/prompts_long_1k.json" \
  --lengths 128 512 1024 --max-draft-tokens "$MAX_DRAFT_TOKENS" --device-id "$DEVICE_ID" \
  --low-memory --allow-output-differences
```

共 24 个组合；两条路线使用相同的完整输入。这里 1K 指输入长度，`--lengths` 指输出上限。
这组输入加 1024 输出可放入容量 2048；更换 tokenizer 后仍以启动前的容量检查为准。
只测“1K 跨段检索 × 输出 512 × MTP”：上面命令增加
`--prompt-id long_zh_qa --verify-gdr mtp`，并把 `--lengths 128 512 1024` 改成 `--lengths 512`；
可省略 `--chunk-deployment-manifest`。仍执行普通 / DFlash 各 3 次预热、10 次测量。
`benchmark_prompts.py` 也支持同一个 `--prompts` 和 `--prompt-id`，路线由所给部署清单决定。

## 查看文字、接受率和重新汇总

在环境配置中填写 `SAVED_BATCH`（已有 `runner-batch.json` 的路径），重新 `source` 后执行：

```bash
"$MODEL_PYTHON" -B "$REPO_ROOT/tools/decode_outputs.py" \
  --model-dir "$TARGET_DIR" --report "$SAVED_BATCH" --prompt-id zh_explain

"$MODEL_PYTHON" -B "$REPO_ROOT/tools/decode_outputs.py" \
  --report "$SAVED_BATCH" --acceptance-only

"$MODEL_PYTHON" -B "$REPO_ROOT/tools/benchmark_prompts.py" \
  --run-dir "$AI_RUN_DIR" --summarize-existing "$SAVED_BATCH" \
  --allow-output-differences
```

这些命令读取已保存报告，无需重跑模型。汇总另存为 `prompt-summary-*`。

## msprof

```bash
for MODE in ordinary dflash; do
  "$MODEL_PYTHON" -B "$REPO_ROOT/tools/profile_om.py" \
    --run-dir "$AI_RUN_DIR" --runner "$CPP_RUNNER" \
    --deployment-manifest "$DEPLOYMENT_MANIFEST" \
    --prompt-report "${SAVED_BATCH}.cases/zh_explain.json" \
    --profile-mode "$MODE" --profile-stage all --device-id "$DEVICE_ID" \
    --max-new-tokens "$MAX_NEW_TOKENS" --max-draft-tokens "$MAX_DRAFT_TOKENS" --profile-warmup 3
done
```

普通模式各采一次 prefill、decode；DFlash 各采一次 prefill、draft、verify。
单项将 `all` 改为 `decode`、`draft` 或 `verify`，并选择对应模式。
查看 `capture/all-stage-summary.csv` 和 `capture/all-operator-types.csv`。
OM Verify 已包含接受判断和状态提交；此采集是单阶段窗口，不是完整生成总时延。

## 导出与编译

<details>
<summary>首次部署：准备配置，导出 Chunk / MTP，构建 runner</summary>

先完成 [Torch-NPU 首次准备](DFLASH_RUN_AND_VALIDATE.md#首次准备)和 W8A8 配置。
还需匹配版本的 TorchAir、ATC、AscendCL、CMake/C++17；MTP 路线需要对应设备算子和 GE 注册。
在环境配置中填写实际 `ATC_BIN`、`SOC_VERSION` 和 `KV_CAPACITY=2048` 后重新 `source`。
新版普通模型与 DFlash 的 recurrent state 都改为 FP32，conv/KV 保持 FP16。
旧 FP16 状态 OM 不能仅靠改配置升级；需重新导出、编译并重建 runner。
下列命令使用新的 `artifacts-fp32` 目录；已有 `factory.json` 且容量足够、
`include_ordinary_decode=true`，可直接从第 1 步开始。

```bash
cd "$AI_RUN_DIR"

"$MODEL_PYTHON" -B "$REPO_ROOT/framework/scripts/lock_quant_inputs.py" \
  --target-dir "$TARGET_DIR" --draft-dir "$DRAFT_DIR" \
  --quant-config "$QUANT_CONFIG" --receiver-models-dir "$RECEIVER_MODELS_DIR" \
  --output "$AI_RUN_DIR/quant-input-manifest.json"

"$MODEL_PYTHON" -B - <<'PY'
import json, os
from pathlib import Path
run = Path(os.environ["AI_RUN_DIR"])
config = {
    "target_dir": os.environ["TARGET_DIR"],
    "draft_dir": os.environ["DRAFT_DIR"],
    "quant_config": os.environ["QUANT_CONFIG"],
    "input_manifest": str(run / "quant-input-manifest.json"),
    "receiver_models_dir": os.environ["RECEIVER_MODELS_DIR"],
    "max_sequence_length": int(os.environ["MAX_SEQUENCE_LENGTH"]),
    "include_ordinary_decode": True,
    "draft_attention_matmul_dtype": "float16",
    "dtype": "float16", "device": "npu:" + os.environ["DEVICE_ID"],
    "adn_rms_norm_ge_op_type": "AdnRmsNorm",
}
with (run / "factory.json").open("x") as f:
    json.dump(config, f, indent=2)
PY
```

下面四步分别执行，每步成功并返回命令提示符后再继续。
`export-air` 的 PASS 表示 AIR 导出完成；`compile-om` 的 PASS 和对应目录下的
`deployment-manifest.json` 才表示整套 OM 编译完成，单个 `.om` 文件不能代表整套完成。

**1. 导出 Chunk AIR**

```bash
"$MODEL_PYTHON" -B -m qwen35_dflash.ascend310p export-air \
  --factory qwen35_dflash.ascend310p.quant_factory:create_quant_incremental_graphs \
  --factory-config "$AI_RUN_DIR/factory.json" --verify-gdr chunk \
  --bundle-dir "$AI_RUN_DIR/artifacts-fp32"
```

**2. 编译 Chunk OM**

```bash
"$MODEL_PYTHON" -B -m qwen35_dflash.ascend310p compile-om \
  --air-manifest "$AI_RUN_DIR/artifacts-fp32/air-manifest.json" \
  --atc "$ATC_BIN" --soc-version "$SOC_VERSION"
```

**3. 只导出 MTP Verify，复用公共图**

```bash
"$MODEL_PYTHON" -B -m qwen35_dflash.ascend310p export-air \
  --factory qwen35_dflash.ascend310p.quant_factory:create_quant_incremental_graphs \
  --factory-config "$AI_RUN_DIR/factory.json" --verify-gdr mtp \
  --reuse-common-from "$AI_RUN_DIR/artifacts-fp32/deployment-manifest.json" \
  --bundle-dir "$AI_RUN_DIR/artifacts-fp32"
```

**4. 只编译 MTP Verify**

```bash
"$MODEL_PYTHON" -B -m qwen35_dflash.ascend310p compile-om \
  --air-manifest "$AI_RUN_DIR/artifacts-fp32/air-manifest-mtp.json" \
  --atc "$ATC_BIN" --soc-version "$SOC_VERSION"
```

完成后 `artifacts-fp32/om/` 只有 **5 个文件**：
`prefill.om`、`decode.om`、`draft.om`、`verify_chunk.om`、`verify_mtp.om`。
两个部署清单分别是同目录的 `deployment-manifest.json`（Chunk）和
`deployment-manifest-mtp.json`（MTP）。将环境配置的两个 manifest 路径改为这两个文件，重新 `source`，
完成第 5 步重建 runner 后，再执行上面的[双路线多长度测试](#多长度与双路线测试)。公共图不重复编译、不复制存储。

复用要求两次导出使用同一源码、权重、配置（仅 `verify_gdr` 不同）、工具链及编译选项；不匹配会报错。
只用 MTP 时可向空目录导出，省略 `--reuse-common-from`，随后编译该目录的 `air-manifest.json`。

遇到 `Common reuse export differs` 时，新版会打印具体字段。
旧版曾将审计信息中 JSON 的 list 与内存 tuple 误判为不一致，现已修复；
源码哈希、张量接口或配置确实不同时仍会拒绝复用。更新源码后，旧清单的源码哈希也会变化，
应在新目录用同一版本重新执行上述四步，不修改旧清单或覆盖已有 OM。

当前 ATC 输出会被收集，每张图结束后才写入 `$AI_RUN_DIR/log/dflash-atc/<本次编译目录>/<图名>.log`；
编译期间可能长时间没有终端输出，不能仅据此判断卡死。
编译默认给 Draft 设置 `--deterministic=1`；详见[漂移问题](DFLASH_CURRENT_USAGE_AND_RESULTS.md#deterministic-与-fc-漂移)。

中断后：对应路线已有 PASS 部署清单则无需重编。首次编译的 `om/` 为空，或增补 MTP 时尚未生成
`verify_mtp.om`，可重跑对应编译命令；若已有未完成的 OM，不会覆盖或自动续编，请保留原产物并使用新目录。

**5. 构建 runner**

```bash
"$MODEL_PYTHON" -B -m qwen35_dflash.ascend310p build-cpp \
  --build-dir "$AI_RUN_DIR/build/cpp" --ascendcl-root "$CANN_ROOT" \
  --output "$AI_RUN_DIR/reports/cpp-build.json"
```

创建 `runner.json`，填写真实设备和软件版本：

```bash
cat > "$RUNNER_CONFIG" <<'JSON'
{
  "device_model": "填写设备型号",
  "cann": "填写CANN版本",
  "driver": "填写驱动版本",
  "firmware": "填写固件版本",
  "runtime": "AscendCL C++",
  "pad_token_id": 0
}
JSON
```

</details>

<details>
<summary>已有 512 容量：扩容到 2048</summary>

复制已有配置，保留权重、量化和算子设置：

```bash
"$MODEL_PYTHON" -B - <<'PY'
import json, os
from pathlib import Path
run = Path(os.environ["AI_RUN_DIR"])
config = json.loads((run / "factory.json").read_text())
config["max_sequence_length"] = 2048
config["include_ordinary_decode"] = True
with (run / "factory-lengths.json").open("x") as f:
    json.dump(config, f, indent=2)
PY
```

分别执行上面的四步，将两条导出命令中的 `factory.json` 换为 `factory-lengths.json`，
所有 `artifacts-fp32` 换为尚未使用的 `artifacts-fp32-lengths`；
完成后更新环境配置中的两个 manifest 路径并重新 `source`。
容量变大可能增加显存和耗时，两条路线须使用同一容量重新比较。

</details>

<details>
<summary>只重编已有 Draft，开启 deterministic</summary>

```bash
"$MODEL_PYTHON" -B -m qwen35_dflash.ascend310p recompile-draft-om \
  --deployment-manifest "$DEPLOYMENT_MANIFEST" --atc "$ATC_BIN" \
  --output "${DEPLOYMENT_MANIFEST%/*}/deployment-manifest-deterministic.json"
```

新清单必须与原清单同目录、文件名未被使用；三个 Target OM 保持原文件。
把环境配置中的 `CHUNK_DEPLOYMENT_MANIFEST` 或 `MTP_DEPLOYMENT_MANIFEST` 改为新清单，重新 `source`。

</details>
