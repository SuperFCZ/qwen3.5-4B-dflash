# OM/C++ 使用

先按 [README](../README.md#环境配置)保存环境配置。新终端执行：

```bash
source /absolute/path/dflash-env.sh
cd "$AI_RUN_DIR"
```

配置里的 `VERIFY_GDR=chunk|mtp` 自动选择对应的 `DEPLOYMENT_MANIFEST`。
两个 manifest 填实际已编译路径；共用 Prefill、Decode、Draft，各自引用自己的 Verify。
Draft 采用 16/64 双档上下文：长输入建缓存用 64 行，生成用 16 行，共用一个 `draft.om`，自动选择档位。
首次部署见文末[导出与编译](#导出与编译)。`QUANT_MODE` 只控制原生推理，OM 精度由编译产物决定。

## 统一测试：短 / 1K 上下文、Chunk / MTP、多长度

默认 **8 条短 prompt + 12 条约 1K 输入 prompt**，每模式 **1 轮预热 + 3 轮测量**。
长输入包括 **6 条中文任务（含英译中）+ 6 条英文任务**，覆盖分析、检索、规划、数学、代码、翻译、抽取、比较、事件排序、规则和创作。
两种输入共用下面一条命令；投机始终开启。选择 `both` 时，**每个输出长度只测一次普通模型**：
低内存模式依次运行普通模型、Chunk、MTP，MTP 复用同一份普通模型输出和时延。每模式仍按指定轮数预热、测量。

使用当前 runner **1.11.0+**，按文末步骤在新目录导出、编译并重建 runner。
部署清单须包含 `draft_prefill_policy=single_draft16_64_gears` 和 `draft_context_gears=[16,64]`。
不要手改旧清单或覆盖已有 OM；单纯重编静态 Draft AIR 不能得到双档模型。
runner 默认预绑定 current/next 两套 I/O；报告中 `protocol.om_io_binding=prebound_ping_pong` 表示已启用。

```bash
"$MODEL_PYTHON" -B "$REPO_ROOT/tools/benchmark_gdr_lengths.py" \
  --run-dir "$AI_RUN_DIR" --runner "$CPP_RUNNER" \
  --runner-config "$RUNNER_CONFIG" --model-dir "$TARGET_DIR" \
  --chunk-deployment-manifest "$CHUNK_DEPLOYMENT_MANIFEST" \
  --mtp-deployment-manifest "$MTP_DEPLOYMENT_MANIFEST" \
  --verify-gdr both --prompt-group all --lengths 128 \
  --warmup 1 --repetitions 3 \
  --max-draft-tokens "$MAX_DRAFT_TOKENS" --device-id "$DEVICE_ID" \
  --low-memory --allow-output-differences
```

| 选择 | 修改参数 |
|---|---|
| 只测短 / 长输入 | `--prompt-group short` / `--prompt-group long` |
| 只测 Chunk / MTP | `--verify-gdr chunk` / `--verify-gdr mtp`；另一条清单可省略 |
| 单个 / 多个输出上限 | `--lengths 512` / `--lengths 128 512 1024` |
| 单条 prompt | `--prompt-id long_zh_code`；可重复参数，可与短 prompt ID 混选 |
| 恢复原测量次数 | `--warmup 3 --repetitions 10` |
| 只检查不运行 | 增加 `--plan-only` |

上面是 20 条 × 2 路线 × 1 长度，共 40 个组合；改成三个长度就是 120 个。
同一长度的两条路线共用普通基线，记录来源路径和哈希；Prefill/Decode OM 或接口不一致时提前报错。
单独选择 `chunk` 或 `mtp` 时，照常测一次普通基线和所选 DFlash 路线。
省略 `--lengths` 则测试 32、64、128、256、512、1024 六个输出上限。
两条路线使用相同完整输入，必须都能容纳输入加输出；内置长输入含聊天模板为 **978–1024 token**，
配 1024 输出需要容量 2048。运行时再用当前 tokenizer 检查，提前 EOS 按实际输出计数。

自定义输入仍可用 `--prompts /path/prompts.json`：
`[{"id":"my_case","prompt":"你的问题","category":"自定义","group":"long"}]`。
`group` 可省略；只有筛选 short/long 时才需要标注。
原 `benchmark_prompts.py` 入口仍支持这些输入选择和轮数参数，单个输出上限用 `--max-new-tokens`。

结果看 `gdr-lengths-*/summary.md` 和 `cases.csv`：整体及 short/long 分组的接受率、tok/s、加速比，
普通 **Prefill / Decode**、DFlash **Prefill / Draft / Verify** 的平均 ms/call 与累计平均 ms/次生成。
另列完整 Prefill / Decode 阶段耗时；DFlash Prefill 包含分段建缓存的 Draft 调用，
这些调用也计入 Draft 图累计耗时，两表不能重复相加；准备阶段候选不计入接受率。
统计均排除预热、模型加载和请求重置，缺失分项显示 N/A。
每组子目录的 `generations.txt` 保存文字输出，`runner-batch.json.cases/` 保存逐轮原始记录。
`--allow-output-differences` 接受跨模式输出差异；去掉则比较两模式第 0 次正式输出的 token/EOS。
多轮漂移独立记录为 `DRIFT_OBSERVED`，不导致测试失败；结果完整性、设备执行错误仍报错。

<details>
<summary>12 条长 prompt 的 ID 和实际输入长度</summary>

| ID | 类型 | 输入 token |
|---|---|---:|
| long_zh_summary | 中文摘要 | 983 |
| long_zh_qa | 跨段检索 | 998 |
| long_zh_plan | 约束规划 | 1001 |
| long_en_analysis | 英文分析 | 1008 |
| long_en_math | 英文数学核算 | 978 |
| long_zh_code | 代码修复 | 984 |
| long_zh_translate | 技术翻译 | 1004 |
| long_zh_extract | 结构化抽取 | 1024 |
| long_en_compare | 英文方案比较 | 980 |
| long_en_timeline | 英文事件排序 | 1022 |
| long_en_rules | 英文规则判断 | 1007 |
| long_en_story | 英文约束创作 | 987 |

长度使用锁定的 Qwen tokenizer 和默认聊天模板；[完整文本](../config/prompts_long_1k.json)。
1K 指输入上下文，`--lengths` 指输出上限。

</details>

## 离线开源测试集

直接读取 GPU 测试使用的 `gsm8k.jsonl`、`math500.jsonl`、`humaneval.jsonl` 等文件，
每行 `{"question":"完整问题"}`；也支持相同记录组成的 JSON 数组。不会联网下载数据。

```bash
"$MODEL_PYTHON" -B "$REPO_ROOT/tools/benchmark_gdr_lengths.py" \
  --run-dir "$AI_RUN_DIR" --runner "$CPP_RUNNER" \
  --runner-config "$RUNNER_CONFIG" --model-dir "$TARGET_DIR" \
  --chunk-deployment-manifest "$CHUNK_DEPLOYMENT_MANIFEST" \
  --mtp-deployment-manifest "$MTP_DEPLOYMENT_MANIFEST" \
  --dataset-dir /absolute/path/datasets --num-questions 10 \
  --verify-gdr both --lengths 128 --warmup 1 --repetitions 3 \
  --max-draft-tokens "$MAX_DRAFT_TOKENS" --device-id "$DEVICE_ID" \
  --low-memory --allow-output-differences
```

- `--num-questions 10`：**每个文件**取前 10 条；省略则读取全部，保持文件顺序。
- 只测指定文件：将 `--dataset-dir ...` 替换成 `--dataset-files /path/gsm8k.jsonl /path/math500.jsonl`。
- 文本字段为 `prompt` 时增加 `--dataset-field prompt`；默认使用 `question`，不把答案字段加入输入。
- 单路线、多长度、`--plan-only` 仍可用。离线文件替代内置 prompt，不与 `--prompts`、short/long 或 prompt ID 筛选混用。

每个**文件 × 路线 × 输出长度**单独汇总接受率、接受/提议数、每轮 token、吞吐、加速比，
以及普通 Prefill/Decode、DFlash Prefill/Draft/Verify 时延。
查看 `gdr-lengths-*/datasets.csv`，或 `datasets/<数据集ID>/summary.md`、`summary.json`；
逐题结果在 `cases.csv`，文件哈希和样本行号随报告保存。接受率是总接受数/总提议数，排除预热。
这些是推理效率指标，不是 GSM8K 正确率或 HumanEval pass@1。

同一长度、路线下所有文件复用一次模型加载；`both` 共用一次普通基线，不会为 MTP 重跑普通模型。
先用少量样本检查；文件格式和全部所选输入的容量会在设备运行前校验，不会静默跳过或截短问题。
`benchmark_prompts.py` 也支持上述离线参数，输出上限使用 `--max-new-tokens`。

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

只采集选中的 OM 阶段，不跑完整生成。下面定义一次快捷命令，参数依次为 **路线、模式、阶段**。
`--prompt-report` 改成实际存在的单条报告；示例使用短输入 `zh_explain`。

```bash
profile_om() {
  local manifest="$CHUNK_DEPLOYMENT_MANIFEST"
  if [[ "$1" == mtp ]]; then manifest="$MTP_DEPLOYMENT_MANIFEST"; fi
  "$MODEL_PYTHON" -B "$REPO_ROOT/tools/profile_om.py" \
    --run-dir "$AI_RUN_DIR" --runner "$CPP_RUNNER" \
    --deployment-manifest "$manifest" --verify-gdr "$1" \
    --prompt-report "${SAVED_BATCH}.cases/zh_explain.json" \
    --profile-mode "$2" --profile-stage "$3" --device-id "$DEVICE_ID" \
    --max-new-tokens 16 --max-draft-tokens 15 --profile-warmup 0
}
```

单项测试，选一行执行：

| OM | 命令 |
|---|---|
| Prefill | `profile_om chunk ordinary prefill` |
| Decode | `profile_om chunk ordinary decode` |
| Draft | `profile_om chunk dflash draft` |
| Verify Chunk | `profile_om chunk dflash verify` |
| Verify MTP | `profile_om mtp dflash verify` |

**采集五个主要 OM**：公共的 Prefill、Decode、Draft 各采一次，两条 Verify 分别采集。

```bash
(
  set -e
  profile_om chunk ordinary all
  profile_om chunk dflash draft
  profile_om chunk dflash verify
  profile_om mtp dflash verify
)
```

`all` 只展开当前模式：ordinary 为 Prefill/Decode，dflash 为 Prefill/Draft/Verify，**不会自动切换 Chunk/MTP**。
`--profile-warmup 0` 不预热；改为 `1` 则在窗口外预热一次。输出上限 `16` 用于设置 15 个候选的预算，不会跑满生成。
Decode/Draft/Verify 的窗口只含一次对应 OM 调用，必要的 Prefill、缓存和候选准备在窗口外执行。
Prefill 采集整个输入：含聊天模板不超过 64 token 时一次 OM 调用，约 1K 输入则约 16 次。
分段建缓存的 Draft 调用在 Draft/Verify 采集窗口外执行。
Draft 窗口采集首次正式调用：输入最后一块超过 16 token 时用 64 档，否则用 16 档；
完整生成的后续轮均用 16 档。比较时须使用相同输入和档位。

每次输出到终端打印的 `Output` 目录，查看 `capture/<stage>-stage-summary.csv`、
`capture/<stage>-operator-types.csv` 和 `capture/<stage>-hotspots.txt`，`<stage>` 为所选阶段或 `all`。
Verify 包含接受判断和图内状态提交；msprof 时延是采集窗口耗时，不是完整生成时延。

## 导出与编译

在环境配置中将 `OM_BUNDLE_DIR` 设为未使用的目录，重新 `source` 后执行第 1–5 步。
只有一个 Draft OM，复用现有 64 行特征缓冲区和 KV 缓冲区。
1024-token 输入需 15 次 Draft 准备调用，最后 64 行随首次正式 Draft 处理；后续投机用 16 行档位。
候选块始终为 16 行，最多输出 15 个候选。准备阶段仍执行完整 Draft，候选丢弃。
编译器自动为 Draft 添加 `--dynamic_dims="16;64"`，Target 三图保持静态；用户无需填写档位参数。
运行日志记录每图的 `weight_bytes/work_bytes`、`dynamic_control_bytes` 和设备可用内存。
单 OM 不保证峰值显存不变：CANN 需要档位控制缓冲区，多档权重布局与工作区也须实测。
当前双档的设备编译、显存、时延和接受率待验证；以同输入、同容量的实测决定是否采用。
模型结构与精度设置不变；档位可能改变算子选择，设备上的候选与有效 KV 仍需对照验证。
配置不再需要 `DRAFT_CONTEXT_ROWS`，已有 `factory.json` 中的 `draft_context_rows` 字段请删除。

<details>
<summary>首次部署：准备配置，导出 Chunk / MTP，构建 runner</summary>

先完成 [Torch-NPU 首次准备](DFLASH_RUN_AND_VALIDATE.md#首次准备)和 W8A8 配置。
还需匹配版本的 TorchAir、ATC、AscendCL、CMake/C++17；MTP 路线需要对应设备算子和 GE 注册。
在环境配置中填写实际 `ATC_BIN`、`SOC_VERSION` 和 `KV_CAPACITY=2048` 后重新 `source`。
普通模型与 DFlash 的 recurrent state 使用 FP32，conv/KV 保持 FP16。
下列命令输出到配置中的 `$OM_BUNDLE_DIR`（默认 `artifacts`）；已有 `factory.json` 且容量足够、
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
  --bundle-dir "$OM_BUNDLE_DIR"
```

**2. 编译 Chunk OM**

```bash
"$MODEL_PYTHON" -B -m qwen35_dflash.ascend310p compile-om \
  --air-manifest "$OM_BUNDLE_DIR/air-manifest.json" \
  --atc "$ATC_BIN" --soc-version "$SOC_VERSION"
```

**3. 只导出 MTP Verify，复用公共图**

```bash
"$MODEL_PYTHON" -B -m qwen35_dflash.ascend310p export-air \
  --factory qwen35_dflash.ascend310p.quant_factory:create_quant_incremental_graphs \
  --factory-config "$AI_RUN_DIR/factory.json" --verify-gdr mtp \
  --reuse-common-from "$OM_BUNDLE_DIR/deployment-manifest.json" \
  --bundle-dir "$OM_BUNDLE_DIR"
```

**4. 只编译 MTP Verify**

```bash
"$MODEL_PYTHON" -B -m qwen35_dflash.ascend310p compile-om \
  --air-manifest "$OM_BUNDLE_DIR/air-manifest-mtp.json" \
  --atc "$ATC_BIN" --soc-version "$SOC_VERSION"
```

完成后 `$OM_BUNDLE_DIR/om/` 有 **5 个文件**：
`prefill.om`、`decode.om`、`draft.om`、`verify_chunk.om`、`verify_mtp.om`。
前三个由 Chunk/MTP 共用，不重复存储；DFlash 运行时只加载 Prefill、Draft 和所选 Verify。
两个部署清单分别是同目录的 `deployment-manifest.json`（Chunk）和
`deployment-manifest-mtp.json`（MTP）。将环境配置的两个 manifest 路径改为这两个文件，重新 `source`，
完成第 5 步重建 runner 后，再执行上面的统一测试。公共图不重复编译、不复制存储。

复用要求两次导出使用同一源码、权重、配置（仅 `verify_gdr` 不同）、工具链及编译选项；不匹配会报错。
只用 MTP 时可向空目录导出，省略 `--reuse-common-from`，随后编译该目录的 `air-manifest.json`。

遇到 `Common reuse export differs` 时，新版会打印具体字段。
源码哈希、张量接口或配置确实不同时仍会拒绝复用。更新源码后，旧清单的源码哈希也会变化，
应在新目录用同一版本重新执行上述四步，不修改旧清单或覆盖已有 OM。

当前 ATC 输出会被收集，每张图结束后才写入 `$AI_RUN_DIR/log/dflash-atc/<本次编译目录>/<图名>.log`；
编译期间可能长时间没有终端输出，不能仅据此判断卡死。
Draft 默认 `--deterministic=0`（关闭）；已有 OM 不会随代码更新自动改变，需重编。
多轮输出不一致只记录 `DRIFT_OBSERVED`，继续统计接受率和时延；详见[漂移问题](DFLASH_CURRENT_USAGE_AND_RESULTS.md#deterministic-与-fc-漂移)。

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
在环境配置中将 `OM_BUNDLE_DIR` 改为尚未使用的 `$AI_RUN_DIR/artifacts-lengths` 并重新 `source`；
完成后更新环境配置中的两个 manifest 路径并重新 `source`。
容量变大可能增加显存和耗时，两条路线须使用同一容量重新比较。

</details>

<details>
<summary>只重编 Draft，选择 deterministic（默认关闭）</summary>

```bash
"$MODEL_PYTHON" -B -m qwen35_dflash.ascend310p recompile-draft-om \
  --deployment-manifest "$DEPLOYMENT_MANIFEST" --atc "$ATC_BIN" \
  --deterministic 0 \
  --output "${DEPLOYMENT_MANIFEST%/*}/deployment-manifest-det0.json"
```

新清单必须与原清单同目录、文件名未被使用；三个 Target OM 保持原文件。
只重编 Draft；此命令只改编译选项，不改变 AIR。
把环境配置中的 `CHUNK_DEPLOYMENT_MANIFEST` 或 `MTP_DEPLOYMENT_MANIFEST` 改为新清单，重新 `source`。
开启时将 `0` 改为 `1`，输出文件名也改为未使用的名称；省略参数默认为 `0`。
更新后按第 5 步重建 C++ runner，多轮漂移会显示变化轮数、token 差异数和首个差异位置，
标为 `PASS_WITH_OBSERVATIONS` 并保留各轮结果。接受率、吞吐和时延使用全部正式轮次；
普通模型与 DFlash 的输出比较仍由 `--allow-output-differences` 控制。

</details>
