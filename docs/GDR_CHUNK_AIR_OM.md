# OM/C++ 使用

已有 OM 和 runner：设置下面变量后直接运行。首次部署见文末[导出与编译](#导出与编译)。

```bash
export REPO_ROOT=/absolute/path/qwen3.5-4B-dflash
export AI_RUN_DIR=/absolute/path/run
export MODEL_PYTHON=/absolute/path/npu-env/bin/python
export TARGET_DIR=/absolute/path/Qwen3.5-4B
export CPP_RUNNER=/absolute/path/qwen35_dflash_acl_runner
export CHUNK_DEPLOYMENT_MANIFEST=/absolute/path/chunk/deployment-manifest.json
export MTP_DEPLOYMENT_MANIFEST=/absolute/path/mtp/deployment-manifest.json
export DEPLOYMENT_MANIFEST="$CHUNK_DEPLOYMENT_MANIFEST"
export PYTHONPATH="$REPO_ROOT/framework/python:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$AI_RUN_DIR"
cd "$AI_RUN_DIR"
```

先加载设备 CANN 环境。切换 MTP 时把 `DEPLOYMENT_MANIFEST` 指向
`"$MTP_DEPLOYMENT_MANIFEST"`；已有 OM 的路线由编译产物决定。

## 多 prompt 测试

```bash
"$MODEL_PYTHON" -B "$REPO_ROOT/tools/benchmark_prompts.py" \
  --run-dir "$AI_RUN_DIR" --runner "$CPP_RUNNER" \
  --deployment-manifest "$DEPLOYMENT_MANIFEST" \
  --runner-config "$AI_RUN_DIR/runner.json" --model-dir "$TARGET_DIR" \
  --max-new-tokens 128 --max-draft-tokens 15 --device-id 0 \
  --low-memory --allow-output-differences
```

默认 8 条 prompt，每模式 3 次预热 + 10 次测量，投机始终开启。
`--allow-output-differences` 用于当前允许输出差异的实验；去掉则严格比较 token/EOS。
`--low-memory` 先测全部普通模式，再测全部 DFlash。

常用调整：`--prompt-id math` 只测一条；`--max-new-tokens 512` 改输出上限；
`--prompts /path/prompts.json` 自定义输入，格式：
`[{"id":"my_case","prompt":"你的问题","category":"自定义"}]`。

结果在终端打印的 `prompt-suite-*`：`summary.md` 看速度/接受率，
`generations.txt` 看文字，`runner-batch.json.cases/` 保存每轮和分项计时。

## 多长度与双路线测试

```bash
"$MODEL_PYTHON" -B "$REPO_ROOT/tools/benchmark_gdr_lengths.py" \
  --run-dir "$AI_RUN_DIR" --runner "$CPP_RUNNER" \
  --runner-config "$AI_RUN_DIR/runner.json" --model-dir "$TARGET_DIR" \
  --chunk-deployment-manifest "$CHUNK_DEPLOYMENT_MANIFEST" \
  --mtp-deployment-manifest "$MTP_DEPLOYMENT_MANIFEST" \
  --lengths 32 64 128 256 512 1024 --max-draft-tokens 15 --device-id 0 \
  --low-memory --allow-output-differences
```

- 两套模型须有相同容量，并容纳 `prompt tokens + max_new_tokens`；短 prompt 套件建议 2048。
- 加 `--plan-only` 只检查清单、哈希和容量，不执行设备模型。旧 512 容量 OM 需扩容重导出。
- 6 个长度 × 2 条路线 × 8 条 prompt，共 96 个组合，按组串行运行；提前 EOS 按实际长度统计。
- `gdr-lengths-*/summary.md` 汇总接受率、加速、普通 Decode / Draft / Verify 的 ms/call；
  `cases.csv` 每个组合一行。分项缺失显示 N/A。

## 查看文字、接受率和重新汇总

```bash
export SAVED_BATCH=/absolute/path/prompt-suite-xxx/runner-batch.json

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
export MSPROF_BIN=/absolute/path/msprof
for MODE in ordinary dflash; do
  "$MODEL_PYTHON" -B "$REPO_ROOT/tools/profile_om.py" \
    --run-dir "$AI_RUN_DIR" --runner "$CPP_RUNNER" \
    --deployment-manifest "$DEPLOYMENT_MANIFEST" \
    --prompt-report "${SAVED_BATCH}.cases/zh_explain.json" \
    --profile-mode "$MODE" --profile-stage all --device-id 0 \
    --max-new-tokens 128 --max-draft-tokens 15 --profile-warmup 3
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
下列路径使用新的运行目录，已部署用户保留原配置。

```bash
export ATC_BIN=/absolute/path/atc
export SOC_VERSION=Ascend310P3    # 改为实际 SoC 型号
export MAX_SEQUENCE_LENGTH=2048
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
    "dtype": "float16", "device": "npu:0",
    "adn_rms_norm_ge_op_type": "AdnRmsNorm",
}
with (run / "factory.json").open("x") as f:
    json.dump(config, f, indent=2)
PY

(
  set -e
  for VERIFY_GDR in chunk mtp; do
    "$MODEL_PYTHON" -B -m qwen35_dflash.ascend310p export-air \
      --factory qwen35_dflash.ascend310p.quant_factory:create_quant_incremental_graphs \
      --factory-config "$AI_RUN_DIR/factory.json" --verify-gdr "$VERIFY_GDR" \
      --bundle-dir "$AI_RUN_DIR/artifacts-$VERIFY_GDR"
    "$MODEL_PYTHON" -B -m qwen35_dflash.ascend310p compile-om \
      --air-manifest "$AI_RUN_DIR/artifacts-$VERIFY_GDR/air-manifest.json" \
      --atc "$ATC_BIN" --soc-version "$SOC_VERSION"
  done
)

"$MODEL_PYTHON" -B -m qwen35_dflash.ascend310p build-cpp \
  --build-dir "$AI_RUN_DIR/build/cpp" --ascendcl-root "$CANN_ROOT" \
  --output "$AI_RUN_DIR/reports/cpp-build.json"

export CPP_RUNNER="$AI_RUN_DIR/build/cpp/qwen35_dflash_acl_runner"
export CHUNK_DEPLOYMENT_MANIFEST="$AI_RUN_DIR/artifacts-chunk/deployment-manifest.json"
export MTP_DEPLOYMENT_MANIFEST="$AI_RUN_DIR/artifacts-mtp/deployment-manifest.json"
export DEPLOYMENT_MANIFEST="$CHUNK_DEPLOYMENT_MANIFEST"
```

只需要一种路线，将 `for VERIFY_GDR in chunk mtp` 改为 `in chunk` 或 `in mtp`。
编译默认给 Draft 设置 `--deterministic=1`；详见[漂移问题](DFLASH_CURRENT_USAGE_AND_RESULTS.md#deterministic-与-fc-漂移)。

创建 `runner.json`，填写真实设备和软件版本：

```bash
cat > "$AI_RUN_DIR/runner.json" <<'JSON'
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

用上面的导出/编译循环，将 `factory.json` 换为 `factory-lengths.json`，
bundle 换为尚未使用的 `artifacts-lengths-$VERIFY_GDR`；完成后更新两个 manifest 变量。
容量变大可能增加显存和耗时，两条路线须使用同一容量重新比较。

</details>

<details>
<summary>只重编已有 Draft，开启 deterministic</summary>

```bash
"$MODEL_PYTHON" -B -m qwen35_dflash.ascend310p recompile-draft-om \
  --deployment-manifest "$DEPLOYMENT_MANIFEST" --atc "$ATC_BIN" \
  --output "${DEPLOYMENT_MANIFEST%/*}/deployment-manifest-deterministic.json"
export DEPLOYMENT_MANIFEST="${DEPLOYMENT_MANIFEST%/*}/deployment-manifest-deterministic.json"
```

新清单必须与原清单同目录、文件名未被使用；三个 Target OM 保持原文件。
运行矩阵时也要更新对应的 `CHUNK_DEPLOYMENT_MANIFEST` 或 `MTP_DEPLOYMENT_MANIFEST`。

</details>
