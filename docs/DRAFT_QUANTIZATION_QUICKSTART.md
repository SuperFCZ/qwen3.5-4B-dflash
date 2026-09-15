# Draft 量化：内网运行指南

本分支为 `feature/draft-quantization`，代码基线是 `gdr-chunk-verify` 的 `8c44da7`。
文档沿用该分支的“外置配置 → Torch-NPU → AIR/OM/C++”组织方式；以下命令对应本量化分支。
CPU 真实权重对照和主机回归已通过，Torch-NPU、ATC 和 310P OM 结果由内网执行后确认。

## 1. Clone 和配置

```bash
git clone --branch feature/draft-quantization --single-branch \
  https://github.com/day8reak/qwen3.5-4B-dflash.git
cd qwen3.5-4B-dflash
cp -n config/draft_quantization_env.sh.example /absolute/path/draft-quant-env.sh
vi /absolute/path/draft-quant-env.sh
source /absolute/path/draft-quant-env.sh
```

先替换命令中的外置配置路径，再填写配置里的 Python、CANN、receiver、Target、Draft 和输出目录。
沿用内网 `gdr-chunk-verify` 的设备环境和已有自定义算子；Python 3.10，
`transformers==5.14.1`，PyTorch/torch_npu/TorchAir 与 CANN 配套。
receiver 须含 `models/export_model_wrapper_qwen3_5.py`；配置会补齐搜索路径。
原 GDR 须支持 `[B] INT16 effective_length`，本量化实现没有新增自定义算子。

每次打开 Bash 终端重新 `source`，并在本仓库目录执行后续命令。`AI_RUN_DIR` 使用新的外部目录；
权重、缓存、构建、日志和 OM 都放源码之外。这里可独立 clone 运行，无需本机 workspace 工具。

## 2. 准备 Draft 权重

| `DRAFT_VARIANT` | `DRAFT_DIR` 对应权重 |
| --- | --- |
| `w8a16` | `naveenrajk/Qwen3.5-4B-DFlash-W8A16`，revision `b4bf9f0f7f79bd8589d34b457e136f644c122180` |
| `w4a16` | `nota-ai/Qwen3.5-4B-DFlash-GPTQ-W4A16`，revision `c5fb290e47e30c81d06e48b0495ec06f2560dd4e` |
| `fp16` | 原六层 Draft，revision 和路径要求见 [量化说明](DRAFT_QUANTIZATION.md) |

在能访问 Hugging Face 的机器上，从本分支源码目录下载，之后把整个输出目录拷进内网。
下载终端无需加载 CANN 配置；使用外部下载目录，workspace 用户使用分配的 run：

```bash
export AI_RUN_DIR=/absolute/path/draft-download-run
python3 -B -m models.dflash_v1.prepare_draft \
  --draft-quantization w8a16 --output "$AI_RUN_DIR/checkpoints/w8a16"
python3 -B -m models.dflash_v1.prepare_draft \
  --draft-quantization w4a16 --output "$AI_RUN_DIR/checkpoints/w4a16"
```

下载命令使用装有 PyTorch、Transformers 和 safetensors 的 Python，输出目录须尚不存在。
内网已有相同 revision 的 `config.json`、`model.safetensors` 时可直接指定该目录；加载会校验哈希。
两种量化 Draft 均为五层，配置和 feature 层自动匹配，不需要手工改模型或重新量化。

## 3. Torch-NPU 验证

先检查设备，再执行小规模及 32-token 对照；命令会保留完整日志：

```bash
set -o pipefail
git rev-parse HEAD > "$AI_RUN_DIR/reports/commit.txt"
npu-smi info | tee "$AI_RUN_DIR/reports/npu-smi.txt"
"$MODEL_PYTHON" -B -c 'import torch, torch_npu, transformers; print(torch.__version__, torch_npu.__version__, transformers.__version__); assert torch.npu.is_available()' \
  2>&1 | tee "$AI_RUN_DIR/reports/python-env.txt"

"$MODEL_PYTHON" -B -m models.dflash_v1.run_npu \
  "${NPU_ARGS[@]}" "${QUANT_ARGS[@]}" \
  --execution-mode validate --block-size 2 --max-new-tokens 4 \
  --report "$AI_RUN_DIR/reports/npu-smoke.json" \
  2>&1 | tee "$AI_RUN_DIR/log/npu-smoke.log"

"$MODEL_PYTHON" -B -m models.dflash_v1.run_npu \
  "${NPU_ARGS[@]}" "${QUANT_ARGS[@]}" \
  --execution-mode validate --block-size 16 --max-new-tokens 32 \
  --report "$AI_RUN_DIR/reports/npu-validate.json" \
  2>&1 | tee "$AI_RUN_DIR/log/npu-validate.log"
```

每一步成功后再继续。报告应有 `correctness_gate.status=PASS`、
`strict_greedy_exact_match=true`、`operator_fallback_enabled=false`，且实际进入 Draft/verify。
`INCONCLUSIVE_NO_DRAFT_ROUND` 不算通过。`QUANT_MODE` 只控制 Target，Draft 由
`DRAFT_VARIANT` 独立选择；测 Target W8A8 时填写原三字段 YAML 并设 `QUANT_MODE=enable`。
性能测试沿用 [NPU benchmark](DFLASH_RUN_AND_VALIDATE.md#7-npu-性能基准与-msprof)，
两组命令都加入 `--draft-quantization "$DRAFT_VARIANT"`。

## 4. AIR → ATC → OM/C++

此路径使用 **Target W8A8 + 所选 Draft**，需要 `QUANT_CONFIG` 的原量化产物。
OM 基线为固定 gear 的全前缀重算；使用本分支重新构建的 runner 1.1.0。
先生成对应版本的输入锁，再从模板生成 factory 配置：

```bash
"$MODEL_PYTHON" -B framework/scripts/lock_quant_inputs.py \
  --target-dir "$TARGET_DIR" --draft-dir "$DRAFT_DIR" \
  --quant-config "$QUANT_CONFIG" --receiver-models-dir "$RECEIVER_MODELS_DIR" \
  --output "$AI_RUN_DIR/quant-input-manifest.json" \
  2>&1 | tee "$AI_RUN_DIR/log/input-lock.log"

"$MODEL_PYTHON" -B - <<'PY'
import json, os
from pathlib import Path
root, run = Path(os.environ['REPO_ROOT']), Path(os.environ['AI_RUN_DIR'])
cfg = json.loads((root / 'config/quant_air_om_factory.example.json').read_text())
for key, env in [('target_dir', 'TARGET_DIR'), ('draft_dir', 'DRAFT_DIR'),
                 ('draft_quantization', 'DRAFT_VARIANT'), ('quant_config', 'QUANT_CONFIG'),
                 ('receiver_models_dir', 'RECEIVER_MODELS_DIR')]:
    cfg[key] = os.environ[env]
cfg.update(input_manifest=str(run / 'quant-input-manifest.json'),
           max_sequence_length=64, device='npu:' + os.environ['DEVICE_ID'])
with (run / 'factory.json').open('x') as stream:
    json.dump(cfg, stream, indent=2)
PY
cp -n config/quant_air_om_runner.example.json "$AI_RUN_DIR/runner.json"
vi "$AI_RUN_DIR/runner.json"
```

将 runner JSON 中的 `device_model`、`cann`、`driver`、`firmware` 填为真实值。
确认配置中的 `SOC_VERSION` 为实际精确型号，`CANN_ROOT` 含 AscendCL 头文件与库。
下面的短 raw prompt 使用 gear 64；换长 prompt 时先调大 factory 的 `max_sequence_length`
至 64 的倍数，使其覆盖 prompt、输出和 proposal 空间，并重新编译。

```bash
"$MODEL_PYTHON" -B -m qwen35_dflash.ascend310p build-cpp \
  --build-dir "$AI_RUN_DIR/build/cpp" --ascendcl-root "$CANN_ROOT" \
  --output "$AI_RUN_DIR/reports/cpp-build.json" \
  2>&1 | tee "$AI_RUN_DIR/log/cpp-build.log"

"$MODEL_PYTHON" -B -m qwen35_dflash.ascend310p probe-pytorch \
  --factory-config "$AI_RUN_DIR/factory.json" --draft-quantization "$DRAFT_VARIANT" \
  --input-token-ids 1,2,3,4 --output "$AI_RUN_DIR/reports/pytorch-probe.json" \
  2>&1 | tee "$AI_RUN_DIR/log/pytorch-probe.log"

"$MODEL_PYTHON" -B -m qwen35_dflash.ascend310p run-e2e-cpp \
  --factory-config "$AI_RUN_DIR/factory.json" --draft-quantization "$DRAFT_VARIANT" \
  --bundle-dir "$AI_RUN_DIR/om" --atc "$ATC_BIN" --soc-version "$SOC_VERSION" \
  --runner "$AI_RUN_DIR/build/cpp/qwen35_dflash_acl_runner" \
  --runner-config "$AI_RUN_DIR/runner.json" --model-dir "$TARGET_DIR" \
  --prompt "$PROMPT" --max-new-tokens 32 --max-draft-tokens 15 --device-id "$DEVICE_ID" \
  --report-dir "$AI_RUN_DIR/reports/om" \
  2>&1 | tee "$AI_RUN_DIR/log/om-e2e.log"
```

最后一条依次执行真机预检、AIR 导出、ATC 编译和 C++ ordinary/DFlash 各 3+10 次生成。
`reports/om/summary.json` 应为 `status=PASS`、`cpu_fallback=false`，
`ordinary_parity` 的 token/EOS mismatch 均为 0。失败时停在首个失败阶段，保留完整日志。
分阶段命令见 [AIR/OM 指南](QUANT_AIR_OM_FRAMEWORK.md)。

量化 bundle 包含 `.om`、manifest、72 个只读压缩权重/scale 输入及 `constant-inputs.tsv`，
运行时自动加载一次。移动产物请保留整个 `om/` 目录。
切换 `w8a16`/`w4a16` 时同步改 Draft 目录和新的 `AI_RUN_DIR`，重新 `source`，然后分别验证、导出和编译。

## 5. 回传结果

每个版本回传 `reports/`、`log/`、`factory.json`、`runner.json`、
`quant-input-manifest.json`、`om/air-manifest.json` 和 `om/deployment-manifest.json`（存在的部分即可）。
附执行命令；失败时给首个报错和完整堆栈。暂时无需回传权重、大型 AIR 或 OM。
先据此确认正确性和兼容性，再分析接受率、时延和峰值内存。
