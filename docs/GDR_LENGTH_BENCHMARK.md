# 多长度、多 prompt、两种 GDR 的一次性对照

`tools/benchmark_gdr_lengths.py` 用于延续已有的 AIR/OM/C++ 128-token 测试。
一次命令运行多种生成长度，每个长度依次测 GDR Chunk 两遍和 GDR MTP；
默认是 **32、64、128、256、512、1024** 个新 token，以及现有 8 条 prompt，共 96 个组合。
每个组合都测普通生成与 DFlash，各 3 次预热、10 次正式测量。
Torch-NPU 直接推理的 `--verify-gdr` 参数见 [两条验证路径](GDR_VERIFY_ROUTES.md)。

## 1. 准备两套已编译模型与足够的 KV 容量

沿用 [部署手册](GDR_CHUNK_AIR_OM.md)的环境、runner 和量化输入。
两份清单分别对应 `chunk` 与 `mtp`，都需要包含 ordinary 的
`target_decode`，以及 DFlash 的 `target_prefill/target_verify/draft`。
如尚未编译 MTP，先按 [路线导出命令](GDR_VERIFY_ROUTES.md)生成它；
运行参数不能把已有 Chunk OM 变成 MTP。

生成 1024 个新 token 还要容纳 prompt。当前短 prompt 套件可使用
`max_sequence_length=2048`；自定义长 prompt 仍以容量预检为准。
如果已有 OM 只有 512 的逻辑 KV 容量，需要从新的 factory 配置重导出两套模型。
下面从已有配置复制，保留权重、量化、receiver 和其他设置：

```bash
cd "$AI_RUN_DIR"
"$MODEL_PYTHON" -B - <<'PY'
import json, os
from pathlib import Path
run = Path(os.environ["AI_RUN_DIR"])
config = json.loads((run / "factory.json").read_text())
config["max_sequence_length"] = 2048
config["include_ordinary_decode"] = True
with (run / "factory-lengths.json").open("x") as stream:
    json.dump(config, stream, indent=2)
    stream.write("\n")
PY

for VERIFY_GDR in chunk mtp; do
  "$MODEL_PYTHON" -B -m qwen35_dflash.ascend310p export-air \
    --factory qwen35_dflash.ascend310p.quant_factory:create_quant_incremental_graphs \
    --factory-config "$AI_RUN_DIR/factory-lengths.json" \
    --verify-gdr "$VERIFY_GDR" \
    --bundle-dir "$AI_RUN_DIR/artifacts-lengths-$VERIFY_GDR"
  "$MODEL_PYTHON" -B -m qwen35_dflash.ascend310p compile-om \
    --air-manifest "$AI_RUN_DIR/artifacts-lengths-$VERIFY_GDR/air-manifest.json" \
    --atc "$ATC_BIN" --soc-version "$SOC_VERSION"
done
```

导出/编译需要真实 CANN、TorchAir 和 MTP 算子环境；目录需未被占用。
这里只提供扩容命令，本地没有运行设备导出。若已有足够容量的清单，直接复用。
更大的 KV/attention 容量会改变模型显存和计算开销；本矩阵要求两条路线使用相同逻辑容量，
并记录新结果，不能把旧的 512-capacity 时延当作新模型的实测。

## 2. 一次测试 6 种长度与两条路线

保持 checkpoint、量化输入、Draft 确定性设置和 K 一致。
下面路径对应上一节的新 bundle；复用已有模型时换成实际清单路径：

```bash
export CHUNK_DEPLOYMENT_MANIFEST="$AI_RUN_DIR/artifacts-lengths-chunk/deployment-manifest.json"
export MTP_DEPLOYMENT_MANIFEST="$AI_RUN_DIR/artifacts-lengths-mtp/deployment-manifest.json"

"$MODEL_PYTHON" -B "$REPO_ROOT/tools/benchmark_gdr_lengths.py" \
  --run-dir "$AI_RUN_DIR" --runner "$CPP_RUNNER" \
  --runner-config "$AI_RUN_DIR/runner.json" --model-dir "$TARGET_DIR" \
  --chunk-deployment-manifest "$CHUNK_DEPLOYMENT_MANIFEST" \
  --mtp-deployment-manifest "$MTP_DEPLOYMENT_MANIFEST" \
  --lengths 32 64 128 256 512 1024 --max-draft-tokens 15 --device-id 0 \
  --low-memory --allow-output-differences
```

如需先检查清单、哈希、prompt token 和容量，在相同命令最后加 `--plan-only`。
它不会加载或执行设备模型。两条路线及全部长度会在第一项设备任务之前检查；
将 Chunk 清单误填为 MTP、缺少图、或者任一长度超出容量都会提前报错。

可用 `--lengths` 指定其他不重复的正整数。必须满足每条 prompt 的
`prompt_token_count + max_new_tokens <= capacity`，且逻辑容量按 64 对齐。
容量不足时会给出整个矩阵所需的最小 `max_sequence_length`；
不能只改运行参数扩大已编译 OM 的容量。

`max_new_tokens` 是上限。模型提前 EOS 时按实际长度统计，不强制继续生成。
比较时同时看实际 token 数、停止原因、模型耗时和吞吐率。

## 3. 调度与统计口径

- 按传入长度的顺序执行，每个长度先 Chunk 后 MTP；同一时刻只有一个 C++ 测试进程。
- 每个“长度 × 路线”复用该组内全部 prompt 的模型，组间退出进程并释放显存。
  默认 12 组会分别加载模型，加载时间单独记录，不进入模型循环加速比。
- 投机始终开启。零接受后提交 anchor，后续轮继续 Draft＋Verify。
- `--allow-output-differences` 允许记录各自稳定但跨模式不同的输出；
  普通/DFlash 独立 3+10 检查、真实设备标记、计数及报告校验仍执行。
  去掉此参数恢复严格 token/EOS 对照。
- 一组失败会保留其报告，并继续剩余组合；每完成一组就更新总汇总。
  Ctrl-C 会保存已经完成的组合；重新启动会生成新的目录，不覆盖旧证据。

可附加 `--prompt-id math --prompt-id zh_plan` 先测指定 prompt；
也可用 `--prompts "$AI_RUN_DIR/custom-prompts.json"` 指定现有格式的 prompt JSON 列表。

## 4. 输出和时延

新目录 `$AI_RUN_DIR/gdr-lengths-*/` 包含：

| 文件 | 内容 |
|---|---|
| `request.json` | 长度、prompt tokens、两套 ABI/OM 哈希、ATC 命令、运行设置 |
| `summary.md` | 每个长度/路线的接受率、tokens/round、吞吐率、相对普通模型加速、decode/Draft/Verify 时延 |
| `summary.json` | 全部统计、各 prompt 实际长度/停止原因、按生成位置的接受率、两路线匹配 prompt 对照 |
| `cases.csv` | 每个长度/路线/prompt 一行，便于画图和筛选慢例 |
| `chunk-128/prompt-suite-*/` 等 | 原始 request、runner 日志、case 报告、summary 和解码后的 `generations.txt` |

接受率采用 `sum(accepted) / sum(proposed)`，不是各 prompt 百分比的平均值。
相对普通模型的整体加速比采用 `sum(ordinary model_ms) / sum(dflash model_ms)`。
MTP/Chunk 对照只使用两边均通过所选检查的相同 prompt，并同时列出耗时比和吞吐比；
不同 EOS 长度可能使两者不同。未通过检查的组合不会被填成 0% 或纳入性能均值。

分项只读取原始 `measurements[].stage_ms`，排除 warmup：

| 汇总列 | 数据源 | 含义 |
|---|---|---|
| Decode ms/call | ordinary 的 `target_decode` | 普通模型单 token 图调用 |
| Draft ms/call | dflash 的 `draft` | Draft 整图调用；长 prompt 时也包括中间 prefill 的 Draft KV 初始化 |
| Verify ms/call | dflash 的 `target_verify` | 所选 Chunk 或 MTP 的验证与图内接受/状态提交 |

这里是同步 OM 调用的墙钟时间，包含必要的数据绑定、小数据传输和同步，
不是单个 GDR kernel 的时间。均值按实际调用次数加权；缺少计时数组显示 N/A。
旧的总时延不能反推出这些分项。完整计时边界见
[现有结果的时延说明](DFLASH_CURRENT_USAGE_AND_RESULTS.md#43-不重新跑模型提取已有分项计时)。

## 5. 当前结果范围

目前已有的 128-token 设备结果仍为 Chunk：8 个 prompt 的加权接受率 20.69%，
整体模型循环加速 1.50075×。这次增加的是对照工具，没有补造新的长度或 MTP 设备数据。
主机测试覆盖长度/路线调度、容量提前检查、失败后继续、EOS 实际长度计数、
计时数组处理和 fake-ACL 不被当成设备结果。
长输出是否降低接受率、MTP 是否更快，需要运行上述命令后按各 prompt 的分段记录判断。
