# DFlash 测试结果与已知问题

运行命令见 [统一测试](GDR_CHUNK_AIR_OM.md#统一测试)。以下均为用户提供的 Chunk 运行结果，任务质量未评估。

**表中加速比按模型生成总时延计算，包含 Prefill + Decode 循环：**
`sum(普通 Prefill + 普通 Decode) / sum(DFlash Prefill + DFlash Decode)`。
DFlash Decode 包含 Draft、Verify、提交与循环调度。它不计模型加载、分词、请求重置、
预热、文本解码和结果写文件，因此不是从输入文本到返回文本的完整端到端时延。
仅 Decode 的加速比需另算 `sum(普通 Decode) / sum(DFlash Decode)`。

## 开源数据集：输出上限 512 token

**FP16 Draft、Chunk 验证、thinking 开启；每个模式/每题 0 次预热、1 次测量。**
5 个文件共 **2563 / 2563 条问题**完成，文件级状态均为 `PASS_WITH_DIFFERENCES`。
512 是输出上限，EOS 可提前结束；吞吐按实际生成 token 计算。

| 数据集文件 | 完成 / 选择 | 接受 / 提出 | 接受率 | token/投机轮 | 普通 tok/s | DFlash tok/s | 加速比 |
|---|---:|---:|---:|---:|---:|---:|---:|
| gsm8k.jsonl | 1319 / 1319 | 540146 / 1864893 | 28.96% | 5.28 | 28.54 | 72.92 | 2.59× |
| humaneval.jsonl | 164 / 164 | 64701 / 284489 | 22.74% | 4.37 | 28.41 | 59.68 | 2.07× |
| math500.jsonl | 500 / 500 | 209575 / 683780 | 30.65% | 5.53 | 28.53 | 76.16 | 2.67× |
| mbpp.jsonl | 500 / 500 | 194904 / 904058 | 21.56% | 4.19 | 28.62 | 58.48 | 2.04× |
| mtbench.jsonl | 80 / 80 | 30994 / 145503 | 21.30% | 4.15 | 28.53 | 57.46 | 2.02× |
| **全部** | **2563 / 2563** | **1040320 / 3882723** | **26.79%** | **4.96** | **28.54** | **68.61** | **2.42×** |

接受率 = 总接受数 / 总提出数；加速比 = 普通总耗时 / DFlash 总耗时，不平均各文件的百分比或倍数。
加速比使用 Prefill + Decode 的模型生成总时延（口径见上文）。两模式各自生成输出，
实际 token 数可不同，吞吐比与时间加速比不一定相同。
输出一致性仍未通过，任务质量未评估；单次测量未验证多轮稳定性。

### 阶段与 OM 时延

普通 Decode **34.85 ms/次**，Draft **19.80 ms/次**，Verify **51.35 ms/次**。

**整体阶段耗时：平均 ms/次生成。**

| 普通 Prefill | 普通 Decode 循环 | DFlash Prefill | DFlash Decode 循环 |
|---:|---:|---:|---:|
| 117.92 | 17787.30 | 130.71 | 7278.87 |

**整体 OM 调用耗时：平均 ms/次调用 / 累计平均 ms/次生成。**

| 普通 Target Prefill | 普通 Target Decode | DFlash Target Prefill | Draft | Verify |
|---:|---:|---:|---:|---:|
| 74.80 / 117.84 | 34.85 / 17775.07 | 75.17 / 118.42 | 19.80 / 2036.55 | 51.35 / 5250.75 |

<details>
<summary>各数据集阶段耗时与 OM 调用明细</summary>

**阶段耗时：平均 ms/次生成。**

| 数据集文件 | 普通 Prefill | 普通 Decode 循环 | DFlash Prefill | DFlash Decode 循环 |
|---|---:|---:|---:|---:|
| gsm8k.jsonl | 118.08 | 17816.29 | 130.86 | 6796.74 |
| humaneval.jsonl | 210.48 | 17564.14 | 249.54 | 8324.82 |
| math500.jsonl | 129.50 | 17815.70 | 145.60 | 6572.40 |
| mbpp.jsonl | 74.03 | 17751.00 | 74.49 | 8680.01 |
| mtbench.jsonl | 127.27 | 17816.20 | 142.81 | 8742.21 |

**OM 调用耗时：平均 ms/次调用 / 累计平均 ms/次生成。**

| 数据集文件 | 普通 Target Prefill | 普通 Target Decode | DFlash Target Prefill | Draft | Verify |
|---|---:|---:|---:|---:|---:|
| gsm8k.jsonl | 74.86 / 118.00 | 34.85 / 17803.11 | 75.21 / 118.55 | 19.81 / 1902.63 | 51.35 / 4902.68 |
| humaneval.jsonl | 75.17 / 210.39 | 34.84 / 17553.17 | 75.50 / 211.30 | 19.81 / 2352.93 | 51.34 / 6005.67 |
| math500.jsonl | 74.91 / 129.44 | 34.84 / 17804.24 | 75.29 / 130.10 | 19.80 / 1842.87 | 51.36 / 4741.65 |
| mbpp.jsonl | 73.99 / 73.99 | 34.85 / 17739.91 | 74.45 / 74.45 | 19.79 / 2414.18 | 51.33 / 6261.54 |
| mtbench.jsonl | 74.82 / 127.20 | 34.84 / 17805.22 | 75.23 / 127.89 | 19.80 / 2446.29 | 51.32 / 6306.30 |

</details>

DFlash Prefill 阶段包含 Target Prefill 和构建上下文的调用；这些调用也计入对应的 OM 累计耗时，
两张表不能相加。Verify 包含提交操作；OM 时延是同步图调用耗时，不是单个算子时延。

## 自定义 prompt：输出上限 128 token

**FP16 Draft、Chunk 验证、thinking 开启；每个模式/每条 prompt 0 次预热、1 次测量。**
每轮最多提出 15 个候选；20 / 20 条完成，两种模式均各生成 128 token。
全部用例最终状态为 `PASS_WITH_DIFFERENCES`：输出一致性仍为 `FAIL`，任务质量未评估；
单次测量未验证重复运行的稳定性。

### 短 prompt

| Prompt | 输入 token | 接受 / 提出 | 接受率 | token/投机轮 | DFlash tok/s | 加速比 |
|---|---:|---:|---:|---:|---:|---:|
| zh_explain | 28 | 101 / 391 | 25.83% | 4.70 | 64.33 | 2.26× |
| zh_plan | 29 | 63 / 944 | 6.67% | 1.95 | 27.30 | 0.96× |
| math | 47 | 107 / 289 | 37.02% | 6.35 | 85.50 | 3.01× |
| code | 46 | 100 / 416 | 24.04% | 4.54 | 61.98 | 2.18× |
| translate | 59 | 108 / 290 | 37.24% | 6.35 | 85.43 | 3.00× |
| summary | 59 | 91 / 517 | 17.60% | 3.43 | 47.33 | 1.66× |
| en_explain | 34 | 99 / 380 | 26.05% | 4.54 | 62.01 | 2.18× |
| creative | 42 | 98 / 436 | 22.48% | 4.23 | 57.98 | 2.04× |

### 约 1K 上下文 prompt

12 条，输入为 978～1024 token；6 条中文、6 条英文任务。

| Prompt | 输入 token | 接受 / 提出 | 接受率 | token/投机轮 | DFlash tok/s | 加速比 |
|---|---:|---:|---:|---:|---:|---:|
| long_zh_summary | 983 | 86 / 565 | 15.22% | 3.02 | 28.38 | 1.25× |
| long_zh_qa | 998 | 97 / 411 | 23.60% | 4.23 | 34.97 | 1.54× |
| long_zh_plan | 1001 | 94 / 484 | 19.42% | 3.85 | 33.03 | 1.45× |
| long_en_analysis | 1008 | 93 / 487 | 19.10% | 3.63 | 31.87 | 1.40× |
| long_en_math | 978 | 99 / 401 | 24.69% | 4.38 | 35.61 | 1.56× |
| long_zh_code | 984 | 98 / 400 | 24.50% | 4.23 | 34.92 | 1.53× |
| long_zh_translate | 1004 | 99 / 392 | 25.26% | 4.54 | 36.36 | 1.60× |
| long_zh_extract | 1024 | 58 / 982 | 5.91% | 1.81 | 19.68 | 0.87× |
| long_en_compare | 980 | 111 / 243 | 45.68% | 7.47 | 46.73 | 2.06× |
| long_en_timeline | 1022 | 99 / 424 | 23.35% | 4.38 | 35.60 | 1.57× |
| long_en_rules | 1007 | 107 / 299 | 35.79% | 6.05 | 42.30 | 1.86× |
| long_en_story | 987 | 98 / 398 | 24.62% | 4.38 | 35.64 | 1.57× |

### 汇总

| 输入组 | 完成 / 选择 | 接受 / 提出 | 加权接受率 | token/投机轮 | 普通 tok/s | DFlash tok/s | 加速比 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 短输入 | 8 / 8 | 767 / 3663 | 20.94% | 3.98 | 28.45 | 54.71 | 1.92× |
| 约 1K 输入 | 12 / 12 | 1139 / 5486 | 20.76% | 3.88 | 22.74 | 33.17 | 1.46× |
| 全部 | 20 / 20 | 1906 / 9149 | 20.83% | 3.92 | 24.72 | 39.37 | 1.59× |

接受率 = 总接受候选 / 总提出候选；分组吞吐 = 实际生成 token 总数 / 总耗时，
加速比 = 普通模型生成总耗时 / DFlash 模型生成总耗时，包含 Prefill + Decode（口径见上文）。
`token/投机轮` 不计 Prefill 和仅执行 Target 的轮次。两种模式各自生成输出，性能数据不代表输出一致性通过。

### 阶段与 OM 时延

**阶段耗时：平均 ms/次生成。**

| 输入组 | 普通 Prefill | 普通 Decode 循环 | DFlash Prefill | DFlash Decode 循环 |
|---|---:|---:|---:|---:|
| 短输入 | 74.06 | 4424.66 | 74.73 | 2264.74 |
| 约 1K 输入 | 1199.86 | 4429.27 | 1528.90 | 2329.47 |
| 全部 | 749.54 | 4427.42 | 947.23 | 2303.58 |

**OM 调用耗时：平均 ms/次调用 / 累计平均 ms/次生成。**

| 输入组 | 普通 Target Prefill | 普通 Target Decode | DFlash Target Prefill | Draft | Verify |
|---|---:|---:|---:|---:|---:|
| 短输入 | 74.00 / 74.00 | 34.82 / 4421.51 | 74.66 / 74.66 | 19.82 / 631.84 | 51.18 / 1631.28 |
| 约 1K 输入 | 74.96 / 1199.41 | 34.85 / 4426.06 | 75.64 / 1210.25 | 20.27 / 967.70 | 51.24 / 1678.03 |
| 全部 | 74.92 / 749.24 | 34.84 / 4424.24 | 75.60 / 756.02 | 20.13 / 833.36 | 51.21 / 1659.33 |

DFlash Prefill 阶段包含 Target Prefill 和构建上下文的调用；Draft 累计耗时包含 Prefill 和生成阶段的调用，
阶段与 OM 表不能相加。Verify 包含提交操作；OM 时延是同步图调用耗时，不是单个算子时延。

长输入的 DFlash Prefill 平均多 **329.04 ms**，Decode 循环少 **2099.80 ms**，整体加速 **1.46×**。
`zh_plan` 和 `long_zh_extract` 分别只有 **1.95 / 1.81 token/投机轮**，整体加速比分别为 **0.96× / 0.87×**。

<details>
<summary>按生成位置统计的接受率</summary>

| Prompt | [0, 32) | [32, 64) | [64, 96) | [96, 128) |
|---|---:|---:|---:|---:|
| zh_explain | 50.00% | 26.67% | 15.33% | 26.37% |
| zh_plan | 6.25% | 5.88% | 4.56% | 12.20% |
| math | 37.33% | 29.52% | 34.67% | 64.71% |
| code | 25.71% | 22.86% | 17.04% | 36.62% |
| translate | 50.00% | 100.00% | 22.86% | 25.26% |
| summary | 20.00% | 13.33% | 14.67% | 28.05% |
| en_explain | 51.67% | 37.33% | 12.73% | 23.75% |
| creative | 35.56% | 14.17% | 23.33% | 19.81% |
| long_zh_summary | 16.30% | 17.78% | 12.78% | 14.78% |
| long_zh_qa | 23.81% | 13.33% | 33.33% | 37.88% |
| long_zh_plan | 34.67% | 12.78% | 15.56% | 25.53% |
| long_en_analysis | 36.00% | 13.33% | 14.67% | 22.68% |
| long_en_math | 40.00% | 14.81% | 18.52% | 42.86% |
| long_zh_code | 30.00% | 33.33% | 19.05% | 18.26% |
| long_zh_translate | 34.67% | 23.33% | 33.33% | 16.82% |
| long_zh_extract | 8.10% | 5.88% | 3.49% | 7.43% |
| long_en_compare | 40.00% | 40.00% | 56.00% | 45.45% |
| long_en_timeline | 35.56% | 17.14% | 13.33% | 42.19% |
| long_en_rules | 28.89% | 44.00% | 35.00% | 36.49% |
| long_en_story | 55.00% | 19.05% | 16.30% | 23.47% |

每个完整投机轮按其首个输出 token 的索引归入区间，跨区间的轮次不拆分；
不计 Prefill 和仅执行 Target 的轮次。表中百分比不能直接平均为总体接受率。

</details>

## 自定义测试：FP16 / W4A16 / W8A16，输出上限 128 token

### 早期解量化实现

来自 `gdr-lengths-isajibrf/summary.json`：Chunk、thinking 开启，0 次预热、1 次测量，
20 条自定义问题。旧日志说明为 group 解量化标准算子 + FP16 MatMul，量化权重常驻。
这轮与后面的 CANN 原生 WeightQuant 结果分开保留，不覆盖历史数据。

| Draft | 完成 / 选择 | 接受率 | token/投机轮 | 普通 tok/s | DFlash tok/s | 加速比 vs 普通 | Draft ms/call | Verify ms/call |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FP16 | 20 / 20 | 20.83% | 3.92 | 24.72 | 39.37 | 1.59× | 20.13 | 51.21 |
| W4A16 | 20 / 20 | 18.61% | 3.60 | 24.72 | 7.85 | 0.32× | 310.08 | 51.25 |
| W8A16 | 20 / 20 | 21.03% | 3.93 | 24.72 | 9.06 | 0.37× | 283.63 | 51.28 |

相对同轮 FP16，W4/W8 的接受率分别变化 **-2.22 / +0.20 个百分点**，
吞吐比和模型生成时间加速比均为 **0.20× / 0.23×**。输出状态为 `PASS_WITH_DIFFERENCES`。

### 原生 WeightQuantBatchMatmulV2 实现

来自 `gdr-lengths-waw8h36d/summary.json`，Chunk 路线，短输入 8 条、约 1K 输入 12 条。
本表保留用户新一轮的原始汇总值，与上面的 `isajibrf` 旧运行分别记录。

| Draft | 分组 | 完成 / 选择 | 接受率 | token/投机轮 | 普通 tok/s | DFlash tok/s | 加速比 | Draft ms/call | Verify ms/call |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FP16 | all | 20 / 20 | 20.83% | 3.92 | 24.74 | 39.35 | 1.59× | 20.22 | 51.14 |
| FP16 | short | 8 / 8 | 20.94% | 3.98 | 28.47 | 54.61 | 1.92× | 20.00 | 51.11 |
| FP16 | long | 12 / 12 | 20.76% | 3.88 | 22.75 | 33.17 | 1.46× | 20.32 | 51.15 |
| W4A16 | all | 20 / 20 | 18.96% | 3.65 | 24.74 | 19.13 | 0.77× | 94.68 | 51.38 |
| W4A16 | short | 8 / 8 | 18.70% | 3.64 | 28.47 | 24.79 | 0.87× | 94.48 | 51.34 |
| W4A16 | long | 12 / 12 | 19.14% | 3.65 | 22.75 | 16.60 | 0.73× | 94.77 | 51.40 |
| W8A16 | all | 20 / 20 | 21.03% | 3.93 | 24.74 | 25.41 | 1.03× | 63.52 | 51.25 |
| W8A16 | short | 8 / 8 | 21.20% | 3.98 | 28.47 | 34.34 | 1.21× | 63.28 | 51.23 |
| W8A16 | long | 12 / 12 | 20.91% | 3.90 | 22.75 | 21.66 | 0.95× | 63.63 | 51.26 |

相对 FP16 Draft 的模型生成时间加速比：W4 **0.49×**、W8 **0.65×**；
两者均匹配 20 条，接受率分别变化 -1.87 / +0.20 个百分点。
这两个比值不能与表中“相对普通生成”的加速比混用。
W8 的接受率 **21.03%** 与 FP16 **20.83%** 接近，W4 为 **18.96%**；
从当前接受统计看，量化版仍有优化执行速度的价值。W8 整体相对普通为 1.03×、长输入为
0.95×，尚无稳定提速结论；相对 FP16 Draft，两种量化实现都更慢。
接受率只反映当前 Verify 接受候选的比例，不代表答案正确率，也未经过多次测量的稳定性验证。

整体阶段时间，单位 ms/次生成：

| Draft | 普通 Prefill | 普通 Decode | DFlash Prefill | DFlash Decode |
|---|---:|---:|---:|---:|
| FP16 | 749.57 | 4425.19 | 948.30 | 2304.80 |
| W4A16 | 749.57 | 4425.19 | 1610.21 | 5081.96 |
| W8A16 | 749.57 | 4425.19 | 1333.36 | 3703.48 |

量化 checkpoint 为五层，FP16 为六层，不是纯 bitwidth 消融。两种模式各自生成输出，
这些速度不证明 ordinary parity 或任务质量。最新 W8 profile 的 26 次 WeightQuant 合计
40.22 ms、TransData 10.45 ms、FP16 head MatMul 6.73 ms；详细映射与精度需求见
[量化 Draft 优化分析](../framework/custom_ops/draft_quant/README.md)。
该 profile 的算子合计 62.65 ms 不是完整请求时延，也不与本节阶段表相加。

## deterministic 与 FC 漂移

已定位到 **`draft.fc(features)`，FP16 Linear 20480 → 2560**。
固定输入和权重，各执行 20 次：

| 路径 | 关闭时变化次数 | 开启时变化次数 | 开关 |
|---|---:|---:|---|
| Torch-NPU FC | 19/20 | 0/20 | `torch.use_deterministic_algorithms(False/True, warn_only=False)` |
| AIR/OM FC | 19/20 | 0/20 | ATC `--deterministic=0/1`，需重编 OM |

关闭时样例出现少量 1 FP16 ULP 变化，可传播到 norm、KV 和候选。
冻结 FC 输出后，两种 RMSNorm 均稳定且逐位一致；冻结 norm 后 V projection 也稳定。
尚未确定具体 kernel，不能归因于 AdnRmsNorm。

`compile-om` 默认给 Draft 加 `--deterministic=0`（关闭）；
`recompile-draft-om --deterministic 0/1` 切换该图的编译选项，默认 `0`。
多轮输出不一致记录为 `DRIFT_OBSERVED`，保留各轮 token、停止原因、首个差异，
继续汇总接受率和时延，状态为 `PASS_WITH_OBSERVATIONS`。吞吐使用各轮实际 token 总数；
不将漂移标为稳定通过。已有结果表不会因更改默认设置而重算。
Python 开关不影响已有 OM。FC 探针稳定不保证完整 Decode/Verify 输出一致，
该开关的独立性能代价仍未测定。[FC 探针命令](../tools/debug_draft_context/README.md)

## 数据来源

数据按各节所列运行配置记录；仅整理用户提供的日志，未重新执行设备测试或读取远端原始 JSON。
日志摘录未含 OM 哈希、state dtype 和实际 deterministic 编译参数。

| 记录 | 运行目录 | 协议 |
|---|---|---|
| FP16 开源数据集，512 输出上限 | `gdr-lengths-x2l5aydx/fp16` | thinking 开启；0 次预热 + 1 次测量 |
| FP16 自定义短/1K prompt，128 输出上限 | `gdr-lengths-isajibrf/fp16` | thinking 开启；0 次预热 + 1 次测量 |
| 早期 FP16/W4/W8 解量化对比，128 输出上限 | `gdr-lengths-isajibrf` | thinking 开启；0 次预热 + 1 次测量 |
| FP16/W4/W8 原生 MatMul 对比，128 输出上限 | `gdr-lengths-waw8h36d` | 用户提供最终汇总；单算子 profile 另行采集 |

```text
$AI_RUN_DIR/gdr-lengths-x2l5aydx/summary.json
$AI_RUN_DIR/gdr-lengths-x2l5aydx/fp16/summary.json
$AI_RUN_DIR/gdr-lengths-x2l5aydx/fp16/cases.csv
$AI_RUN_DIR/gdr-lengths-x2l5aydx/fp16/chunk-512/prompt-suite-isyol8hb/summary.json
$AI_RUN_DIR/gdr-lengths-x2l5aydx/fp16/chunk-512/prompt-suite-isyol8hb/generations.txt
$AI_RUN_DIR/gdr-lengths-isajibrf/fp16/summary.json
$AI_RUN_DIR/gdr-lengths-isajibrf/summary.json
$AI_RUN_DIR/gdr-lengths-isajibrf/fp16/cases.csv
$AI_RUN_DIR/gdr-lengths-isajibrf/fp16/chunk-128/prompt-suite-_swok3lt/summary.json
$AI_RUN_DIR/gdr-lengths-isajibrf/fp16/chunk-128/prompt-suite-_swok3lt/generations.txt
$AI_RUN_DIR/gdr-lengths-waw8h36d/summary.json
```

分文件报告保留于对应汇总目录的 `datasets/<dataset-id>/summary.json` 和 `summary.md`，文件汇总见 `datasets.csv`。
