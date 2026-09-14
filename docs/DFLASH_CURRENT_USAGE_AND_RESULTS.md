# 128 token 测试结果与已知问题

Chunk 验证；每轮最多 15 个候选，投机持续开启；每个模式、每条 prompt 预热 1 次、测量 3 次。
20 条均生成 128 token，子用例和外层汇总均为 `PASS_WITH_DIFFERENCES`：
两种模式各自通过重复性检查，输出不完全一致，任务质量未评估。

## 短 prompt

| Prompt | 输入 token | 接受率 | token/投机轮 | DFlash tok/s | 加速比 |
|---|---:|---:|---:|---:|---:|
| zh_explain | 28 | 25.83% | 4.70 | 49.52 | 1.74× |
| zh_plan | 29 | 6.67% | 1.95 | 20.96 | 0.73× |
| math | 47 | 37.02% | 6.35 | 66.13 | 2.32× |
| code | 46 | 24.04% | 4.54 | 47.80 | 1.68× |
| translate | 59 | 37.24% | 6.35 | 66.16 | 2.33× |
| summary | 59 | 17.60% | 3.43 | 36.38 | 1.28× |
| en_explain | 34 | 26.05% | 4.54 | 47.76 | 1.68× |
| creative | 42 | 22.48% | 4.23 | 44.64 | 1.57× |

## 约 1K 上下文 prompt

本轮长输入为 11 条中文任务（含英译中）、1 条英文任务；新增英文用例尚未计入本表。

| Prompt | 输入 token | 接受率 | token/投机轮 | DFlash tok/s | 加速比 |
|---|---:|---:|---:|---:|---:|
| long_zh_summary | 983 | 15.22% | 3.02 | 22.31 | 0.98× |
| long_zh_qa | 998 | 23.60% | 4.23 | 27.69 | 1.22× |
| long_zh_plan | 1001 | 19.42% | 3.85 | 26.12 | 1.15× |
| long_en_analysis | 1008 | 19.10% | 3.63 | 25.16 | 1.11× |
| long_zh_math | 975 | 15.03% | 3.10 | 22.67 | 1.00× |
| long_zh_code | 984 | 24.50% | 4.23 | 27.70 | 1.22× |
| long_zh_translate | 1004 | 25.26% | 4.54 | 28.86 | 1.27× |
| long_zh_extract | 1024 | 5.91% | 1.81 | 15.34 | 0.68× |
| long_zh_compare | 971 | 13.43% | 2.89 | 21.62 | 0.95× |
| long_zh_timeline | 1004 | 14.65% | 3.10 | 22.70 | 1.00× |
| long_zh_rules | 979 | 23.77% | 4.23 | 27.69 | 1.22× |
| long_zh_story | 960 | 20.96% | 3.97 | 27.28 | 1.18× |

## 汇总

| 输入组 | Prompt 数 | 接受 / 提出（3 次合计） | 加权接受率 | token/投机轮 | DFlash tok/s | 加速比 |
|---|---:|---:|---:|---:|---:|---:|
| 短输入 | 8 | 2301 / 10989 | 20.94% | 3.98 | 42.13 | 1.48× |
| 约 1K 输入 | 12 | 3225 / 19134 | 16.85% | 3.34 | 23.90 | 1.05× |
| 全部 | 20 | 5526 / 30123 | 18.34% | 3.57 | 28.90 | 1.17× |

接受率 = 接受候选 / 提出候选。tok/s 和加速比均计入 Prefill，排除加载、预热与请求重置。
单条加速比比较两种模式的时延中位数；分组加速比为普通总耗时 / DFlash 总耗时。
两种模式比较各自生成的输出，倍数大于 1 表示更快；1.00× 为四舍五入后的近似持平。

## 阶段与 OM 时延

**阶段耗时：平均 ms/次生成。**

| 输入组 | 普通 Prefill 阶段 | 普通 Decode 循环 | DFlash Prefill 阶段 | DFlash Decode 循环 |
|---|---:|---:|---:|---:|
| 短输入 | 74.40 | 4422.95 | 75.02 | 2963.30 |
| 约 1K 输入 | 1200.69 | 4425.03 | 1820.59 | 3534.95 |
| 全部 | 750.17 | 4424.20 | 1122.36 | 3306.29 |

**OM 调用耗时：平均 ms/次调用 / 累计平均 ms/次生成。**

| 输入组 | 普通 Target Prefill | 普通 Target Decode | DFlash Target Prefill | Draft | Verify |
|---|---:|---:|---:|---:|---:|
| 短输入 | 74.35 / 74.35 | 34.80 / 4419.34 | 74.97 / 74.97 | 41.66 / 1327.96 | 51.25 / 1633.52 |
| 约 1K 输入 | 75.40 / 1200.17 | 34.81 / 4421.42 | 75.29 / 1198.38 | 41.70 / 2206.49 | 51.27 / 1948.10 |
| 全部 | 75.36 / 749.84 | 34.81 / 4420.59 | 75.28 / 749.01 | 41.69 / 1855.08 | 51.26 / 1822.27 |

DFlash Prefill **阶段**包含 Target Prefill 和 Draft 上下文缓存构建；OM 表中的 DFlash Target Prefill
只计 Target Prefill 图。Draft 累计耗时同时包含 Prefill 和生成阶段的调用，两张表不能相加。
Verify 已包含提交操作。OM 计时包含同步与调用开销，不是单个算子的耗时。

### 为什么长输入收益较小

普通与 DFlash 共用 Target Prefill OM，长输入的该图累计耗时分别为 **1200.17 / 1198.38 ms**，基本相同。
DFlash 每处理一个非末尾的 64-token 输入块，还调用完整 Draft 建立上下文 KV，并丢弃这次候选。
本轮长输入需要 15～16 个 Prefill 块，因此多出 14～15 次 Draft 调用，每次约 **41.7 ms**。
[Prefill 实现](../framework/runtime/cpp/src/acl_chunk.cpp#L610)

这使 DFlash Prefill 阶段从普通的 **1200.69 ms** 增至 **1820.59 ms**，多约 **620 ms**。
长输入 Decode 循环节省约 **890 ms**（4425.03 → 3534.95），抵消额外 Prefill 后，
整段生成只节省约 **270 ms**，加速比为 **1.05×**。

按本轮计时估算，一轮 Draft + Verify 约 **93 ms**，相当于 **2.67 次**普通 Decode。
忽略 Prefill 和其他开销，每轮需产出约 2.7 token 才能加速 Decode；
`zh_plan` 仅 1.95、`long_zh_extract` 仅 1.81，因此分别降至 **0.73× / 0.68×**。
接受率低的完整原因仍待定位；这些时延不能证明是长上下文或精度损失导致。

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

`compile-om` 默认只给 Draft 加 `--deterministic=0`（关闭）；
`recompile-draft-om --deterministic 0/1` 可单独切换 Draft，默认 `0`。
多轮输出不一致记录为 `DRIFT_OBSERVED`，保留各轮 token、停止原因、首个差异，
继续汇总接受率和时延，状态为 `PASS_WITH_OBSERVATIONS`。吞吐使用各轮实际 token 总数；
不将漂移标为稳定通过。已有结果表不会因更改默认设置而重算。
Python 开关不影响已有 OM。FC 探针稳定不保证完整 Decode/Verify 输出一致，
该开关的独立性能代价仍未测定。[FC 探针命令](../tools/debug_draft_context/README.md)

## 数据来源

用户提供的 `gdr-lengths-vwh9cbqo` 运行报告；本次仅整理文档，未重新执行设备测试。
日志摘录未含 OM 哈希和 state dtype，不能据此确认部署精度配置。

```text
$AI_RUN_DIR/gdr-lengths-vwh9cbqo/summary.json
$AI_RUN_DIR/gdr-lengths-vwh9cbqo/cases.csv
$AI_RUN_DIR/gdr-lengths-vwh9cbqo/chunk-128/prompt-suite-05akpid6/summary.json
$AI_RUN_DIR/gdr-lengths-vwh9cbqo/chunk-128/prompt-suite-05akpid6/generations.txt
```
