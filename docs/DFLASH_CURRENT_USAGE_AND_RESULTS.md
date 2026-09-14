# 结果与已知问题

记录用户提供的实测日志与报告。两组均允许普通 / DFlash 输出不同，任务质量未评估；
整理文档不代表新增设备测量。

## 1K 输入 / 128 输出：Chunk

运行 `gdr-lengths-3a3e3701`：4 条完整长 prompt，`--verify-gdr chunk --lengths 128`，
最多 15 个候选，3 次预热 + 10 次测量，low-memory、always-on，允许输出差异。
四条均生成 128 token，以 `max_new_tokens` 结束；子套件均为 `PASS_WITH_DIFFERENCES`。
日志未展示本次 OM 的 ABI、recurrent state dtype 和编译选项，暂不据此标注 FP32 实测。

| Prompt | 输入 token | 接受 / 提出 | 接受率 | token/投机轮 | DFlash tok/s | 加速比 |
|---|---:|---:|---:|---:|---:|---:|
| long_zh_summary 中文摘要 | 983 | 860 / 5650 | 15.22% | 3.02 | 22.34 | 0.98× |
| long_zh_qa 跨段检索 | 998 | 970 / 4110 | 23.60% | 4.23 | 27.70 | 1.22× |
| long_zh_plan 约束规划 | 1001 | 940 / 4840 | 19.42% | 3.85 | 26.11 | 1.15× |
| long_en_analysis 英文分析 | 1008 | 930 / 4870 | 19.10% | 3.63 | 25.16 | 1.11× |

加权接受率 **3700 / 19470 = 19.00%**，3 条更快，中文摘要略慢。
加速比为普通 / DFlash 模型循环时延中位数之比，包含 prefill，排除模型加载与预热。
**整体总时间加速比、普通 Decode / Draft / Verify 的 ms/call 暂缺**：
外层汇总误报 `fake ACL cannot supply matrix device measurements`，显示 0/4 和 N/A；
上表取自已完成的子套件日志，汇总问题见文末。

**为什么比短 prompt 加速小：** 接受率与旧短 prompt 的 20.69% 相近，但计时还包含处理上下文的成本。
本次每条输入需要 16 个 64-row prefill 块；DFlash 在前 15 块还调用完整 `draft.om` 建立上下文 KV，
这些调用产生的候选丢弃，耗时计入模型循环、候选数不计入接受率。
这一额外开销已从 [Prefill 实现](../framework/runtime/cpp/src/acl_chunk.cpp)确认；
具体占比仍需读取分项计时。两组 prompt 和部署信息不同，不能将全部差距归因于上下文长度。

<details>
<summary>按生成位置的接受率与报告位置</summary>

| Prompt | [0,32) | [32,64) | [64,96) | [96,128) |
|---|---:|---:|---:|---:|
| long_zh_summary | 16.30% | 17.78% | 12.78% | 14.78% |
| long_zh_qa | 23.81% | 13.33% | 33.33% | 37.88% |
| long_zh_plan | 34.67% | 12.78% | 15.56% | 25.53% |
| long_en_analysis | 36.00% | 13.33% | 14.67% | 22.68% |

按整轮起点归档，不拆跨区间轮次；末段提出候选较少，接受率分母也随之变化。
首个输出分叉位置（从 0 编号）依次为 22、17、31、30；各模式内部均通过 3+10 重复性检查。

子套件：`$AI_RUN_DIR/gdr-lengths-3a3e3701/chunk-128/prompt-suite-y3ynevf0/`。
`summary.json` 保存各 prompt 指标，`generations.txt` 保存文字，
`runner-batch.json.cases/*.json` 保存每次 `latency_ms.prefill/decode` 和 `stage_ms`。
外层 `gdr-lengths-3a3e3701/summary.json`、`cases.csv` 当前未完成有效汇总。

</details>

## 短 prompt / 128 输出：旧 Chunk v3

以下为用户提供的 `prompt-suite-5son4ozl`，按允许输出差异策略重汇总为
`prompt-summary-s4cu60r4`。Ascend 310P、Chunk 路线、W8A8 Target、确定性 FP16 Draft；
8 条 prompt，每条生成 128 token，3 次预热 + 10 次测量，投机始终开启。
**这是旧 Chunk v3、recurrent state 写回 FP16 的结果。** 新版原版/Chunk/MTP 统一 FP32，
已有主机接口测试，绑定新版 ABI 与编译产物的设备结果尚待核验；下表不能当成新版 FP32 的成绩。

两种模式各自重复稳定，跨模式输出不同，状态为 `PASS_WITH_DIFFERENCES`；
任务质量未评估。

普通模型约 **29 tok/s**，生成 128 token 用时 **4405～4421 ms**（中位数）。

| Prompt | DFlash tok/s | 吞吐增幅 | 加速比 | DFlash 128 token ms | 接受率 | token/投机轮 |
|---|---:|---:|---:|---:|---:|---:|
| zh_explain 中文解释 | 46.04 | +58.41% | 1.58× | 2781.29 | 22.61% | 4.10 |
| zh_plan 中文规划 | 21.75 | -25.01% | 0.75× | 5893.52 | 6.39% | 1.92 |
| math 数学 | 69.35 | +139.51% | 2.39× | 1846.56 | 38.30% | 6.35 |
| code 代码 | 43.28 | +49.41% | 1.50× | 2954.61 | 21.16% | 3.85 |
| translate 翻译 | 73.83 | +155.05% | 2.55× | 1734.48 | 38.85% | 6.68 |
| summary 摘要 | 38.67 | +33.49% | 1.34× | 3305.12 | 18.00% | 3.43 |
| en_explain 英文解释 | 46.07 | +59.05% | 1.59× | 2779.94 | 22.59% | 4.10 |
| creative 中文创作 | 56.78 | +96.03% | 1.96× | 2256.16 | 29.77% | 5.08 |

整体 **28.98 → 43.49 tok/s，1.50075× 加速**，7 条更快、1 条变慢。
加权接受率 **7580 / 36630 = 20.69%**；已接受候选占最终输出 **74.02%**，两者分母不同。
整体加速按总模型时间之比计算，不平均各行倍数；加载、模式切换不计入模型循环。
表内加速比使用时延中位数，吞吐增幅按全部测量的 tok/s 计算。

## Decode、Draft、Verify 时延

| 项目 | 当前数据 / 读取位置 |
|---|---|
| 旧短 prompt 普通全程摊销 | 约 34.4～34.5 ms/token，含 prefill，不能当成纯 decode |
| 普通 Decode 图 ms/call | 原始 ordinary `measurements[].stage_ms.target_decode[]` |
| Draft 图 ms/call | 原始 dflash `measurements[].stage_ms.draft[]` |
| Verify 图 ms/call | 原始 dflash `measurements[].stage_ms.target_verify[]` |

目前提供的两组日志均未展示这三项图计时数组，**暂缺单独实测值**。
[多长度测试](GDR_CHUNK_AIR_OM.md#多长度与双路线测试)的外层汇总问题需先修复；
原始子报告中的计时仍可读取。需要算子级热点时使用 [msprof](GDR_CHUNK_AIR_OM.md#msprof)。

`stage_ms` 为同步 OM 调用墙钟时间，包含必要的数据绑定、传输和同步；
Verify 包含图内状态提交，不能再加一份 commit。长 prompt 的 Draft 调用还可能包括
prefill 中的 KV 初始化。以上均区别于纯 kernel 时间。

## 长度增加后的接受率

旧短 prompt 的同一次 128-token 运行，按整轮起点分段：

| Prompt | [0,32) | [32,64) | [64,96) | [96,128) |
|---|---:|---:|---:|---:|
| zh_explain | 48.33% | 17.04% | 24.17% | 14.04% |
| zh_plan | 7.56% | 5.88% | 3.67% | 10.34% |
| math | 37.33% | 21.90% | 53.33% | 62.96% |
| code | 20.83% | 22.86% | 17.33% | 27.03% |
| translate | 51.67% | 100.00% | 26.67% | 23.47% |
| summary | 20.00% | 11.67% | 14.67% | 46.00% |
| en_explain | 51.67% | 20.83% | 12.73% | 23.75% |
| creative | 33.33% | 46.67% | 14.81% | 44.26% |

并非统一随长度下降：math 后段提高，zh_plan 从开头就低。
跨区间的轮次不拆分；末段候选缩短也会改变接受率分母。
已有独立 32-token 测试的全套接受率为 29.33%；不能直接相减两次计数推算后 96 token。
**256/512/1024 和 MTP 的设备结果待测。**

## deterministic 与 FC 漂移

已定位到 Draft 上下文投影 **`draft.fc(features)`，FP16 Linear 20480 → 2560**。
固定同一 AIR、权重、输入，各运行 20 次：

| 路径 | 关闭时变化次数 | 开启时变化次数 | 开关 |
|---|---:|---:|---|
| Torch-NPU native FC | 19/20 | 0/20 | `torch.use_deterministic_algorithms(False/True, warn_only=False)` |
| AIR/OM FC | 19/20 | 0/20 | ATC `--deterministic=0/1`，需重编 OM |

det0 样例是少量元素的 1 FP16 ULP 变化，可传播到 norm、KV 和候选。
冻结 FC 输出后的 AdnRmsNorm / Tensor norm 均稳定且逐位一致，冻结 norm 后的 V projection 也稳定。
**没有证据归因于 AdnRmsNorm；具体 split-K/atomic kernel 尚未确定。**

- Python 开关只影响原生执行，不修改已有 OM；原生推理入口未承诺默认开启该开关。
- `compile-om` 默认只给 Draft 补 `--deterministic=1`，实际配置看 manifest 的 `graphs[].atc_command`。
- 公共 `--atc-arg=--deterministic=0` 会覆盖 Draft 默认并传给所有待编译图。
- 开启确定性解决了该 FC 探针的重复性；不同模式间仍有舍入差异，也不保证 decode/verify 一致。
- 确定性开关的独立性能代价尚未做同范围 A/B 测量。

旧部署使用[只重编 Draft 命令](GDR_CHUNK_AIR_OM.md#导出与编译)；
探针复现见[FC 调试工具](../tools/debug_draft_context/README.md)。

## 其他已知问题

- **多长度外层误报 fake ACL**：当前 C++ 把 `fake_acl` 写在批次索引中，单个 prompt 子报告没有此字段；
  `benchmark_gdr_lengths.py` 却要求子报告的该字段严格为 `false`，将缺失误判为 fake ACL。
  本次子套件已完成，外层汇总失败；修复与离线重新汇总待处理。
- **普通 decode 与 verify 输出分叉**：已观察到，尚未分离数值路径与状态更新的影响。
- **低接受率**：zh_plan 仅 6.39%，实际慢于普通模型；暂未定位统一根因。
- **Chunk 单输出性能退化**：历史单 GDR 约 22～29 ms，双输出约 0.26～0.27 ms；
  当前保留 discard 输出。此数字不代表整张 Verify 时延。
- **零接受后停投机**：已改为 always_on，本套关闭事件为 0。
- **后续验证**：双路线长输出、任务质量，以及冻结同一状态下的 decode/verify 和 Draft KV 对照。
