# 结果与已知问题

以下为用户提供的 Chunk 实测：最多 15 个候选，3 次预热 + 10 次测量，low-memory，
允许普通 / DFlash 输出不同。任务质量未评估，整理文档未新增设备测量。

## 短 prompt：32 / 64 / 128 输出

运行 `gdr-lengths-ng09f5zx`，8 条 prompt，输入 28～59 token。各条均达到输出上限。

| 输出 token | 接受 / 提出（10 次测量合计） | 加权接受率 | 严格一致 / 允许差异（条） |
|---|---:|---:|---:|
| 32 | 2010 / 6770 | 29.69% | 3 / 5 |
| 64 | 3920 / 15110 | 25.94% | 1 / 7 |
| 128 | 7670 / 36630 | 20.94% | 0 / 8 |

各格为 **接受率 / DFlash tok/s / 加速比**。

| Prompt | 输入 token | 32 输出 | 64 输出 | 128 输出 |
|---|---:|---|---|---|
| zh_explain | 28 | 48.28% / 71.46 / 2.58× | 38.69% / 63.78 / 2.26× | 25.83% / 49.20 / 1.73× |
| zh_plan | 29 | 7.21% / 20.39 / 0.74× | 6.85% / 20.39 / 0.72× | 6.67% / 20.76 / 0.73× |
| math | 47 | 50.00% / 59.05 / 2.14× | 33.33% / 53.74 / 1.90× | 37.02% / 65.54 / 2.31× |
| code | 46 | 25.25% / 43.86 / 1.59× | 30.86% / 46.47 / 1.65× | 24.04% / 47.37 / 1.67× |
| translate | 59 | 48.28% / 71.08 / 2.57× | 65.91% / 100.76 / 3.57× | 37.24% / 65.51 / 2.31× |
| summary | 59 | 27.59% / 38.88 / 1.41× | 18.00% / 34.73 / 1.23× | 17.60% / 36.08 / 1.27× |
| en_explain | 34 | 62.22% / 71.38 / 2.59× | 47.83% / 70.16 / 2.49× | 26.05% / 47.38 / 1.67× |
| creative | 42 | 38.24% / 50.35 / 1.82× | 29.70% / 46.46 / 1.65× | 22.48% / 44.29 / 1.56× |

三个长度均有 7 条更快、1 条更慢。128 输出时 math、translate 为 **2.31×**；
zh_plan 为 **0.73×**，接受率 6.67%，每轮仅产出 1.95 token。
整体接受率随输出预算降低，各任务并非单调下降；末轮候选缩短也会改变接受率分母。

## 长 prompt：1K 输入 / 128 输出

运行 `gdr-lengths-3a3e3701`，4 条 prompt，均生成 128 token，状态为 `PASS_WITH_DIFFERENCES`。

| Prompt | 输入 token | 接受 / 提出 | 接受率 | token/投机轮 | DFlash tok/s | 加速比 |
|---|---:|---:|---:|---:|---:|---:|
| long_zh_summary 中文摘要 | 983 | 860 / 5650 | 15.22% | 3.02 | 22.34 | 0.98× |
| long_zh_qa 跨段检索 | 998 | 970 / 4110 | 23.60% | 4.23 | 27.70 | 1.22× |
| long_zh_plan 约束规划 | 1001 | 940 / 4840 | 19.42% | 3.85 | 26.11 | 1.15× |
| long_en_analysis 英文分析 | 1008 | 930 / 4870 | 19.10% | 3.63 | 25.16 | 1.11× |

加权接受率 **3700 / 19470 = 19.00%**，3 条更快，中文摘要略慢。

每条输入需要 16 个 64-row Prefill 块。DFlash 在前 15 块还调用完整 Draft 建立上下文 KV，
耗时计入模型循环，丢弃的候选不计入接受率。[Prefill 实现](../framework/runtime/cpp/src/acl_chunk.cpp)
这会增加长输入成本；具体占比缺少分项计时。长短两组任务和部署信息不同，不能直接隔离长度影响。

## 计时口径与待补数据

- 接受率 = 接受候选 / 提出候选，统计正式测量，排除预热。
- 加速比 = 普通 / DFlash 模型循环时延中位数，包含 Prefill，排除加载与预热。
- **分项时延暂缺**：普通 Prefill、Decode，以及 DFlash Prefill、Draft、Verify。
  整体总时间加速比也未提供，不由 tok/s 或各行倍数反推。
- **256 / 512 / 1024 输出及 MTP 尚无本轮结果**。日志未提供 OM 哈希、ABI 或 state dtype。
- 子套件均通过各模式独立重复性检查。短 prompt 外层 32/64 显示 `FAIL`，原因未展示；
  128 外层状态未展示。长 prompt 外层误报 fake ACL，因此这些矩阵尚无有效总汇总。
  [统一测试](GDR_CHUNK_AIR_OM.md)已修复 fake ACL 字段读取，并输出分项时延，默认 1 次预热 + 3 次测量。

## deterministic 与 FC 漂移

已定位到 **`draft.fc(features)`，FP16 Linear 20480 → 2560**。
固定同一 AIR、权重和输入，各执行 20 次：

| 路径 | 关闭时变化次数 | 开启时变化次数 | 开关 |
|---|---:|---:|---|
| Torch-NPU FC | 19/20 | 0/20 | `torch.use_deterministic_algorithms(False/True, warn_only=False)` |
| AIR/OM FC | 19/20 | 0/20 | ATC `--deterministic=0/1`，需重编 OM |

关闭时样例出现少量 1 FP16 ULP 变化，可传播到 norm、KV 和候选。
冻结 FC 输出后，两种 RMSNorm 均稳定且逐位一致；冻结 norm 后 V projection 也稳定。
尚未确定具体 kernel，不能归因于 AdnRmsNorm。

`compile-om` 默认只给 Draft 加 `--deterministic=1`；
显式 `--atc-arg=--deterministic=0` 会覆盖该默认值并传给所有待编译图。
Python 开关不影响已有 OM。确定性解决了 FC 探针重复性，但不保证 Decode/Verify 输出一致；
该开关的独立性能代价、低接受率和输出分叉的完整原因仍待定位。
[FC 探针命令](../tools/debug_draft_context/README.md)

<details>
<summary>原始报告目录</summary>

各目录位于运行时的 `$AI_RUN_DIR` 下，包含 `summary.json`、`generations.txt` 和逐次测量报告。

| 测试 | 目录 |
|---|---|
| 短 / 32 | `gdr-lengths-ng09f5zx/chunk-32/prompt-suite-ft5jjazv/` |
| 短 / 64 | `gdr-lengths-ng09f5zx/chunk-64/prompt-suite-q4dibwtp/` |
| 短 / 128 | `gdr-lengths-ng09f5zx/chunk-128/prompt-suite-ki5odl8f/` |
| 长 / 128 | `gdr-lengths-3a3e3701/chunk-128/prompt-suite-y3ynevf0/` |

</details>
