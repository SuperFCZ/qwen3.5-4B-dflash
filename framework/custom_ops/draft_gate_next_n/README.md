# C16 Layer0 Gate/Up 跨 N64 调度

310P3 单算子选择 `panel-k2560-w2-next-n`。相对当前整图使用的 `panel-k2560-w2`，无插桩 Median/P95 从 **0.469935/0.483410 ms 降到 0.412700/0.419090 ms**。[正式配对和 msprof 结果](#310p3-实测)

在 310P3 服务器仓库根目录执行。

```bash
git pull --ff-only origin feature/ascend310p-custom-ops && bash framework/custom_ops/draft_gate_next_n/run.sh
```

默认测单个 Gate/Up OM 的正式 Execute+Sync，再对每个正确版本独立运行 msprof。终端打印每组配对的 Median/P95、各轮倍率、设备 Gate 任务中位数、P95 和 30 次窗口中能匹配到的流水计数。结果目录为 `/home/w00949577/z50058744/qwen35-dflash-run/runs/gate-next-n-XXXXXX/`。使用原有独立 `gate310p` 环境。可用 `--no-profile` 或 `--compile-only` 限定范围，`--resume gate-next-n-XXXXXX` 在新目录复用通过校验的成功阶段。

## 比较的排程

三份单算子 OM 使用同一份 AIR 和真实打包常量，8 个实际 block 各拥有 38 个 N64。每个 N64 进行一次完整 K2560 归约。每核仍然按相同的 160 次 K16 gather 将激活放入 A1。一个 raw 区、两个 80 KiB W16、一个 B1、46,592 B MatMul scratch、三道原有全流水屏障以及 Group128 舍入顺序均相同。

| 实现 | 跨 N64 的首个 N16 |
|---|---|
| `panel-k2560-w2` | 采用 TglfhB 的私有 OPP，在本次单 Gate OM 中复核其实际机器执行段。上一 N64 收尾后开始准备下一块 |
| `panel-k2560-w2-next-n` | 当前 B1 就绪后预取下一块 raw 和 scale。当前 Cube 提交后、原有屏障前在独立 UB 中处理下一块的首片 Cast/Mul |
| `panel-k2560-w2-next-n-early-dma` | 提前到当前最后一次 UB→B1 搬运发出后就装入下一块的 raw 和 scale。V→MTE2 先保护旧数据的最后一次 Vector 读取，随后同一道 B1 就绪屏障覆盖两侧搬运。其余与上一候选一致 |

本核按 `next = nt + blocks` 取后继，每核最后一个 tile 不做前视。当前 B1 只供当前 Cube 读取，下一块的首片先留在 W0，待原有全流水屏障、输出写回和末尾屏障后才写入 B1。Iterate 失败时先排空再退出。上述是源码发出顺序，实际重叠程度由设备决定。

## 输入和计时

对照为 `full-all-w8-TglfhB` 已通过整图验收的 C16 Gate。复制其 Layer0 投影和原常量，逐位逆验原始 NZ 权重及 GN scale，编译后检查 OM 内两个常量、实际 ELF 和完整 Cube/UB 计划。真实激活来自封存的 Layer0 embedding 和 `o-bandwidth-F1HnBO` 的设备 O 输出。单独的小参考图保留 UFqlys 原始 residual、post-norm 和原版 Gate，捕获 FP16 输入及独立参考输出。该回放没有计时，正式单算子图只含一个计算节点。

每个参与计时的版本都通过回放、稠密压力及跨分组边界三种输入，两种 poison 下的全部 311,296 个 FP16 输出、Readonly、Guard、Repeat、Sync 和 pre-execute DSO 检查。每组采用三轮正反序，每轮每侧 5 次预热与 30 次正式调用。Median 至少下降 3%、三轮同向且 P95 无回退才判 GO。两候选均优于对照时，另做两候选的新鲜直接配对。

msprof 使用 task-based PipeUtilization，保留原始 timeline、op_summary、窗口归属和匹配的 Vector、Scalar、MAC、MTE1/MTE2/MTE3 指标。每次采集使用 38 次调用中的 30 个正式窗口。严格设备归属不足时报告 `PROVISIONAL_HOST_ENCLOSURE`。逻辑吞吐分子为 51,290,112 B，按 204.8 GB/s 的单遍参照为 250.44 µs。流水时间可重叠，逻辑吞吐与实际 DRAM 事务分开解读。

`latency.json` 保存正式配对，`profile-*/report.json` 和 `profile-*/counters.json` 保存插桩结果，`result.json` 汇总选型。错误直接输出到终端和 `traceback.txt`。CPU 合同测试入口为 `python -B -m unittest discover -s framework/custom_ops/draft_gate_next_n/test -v`，可通过 `DFLASH_GE_DESCRIPTOR` 和 CPU Torch/Transformers 启用原始五层图边界验证。

## 310P3 实测

实验实现提交 `030f8b7`。服务器运行 `gate-next-n-669ScA` 返回 `MEASURED_AND_PROFILED`。三份 OM 采用相同 AIR、8-block 调度、完整 Cube tiling 和 46,592 B MatMul scratch。三组输入、每组两种 poison 的 311,296 个 FP16 输出均逐位匹配参考，Guard、Readonly、Repeat 和 Sync 通过。

### 正式单 Gate/Up Execute+Sync

每边每组都是三轮各 30 个新鲜正式样本。下表单位为 ms，倍率为左侧 Median 除以右侧 Median。表中 Next-N 和 Early DMA 分别指 `panel-k2560-w2-next-n` 和 `panel-k2560-w2-next-n-early-dma`。

| 直接配对 | 左侧 Median / P95 | 右侧 Median / P95 | 三轮倍率 | 总体倍率 | 判定 |
|---|---:|---:|---|---:|---|
| 已部署版本 → Next-N | 0.469935 / 0.483410 | **0.412700 / 0.419090** | 1.1396 / 1.1399 / 1.1376 | 1.1387× | GO |
| 已部署版本 → Early DMA | 0.469320 / 0.477190 | 0.426595 / 0.433100 | 1.0983 / 1.1039 / 1.0992 | 1.1002× | GO |
| Next-N → Early DMA | 0.414475 / 0.421600 | 0.426725 / 0.442070 | 0.9709 / 0.9680 / 0.9743 | 0.9713× | NO_GO |

Next-N 对已部署版本节省 **57.235 µs（12.18%）**，P95 节省 64.320 µs。两候选直接比较，Early DMA 的中位数比 Next-N 长 12.250 µs。

### msprof 设备任务与流水

下表时间单位为 µs。每个版本有 30 个正式窗口，每个窗口恰有一个 Gate 任务，归属均为 `PROVISIONAL_HOST_ENCLOSURE`。三个 op_summary 均成功匹配 30/30 个任务。

| 版本 | 设备 Median / P95 | 逻辑 GB/s | AICore | Vector | MAC | Scalar | MTE1 | MTE2 | MTE3 | Cube 利用率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 已部署版本 | 397.19 / 402.42 | 129.13 | 369.75 | 153.95 | 26.77 | 90.98 | 92.50 | 110.91 | 91.49 | 93.03% |
| Next-N | **341.55 / 345.75** | **150.17** | 317.12 | 153.95 | 26.77 | 87.82 | 92.50 | 125.54 | 94.86 | 92.83% |
| Early DMA | 354.47 / 357.86 | 144.70 | 329.05 | 153.95 | 26.77 | 88.39 | 92.50 | 120.20 | 93.09 | 92.92% |

Next-N 的设备任务缩短 **55.64 µs（14.01%）**。Vector、MAC、MTE1 活动时间在此次采集的显示精度下相同，MTE2 活动时间增加。结果支持在保持计算量时通过调整数据准备时机缩短任务。计数器区间可重叠，不能据此分解 55.64 µs 的逐流水贡献。341.55 µs 是理想单遍参照 250.44 µs 的 1.364 倍。实际 DDR 字节数尚未测定。

原始报告为 `/home/w00949577/z50058744/qwen35-dflash-run/runs/gate-next-n-669ScA/result.json`。同目录的 `latency.json` 含逐轮样本，`profile-*/profiler/` 含时间线及 op_summary。msprof SHA256 为 `a168e827ebfeced390b62c062de4a42214fe2cfbcc327178708b5224519646cb`。本次单 Gate Next-N OM 所选 ATC ELF 的 SHA256 为 `efde2ab507a0d6fd6bd1c2220cd867f524551d6f539a6b8bfa7451c0deaf2f92`。数据范围为真实 checkpoint 的 C16 Layer0 单投影，未进行五层或完整 Draft 的性能复测。
