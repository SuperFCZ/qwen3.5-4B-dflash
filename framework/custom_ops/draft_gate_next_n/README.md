# C16 Layer0 Gate/Up 跨 N64 调度

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
