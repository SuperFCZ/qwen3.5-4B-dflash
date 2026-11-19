# 当前决策与证据缺口

**已测事实（用户提供，admit 在服务器重验原件）**：gate-raw-atc8-Np20yu/full-benchmark/report.json 的 GATE8_FULL_BENCHMARK_PASS 是完整五层 Draft Execute+Sync，非 decode。C16 Off/Raw median 31.217700/30.569870 ms、P95 31.272960/30.620470 ms；C64 median 32.091310/31.426295 ms、P95 32.143220/31.468800 ms。净差 0.647830/0.665015 ms，约 1.0212×，只属于整图本次配对。不把五个单投影 median 加到整图上。

**当前 Raw 真实性能热点和 Off/Raw 管线差值：待测，不可从上述净差反解。** 历史 Off provisional 的五层 Gate/Up 为 10.3143/10.3053 ms/Execute，Down 为 4.7072/4.7040，Head 单任务约 6.78/6.76。Off WAIT interval union 23.103/23.956 ms，其中与 GroupQuant union 的交集 20.937/21.616 ms。说明大量时间区间重叠，不说明可以删除 23 ms 等待；具体 Notify 对象、可消除等待、关键路径均未证实。Off 的首个计算族调查对象是 Gate/Up，Head 是最大的已识别单任务之一；不能把这两个判断直接换成 Raw 的排名。docs/OPTIMIZATION.md 的早期 Native Runtime WeightQuant/TransData profile 也不是当前 A-only profile。

**工程决策**：
1. 最小的下一次设备工作是只采一次当前 Raw C16 的受控 profile；不是新 Benchmark。必须取得真实 PID/exec/FD、原预放行和 post-maps、全量 ID/KV 位型、两次前检、Tail、Postcheck 及 38 对窗口。之后独立采 C64。要归因 Off→Raw 差异，再用本次正式配对中的两个 Off OM，同工具/设备/输入协议采集，并在相同归属等级下比较。
2. 算子看 26 个实际映射、每次覆盖率及任务区间。若 Raw 下 Gate 仍排首位，首选研究 W16→L1 最后读的生命周期；若 Down 的可定位搬运/Vector 工作相当或更重，改优先考察 Down。不能按 Off 排名或某单一 pipe time 固定优先级。同步先研究 wait stream、record ID/世代、与有效计算的交集和靠近 Sync 返回的尾段；没有对象 ID 的同时性仅为线索。分析入口将新结果写入输出目录自己的 optimization-next-step.md。
3. W16：真实本地 SDK 对两预算均返回 0、Cube 200 B 相同；用户 bank 释放和故障排空合同未证，BLOCKED。未来先闭合源代码/硬件事件合同及正式 Host，才安排单个真实 C16 Gate 验证；不改原屏障、Scratch 或 reserve 来跳过证明。
4. Head 值得独立并行研究实际 immutable NZ 输入布局及 15 行共享，先建立最小同图 Head 参考，不采用旧融合包。验收全词表、完整 FP16 logits 位型和 15 IDs，以及跨分片并列、NaN/Inf、Guard 和重复性；再考察单变量，最后才是新完整 Draft 未插桩效果。此处没有承诺节省时间。

| Gear | 距 20 ms 还差 | 距 13.33 ms | 距 13 ms |
|---|---:|---:|---:|
| C16 | 10.569870 ms | 17.239870 ms | 17.569870 ms |
| C64 | 11.426295 ms | 18.096295 ms | 18.426295 ms |

这是基线与目标的算术差（严格低于 20 ms 还需略超过该差值）；13.33 ms 需约 2.29×/2.36× 整图速度比，不是预测。要同时面对 26 个投影、完整 FP16 词表权重、非线性/状态和同步；即使 Gate 单算子改善，也不能直接累加到图上。当前没有证明达到 20 或 13 ms 的实现。
