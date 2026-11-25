# W8A16 投影间的优化迁移审计

审计基于 2026-11-25 的实验分支 `bce1dca`。整图基线为 310P3 的 `full-all-w8-TglfhB`，C16/C64 无插桩 Median 为 13.601305/13.964895 ms。此外核对了此后获胜的 Layer0 Gate 单算子 `gate-next-n-669ScA`。两份证据对应不同的 Gate 版本。本次做了源码核对、已封存计数器的离线关联和本地 SDK 主机 tiling 查询。[整图选型](selection.py) · [完整 Draft 测量](MEASURED.md) · [Gate 单算子配对](../draft_gate_next_n/README.md#310p3-实测)

**第一项新单算子实验应选 Down。** 将 FC 的“当前 Cube 执行时准备后继面板首片”迁入已验收的单 W16 Down。Gate 的 Next-N 已把真实 Layer0 单算子正式 Median 从 0.469935 降到 0.412700 ms。最直接的整图验证是只更新五个 Gate，与 TglfhB 配对。

## 实际实现覆盖

下表的 N/K 是每次外层 Matmul 使用的输出列数和 K 面板长度。“首片前移”指在当前 `Iterate` 成功提交后、原有全流水屏障前，对后继面板的首片做 Cast/Mul。所有行都已有同一面板内的分片搬运。表中 W16 是 UB 内的 FP16 权重暂存，B1 是交给 Cube 消费的 L1 张量。

| 实际路径 | N/K | 每核输出块 × 每块 K 面板 | W16 份数 | 激活进入 A1 | 后继面板首片前移 |
|---|---|---:|---:|---|---|
| Gate/Up，两档整图 | 64/2560 | 38 × 1 | 2 | 完整 A，每 K16 跨 16 行 gather | TglfhB 逐块准备。669ScA 单算子已通过跨 N 前移 |
| Down，两档整图 | 64/4864 | 5 × 2 | 1 | 完整 A，按 16 行 scatter | 每个 K 面板从头搬首片 |
| Q，两档整图 | 64/2560 | 8 × 1 | 2 | 完整 A，按 16 行 scatter | 跨 N |
| O，两档整图 | 64/4096 | 5 × 1 | 2 | 完整 A，按 16 行 scatter | 跨 N。每个 N16 再分两段 K2048 |
| KV C16，M32 | 64/2560 | 4 × 1 | 2 | 完整 A，按 32 行 scatter | 跨 N |
| KV C64，M80 | 128/1280 | 2 × 2 | 2 | 完整 A，按 80 行 scatter | 跨 K 及跨 N |
| FC C16，M16 | 64/2560 | 5 × 5 | 2 | 完整 A，按 16 行 scatter | 跨 K 及跨 N |
| FC C64，M64 | 160/1280 | 2 × 10 | 2 | 图内付费转置为 `[K/16,M,16]`，按面板连续装入 | 跨 K 及跨 N |

六类胜者都已有离线无损重排 INT8 与 GN scale、单份原始码缓冲、片上 Cast、用零步长复用 16 路 Group128 scale 的 FP16 Mul，以及 A1/B1 的 NZ 输入。每个权重面板供本块全部 M 行使用。O 和窄列 KV 从 Q 的 Next-N 源码生成，宽列 KV 从 FC 的多 K 面板源码生成。所以 O、KV 和 FC 实际已具备后继首片前移。早期 Brcb→B16 解决的独立 scale 展开，在这些胜者里已由直接 Mul 取代。[Q](../draft_q_bandwidth/op_kernel/next_n.cpp) · [O 源码生成](../draft_o_bandwidth/o_panel.py) · [KV 两种源码来源](../draft_kv_bandwidth/kv_panel.py) · [多面板 FC](../draft_fc_bandwidth/op_kernel/d_flash_fc_panel.cpp) · [C64 FC 装载方式](../draft_fc_bandwidth/packed_panel.py)

## 值得验证的迁移

### 1. FC 的跨 K/N 首片前移 → Down

Down 当前每个 N64 做两次 K4864 Iterate，每个 K4864 分八片。每片 Cast 后已经能预取本面板下一片 INT8。但当前 B1 填满后，`Iterate(kp != 0)` 后立即执行 `PIPE_ALL`，下一个面板随后才搬 raw 和 scale。FC 则显式区分同一 N 的下一 K、下一归属 N 的第一个 K，以及本核尾块，在当前 Cube 使用 A1/B1 时，先把下一面板的首片反量化到独立 W16。[Down 循环](../draft_down_bandwidth/op_kernel/d_flash_down_panel.cpp) · [FC 的后继索引和 W16 写入](../draft_fc_bandwidth/op_kernel/d_flash_fc_panel.cpp#L118)

Down 每核有 **5 次 K0→K1、4 次跨 N，共 9 个可利用的交接**。当前 B1 就绪屏障已完成最后一笔 UB→B1 搬运，随后 Cube 读取 B1，这给单 W16 准备后继首片留下了位置。第一版保留 1 份 raw、1 份 W16、现有 A1/B1、K4864、完整 Cube tiling 和所有已有消费者屏障。后继 W16 写入 B1 必须在当前 B1 读完之后，当前 W16 到 B1 的搬运也必须先于同一 W16 的下一次覆写。K0 初始化、K1 partial 续加，K1 结束才写回 FP16。首面板预热、本核尾块、Iterate 失败分别处理。

迁移依据是 Q 的 Next-N 直接配对使正式 Median 下降 8.02%，Gate 的同类直接配对下降 12.18%。两者都固定了各自的布局和 Cube 计划。Down 每核可前移 80 个分片中的 9 个，Gate 是 152 个中的 37 个。这是源码计数，无法用来折算 Down 的节省时间。TglfhB 的 Layer0 Down 在两个 gear 都约 209 µs，MTE2 活动约 92 µs，Vector 约 79 µs。它是目前证据最清楚的未测调度差异。[Q 直接配对](../draft_q_bandwidth/README.md#zygjul-直接配对结果) · [Gate 直接配对](../draft_gate_next_n/README.md#310p3-实测) · [五层 Down 时长](MEASURED.md#全-w8-图内各部分)

### 2. Q 的按行 A1 装载 → 新 Gate/Up

Gate 的 Next-N 试验特意保留原装载方法。Gate 每核发 160 次 gather，每次取 16 行的 K16。Q 每核发 16 次 scatter，每次取一行的 160 个 K16。已枚举两者的全部源、目的地址，都是 2,560 个 32 B 块，共 81,920 B，最终 A1 坐标均为 `((k/16)*M + row)*16 + k%16`。可以在 `panel-k2560-w2-next-n` 上只替换装载方向，保持 AIR、权重、UB、Cube 和 Next-N 的顺序。优势是变量干净、验证成本低。限制是每核只装一次 A，之后会计算 38 个 N64，命令数的 10:1 不能换算为速度比。[Gate 装载](../draft_gate_bandwidth/op_kernel/full_l1.cpp#L18) · [Q 装载](../draft_q_bandwidth/op_kernel/next_n.cpp#L19) · [Gate Next-N 明确保留原装载](../draft_gate_next_n/gate_next.py#L23)

### 3. KV/FC 的宽输出列 → 优先验证 Gate/Up

C64 KV 从 N64/K2560 换成 N128/K1280，在相同公开 ND 输入下正式 Median 从 0.166750 降至 0.147870 ms。C16 KV 做同类 N128/K1280 比较只得到 1.0284×，且 P95 回退。C64 FC 在包含输入重排的 N64→N160 直接配对中得到 1.0396×。这些结果支持按目标形状重新选宽度。它们同时包含 Cube 分块或 K 面板变化，无法单独归功于写回次数。[KV 宽列配对](../draft_kv_bandwidth/README.md#nafzjo-宽列设备结果) · [FC N160 配对](../draft_fc_bandwidth/README.md#最终设备结果)

Gate 的 N19456 可整分为 152 个 N128，8 个 block 恰好每核 19 块，Gate 与 Up 两半的边界 9728 也整除 128。首选比较 **新 Gate Next-N 的 N64/K2560 → N128/K2560**。保留完整 K、相同激活装载和 2 份 W16。每核输出写回与 Iterate 从 38 次变成 19 次。代价是每块 N16 分片从 4 片变 8 片，跨 N 前移首片的交接从 37 次变 18 次，Cube 的 baseK 也需重新选择。需要适配权重/scale 布局与首尾索引，并直接复测。N128/K1280 可另作对照，它同样每核写回 19 次，但有 38 次 Iterate。

### 4. Gate 整条 N16 连续读 → O 的 raw 搬运粒度

O 已采用 Next-N，仍将每条 N16 的连续 K4096 INT8 分成两个 32 KiB raw DMA。一个可隔离的假设是一次搬入 64 KiB raw，W16 仍按两个 K2048 分片 Cast/Mul。每个 N64 的权重 DMA 从 8 次变 4 次，传输字节相同。现有权重顺序允许这样读取。本地 SDK 在 raw 从 32 KiB 增至 64 KiB 时给出的 **200 B Cube tiling 完全一致**。潜在代价是首片等候更大的 DMA，同时要等两个半片的最后读者释放 raw 才能复用。Layer0 O 的 MTE2 两档约 59 µs，值得辨别权重 DMA 粒度的影响，现有计数无法区分其中的激活、权重与缓存效应。[O 分片地址](../draft_o_bandwidth/o_panel.py#L36) · [O 的资源](../draft_o_bandwidth/o_spec.h) · [Gate 的 N16 权重搬运](../draft_gate_bandwidth/op_kernel/full_l1.cpp)

Q N128/K2560、O N160/K2048、C16 FC N160/K1280 也有容量可行的配置。Q、O 的单算子已约 72.60、73.70 µs，C16 FC 在整图中每次只调用一次。可将这些形状放在 Down 调度和 Gate 对照之后。

## 剔除重复和低优先级试验

| 容易误判的迁移 | 已有证据及决策 |
|---|---|
| Gate 双 W16 → Down | Down 已用同一 AIR 直接比较 W1/W2。稠密合成与边界用例正确，正式 0.275080 → 0.268345 ms，下降 2.45%，低于 3% 门槛。首轮跨面板实验固定 W1，之后如需研究 W2 与新排程的交互再单独配对。[Down 原始对照](../draft_down_bandwidth/README.md#310p3-服务器结果) |
| FC C64 的图内激活打包 → KV | FC C64 的 N64 配对含打包仍从 0.730535 降至 0.396705 ms。它按 K 面板反复装 A，KV 的 N64 版本每核只装一次整 A。C64 KV N64 打包后本体 84.80 µs，计入准备的区间 109.95 µs，高于直接装载的 102.94 µs。N128 的片上位重排另测为 0.149090 → 0.215060 ms。精确的“N128 + 外部打包”并未由这两组对照单独裁决，优先级低。若上游能跨层复用已打包的相同 context 行，需要把各层不同的 16 行提议激活、生产者改动和重排全部计时。[FC 数据](../draft_fc_bandwidth/README.md#ss7hq7-与-c64-尾延迟) · [KV 数据](../draft_kv_bandwidth/README.md#最终选型与证据) |
| 更早 DMA、双 raw、双 L1 | Gate Early DMA 比普通 Next-N 的正式 Median 长 12.250 µs。Q 的双 raw 与单 raw 直接配对为 0.9705×。C64 FC 的双 L1/K640 对 N64 的配对为 1.0166×，还同时改变了 K 面板等参数。每项结论只适用于对应组合。[Gate](../draft_gate_next_n/README.md#正式单-gateup-executesync) · [Q](../draft_q_bandwidth/README.md#zsclio-设备结果与跨-n64-对照) · [FC](../draft_fc_bandwidth/README.md#最终设备结果) |
| 照搬完整 K、宽 N 或整 A 常驻 | Down 的 N64×K9728 单份 FP16 B1 已需 1,245,184 B，超过 1 MiB L1。Down 若选 N160/K2432 且保留整 A，显式 A1+B1 已需 1,089,536 B。FC C64 的完整 A 单独即需 1,638,400 B。缩短面板或改装载策略意味着新的数据流和成本。 |

已有 B1 消费者、W16 末次读取和 FP32 L0C 跨 K 续加的屏障按各自合同保留。历史窄 fence 方案的证据状态仍为 `BLOCKED_MACHINE_ADMISSION`。[生命周期审计](../draft_gate_w16_narrow_fence/narrow-fence-design.md)

## 主机 SDK 资源筛查

本次使用本地 CANN 9.0.0 主机库查询 `PlatformAscendCManager::GetInstance("Ascend310P3")` 与 `MatmulApiTiling::GetTiling`。取 A1/B1 NZ FP16、GM 输出 ND FP16，无 Bias，M=16。SDK 返回 L1/UB/L0A/L0B/L0C 容量分别为 1,048,576 / 262,144 / 65,536 / 65,536 / 262,144 B。按现有算子口径先扣显式 A1/B1 和用户 UB（含 8 KiB 预留），再核对真实返回的 base、depth、DB、transLength 与总用量。下列成功行均为主机查询，板上 ATC、bitwise 和性能验收仍适用于每个新候选。

| 目标 | 列宽 / K 面板 | 实际 base M/N/K | stepKa/Kb | 用户 UB / SDK scratch（B） | 总 L1（B） | 每核 Iterate / 输出块数 |
|---|---:|---:|---:|---:|---:|---:|
| Down 当前 / 仅改首片时机 | 64 / 4864 | 16/64/256 | 1/1 | 129792 / 132352 | 1015808 | 10 / 5 |
| Gate Next-N / 仅改 A 装载 | 64 / 2560 | 16/64/256 | 10/10 | 215552 / 46592 | 819200 | 38 / 38 |
| Gate 宽列，全 K | 128 / 2560 | 16/128/128 | 20/2 | 218112 / 44032 | 950272 | 19 / 19 |
| Gate 宽列，半 K | 128 / 1280 | 16/128/128 | 10/10 | 113152 / 148992 | 778240 | 38 / 19 |
| Q 宽列，全 K | 128 / 2560 | 16/128/128 | 20/2 | 218112 / 44032 | 950272 | 4 / 4 |
| O 宽列，保留整 A | 160 / 2048 | 16/160/64 | 32/4 | 177152 / 84992 | 1015808 | 4 / 2 |
| FC C16 宽列，保留整 A | 160 / 1280 | 16/160/80 | 16/2 | 113792 / 148352 | 962560 | 20 / 2 |

Gate N128 若同时保持 baseN=128、原 baseK=256 和双 L0B，则 L0B 需要 131,072 B。表中实际 baseK=128 时 L0B 为 65,536 B。O 的独立 raw 合并查询固定 N64/K4096、K2048 向量分片和双 W16，用户 UB 从 176,128 B 变为 208,896 B，scratch 从 86,016 B 变为 53,248 B，两次 Cube 序列化相同。Down 若将 N160/K2432 改成按面板装 A，主机返回总 L1 1,015,808 B，但会改变激活的复用方式。查询方法沿用项目的 SDK 主机样例。[SDK 查询调用示例](../draft_kv_bandwidth/test/sdk_plan.cpp)

## 同一整图的流水证据

下表来自服务器已封存 `full-all-w8-TglfhB` 的 `modules-c*/timed-tasks.csv` 与 `profile-c*/profiler/` 下的 op_summary。按已审计的投影节点、正式窗口、Stream/Task、类型和记录的开始时间关联。每个档位每行均为 30/30 唯一匹配。单元格为 **C16 / C64**，单位 µs。FC 为入口，其余为 Layer0。时长列从已归档的 FC 或 Layer0 模块时长表换算。六个流水列取每窗唯一对应的 W8 任务的计数器中位数。

| 投影 | 图内模块 | MAC | MTE1 | MTE2 | MTE3 | Vector | Scalar |
|---|---:|---:|---:|---:|---:|---:|---:|
| FC | 254.4 / 291.5 | 17.61 / 60.00 | 60.85 / 71.98 | 131.61 / 145.79 | 35.74 / 43.30 | 100.75 / 107.61 | 50.84 / 62.47 |
| Gate/Up（整图旧版） | 407.5 / 406.1 | 26.77 / 26.77 | 92.50 / 92.49 | 135.04 / 131.92 | 80.15 / 80.51 | 153.95 / 153.95 | 91.07 / 91.22 |
| Down | 208.6 / 208.8 | 13.38 / 13.38 | 45.78 / 45.78 | 91.76 / 92.04 | 27.65 / 27.81 | 78.61 / 78.61 | 42.46 / 42.46 |
| Q | 80.0 / 97.5 | 5.63 / 5.63 | 19.51 / 19.49 | 31.35 / 51.34 | 16.44 / 17.27 | 32.41 / 32.41 | 18.97 / 19.00 |
| O | 104.0 / 104.7 | 5.63 / 5.63 | 19.33 / 19.33 | 58.94 / 59.44 | 13.94 / 13.96 | 32.97 / 32.97 | 18.36 / 18.38 |
| KV | 51.7 / 82.7 | 5.04 / 11.51 | 10.97 / 11.99 | 22.30 / 38.52 | 10.40 / 21.72 | 16.26 / 17.46 | 11.28 / 13.71 |

C64 的任务归属是 `STRICT_DEVICE_ATTRIBUTION`，C16 是 `PROVISIONAL_HOST_ENCLOSURE`。FC C64 另有 21.0 µs 的图内激活布局任务。流水可重叠，MTE2 时间也不能直接给出 DRAM 读字节。此次读取没有产生候选的新延迟。[整图任务、CSV 身份及归属证据](MEASURED.md#原始证据)

## 第一组配对的边界

取真实 C16 Layer0 Down。可复用 `gate-next-n-669ScA` 已核验的 Gate 输出，经原图的 split 和 SwiGLU 生成 Down 激活，再由已验收 Down 核生成独立设备参考。对照与候选使用相同单算子 AIR、NZ/GN 对应的打包常量、8-block、W1、SDK tiling、workspace 和输入。三组输入覆盖稠密回放、Group128、K2432、K4864 边界以及 K 末坐标 9727，两种 poison 下检查全部 40,960 个 FP16 输出、Guard、Readonly、Repeat 和 Sync。

正式 Execute+Sync 采用三轮正反序，每轮每边 5 次预热及 30 次正式样本。Median 下降至少 3%、三轮同向、P95 无回退才推进。另起 msprof 比较 30 个正式窗口中设备任务的 Median/P95 及流水活动。已获胜的 Gate Next-N 与新 Down 应分别验证整图净收益。
