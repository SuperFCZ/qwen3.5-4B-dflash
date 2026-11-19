# 六类 W8A16 投影的 Scalar、Vector 和搬运工作

数据来自 `w8a16-deep-profile-acJHJJ/profile-paired-analysis/{raw-c16,raw-c64}/case-analysis.json`。每档 30 个正式插桩窗口，单位为 **µs/Execute 的同类 Task Duration Sum 均值**。每层角色由 projection-map 对齐图节点和冻结投影。FC 一个，其余各五个。严格 Device Step 均未解决。二进制级动态指令计数和 stall 计数均未取得。

| 角色 | M(C16/C64) × K × N | Raw C16 / C64 (µs) | 每个投影 N64 × K 步数 | 每个投影权重/scale 搬运和反量化路径 |
|---|---|---:|---:|---|
| Gate/Up | 16 × 2560 × 19456 | 9641.374 / 9643.620 | 304 × 10 | K256、两 raw bank、broadcast |
| Down | 16 × 9728 × 2560 | 4707.959 / 4705.750 | 40 × 38 | K256、两 raw bank、broadcast |
| Q | 16 × 2560 × 4096 | 2205.911 / 2201.692 | 64 × 10 | K256、单 raw bank、serial、broadcast |
| O | 16 × 4096 × 2560 | 2179.134 / 2173.077 | 40 × 16 | K256、单 raw bank、serial、broadcast |
| KV | 32/80 × 2560 × 2048 | 1151.500 / 1767.229 | 32 × 10 / 32 × 20 | C16 K256 serial/broadcast。C64 M80 A-UB、K128、scalar scale |
| FC | 16/64 × 12800 × 2560 | 1331.564 / 1414.721 | 40 × 50 | K256、单 raw bank、serial、broadcast |

计算以下工作量使用已验收 Raw 生成配置：batched、pipeline=prefetch、scale=broadcast、Gate=raw、copy=loop、KV M80=a-ub、core_limit=0。pipeline 配置由形状谓词再选择实际路径。源码为 `draft_quant/op_kernel/d_flash_group_quant_linear.cpp` 第 34–62、79–104、126–220、239–287 行及配套配置宏。[Kernel](../draft_quant/op_kernel/d_flash_group_quant_linear.cpp) · [K256 与 KV 例外](../draft_quant/A5_1.md)

## 每个 N64 的一次 K 处理

| 动态调用或配对 | Gate、Down K256 | Q、O、KV32、FC K256 | KV80 K128 |
|---|---:|---:|---:|
| GM→UB DataCopy：q / s / 显式 stage A | 8 / 2 / 0 | 8 / 2 / 0 | 4 / 1 / 1 |
| Cast，K32→[64,tileK] FP16 的分片 | 8 | 8 | 4 |
| Brcb / Mul | 2 / 2 | 2 / 2 | 0 / 0 |
| scale GetValue / Muls | 0 / 0 | 0 / 0 | 64 / 64 |
| 应用显式 PIPE_V | 6 | 6 | 2 |
| B16 Brcb 实现内部 PIPE_V | 48 | 48 | 0 |
| MTE2_S、MTE2_V 各自的 Set/Wait 对 | 各 1，ID 跨 K 保留 | 各 1，即取即用 | 各 1，即取即用 |
| 应用 V_S 的 Set/Wait 对 | 1 | 0 | 0 |
| Iterate / K 内 PIPE_ALL | 1 / 1 | 1 / 3 | 1 / 3 |

另计每 N 的一次 GetTensorC、一次 PIPE_ALL。每个使用广播的 block 启动前一次 Duplicate 和 PIPE_V。表统计应用调用与指定 SDK 循环，未展开的 Matmul 内部流水另计。工具链会展开、合并或插入指令，需要最终 ATC 专用 ELF/反汇编才能报告实际 ISA 数。应用行 54–61 每块 q 的八次 Copy 都是 2048 B，不是八次 32 B。每组 scale Copy 为 128 B。

每次 B16 Brcb(8,{1,8}) 在 310P 的 SDK 中做 8 轮保存 rear、构造向量、转置、恢复 rear。每轮直接调用 3 个 vadds、1 个 vtranspose、3 个 PIPE_V，并切换 mask。一次 K256 的两个 Brcb 因而有 48 个 vadds、16 个 transpose、48 个内部 PIPE_V。外层 6 个来自 Cast 后 1、两个组各 2、最后 1。当前 Raw Gate 的 1130 µs Scalar、743 µs Vector、453 µs MTE2 每 task 与 93.7% Cube Utilization 来自不同计数口径。它们可以重叠，且 Gate scale 的显式 GetValue 实际是 **0**。不能把 Scalar 时间解释成 GetValue 或可消除的等待。[SDK 行号和哈希](w16-event-lifecycle-revised.md)

一个 Gate 投影有 3040 次 Iterate、24320 次 Cast、各 6080 次 Brcb/Mul、18240 个循环内显式 PIPE_V、145920 个 Brcb 内部 PIPE_V、3344 次应用 PIPE_ALL。一个 Down 是 1520 次 Iterate、各 3040 次 Brcb/Mul、9120 个循环内显式 PIPE_V、72960 个 Brcb 内部 PIPE_V、1560 次应用 PIPE_ALL。一个 KV80 投影有 40960 次 scale GetValue 和相同次数的 Muls。表内总数为所有 block 合计。

## 单变量候选与反例

下表的前后数量是每个 N64/K 步的**源码发射点按循环展开后的次数**。具备协议依据与能得到实测加速是两个独立门槛。

| 单项变化 | 原 → 候选 | 原因、依赖和资源 | 判定与风险 |
|---|---|---|---|
| Gate 使用另一 W，把 D(k+1) 前移 | 8 Cast、2 Brcb、2 Mul、1 Iterate、1 V_S、1 K 全流水 fence 全部 1:1 保留。每 N 有 9 个可改变先后关系的边界 | W 增 32768 B，SDK scratch 同量减少。同代 DMA 必须先就绪，下一次 rebind 仍在 fence 之后 | 首要研究项。若 SDK 已经提前等完 Cube，可隐藏的工作量会很小。 |
| 减少 scalar scale 读取 | Gate/Down/Q/O/FC/KV32 已是 0→0。KV80 如另做 broadcast，可由 64 GetValue+64 Muls → 1 Brcb(8)+1 Mul | 仍用每通道原 FP16 scale 和一次 FP16 乘法。KV80 有独立 80×128 A staging。需新 2304 B broadcast 和 8192 B 顶端保留并重新 tiling，新增 SDK 内部 24 个 PIPE_V | 只在 KV80 另立实验。历史选择标明排除该 shape。不能把 Gate 的资源或验收代入。 |
| 合并两组 scale 的 Brcb | 2×Brcb(8) → 1×Brcb(16)，2 Mul → 2 Mul。内部 16 个 repeat 和 48 个 PIPE_V 均保留 | 源 s 为连续 128 half，目的两组各 1024 half，第二次 Mul 的 scale 起点 +1024 half。目的需 4096+256=4352 B，增 2048 B。尾部必须可读写，初始化覆盖整个区 | 只减少一次 API 进入，不能按 2→1 宣称核心工作减半。不同时改数学、共享 reserve 或 K Tile。 |
| 仅折叠 broadcast 在最后一组后的重复 PIPE_V | 外层 6→5。内部 48→48 | 210 行最后一组 Mul 后已有 PIPE_V，至 220 行只有循环控制。保留 194、202、210 及下一消费者 V_S/PIPE_ALL。scalar 路的 220 仍需保留。UB 增量 0 | 必须绑定编译后的真实指令。若原编译器已折叠，停止此项。不能移除保护两个组共用 expanded 的 210。 |
| 原 q 地址循环改用递推 | K256 的 8 次完整 q 源偏移表达式 → 1 个初值 + 7 次常量增量。scale 2 次完整表达式 → 1 初值 + 1 增量。DMA 10→10 | 每 K32 平面跨 globalN×32 B，每组 scale 跨 globalN×2 B。保持源有符号顺序、strides、32 B 对齐。UB 增量 0 | 编译器可能已做同样的强度削减。须查专用 ELF。源乘法个数不能直接作为 Scalar 周期。 |
| **后备：Down 专用二维 raw 搬运** | q 8→1，s 2→1，即每个 Down 的 15200→3040 次 Copy。拷贝字节数不变 | 原 Down NZ 平面间距大。用两条 basic DataCopy 的 blockCount/blockLen/srcStride/dstStride 分别为 (8,64,2496,0)、(2,4,156,0)，len/gap 均以 32 B 计。保留双 raw bank、单 W 和两个 Wait。UB 增量 0 | 参数以 Down N=2560 推导。Gate N=19456 的 stride 不能照搬。必须验证全部 40×38 块及末端边界。旧 Gate 二维拷贝只有约 0.36% 收益，故不重复 Gate 实验。 |
| 去掉 raw MTE2_S 或依靠 Iterate 返回代替交接 | 假设每 K 某方向 1 次 Set/Wait→0。批准数量仍为 1→1 | 改变 Scalar 发射和第三方 SDK 输入的先后关系。复用 UB 的 MTE3 读与 Vector 写需分别证明。单凭外部 GetValue=0 不能删除 Scalar 依赖 | 不作为本轮候选。保持原 V_S、PIPE_ALL、每个输入方向的 Wait 和现有失败排空。 |

数值恒等条件为 INT8→half 的精确 Cast、按原通道/Group128 位型执行一次 FP16 乘、每个 N owner 内同序 FP32 K 累加、最终一次 FP16 输出。禁止代换为 FP32 scale 乘、跨组共用 scale、提前 GetTensorC 或拆 K 后累加 FP16。CPU 覆盖了 bank 轮换、Group128 分界两侧、各 K32 平面的 ND 目的索引以及舍入顺序反例。正式 Golden 完整逐位比较仍是设备门槛。
