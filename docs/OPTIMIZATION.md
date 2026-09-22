# 未来优化

## 目标与比较口径

**测试性能基线始终是不用 DFlash 的原版**：Decode 加速比 = ordinary Decode / DFlash Decode。
FP16 Draft 约 20 ms 仅用作 W8 Draft 的优化对照，不能替代 ordinary 基线。

当前 W8 离线 NZ 的完整 Draft 平均 **47.89 ms/call**，FP16 为 **20.22 ms/call**；
W8 至少比 FP16 Draft 快 1.5 倍，按约 20 ms 的参考值规划为 **≤13.33 ms/call**。
这要求 W8 相对当前实现加速约 **3.59 倍**，并非把 47.89 ms 降低三分之一即可。
完整测量见 [测试结果](RESULTS.md)。

最终验收用同设备、同样本、同 C16/C64 调用分布分别测量 FP16 与 W8，要求
`Σ FP16 Draft 时间 / Σ W8 Draft 时间 ≥ 1.5`；分别报告两档和总体。
计时包括整张 Draft 的 FC、全部 block、KV 更新、完整 head、Top-1 和同步调用开销，
不能用单个 gate/up kernel 的倍数代替。FP16 六层、W8 五层的 checkpoint 均保持不变。
这里的 20 ms 对照是当前 FP16 实现；head/attention 优化同样可用于 FP16，
若也集成到 FP16，须另报优化后的 FP16 对照，不能将通用优化的收益全部归因于量化。

## 结构与达到目标的障碍

[DraftGraph](../framework/python/qwen35_dflash/ascend310p/incremental.py) 已把 K/V、gate/up 各合并成一次投影。
W8 每次执行 **1 个 FC + 5 × (KV、Q、O、gate/up、down) = 26 个量化投影**，
随后对 15 个候选做完整 FP16 词表 head。Q/O/MLP 的 M=16，FC 的 M=C，
KV 的 M=C+16，C 为 16 或 64。正常解码已将 context 和 block 共用的 KV 权重合并读取。

### 权重容量账本

| 范围 | FP16 Draft | W8A16 Draft |
|---|---:|---:|
| FC + 各层投影权重 | 1,268,776,960 B | 537,395,200 B |
| group scale | — | 8,396,800 B |
| 完整 FP16 head | 1,271,398,400 B | 1,271,398,400 B |
| 上述合计 | 2,540,175,360 B | 1,817,190,400 B |

不含 embedding 查询、norm、KV、activation 和 workspace；这些是逻辑容量，不是设备实测流量。
head 为 **1212.5 MiB（约 1.18 GiB）**，占 W8 表内字节量约 **70%**。

若各权重只读一遍、两版有效带宽相同且访存主导，字节量之比仅 **1.40**。
这个简化模型下，压缩投影权重本身不足以得到 1.5×；仍须减少重读、提高执行效率或削减其他开销。
它不是硬件上限：实际 cache、调度、片上反量化和累加方式会改变结果。
按 13.33 ms 处理 W8 表内全部字节，对应约 **136 GB/s** 的平均处理速率要求，
尚未计其他读写和计算；不能据此断言设备能达到。

### 优先级

| 优先级 | 方案 | 对 13.33 ms 目标的作用 |
|---|---|---|
| P0 | A：W8 group Linear | 片上分块反量化、复用权重、搬运/计算重叠；必须最终覆盖全部 26 次投影 |
| P0 | C：完整词表 head + Top-1 | 同时优化 FP16 head 的 MatMul 数据流；只省 logits 写回远远不够 |
| P1 | D/E：gate/up + SwiGLU、down + residual | 建立在 A 上的 epilogue，减少中间落显存；与 A 的收益重叠 |
| P2 | F：分段 K/V attention | 减少 cache 拼接与 scores/probability 中间量；作为其余开销的优化 |
| Prefill 专项 | G：context-only 图 | 删除建缓存时无消费者的候选分支，收益只计 Prefill |
| W4 专项 | B：W4 解包 + NZ | 合并解包与转换，不计入 W8 目标 |

已测 W8 runtime C64 profile：26 个 WeightQuant 共 40.22 ms，其中五次 gate/up 共 20.85 ms、
五次 down 共 8.35 ms；FP16 head MatMul 为 6.73 ms。
这组数据用于确定热点，不是当前离线 NZ 的分项；当前版本须重新取得逐算子时间。
已离线化的权重 TransData 不再列为待获取收益。

按该 profile，只做 gate/up、down 不能达到预算：其余 16 个 WeightQuant 加 head
已约 **17.75 ms**，还未计 attention 等。只融合激活/残差也不够：
D/E 五层合计最多避免约 7.05 MB 的指定中间张量写读；C 省下的 logits 写读约 14.90 MB，
都不能替代仍需处理的 1.27 GB head 权重。

## 完整 Draft 的工程时延预算

下面是**为达到目标而分配的预算，不是预测或已测结果**。所有区域按集成后的不重叠范围计时；
若某项超出预算，必须从其他项找到实测余量。

| 区域 | 次数 | 区域合计预算 | 每次平均预算 | 权重 + scale 容量 / 区域时间 |
|---|---:|---:|---:|---:|
| gate/up（含融合 SwiGLU） | 5 | ≤2.40 ms | ≤0.48 ms | 约 105 GB/s |
| down（含融合 residual） | 5 | ≤1.20 ms | ≤0.24 ms | 约 105 GB/s |
| FC + KV + Q + O | 16 | ≤1.40 ms | 按各形状分配 | 约 119 GB/s |
| 完整 FP16 head + Top-1 | 1 | ≤6.00 ms | ≤6.00 ms | 约 212 GB/s |
| attention、norm、RoPE、cache、布局及调用开销 | — | ≤2.00 ms | — | 单独测量 |
| 合计 | — | **≤13.00 ms** | — | 给 13.33 ms 留约 0.33 ms 余量 |

速率列只是“一遍处理权重”的量级要求，不含其他流量，不是芯片带宽测量。
当前缺少匹配此 Ascend310P/CANN 的有效带宽、片上容量/搬运路径测量以及离线 NZ 分项 profile，
因此硬件 roofline 与可实现加速上限为 **N/A**。现在能给出工程路径，尚不能保证达到 1.5×。

实现先后顺序：

1. 用真实 gate/up、down 权重与激活做 A 的 M16 最小内核；先看完整区域是否接近 0.48/0.24 ms，
   再扩展 Q/O/KV/FC，验证 M16/32/64/80，避免只优化十个热点后其余投影成为下限。
2. 同时用完整 248320 词表、M=15 测 C。按 N 分区复用全部候选行的权重 tile，
   连同 partial Top-1 归约计时；若 head 仍占大部分预算，应优先解决 head 数据流。
3. 在 A 的稳定 tile 上逐项加入 D/E；检查是否因缓冲增加、权重重读或同步增多而抵消收益。
   F 和控制量的 AI_CPU Cast 优化用于压低剩余开销，不预记未经测量的节省。
4. 接回 C16/C64 Draft，对比同输入输出/KV、完整图时延，再跑 20 条与数据集。
   更改 head 精度、缩小词表、将激活量化为 INT8 或降低层数属于另一模型/精度方案，
   不计入当前 W8A16、完整 FP16 head 的目标。

## 对完整 Decode 的影响

本次 W8 测量每次生成平均 32.3 轮，即 1 次 C64 + 31.3 次 C16；
ordinary Decode 为 4436.34 ms，Verify 累计为 1652.73 ms。
保持接受轨迹、Verify 与调用次数不变，令完整 Draft 两档耗时为 t64/t16，
其余 Decode 循环开销为 H（不含 Prefill），则：

```text
新 DFlash Decode = 1652.73 + t64 + 31.3 × t16 + H
Decode 加速比 = 4436.34 / 新 DFlash Decode
```

两档若均达到 13.33 ms，H=0 的理想预算约为 **2083.29 ms、2.13× ordinary**；
实际须加入 H。这个结果不能写成“Decode 比 FP16 DFlash 快 1.5 倍”。
只改 Draft 而 Verify 保持不变时，令 Draft 和 H 都为零，相对 ordinary 的理想上限约 **2.68×**。
这些是固定轨迹的算术边界，不是未来测试成绩；C16/C64 混合图均值不得直接当作 t16 或 t64。

## 自定义算子需求

以下均为待开发接口。代码与形状数据放在 [framework/custom_ops/](../framework/custom_ops/)；
[workloads.json](../framework/custom_ops/draft_quant/workloads.json) 保留逐形状容量、计算量及测量来源。

公共约定：同一 NPU，输入只读；输出与输入、workspace 不重叠，无隐藏状态或设备端分配。
未特别注明的 Tensor 为连续 ND，stride 按最后一维为 1 推导。
host 校验 shape/dtype/layout，给出 workspace 字节数与对齐；输出完成后才可复用缓冲区。
以下 `RN16` 表示基线的 FP16 舍入，具体数值验收见本页末尾。

### A：DFlashGroupQuantLinear

替换 `GroupQuantLinear.forward` / AIR WeightQuant 节点，保持 group-128 的量化数值：

```text
(X, W_nz, S; M, N, K, group_size=128, layout="nz_int8_v1") -> Y
W_real[n,k] = q[n,k] * S[k//128,n]
Y_real[m,n] = Σk X[m,k] * W_real[n,k]
```

| Tensor | shape | dtype / 含义 |
|---|---|---|
| X | [M,K] | FP16，stride=[K,1] |
| W_nz | [ceil(K/32),ceil(N/16),16,32] | INT8 FRACTAL_NZ，逻辑/origin=[N,K] |
| S | [K/128,N] | FP16 GN，正且有限，stride=[N,1] |
| Y | [M,N] | FP16，stride=[N,1] |

`W_nz[k//32,n//16,n%16,k%32]=q[n,k]`，q 为 signed INT8，范围 [-128,127]，padding=0。
NZ stride 为 `[ceil(N/16)*512,512,32,1]`，逻辑形状与物理存储形状分别声明。
模块中的 scale 原为 [N,G]，转 GN 只改变布局、保留 FP16 位型。
无 bias、antiquant_offset、输出量化；不把激活转 INT8，不在 forward 中转换整张权重。

固定工作负载，C=16/64：

| 投影 | 每次 Draft 次数 | X | W_nz | S | Y |
|---|---:|---|---|---|---|
| gate/up | 5 | [16,2560] | [80,1216,16,32] | [20,19456] | [16,19456] |
| down | 5 | [16,9728] | [304,160,16,32] | [76,2560] | [16,2560] |
| Q | 5 | [16,2560] | [80,256,16,32] | [20,4096] | [16,4096] |
| O | 5 | [16,4096] | [128,160,16,32] | [32,2560] | [16,2560] |
| K/V | 5 | [C+16,2560] | [80,128,16,32] | [20,2048] | [C+16,2048] |
| FC | 1 | [C,12800] | [400,160,16,32] | [100,2560] | [C,2560] |

K 为 128 的倍数，N 为 16 的倍数；tiny 用例 K=256、N=64。
gate/up 输出前 9728 列为 gate、后 9728 为 up；K/V 前 1024 列为 K、后 1024 为 V。

每次 gate/up 的 X/W/S/Y 分别为 81,920 / 49,807,360 / 778,240 / 622,592 字节，
乘加 796,917,760 次；down 分别为 311,296 / 24,903,680 / 389,120 / 81,920 字节，
乘加 398,458,880 次。A 不减少这些逻辑计算量，收益来自更高效的执行。

建议沿 N 分块，一个 owner 完成输出列的整个 K 归约；权重 tile 复用全部 M 行，
片上反量化并流水预取。scale 必须随 K-group 变化，不能改成 GEMM 后乘一个列 scale。
split-K、归约重排及反量化 dtype 的变化须数值验证。基线属性为
`transpose_x=false, transpose_weight=true, antiquant_group_size=128, inner_precise=0`。

**首版内核组织：**

1. 编译时固定 M/N/K；按 N tile 分工，优先试 `M_tile=16、N_tile=64/128、K_tile=128`。
   K tile 对齐一个 scale group；扩到 K_tile=256 时必须使用两个 group 的 scale。
2. 从离线 NZ 直接搬入 INT8 tile 和 GN scale，在片上完成 signed INT8→浮点和逐 group 缩放；
   将符合原生舍入的 FP16 权重 tile 送入矩阵乘路径。激活仍为 FP16，
   禁止把整个 [N,K] 反量化为 FP16 后写回全局内存。
3. 同一权重 tile 服务该块全部 16 行。FC/KV 的 M=32/64/80 另外调度，
   在片上资源允许时复用权重 tile 处理多个 M 子块；不能为每个 M 子块无条件重读整张权重。
4. 预取下一块、反量化当前块、矩阵乘已就绪块重叠执行。DMA 完成才能反量化，
   反量化完成才能矩阵乘，矩阵乘的最后一个消费者完成后才能复用缓冲；
   具体存储级间路径和事件须按本机 310P 工具链实现，不照搬其他芯片。
5. 每个 owner 独占 `Y[m0:m1,n0:n1]` 并完成整个 K 归约；首版不做跨 owner split-K。
   只在归约结束后执行基线规定的输出舍入、写回。不得用 FP16 累加替代未经确认的原生累加。

以 M_tile=16、N_tile=64、K_tile=128 为例，单个逻辑 tile 的最低 payload：

| 缓冲 | shape / dtype | 字节 |
|---|---|---:|
| X | [16,128] FP16 | 4096 |
| q | [64,128] INT8 | 8192 |
| group scale | [64] FP16 | 128 |
| 反量化 W | [64,128] FP16 | 16384 |
| 累加器候选 | [16,64] FP32 | 4096 |

前四项双缓冲加一份累加器为 61,696 B，但它们分属不同存储级，
**不代表“有 64 KiB UB 就能放下”**。host/tiling 必须另列 UB/L1/L0A/L0B/L0C 的实际分配、
转排布临时区、对齐和生命周期；归约顺序/累加模式以同输入原生输出验证。
若资源不够，先减 tile 或调整流水，不隐式溢出到全局内存，也不引入精度降级。

### D：DFlashW8GateUpSwiGLU

接入 gate/up 投影到 SwiGLU 区域，每层一次，输出直接供 down 使用。

```text
(X, W_gate_up_nz, S_gate_up; M=16, H=2560, I=9728, group_size=128) -> Z
[G16,U16] = split(A(X,W_gate_up_nz,S_gate_up), I)
A16 = SiLU_baseline(G16)
Z = RN16(A16 * U16)
```

X/W/S 采用 A 的 gate/up 契约，Z 为 **FP16 ND [16,9728]**、311,296 字节。
保留 Linear→FP16、SiLU→FP16、乘 up→FP16 三个边界，不直接使用 FP32 累加器做激活。
不输出 gate/up 临时量，不融合输入 RMSNorm。

若分离图物化 gate/up 输出，每层可避免其写入与重读 **1,245,184 字节**，五层约 6.23 MB。
矩阵乘收益归 A；D 的额外收益仅计激活融合，并扣除片上资源增加的成本。

按输出列 j 配对 gate 的 j 列和 up 的 j+9728 列；它们在现有合并权重中并不相邻。
两路 tile 都完成 K 归约和 FP16 舍入后，才做 SiLU 与相乘。
可比较两次对应 tile 搬运与版本化的离线交错布局；后者必须逐字节逆变换，
不改变 GN scale 对应关系。完整 [16,19456] 中间结果不出片上，
但不能因两套累加器让权重重读增加。该区域整体受前述五层 2.40 ms 预算约束。

### C：DFlashDraftLmHeadTop1

接入 `AirDFlashOps.top1`，仅替换 Draft head，完整词表逐列计算。

```text
(hidden, head_weight; M, K=2560, N=248320, layout="fp16_nk") -> token_id
```

| Tensor | shape | dtype / 含义 |
|---|---|---|
| hidden | [M,2560] | FP16 ND，已过 final norm；生产 M=15，支持 1..15 |
| head_weight | [248320,2560] | FP16 NK，共享原 head，stride=[2560,1] |
| token_id | [M] | INT64 ND，原始词表 ID，范围 [0,248319] |

先得到基线 **FP16 logits** 再 argmax；相等时取最小 ID。不能比较未舍入的 FP32 logits，
不能裁剪词表。无 bias、softcap、温度；无效候选尾部仍由调用方遮挡。
预排布 FP16 权重需另定索引和逆变换，不能套用 INT8 NZ 布局。

M=15 的 hidden 为 76,800 字节，输出 120 字节，完整计算为 9,535,488,000 次乘加。
按词表分 P 个分区，workspace 声明 FP16 partial_max[P,M] 与 INT64 partial_id[P,M]，
未对齐容量 `10*P*M` 字节，最终归约等待所有分区完成。
可避免 logits 写回/重读 **14,899,200 字节**，但仍需处理 **1,271,398,400 字节** head 权重。
partial 归约和启动计入成本，不能把整个 head 耗时算成节省量。

**首版实现：**按词表 N 分区，每个分区用矩阵乘 tile 同时处理 15 个候选行，
物理可补齐 16 行，补齐行不输出 token。沿 K=2560 完整归约，生成一小块 FP16 logits，
就地更新各行 (max,id)，最后归约各分区的 [P,M] 结果；head 权重 tile 应复用所有候选行，
不能每个 token 独立扫描完整词表。比较 max 时保持 FP16 边界、原始 ID 和最小 ID tie-break。

该算子达到 ≤6 ms 预算的关键是完整 head MatMul 的实际执行效率。
仅删除 14.90 MB logits 写读不等于省去 1.27 GB head 读取；
原生 head 已接近本机带宽极限时，Top-1 融合可能仅有小幅收益。
需同时报告权重实际读取量、tile 重读次数、分区负载、归约耗时和整张 Draft 时间。

### E：DFlashW8DownResidual

替换 down 与残差 Add，删除调用方原 Add，避免重复相加。

```text
(Z, W_down_nz, S_down, R; M=16, K=9728, N=2560, group_size=128) -> H_out
D16 = A(Z,W_down_nz,S_down)
H_out = RN16(R + D16)
```

Z/W/S 采用 A 的 down 契约。R、H_out 均为 **FP16 ND [16,2560]**，各 81,920 字节。
R 为 post-attention residual hidden；不广播、不原地修改。
先舍入 down 输出再加残差，不使用 `RN16(R+FP32_accumulator)`，不融合下一层 norm。
每层可避免 down 中间量写入/重读 **163,840 字节**，五层 819,200 字节；与 A 的收益不重复计数。

残差在最终输出 tile 阶段读取，完成 `down→FP16→residual add→FP16` 后写入独占输出区。
与下一层 RMSNorm 融合会引入跨列归约，首版分开实现。A 的 down 加 E 合计受五层 1.20 ms 预算约束。

### F：DFlashSegmentedGqaAttention

替换 cache/block Concat、QK、mask/softmax、PV；上游 Q/K norm 与 RoPE 保留。

```text
(Q, K_cache, V_cache, K_block, V_block, mask; L, scale=128**-0.5, kv_groups=4) -> O
```

| Tensor | shape | dtype / 含义 |
|---|---|---|
| Q / O | [1,32,16,128] | FP16 ND，Q 已做 norm/RoPE |
| K_cache / V_cache | 各 [1,8,L,128] | FP16 BHLD，L 为部署容量 |
| K_block / V_block | 各 [1,8,16,128] | FP16，临时 anchor/MASK KV |
| mask | [1,1,16,L+16] | BOOL，True 可见，广播到 Q heads |

Q head h 使用 KV head `h//4`。cache/block 分别寻址，head stride 为 `L*128` / `16*128`，
行 stride 均为 128；Q/O stride 为 [65536,2048,128,1]。
逻辑顺序为 cache 全部 L 行，再接 block 16 行；输入只读，不更新 cache。
输出仅 O，调用方恢复 [1,16,4096] 供 O projection。

```text
scores32 = MatMul16_then_float(Q,K.T) * scale
scores32 = where(mask,scores32,-inf)
P16 = RN16(softmax(scores32, dtype=FP32))
O16 = RN16(MatMul16_then_float(P16,V))
```

基线 QK/PV 舍入按冻结 OM 确认，保留 FP32 Softmax 和 PV 前 FP16 概率。
直接改成在线 FP32 attention 累加不保证数值等价。mask 由调用方携带完整的有效区、候选数、
因果/双向和滑窗限制；支持带洞 mask，不按单一前缀长度推断。全 mask 行按基线处理。

Q/O 各 131,072 字节；cache 对共 `4096*L` 字节，block 对 65,536 字节，
mask 为 `16*(L+16)` 字节。若原图物化拼接，可避免每层 **8192*(L+16) 字节**写回/重读；
另可减少 scores/probability 中间量。布局适配、重算、workspace 和启动须一并计时。

### G：DFlashContextOnly（Prefill 图优化）

用于非最后一个完整 prompt 块，仅更新 context KV。首版复用已验证算子，不要求做成一个 kernel。

```text
(features, start_position, valid_rows, d0_key, d0_value, ..., d4_key, d4_value;
 C=64, L, source_layer_ids, draft_layer_ids, norm_epsilons)
 -> (d0_key_next, d0_value_next, ..., d4_key_next, d4_value_next)
```

| Tensor / 常量 | shape | dtype / 含义 |
|---|---|---|
| features | [1,64,28160] | FP16 ND，按 checkpoint 层号选列为 [1,64,12800] |
| start_position / valid_rows | 各 [1] | INT64 / INT16；首版 p>=0、v=64、p+64<=L |
| 输入 / 输出 state | 各 10 个 [1,8,L,128] | FP16 BHLD，按层号、先 K 后 V；current/next 不别名 |
| FC W / S | [400,160,16,32] / [100,2560] | INT8 NZ / FP16 |
| 每层 KV W / S | [80,128,16,32] / [20,2048] | INT8 NZ / FP16 |
| hidden / K norm gamma | [2560] / [128] | FP16，沿对应末维归约 |
| RoPE inv_freq / epsilon | [64] / 标量属性 | FP32 / 原 checkpoint 配置 |

权重、scale、gamma、inv_freq 是只读模型常量。
features 容量 3,604,480 字节，两个控制量为 8 / 2 字节；输入/输出 state 各占 `20480*L` 字节。
层号按 checkpoint 顺序选取，不能直接截前 12800 列。

处理：选列 → FC → hidden norm；各层 context KV 投影 → K norm/RoPE → 写入 [p,p+64)；
V 不做 norm/RoPE，其余 cache 位型不变。RoPE 保留原 rotate_half、位置与 cos/sin 舍入。
输出完整 cache，含未写区复制成本；不输出候选、不提交 block KV。完成后 runner 交换状态并推进 cursor。
首版仅 v=64；扩展部分行需匹配原图物理 scratch 行语义。

投影调用由 26 降至 6，C64 量化投影乘加从约 11.85G 降至 3.77G，
另删除约 9.54G 的 head 乘加。context KV 的 M 从 80 改为 64，需要重新验证舍入和后续候选。
1024-token prompt 有 15 次此类准备调用；收益只计 Prefill，**Decode 收益记 0**。
新增 OM 常量和 workspace 计入显存。

### B：DFlashW4UnpackNZ（W4 专用）

```text
(packed; N, K, layout="nz_int8_v1") -> q_nz
q[n,2j]   = (packed[n,j] & 15) - 8
q[n,2j+1] = (packed[n,j] >> 4) - 8
```

输入 **UINT8 ND [N,K/2]**，偶 K 在低四位，offset-binary 编码；
输出 **INT8 NZ [ceil(K/32),ceil(N/16),16,32]**，origin=[N,K]，padding=0，值域 [-8,7]。
K 为正偶数，使用 A 的投影形状；无 scale、浮点计算或状态。
例如 0x80→(-8,0)，0x88→(0,0)，不是 signed INT4 补码。

若原实现物化 ND INT8 权重，可省去 **2*N*K 字节**中间写回/重读，仍需写读 NZ 输出并执行 GEMM。
每轮转换保留 W4 压缩常驻；离线或加载时缓存 INT8 NZ 会增加约一倍权重存储/常驻量。
此项不计入 W8 收益。

## 精度、性能与交付验收

| 对象 | 无损替换要求 |
|---|---|
| W8/W4 code、scale、离线载体 | 0 位型差异；逆变换逐字节还原，padding 符合约定 |
| A / D / E / F 输出 | 与同输入原生 OM 子图逐元素、逐位型一致；atol=0、rtol=0，报告 ULP |
| C | FP16 logits 舍入一致，Top-1 ID 完全一致；覆盖跨 tile tie、近似并列及特殊值 |
| G | 全部 KV 位型一致，包括未写区；切回正常 Draft 的候选与状态一致 |
| 集成 | 同量化 Draft 的候选/KV 轨迹匹配；完整模型另外以 ordinary 验证 token/EOS/stop reason |

dtype 与 inner_precise 不能单独确定内部累加精度。须记录反量化乘法、Cube 累加、
split-K 次序、各 FP16 舍入及溢出行为；CPU 公式用于定位索引，目标 OM 才是运行基线。
norm 保留 FP32 reduction/rsqrt → normalized 转 FP16 → 乘存储 gamma 的顺序。
接受率相近不能代替数值验收；改变归约或舍入的方案需另列误差和模型影响。

覆盖真实权重/激活、所有档位、group/tile 边界、signed code、±0、极值、padding、输出哨兵；
集成覆盖 63/64/65 边界、C16/C64 切换、零接受/全接受、EOS、重复请求及 current/next 切换。

性能用当前 baseline/candidate 同设备、同输入配对，预热 3 次、测量 10 次；
报告原始时间、median/p95、区域与完整 Draft/Decode、workspace 和峰值显存。
子图数值对照使用同一个 W8 checkpoint 的原生子图，性能报告基线仍是 ordinary。
另用 FP16 Draft 测完整 Draft 的 1.5× 优化目标；C16/C64 分别报告，
最终调用分布加权的时延比 ≥1.5，任何档位的回退和额外调度都计入总时间。
布局适配、归约启动和输出复制均计入；A 与 D/E 收益不重复相加，G 只计 Prefill。
时差未超过波动或完整 Decode 变慢，不宣称提速。

交付 host/tiling、kernel、GE/torch 注册、构建与 compare 入口、版本化接口/载体、
native golden、数值结果与设备性能；编译通过、OM 数值、速度和模型一致性分别验收。
