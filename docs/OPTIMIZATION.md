# 未来优化

优先优化 W8A16 的小 M、group-128 矩阵乘，先覆盖 gate/up、down。
当前基线为离线 NZ、静态 C16/C64、Chunk，20 条输入：Draft **47.89 ms/call**，
Verify **51.17 ms/call**，Decode **1.39×**。完整数据见 [测试结果](RESULTS.md)。

## 优先级与收益依据

| 优先级 | 方案 | 收益来源 |
|---|---|---|
| P0 | A：W8 group Linear | 片上分块反量化、权重复用与搬运/计算流水；先做 gate/up、down |
| P1 | D：gate/up + SwiGLU | 在 A 上融合激活，减少中间张量写回和重读 |
| P1 | C：完整词表 head + Top-1 | 分块计算、归约，避免完整 logits 落显存 |
| P2 | E：down + residual | 在 A 上融合残差，减少 down 中间量 |
| P2 | F：分段 K/V attention | 减少 cache 拼接与 scores/probability 中间量 |
| Prefill 专项 | G：context-only 图 | 建缓存时只计算 FC/KV/norm/RoPE，删除无消费者的候选分支 |
| W4 专项 | B：W4 解包 + NZ | 合并解包与格式转换；不影响 W8 |

已提供的 C64 profile 中，gate/up 五次合计 20.85 ms、down 五次 8.35 ms、head 6.73 ms。
该 profile 来自 runtime 权重转换版本，只用于定位优先级；尚无当前离线 NZ 的单算子分解，
不能把这些时间当作 47.89 ms 的组成，也不能再次计入已离线化的权重 TransData 收益。

量化投影的 W8 权重共 512.5 MiB、FP16 scale 约 8.01 MiB；FP16 head 单独为 1212.5 MiB。
这些是张量容量，实际访存量还取决于 tiling 与重读。K/V 和 gate/up 已合并投影，不能重复计收益。

其他代码层机会：减少控制量的 AI_CPU Cast；按实际候选数选择更小计算档位；
或用更多显存缓存热点反量化权重。后两者分别可能改变接受轨迹、浮点舍入，
均需同输入配对验证，不能按权重压缩比例承诺提速。

## Decode 收益预算

本次 DFlash Decode 平均 3193.72 ms，ordinary 为 4436.34 ms；
每次生成平均 32.3 轮，即 1 次 C64 + 31.3 次 C16 Draft。
保持接受轨迹及 Verify 不变，若两档每次分别节省 `δ64 / δ16` ms：

```text
新 Decode 时间 = 3193.72 - δ64 - 31.3 × δ16
新 Decode 加速比 = 4436.34 / 新 Decode 时间
```

| 每次 Draft 两档均节省 | 新 Decode ms | 相对 ordinary 的 Decode 加速比 |
|---|---:|---:|
| 1 ms | 3161.42 | 1.40× |
| 5 ms | 3032.22 | 1.46× |
| 10 ms | 2870.72 | 1.55× |

这是固定轨迹下的敏感性计算，不是性能预测。Verify 累计约 1652.73 ms；
即使其余 Decode 开销全部消失，相对当前 DFlash 的上限也约为 1.93×。
因此，仅优化 Draft 不能支持“完整 Decode 比当前版本快 4 倍”的目标。

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
布局适配、归约启动和输出复制均计入；A 与 D/E 收益不重复相加，G 只计 Prefill。
时差未超过波动或完整 Decode 变慢，不宣称提速。

交付 host/tiling、kernel、GE/torch 注册、构建与 compare 入口、版本化接口/载体、
native golden、数值结果与设备性能；编译通过、OM 数值、速度和模型一致性分别验收。
