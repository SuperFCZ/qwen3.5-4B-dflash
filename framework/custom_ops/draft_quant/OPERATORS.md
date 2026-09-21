# 量化 Draft 自定义算子需求

目标：Ascend310P，接收端 CANN 9.0.0。以下是开发需求，不表示已有对应 kernel。
当前主线为 **W8A16 + 离线 NZ + 静态 C16/C64 OM**，首版优先 A 的 gate/up、down。
固定形状见 [workloads.json](workloads.json)，顺序与收益预算见 [分析](README.md)，
测试与交付见 [验收要求](VALIDATION.md)。B 仅用于后续 W4；G 是模型图改造接口。

约定：`H=2560, I=9728, B=1, T=16, Hq=32, Hkv=8, D=128`；
`C=16/64` 是新增 context 的物理行数，`v<=C` 是有效行数，二者不能混用。
所有运行时 Tensor 位于同一 NPU；权重只读、离线加载，不在 forward 中打包。
Tensor 表未注明时均连续 ND；workspace 由 host 查询，不能与输入、输出或其他实例重叠。
实现需给出每个支持形状的 workspace 字节数、对齐、分块/尾块规则、事件依赖和回退条件。

## A：DFlashGroupQuantLinear

替换 FC、Q、K/V、O、gate/up、down，优先 gate/up 和 down。
接入 `GroupQuantLinear.forward` / AIR WeightQuant 节点；首版 W8 仅做 Linear。
图适配必须复用现有离线 NZ 载体和尺度，不能落回每轮 ND→NZ。

```text
DFlashGroupQuantLinear(X, W, S; bits=4|8, M, N, K,
                      group_size=128, weight_layout) -> Y
```

| 输入/输出 | 逻辑形状 | dtype | 布局与含义 |
|---|---|---|---|
| X | `[M,K]` | FP16 | 连续 ND，row stride=K；上层 `[1,M,K]` 展平 batch |
| W8 | `[N,K]` | INT8 | 有符号量化 code，范围 `[-128,127]` |
| W4 | `[N,K/2]` | UINT8 | 两个 offset-binary 4-bit code/byte，偶数 K 在低四位 |
| S | `[G,N]`，`G=K/128` | FP16 | 连续 ND，stride=`[N,1]`；严格正、有限 |
| Y | `[M,N]` | FP16 | 连续 ND，stride=`[N,1]`；上层恢复 `[1,M,N]` |

M：FC 为 16/64，K/V 为 32/80，其他投影为 16。K/N 为 128/16 的倍数；tiny 为 K=256,N=64。
若实施 G，另增加 context-only K/V 的 M64 用例；它不是当前 combined Draft 的 K/V 档位。
输入只读，Y 与输入、workspace 不重叠。未支持的形状明确报错或由宿主分发至原路径。
无 bias、antiquant_offset、输出 quant_scale/quant_offset；不量化激活。

模块中的 S 为 `[N,G]`，checkpoint scale 转为 FP16 后使用。
本 ABI 采用 `[G,N]`，离线转置保持 FP16 位型；原生图已采用此布局，无需再次转置。

| 调用点 | X | W8 NZ 物理形状 | S | Y | 每次 Draft 调用数 |
|---|---|---|---|---|---:|
| gate/up | `[16,2560]` | `[80,1216,16,32]` | `[20,19456]` | `[16,19456]` | 5 |
| down | `[16,9728]` | `[304,160,16,32]` | `[76,2560]` | `[16,2560]` | 5 |
| Q | `[16,2560]` | `[80,256,16,32]` | `[20,4096]` | `[16,4096]` | 5 |
| O | `[16,4096]` | `[128,160,16,32]` | `[32,2560]` | `[16,2560]` | 5 |
| K/V | `[C+16,2560]` | `[80,128,16,32]` | `[20,2048]` | `[C+16,2048]` | 5 |
| FC | `[C,12800]` | `[400,160,16,32]` | `[100,2560]` | `[C,2560]` | 1 |

### 权重编码与数学定义

W4 沿 K 打包，解包规则：

```text
b = W4[n,j]
q[n,2*j]   = (b & 0x0f) - 8
q[n,2*j+1] = (b >> 4)   - 8
0x80 -> (-8,0)；0x0f -> (7,-8)；0x88 -> (0,0)
```

原始 INT32 checkpoint packed words 需先转为上述 byte ABI。W4 不是 signed INT4 补码；
W8 直接读取 INT8 q，不减 128。所有 q 和 scale 保留原 checkpoint 数值。

| weight_layout | 存储形状与格式 | 索引 |
|---|---|---|
| `nk_int8` | ND INT8 `[N,K]` | `q[n,k]` |
| `nz_int8_v1` | FRACTAL_NZ INT8 `[ceil(K/32),ceil(N/16),16,32]` | `W[k//32,n//16,n%16,k%32] = q[n,k]` |
| `nk_u4_offset_binary` | ND UINT8 `[N,K/2]` | 上述 W4 解包规则 |

NZ padding 为 INT8 0，origin 为逻辑 `[N,K]`；推导 K/N 使用逻辑形状。
NZ 必须实际重排字节。新增 W4 分块载体需定义版本和逆变换；W4 零填充 byte 为 `0x88`。

```text
W_real[n,k] = q[n,k] * S[floor(k/128),n]
Y_real[m,n] = sum(k=0..K-1, X[m,k] * W_real[n,k])
```

scale 随 K-group 变化，不能改成整次 MatMul 后乘单个列 scale。
逐 group 部分和再缩放也可能改变浮点舍入，需按下文精度要求比较。

运行基线为同 q/S 的 CANN `WeightQuantBatchMatmulV2`：X=FP16、W=INT8 NZ、S=FP16 `[G,N]`，
`transpose_x=false`、`transpose_weight=true`、`antiquant_group_size=128`、`inner_precise=0`，Y=FP16。

### 实现约束

按小 M 复用权重 tile；W4 在局部解包/反量化，避免每轮生成整张 INT8/FP16 中间权重。
W8 直接消费离线布局，减少固定权重转换。报告 padded tile、实际读写量和 workspace。
基础字节数：X=`2*M*K`、W8=`N*K`、W4=`N*K/2`、S=`2*G*N`、Y=`2*M*N`。

gate/up 的 W8/W4 权重分别为 49,807,360 / 24,903,680 字节，完整 FP16 为 99,614,720 字节。
gate/up 输出前 9728 列为 gate，后 9728 列为 up；K/V 输出前 1024 列为 K，后 1024 列为 V。
若后续融合 SwiGLU，保留 Linear→FP16、SiLU→FP16、乘 up→FP16 的基线舍入边界。

建议先沿 N 分块，让一个 owner 完成所负责输出列的整个 K 归约；同一权重 tile 复用全部
16 行，K tile 按 group 边界读取对应 S。只将当前/下一 tile 反量化到片上，
流水衔接搬运、反量化与矩阵乘，避免 GM 中的整张 FP16 权重。
不得为了使用整数 GEMM 将 X 隐式量化为 INT8。多核 split-K 会改变归约顺序，不能默认无损。
tile 尺寸与双缓冲是否可行须按实际 310P 的 L0/L1/UB 容量确定，不沿用其他芯片参数。
这只是候选调度，内部累加/反量化舍入必须满足文末数值合同。

## B：DFlashW4UnpackNZ

将 W4 解包与 ND→NZ 合并，输出交给现有 CANN GEMM；不做浮点运算。

| 输入/输出 | shape | dtype / format |
|---|---|---|
| packed | `[N,K/2]` | UINT8 ND，上述 offset-binary 顺序 |
| q_nz | `[ceil(K/32),ceil(N/16),16,32]` | INT8 FRACTAL_NZ；origin `[N,K]` |
| attrs | N、K、layout version | K 为正偶数；首版限定 workloads 中的形状 |

按 A 的 q 定义解包并排列，padding=INT8 0；无 scale、FP16 buffer 或原地写入。
输入输出不重叠，输出须逐字节一致。

| 执行方式 | 成本 |
|---|---|
| 每轮融合转换 | 保留压缩常驻，但仍写入并读取整张 INT8 权重 |
| 加载时转换并缓存 | 增加 INT8 常驻内存及加载耗时 |
| 离线转换并加载 | 设备无需转换，载体存储约为压缩 W4 的两倍 |

## C：DFlashDraftLmHeadTop1

替换 `argmax(F.linear(hidden, head_weight))`，完整词表分块计算，输出 token ID。

| 输入/输出 | shape | dtype / 约束 |
|---|---|---|
| hidden | `[M,2560]` | FP16 ND；生产 M=15，支持 1..15 |
| head_weight | `[248320,2560]` | FP16 NK，共享 Target 原始 head |
| token_id | `[M]` | INT64 ND，范围 `[0,248319]`；上层恢复 `[1,M]` |
| attrs | M、K=2560、N=248320、layout version | 无 bias，完整词表，greedy Top-1 |

预排布 FP16 权重需单独定义索引、stride 和 origin；不能套用 INT8 NZ 的 16×32 排列。
先得到与基线相同的 **FP16 logits** 再比较，最大值相等时取最小词表 ID。
不能对未舍入的 FP32 logits 先 argmax，也不能裁剪词表。诊断版本回传 logits 做逐值比较。

M=15 的完整 logits 为 7,449,600 字节，可省去其 GM 写入和读取；
head 权重仍为 1,271,398,400 字节，完整投影的计算和权重读取仍需执行。

接入 `AirDFlashOps.top1`，首版仅用于 Draft。head 已只输入 MASK 行，不包含 anchor。
`proposal_count` 的无效尾部置零仍由 `DraftGraph` 处理；算子不输出 softmax 概率，
不应用新 logit softcap、重排 token ID 或裁剪词表。
按词表分块时，每个 owner 保留每行 `(FP16 最大值, INT64 原始词表 ID)`；
最终按“值大优先，相等 ID 小优先”归约。跨词表 tile 分块无需对 K 做 split-K。
同一权重 tile 应复用 M 行，避免为 15 行扫描 15 次全词表权重。
生产只输出 ID；调试额外输出 logits 的开销不得计入生产时延。
首先可省去 7,449,600 字节 logits 的写入和重读，合计 14,899,200 字节，以及独立归约调度；
这不等于能消除旧 profile 中完整 LM MatMul 的 6.73 ms。

## D：DFlashW8GateUpSwiGLU

在 A 已验证的 gate/up GEMM 上融合 `split → SiLU(gate) * up`，每层调用一次。
替换 `PackedDraftLayer.project_gate_up` 到 `layer.ops.swiglu` 的区域，输出直接交给 down。

```text
DFlashW8GateUpSwiGLU(X, W_gate_up_nz, S_gate_up;
                    M=16, H=2560, I=9728, group_size=128) -> Z
```

| 输入/输出 | shape | dtype / 含义 |
|---|---|---|
| X | `[16,2560]` | FP16；已经过 post-attention RMSNorm |
| W_gate_up_nz | `[80,1216,16,32]` | INT8 NZ，逻辑 `[19456,2560]`；前 9728 输出列为 gate，后 9728 为 up |
| S_gate_up | `[20,19456]` | FP16 GN，与 A 完全一致 |
| Z | `[16,9728]` | FP16 ND；SwiGLU 结果，作为 down 的 X |

逐元素定义（`RN16` 表示基线到 FP16 的舍入）：

```text
[G16, U16] = split(A(X, W_gate_up_nz, S_gate_up), I)   # FP16 Linear 输出
A16 = SiLU_baseline(G16)                             # 返回 FP16
Z16 = RN16(A16 * U16)
SiLU_real(g) = g / (1 + exp(-g))
```

这里 `SiLU_baseline` 的函数实现、极值行为和舍入以当前 native OM 为准；
不得把 FP32 GEMM 累加器直接送给 SiLU，也不得把 SiLU 与乘 up 合成一次最终舍入。
首版不融合输入 RMSNorm：它还涉及独立 FP32 归约与 cast-before-gamma 规则。

对应 gate/up tile 成对计算，片上暂存 G16/U16，只写 Z16；不改变已有 gate/up 权重次序。
相对分离实现可避免每层 622,592 字节投影结果的写入和重读，即 1,245,184 字节。
若当前 ATC 已消除部分中间量，按实际图计收益。不要为了融合 down 重复读取 W_gate_up；
完整 SwiGLU+down 跨 K 归约留待测量证明片上容量、权重复用与舍入均可行后再设计。

## E：DFlashW8DownResidual

在 A 的 down GEMM 后融合残差加法，替换 `hidden + down(swiglu)`。
这是 down 内核的 epilogue，不另算一次矩阵乘提速。

```text
DFlashW8DownResidual(Z, W_down_nz, S_down, R) -> H_out
```

| 输入/输出 | shape | dtype / 含义 |
|---|---|---|
| Z | `[16,9728]` | FP16 ND；D 或基线 SwiGLU 输出 |
| W_down_nz | `[304,160,16,32]` | INT8 NZ；逻辑 `[2560,9728]` |
| S_down | `[76,2560]` | FP16 GN |
| R | `[16,2560]` | FP16 ND；attention 输出投影与 residual 相加后的 hidden，**不是** post-norm 输入 Z |
| H_out | `[16,2560]` | FP16 ND；下一层输入或最终 RMSNorm 输入 |

`D16=A(Z,W_down_nz,S_down)`，`H_out=RN16(R+D16)`。
保留 GEMM 输出到 FP16 的舍入，不能实现为 `RN16(R+FP32_accumulator)`。
R 只读且与 H_out 不重叠，首版不融合下一层 norm。
每层可避免 down 中间量 81,920 字节的写入+读取；只有最新 OM 测量显示收益才启用。

## F：DFlashSegmentedGqaAttention（低优先级）

替换 cache/block 拼接、GQA QK、mask/softmax、PV 区域；Q/K 的 RMSNorm 与 RoPE 留在上游。
首版不修改 cache，仅消费已经更新的历史 cache 与临时 block。

```text
DFlashSegmentedGqaAttention(Q, K_cache, V_cache, K_block, V_block,
                           mask; scale, kv_groups=4) -> O
```

| 输入/输出 | shape | dtype / 布局与含义 |
|---|---|---|
| Q | `[1,32,16,128]` | FP16 BHQD，已经 Q norm 与 RoPE |
| K_cache / V_cache | 各 `[1,8,L,128]` | FP16 BHLD；L 为当前部署 cache 容量，不是有效长度 |
| K_block / V_block | 各 `[1,8,16,128]` | FP16 BHQD；本轮 anchor/MASK，不能提交进持久 cache |
| mask | `[1,1,16,L+16]` | BOOL ND；True 可见，广播到 32 个 Q head |
| scale | 编译属性 | 与当前 `128**-0.5` 缩放常量一致 |
| O | `[1,32,16,128]` | FP16 ND；调用方转 `[1,16,4096]` 后送 O projection |

Q head `h` 对应 KV head `floor(h/4)`。逻辑 K/V 顺序为整个 cache 容量 L 行、再接 16 个 block 行。
实现用两个基地址寻址，不在 GM 复制这两个大 Tensor；L 从部署清单取，不能硬编码为 prompt 长度。
cache 元素偏移为 `((b*8+hkv)*L+s)*128+d`；cache 的 head stride 为 `L*128`，
block 的 head stride 为 `16*128`，二者的行 stride 均为 128 个元素。
上游若仍需要物化 transpose/contiguous，须把该布局成本计入融合区域的总耗时。

基线数据流：

```text
scores32 = MatMul16_then_float(Q, K.T) * scale
scores32 = where(mask, scores32, -inf)
P32 = softmax(scores32, dim=last, dtype=FP32)
P16 = RN16(P32)
O16 = RN16(MatMul16_then_float(P16, V))
```

`MatMul16_then_float` 指当前 AIR 的 FP16 操作数 MatMul→FP32 Cast；
ATC 可能融合 Cast，必须从冻结 OM 确认是否有中间 FP16 结果舍入，不能只按 Python 推断。
Softmax 的 FP32 归约和 **PV 前概率转 FP16** 均需保留。
直接改成在线 FP32 FlashAttention 累加会改变该数值路径，属于待单独验证的非等价候选。

生产 mask 含 context 有效区、`proposal_count`、因果/双向与滑窗限制；算子逐元素消费 mask，
不能按全因果处理所有层，不能将带洞 mask 解释成一个有效前缀长度。
首版不跳过无效 query 行；全 mask 行的 NaN 等行为需与基线一致，不能自行返回零。
需要保存整行统计量或重算 scores 的方案，须把二次 K 读取、workspace 与额外计算计入性能。

## G：DFlashContextOnly 图接口（只改善 Prefill）

用于 C64 的非最后 prompt 块，替换 runner 中“完整 Draft 执行后丢弃候选”的调用。
这是一张只保留有效消费者的图，先复用 A 与原有 norm/RoPE/cache 算子，不要求首版写成单 kernel。

| 输入/输出 | shape | dtype / 含义 |
|---|---|---|
| features | `[1,64,28160]` | FP16；统一 Target feature，按当前 W8 的 5 个 `target_layer_ids` 选列为 `[1,64,12800]` |
| start_position | `[1]` | INT64；新增 context 的绝对起点 p |
| valid_rows | `[1]` | INT16；首版只允许完整非末块 v=64；预留部分行时要匹配下述 scratch 写入 |
| state | 10 个 `[1,8,L,128]` | FP16；五层 K/V，输入只读 |
| updated_state | 同上 10 个 | FP16；current/next 输出，不输出 token ID |
| FC 权重 / scale | `[400,160,16,32]` / `[100,2560]` | INT8 NZ / FP16 GN，逻辑 W `[2560,12800]` |
| 每层 KV 权重 / scale | `[80,128,16,32]` / `[20,2048]` | INT8 NZ / FP16 GN，逻辑 W `[2048,2560]`；每层一对 |
| hidden norm gamma / 每层 K norm gamma | `[2560]` / `[128]` | FP16；K gamma 广播到 8 个 head |
| RoPE inv_freq / norm epsilon | `[64]` / 标量属性 | FP32 / checkpoint 配置；复用当前常量构造，不另改 theta 或 epsilon |

处理：选择 features → 无效行置零 → FC → hidden norm；五层各自做 context KV 投影、
K norm、context RoPE，然后更新 `[p,p+64)` 的 K/V 行。各层 context 不读取 block hidden，
因此 Q、O、gate/up、down、final norm、head 均没有通往输出 cache 的消费者，可以移出此图。
`p+64<=L`，位置唯一；非写入区位型不变，不得将 block KV 写入输出。
若扩展 v<64，需与原图一致地计算/写入 C 行 scratch，并只将 v 行标记为有效；
不能只比较有效区就宣称整个 cache 等价。

现有 combined K/V 的 M=80 改为 64，虽实数逐行公式相同，kernel 的舍入可能变化；
必须比较 context K/V 位型以及接回正常 Draft 后的候选。迁移 `DraftContextGraph` 时补全
统一 feature 选择、valid_rows、量化合并投影及功能式 whole-row update，不能直接复用旧接口。
除当前 prefill 路径外，正常 decode 仍每轮一次 combined Draft，避免额外图启动和权重重读。

## 精度要求

| 对象 | 首版无损替换门槛 |
|---|---|
| W4/W8 q、scale、离线布局 | 0 位型差异，填充符合约定，逆变换逐字节还原 |
| A：Y | 与同输入、权重和配置的 native OM 逐元素一致，`atol=0, rtol=0`，0 位型差异；报告 ULP |
| B：q_nz | 逐字节一致，差异数 0 |
| C：Top-1 | 每行 ID 完全相同；诊断 FP16 logits 同 A 门槛，并验证 tie |
| D / E | 同输入的组合 native 子图输出逐元素、逐位型一致；分别检查 Linear 和激活/残差中间边界 |
| F | 对冻结 attention 子图逐元素、逐位型一致，QK/Softmax/PV 中间结果用于定位差异 |
| G | 五层全部 KV 输出匹配，包括未写入区与物理 scratch；后续正常 Draft 行为一致 |
| 同量化 Draft 集成 | 候选 ID、KV 写入与后续消费结果匹配基线，无越界或跨缓冲区写入 |
| 完整模型 strict greedy | ordinary Target 为权威，token ID、EOS/stop reason 0 mismatch |

输入输出 dtype 和 `inner_precise=0` 不足以确定内部累加精度。开发方须对照目标版本原生 OM
明确反量化乘法 dtype、FP16 舍入位置、Cube 累加 dtype、split-K 次序、最终转换及溢出行为。
CPU FP64/FP32 公式和显式 `FP16(q*S)` 对照用于定位索引与舍入差异，不能替代 native OM golden。

X 为有限 FP16，S 严格正且有限；测试覆盖 ±0、次正规数、近溢出值及输出 Inf/NaN 的基线行为。
未支持的特殊值明确拒绝或走已验证原路径。若需放宽误差，先提交 max-abs、max-relative、RMS、
ULP、Top-1 翻转及模型结果，另定容差；接受率相近不能代替数值验收。

整数量化 code 解包与可逆重排 B 可通过还原验证证明 exact；A/C/D/E/F 的新 tiling、函数实现或归约顺序，
以及 G 的 M 改变，在证明相同舍入前均为 uncertain。本文只定义交付目标，
未授权改变精度或放宽门槛；需要改变公式、舍入或精度的实验应先明确差异、风险和回退版本。
范数需复用当前规则：FP32 reduction/rsqrt → normalized 转 FP16 → 乘存储 gamma，
不是 `1+gamma`，也不是将 gamma 提前乘进 FP32 后最后只 cast 一次。
