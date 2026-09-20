# 量化 Draft 自定义算子需求 v1

目标接收环境：Ascend310P，CANN 9.0.0，torch_npu `2.8.0.post1+gitc7b6b32`。
实现前记录实际 `soc_version`、驱动、固件和算子包版本；不能仅按“310P”推定所有特性。
本文件为需求与建议 ABI，**尚未实现/注册任何新算子**。精度门槛见第 4 节，交付项见
[VALIDATION.md](VALIDATION.md)。实际形状见 [workloads.json](workloads.json)。

## 1. A：DFlashGroupQuantLinear（主优化算子）

替换范围：量化 Draft 的 FC、Q、K/V、O、gate/up、down，优先实现 gate/up 和 down。
接入点为 `GroupQuantLinear.forward` / AIR 对应的 WeightQuant 节点。
第一版不融合 norm、SiLU、残差或 cache；先隔离 group GEMM 的精度和速度。

### 1.1 逻辑接口

```text
DFlashGroupQuantLinear(X, W, S;
    bits=4|8, M, N, K, group_size=128, weight_layout)
    -> Y
```

| 输入/输出 | 逻辑形状 | 存储 dtype | 布局及含义 |
|---|---|---|---|
| X 输入 | `[M,K]` | FP16 | 连续 ND；上层 `[1,M,K]` 仅展平 batch，row stride=K |
| W8 输入 | `[N,K]` | INT8 | 有符号量化 code，范围 `[-128,127]`；可用下面约定的 NZ 载体 |
| W4 输入 | `[N,K/2]` | UINT8 | 两个 offset-binary 4-bit code/byte，排列见 1.2 |
| S 输入 | `[G,N]`，`G=K/128` | FP16 | 连续 ND，stride=`[N,1]`，严格正、有限；每 K-group 每输出列一个 scale |
| Y 输出 | `[M,N]` | FP16 | 连续 ND，stride=`[N,1]`；上层恢复 `[1,M,N]` |

生产 M：FC 为 16/64，K/V 为 32/80，其他投影为 16。
生产 K/N 均为 128/16 的倍数，具体集合冻结在 workloads.json；tiny 对照为 K=256,N=64。
首版只要求这些形状，未实现的 shape/layout/bitwidth 必须明确报错或由宿主分发至原有正确路径。
不能静默走 CPU、改变精度或在 kernel 内猜测 stride。所有输入只读，Y 与输入、workspace 不重叠。

S 的 checkpoint/模块存储是 `[N,G]`，从 BF16 转为 FP16 后驻留。
新 ABI 固定采用 `[G,N]`；宿主应离线转置并记录哈希，按 FP16 **数值与位型**保持一致。
不得将两种存储顺序当作同一块连续内存。当前原生图已采用 `[G,N]`，不要再次转置。

无 bias，无动态激活量化，无 antiquant_offset，无输出 quant_scale/quant_offset。
不支持通过改属性把 group-128 checkpoint 当作 per-channel 使用。

### 1.2 code、NZ 与数学定义

W4 的字节定义（沿 K 连续，偶数 K 在低四位）：

```text
b = W4[n, floor(k/2)]
q[n,2*j]   = (b & 0x0f) - 8
q[n,2*j+1] = (b >> 4)   - 8

例：0x80 -> q_even=-8, q_odd=0
    0x0f -> q_even=7,  q_odd=-8
    0x88 -> q_even=0,  q_odd=0
```

它不是 two's-complement signed INT4；不能直接把 0x8 解释为 -8。
文件中的原始 checkpoint 是 INT32 compressed-tensors packed words，进入本算子前已转换为
上述 byte 载体；不要把 INT32 文件格式直接当成本 ABI。
W8 直接读取 INT8 q，不减 128。两种模式都保留原 checkpoint 的每一个 q 和 S。

W8 可接受两种明确分发的 weight_layout：

| 标记 | shape/format | 索引关系 |
|---|---|---|
| `nk_int8` | ND `[N,K]` INT8 | `q[n,k]` |
| `nz_int8_v1` | FRACTAL_NZ `[ceil(K/32),ceil(N/16),16,32]` INT8 | `W[k//32,n//16,n%16,k%32] = q[n,k]` |

NZ padding 写 INT8 0，origin 保留逻辑 `[N,K]`；物理 shape 不得用于推导 GEMM 的 K/N。
只有 `pack_int8_nz` 对应的真实字节重排才能声明 NZ，禁止只改 format 标签。
W4 首版采用 `nk_u4_offset_binary`；若增加离线分块载体，需新 layout/version、完整逆变换和
哈希验证。W4 的零 code 是 nibble 8，填充 byte 应为 0x88；UINT8 0 表示两个 -8，并非零。

数学目标（实数表达，仅定义分组与索引，不规定浮点归约次序）：

```text
W_real[n,k] = q[n,k] * S[floor(k/128),n]
Y_real[m,n] = sum(k=0..K-1, X[m,k] * W_real[n,k])
```

一个输出列可能有多个不同的 S，不能改成 `MatMul(X,q) * 单个scale[n]`。
即使采用逐 group 部分和再缩放在实数域等价，也可能改变反量化乘法的 FP16 舍入和归约顺序，
必须标为数值尚未证明的候选，不能按无损布局优化直接替换。

当前权威运行基线为相同 q/S 的 CANN `WeightQuantBatchMatmulV2`：
X=FP16，W=INT8 NZ，S=FP16 `[G,N]`，`transpose_x=false`、`transpose_weight=true`、
`antiquant_group_size=128`、`inner_precise=0`，输出 FP16。
这是 A16 权重量化路线，需求不包含 INT8×INT8 激活量化。

### 1.3 实现关注点与工作量

要求针对小 M 组织权重 tile 的解包/反量化和多行复用，保留计算后的输出块，降低每个 group 的
Scalar/Vector 调度开销。W4 尽量在局部 tile 内解包，不落地整张 INT8 或 FP16 权重到 GM。
W8 尽量直接消费离线 NZ 或 kernel 自己声明的可逆载体，避免每次调用重新转换固定权重。

每个 `(M,N,K)` 必须交付以下账本，不能只用逻辑 FLOPs 估计性能：

- 逻辑 FMA 为 `M*N*K`；Cube 实际 padded M/N/K、tile 数及重复计算另列。
- X 为 `2*M*K` 字节，W8 为 `N*K` 字节，W4 为 `N*K/2` 字节，S 为 `2*G*N` 字节，Y 为 `2*M*N` 字节；NZ padding、重复读及 workspace 另计。
- 每条 GM↔UB/L1、L1↔L0、FIX 路径，解包、反量化、归约和布局开销；串行依赖、填充/排空与真实重叠证据。
- UB/L1/L0A/L0B/L0C 各 buffer 的 dtype、aligned shape/stride、字节数、生命周期；双缓冲和独立输出 owner 都必须覆盖。
- 逻辑 Block Num 与实际物理核的映射，避免按 profile 中的 8 blocks 直接假定 8 倍吞吐。

gate/up 单个 W8 权重 49,807,360 字节，W4 24,903,680 字节；完整 FP16 解量化矩阵
99,614,720 字节，不能因实现方便每轮生成后再读。down 单个 W8 权重 24,903,680 字节。
gate/up 输出的前 9728 列为 gate，后 9728 列为 up；K/V 输出前 1024 列为 K、后 1024 列为 V。
这些列顺序及下游消费者不变。

后续 gate/up+SwiGLU 融合可避免中间 GM，但要保留 Linear→FP16、SiLU→FP16、乘 up→FP16
等基线边界，并冻结 SiLU 实现。当前没有授权替换近似激活，也未实现此融合。

## 2. B：DFlashW4UnpackNZ（较小的独立备选）

如果 A 尚不能交付，先将 W4 的 byte 解包链与 INT8 ND→NZ 合并，继续使用已有 CANN GEMM。
该算子不做浮点运算，输入输出必须逐字节精确。

| 输入/输出 | shape | dtype / format |
|---|---|---|
| packed 输入 | `[N,K/2]` | UINT8 ND，1.2 的 offset-binary 顺序 |
| q_nz 输出 | `[ceil(K/32),ceil(N/16),16,32]` | INT8 FRACTAL_NZ；origin `[N,K]` |
| attrs | N、K、layout version | K 为正偶数；首版限定 workloads 中的形状 |

数学：先按 1.2 得到 q，然后按 `nz_int8_v1` 排列；padding=INT8 0。无 scale、无 FP16 buffer，
没有 bias，也不允许写入输入内存。Y 与输入不可别名。输出交给 native WeightQuant 的 W 口，
scale 和激活完全保持原样。GE infer-shape、format select、tiling 必须知道这是物理 NZ 输出。

三种部署方式需要分别标注：

1. 每轮融合转换：保持压缩常驻，仍付出整张 INT8 输出和读取成本；收益需与原解包+TransData 总成本比较。
2. 加载一次转换并缓存：增加常驻 INT8 权重，测量峰值/稳态内存；避免每轮转换，但加载仍耗时。
3. 离线转换并加载 NZ：设备上无需 B；W4 checkpoint 数值不变，但执行载体是 INT8，存储约翻倍，不能称为原生 W4 GEMM。

用户已允许离线转换；当前已有的转换工具仅支持 W8。上述 W4 转换/常驻策略仍为提案，
不能把 W8 常量预打包工具未经检查直接用于 W4。

## 3. C：DFlashDraftLmHeadTop1（独立热点）

当前 `AirDFlashOps.top1` 为 `argmax(F.linear(hidden, head_weight))`。
LM head 的全 FP16 MatMul 约 6.729 ms，ArgMax 约 0.185 ms。
需求为完整词表分块矩阵乘并归约，保留全词表语义，只输出 token ID。

| 输入/输出 | shape | dtype / 约束 |
|---|---|---|
| hidden 输入 | `[M,2560]` | FP16 ND，生产 M=15，支持 1..15 供尾块/单算子验证 |
| head_weight 输入 | `[248320,2560]` | FP16，逻辑 NK；共享 Target 原始 head，不能改成 W8/W4 |
| token_id 输出 | `[M]` | INT64 ND，每项范围 `[0,248319]`；上层恢复 `[1,M]` |
| attrs | M、K=2560、N=248320、layout version | 无 bias，完整词表，greedy Top-1 |

默认逻辑接口连续 NK。若使用预排布物理权重，必须分别定义 FP16 的索引映射、stride 和
origin；不能复用 INT8 的 16×32 NZ 解释。用户 profile 的 head 权重物理 shape 是
`[15520,160,16,16]`，接入前需从实际 OM 的 transpose/layout 属性确认其逻辑方向。
输入逻辑形状相同不等于此物理载体可直接互换。

精度及 tie 规则：先取得与原 MatMul 一样的 **FP16 logits**，再比较。
相等最大值返回最小词表 ID；归约采用 `(logit 最大, index 最小)`，与 tile 顺序无关。
不能对未舍入 FP32 logits 先 argmax：不同 FP32 值可能舍入到相同 FP16，改变 tie 结果。
所有词表列均参与，不能以 Top-K 子集、候选词表映射或早停阈值替代。
诊断版本可回传完整 logits 做逐值比较，生产版本只返回 ID。

完整 logits 为 `15*248320*2=7,449,600` 字节；避免其 GM 写入及后续读取是直接收益来源。
head 权重仍有 `248320*2560*2=1,271,398,400` 字节，完整投影的权重读取和计算没有消失。
不得把整个 6.729 ms 算作可删除开销。更少 launch 也不自动等于更快。

## 4. 精度和正确性：交付时必须明确的硬要求

| 对象 | 首版无损替换门槛 |
|---|---|
| W4/W8 q、scale、离线排列 | q/scale 位型 0 差异；所有填充符合约定；编码→解码逐字节还原 |
| 算子 A 的 Y | 与同输入、同权重、同配置的 native OM 基线逐元素一致，`atol=0, rtol=0`；另报位型/ULP，严格模式要求 0 位型差异 |
| 算子 B | INT8 NZ 输出逐字节完全相同，差异数必须 0 |
| 算子 C | 每一行 token ID 完全相同；诊断 FP16 logits 按 A 的严格门槛比较，特别验证 tie |
| 集成后同量化 Draft 的候选/KV | 有效候选 ID 0 mismatch；缓存写入值及后续轮消费按冻结基线比较；无越界或跨 owner 污染 |
| 完整模型 strict greedy | 普通 Target 为权威，token ID、EOS/stop reason 0 mismatch；现有 output parity FAIL 不会因算子通过而自动变 PASS |

**不能从输入输出 dtype 推断累加精度。** 当前 native 的 `inner_precise=0` 不足以说明
接收端每个 K tile 的累加、归约和反量化舍入规则。开发方须用匹配版本源码/编译选项和
捕获的 OM 输出冻结：反量化乘法在哪种 dtype 发生、是否先舍入为 FP16、Cube 累加 dtype、
split-K 次序、最终转换的舍入/溢出/次正规数行为。内部改成 FP32 累加并不自动保证与旧结果一致。
在这些规则尚未确认前，不应承诺“同数学公式所以 bit-exact”。

CPU FP64/FP32 实数公式用于发现索引、符号、分组错误；
`FP16(q*S)` 后 `F.linear` 的 dequant 对照用于分解诊断，二者都不能替代 CANN native
实际输出作为严格运行基线。现有 dyadic tiny probe 避免了归约歧义，只能验证特定输入。

首版没有授权任何近似门槛。若更快实现因归约变化达不到零差异，保留原路径并提交
max-abs、稳定分母的 max-relative、RMS、ULP 分布、top1 margin/翻转数和完整 Draft 多轮结果；
提出具体容差及模型质量验收方案后另行决定。不得用 cosine、接受率相近或
`--allow-output-differences` 代替误差审批，也不得自动放宽以上阈值。

X/S 的正常工作域为有限 FP16，S 严格为正；应包含 0、±0、次正规数和接近溢出的测试。
有限输入仍可能产生 Inf 或 NaN（溢出/抵消），异常输出要与原路径行为比较并单独记录。
未支持的 NaN/tie/overflow 行不得静默钳位成正常数，应显式拒绝或走已验证的原路径。

## 5. 开发前必须冻结与交付

冻结实际 checkpoint/config/source/OM/input 哈希、shape 和物理载体、ATC 参数、设备身份。
给出 torch_npu schema、Meta 输出、GE infer-shape/format/dtype 规则、host tiling、kernel、
隔离注册方法及回退条件。使用真实业务权重和至少一轮捕获的激活，合成矩阵只能补覆盖。
不得全局禁用形状推导、替换安装目录算子或默认改变 Target 的输出权威。

性能验收目标先对 **同一量化 checkpoint 的现有 native 路径** 报告中位数、p95、CV、
峰值内存，再对 FP16 Draft 报完整图差异。单算子设计目标可设为相对对应原生算子 ≥1.5×，
这是待实测的目标，不是可实现承诺。Draft 13.48 ms / 5.055 ms 是用户 1.5× / 4× 目标对应
的图级预算，不能按算子数平均分配，也不能仅靠某一矩阵乘加速就判定达标。
