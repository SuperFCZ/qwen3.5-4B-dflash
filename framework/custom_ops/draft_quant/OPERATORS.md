# 量化 Draft 自定义算子需求

目标：Ascend310P，CANN 9.0.0，torch_npu `2.8.0.post1+gitc7b6b32`。
以下为开发接口；固定形状见 [workloads.json](workloads.json)，测试与交付见 [验收要求](VALIDATION.md)。

## A：DFlashGroupQuantLinear

替换 FC、Q、K/V、O、gate/up、down，优先 gate/up 和 down。
接入 `GroupQuantLinear.forward` / AIR WeightQuant 节点；首版仅做 Linear。

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
输入只读，Y 与输入、workspace 不重叠。未支持的形状明确报错或由宿主分发至原路径。
无 bias、antiquant_offset、输出 quant_scale/quant_offset；不量化激活。

模块中的 S 为 `[N,G]`，checkpoint BF16 scale 转为 FP16 后使用。
本 ABI 采用 `[G,N]`，离线转置保持 FP16 位型；原生图已采用此布局，无需再次转置。

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

## 精度要求

| 对象 | 首版无损替换门槛 |
|---|---|
| W4/W8 q、scale、离线布局 | 0 位型差异，填充符合约定，逆变换逐字节还原 |
| A：Y | 与同输入、权重和配置的 native OM 逐元素一致，`atol=0, rtol=0`，0 位型差异；报告 ULP |
| B：q_nz | 逐字节一致，差异数 0 |
| C：Top-1 | 每行 ID 完全相同；诊断 FP16 logits 同 A 门槛，并验证 tie |
| 同量化 Draft 集成 | 候选 ID、KV 写入与后续消费结果匹配基线，无越界或跨缓冲区写入 |
| 完整模型 strict greedy | ordinary Target 为权威，token ID、EOS/stop reason 0 mismatch |

输入输出 dtype 和 `inner_precise=0` 不足以确定内部累加精度。开发方须对照目标版本原生 OM
明确反量化乘法 dtype、FP16 舍入位置、Cube 累加 dtype、split-K 次序、最终转换及溢出行为。
CPU FP64/FP32 公式和显式 `FP16(q*S)` 对照用于定位索引与舍入差异，不能替代 native OM golden。

X 为有限 FP16，S 严格正且有限；测试覆盖 ±0、次正规数、近溢出值及输出 Inf/NaN 的基线行为。
未支持的特殊值明确拒绝或走已验证原路径。若需放宽误差，先提交 max-abs、max-relative、RMS、
ULP、Top-1 翻转及模型结果，另定容差；接受率相近不能代替数值验收。
