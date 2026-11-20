# Gate 的 Group128 Scale 展开

输入来自冻结的 C16 Layer0 Gate。每个 N64/K256 的 raw bank 已包含 2×64 个 GN FP16 Scale。应用先完成 MTE2_S/MTE2_V 双 Wait，原始 INT8 的 8 次 batched Cast 后保留 PIPE_V。对每一组，Scale 的目标是 64 个 32 B 块，每块内的 16 个 half 都具有本通道相同的位型。Mul 的 mask=128，repeat=64，src1BlkStride=0、src1RepStride=1。对通道 c、Group128 内位置 k，Mul 读取 expanded[16c+(k mod 16)]，在 W[c×256+128g+k] 原位进行一次 FP16 乘法。两个组各用自己的 64 个 Scale。

## 原实现与备选

| 方案 | 310P 实现及布局 | UB、准备与依赖 |
|---|---|---|
| 冻结 Raw Brcb | `Brcb(8,{1,8})` 每轮取 8 个 scale，8 轮覆盖 64。m200 B16 使用 3 个整数 vadds 和 1 个 vtranspose 保存 rear、构造重排输入、转置和恢复 rear。每轮有 3 个内部 PIPE_V。 | 每组写 2048 B，分配 2304 B 含 256 B rear。SDK 使用顶端固定临时区 [254976,256000)。每 K 两个 Brcb、48 个内部 PIPE_V，应用 6 个 PIPE_V。 |
| **选择 native B16 TransDataTo5HD** | 16 个只读源行指向同一个 32 B 块。每个源列取出的 16 个值逐位相同，转成一个独立目标块。每组 1 次 API、4 个 B16 repeat。m200 通过 VA0…VA3 和 scatter_vnchwconv_b16 发射。 | 单组仍写 2048 B，rear 保留。两份 16×uint64 的地址列表是调用的标量参数，TPipe 新增分配 0 B。没有索引 GM/UB 表。标量建表、VA 寄存器装载、mask 设置和重复读的成本均进入计时。 |
| 通用 `BroadCast<half,2,1>` | [64,1]→[64,16] 表达相同结果。v200 last-dimension 路径准备 zeroTemp，先进行对齐复制，再做 NCHW→NHWC 转置。 | 需要按 SDK 规则确定并提供共享临时缓冲，还会带来初始化与中间依赖。适合动态 shape，此固定形状先直接测试底层转置。 |
| 按字节偏移 `Gather` | m200 的 half Gather 支持 vgather，可以用 offset[i]=2×floor(i/16) 表示同一映射。该目标的 Gatherb 明确不支持。 | 一次覆盖 1024 个输出所需的完整 uint32 索引表为 4096 B，可分块减少表大小但需多次发射。需要新索引的初始化、可见性和 UB 分账。 |
| 逐通道取值加 Duplicate/Muls | 需要按通道进行标量读取和广播，或改变当前 Mul 的发射方式。 | 每 K 当前 Gate 的显式 scale GetValue 为 0。舍入、时序和标量开销均须单独验证，本次保留原 Mul。 |

本候选只替换 Gate 实际选中的 Brcb 调用，其他形状仍走原分支。不按 API、repeat 或 barrier 数量预测任何延迟。实际代码大小、编译器展开及标量寄存器/栈分配由构建产物决定。[原始 Gate 数学与同步](../draft_quant/op_kernel/d_flash_group_quant_linear.cpp) · [m200 候选实现](kernel/scale_expand.inc)

## 地址与数据依赖

地址的单位是字节。源和目的的 16 项地址列表各自连续存放。重复的是只读源地址，其指向的 UB 区与目的区全程不相交。令 r、c、j 的范围分别为 0…3、0…15、0…15。

```text
srcVA[j] = S                     dstVA[c] = E + 32*c
srcRepStride = 1 个 32 B 块       dstRepStride = 16 个 32 B 块
读 S + 32*r + 2*c                写 E + 512*r + 32*c + 2*j
expanded[16*(16*r+c)+j] 的原始位型 = scales[16*r+c] 的原始位型
```

- 每组从 [S,S+128) 读，唯一写区为 [E,E+2048)。4×16×16 个 half 位置各写一次，既不跨组也不跨 bank。源地址从现有 sLocal[group×64] 的物理地址取得，目标从已分配的 expanded tensor 取得。无固定地址传给设备算法。
- 按 Raw TPipe 合同，两个 q bank 共 32768 B，两个 s bank 共 512 B，单 W 为 32768 B，expanded 连 rear 为 2304 B，SDK scratch 185600 B，顶端 reserve 8192 B。按零起点的规划，E=66048，payload 末端 68096，rear 末端 68352，scratch 末端 253952。每个源、目标和 repeat 起点 32 B 对齐。候选保留全部原有分配、初始化和 reserve。Host 实际预算与编译对象可写节的变化分别检查；此表没有观测运行中的物理地址。
- 原两方向 raw Wait 已经在 Vector 首次读取前发出。地址列表保存位置，构造时没有读取 FP16 scale 的标量 GetValue。使用 uint64_t 地址数组的公开重载，在同一次调用内设置 VA 并发出 Vector 指令。保留正常 mask 模式和 ResetMask，后续 Mul 使用原先显式 128-lane mask。已有 PIPE_V 分别保护转置写→Mul 读、Mul 读完→下一组复用 expanded。候选没有上一个 Brcb 实现的临时矩阵或 rear 保存/恢复依赖。
- Cast、两个 FP16 Mul、每次 Matmul 前 V_S、每 K 的 PIPE_ALL、失败的双 Wait/排空、最后一次 GetTensorC 及 N 尾 fence 均来自原源码。每个 N 只有一个 owner，K=0 使用 partial=false，K=1…9 使用 true。checkpoint、激活精度、Scale 位型和 FP32 K 累加顺序均一致。
- 数据依赖的结论限于此公开 API、已审核的 m200 实现、准确的 stride 与不相交区间。模型检查方向和覆盖，真正的 Gate 数值准入须由冻结 Native Golden 的 NPU Smoke 完成。本实验没有修改或复用 W16 窄 Fence 的机器证明。

## SDK 源码与文档

`sdk-files.json` 给出所选 CANN 根下每个文件的实际相对路径及完整 SHA256。Source 和 review 校验该清单。

| 文件（路径前缀 aarch64-linux/asc） | 核对位置 |
|---|---|
| `impl/basic_api/dav_m200/kernel_operator_vec_brcb_impl.h` | 54–94、97–119。B16 stride-one，rear=128 个 half，临时 1024 B，三处 PIPE_V。 |
| `include/basic_api/kernel_operator_vec_transpose_intf.h`、`include/basic_api/kernel_struct_transpose.h` | 地址序列重载，参数构造第 60–76 行明确依次为 repeatTimes、dstRepStride、srcRepStride。 |
| `impl/basic_api/kernel_operator_vec_transpose_intf_impl.h` | 104–123。地址数组模式在 CPU debug 下逐项检查 32 B 对齐，再进入 TransDataTo5HDImpl。 |
| `impl/basic_api/dav_m200/kernel_operator_vec_transpose_impl.h` | 140–146、201–208、229–239。half 的 B16 专用指令，16 项源和目标 VA 列表，不写源。 |
| `impl/adv_api/detail/transpose/transdata/transdata_impl.h` | 253–281、295–314。SDK 自己将多个只读源列表项置为同一 tmpBufferAddr，随后调用相同的 half API。该示例用于 padding；它补充地址列表证据，候选的全部 16 项重复另由真实 310P 系列 SDK CPU 模型逐位验证。 |
| `include/basic_api/kernel_common.h`、`impl/basic_api/kernel_reg.h`、`impl/basic_api/dav_m200/kernel_operator_vec_binary_impl.h` | mask 的正常模式和 reset，m200 MulImpl 118–134 使用自身的 mask 和原 binary repeat 参数。 |
| `impl/basic_api/dav_m200/kernel_operator_vec_gather_impl.h`、`impl/adv_api/detail/pad/broadcast/broadcast_v200_impl.h` | Gather 的支持类型/offset、Gatherb 不支持，通用 Broadcast 的临时区、初始化和转置。 |

公开 API 说明了 B16 的列到行转换、repeat>1 的首址和后续 stride（单位为 32 B），并列出 Atlas 推理 AI Core 对 half 的支持。一般重叠规则约束源与目的之间的读写冲突。[CANN 9.0 TransDataTo5HD](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0200.html) · [地址对齐与重叠合同](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0004.html) · [CANN 9.0 BroadCast](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0853.html)
