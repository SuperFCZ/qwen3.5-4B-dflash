# DFlash 流程与架构

Target 负责验证和最终预测，Draft 利用 Target 的隐藏特征，一次提出最多 15 个候选。
两种验证路线都支持 Torch-NPU 和 OM/C++；命令见 [Torch-NPU](DFLASH_RUN_AND_VALIDATE.md)、
[OM/C++](GDR_CHUNK_AIR_OM.md)。

## 一轮怎样运行

```mermaid
flowchart TD
    P["Prompt"] --> T["Target Prefill：状态、特征、首 token"]
    T --> D["Draft：anchor + MASK → 候选"]
    D --> V["Target Verify：验证 anchor + 候选"]
    V --> A["取连续匹配前缀，得到接受数 a"]
    A --> C["提交 anchor + a 个候选对应的状态"]
    C --> O["输出接受前缀及修正 / bonus token"]
    O --> E{"EOS 或达到长度上限？"}
    E -->|否：新 token 作 anchor| D
    E -->|是| F["结束"]
```

anchor 是已输出、尚待本轮写入状态的 token。若候选为 `A B C`、Target 判断为
`A B X`，接受 2 个候选并输出 `A B X`；提交旧 anchor、A、B 的状态，X 留作下轮 anchor。
零接受仍提交旧 anchor，并继续投机。拒绝位置之后的预测不属于已提交序列。

## 模型组成

| 部分 | 结构 |
|---|---|
| Target | Qwen3.5-4B，32 层：24 层 Gated DeltaNet + 8 层 full attention；隐藏维度 2560 |
| Target 特征 | 统一 OM 输出第 1、5、8、9、13、15、17、21、22、25、29 层的并集（从 0 编号），每 token 28160 维；Draft 选择 checkpoint 对应的层 |
| Draft | FP16 为 6 层、W4/W8 为 5 层；FC 将选定特征投影到 2560 维，随后为 norm、上下文 K/V、attention/MLP 与 LM head |
| Draft 输入 | 新提交的 Target 特征 + anchor/MASK block；持久 KV 只保留已提交上下文 |
| 精度 | 原生 Target 可选 FP16/W8A8；OM 为 W8A8 Target，Draft 可选 FP16/W4A16/W8A16 |

OM Draft 的 QK/PV 矩阵乘默认 FP16，缩放、Mask、Softmax 为 FP32；
原生 Torch-NPU Draft 使用 FP32 矩阵乘。两者不能视为完全相同的数值路径。

## Draft 内部流程

参照 [DFlash 官方图 2](https://arxiv.org/pdf/2602.06036#page=4) 的
“Target 特征注入各层 KV、MASK 块并行生成”结构，按本仓库的 6 层 Draft 重画。
**从左向右看：绿色主线生成候选，蓝色支路提供上下文。**

![Draft 的三个输入、两路计算及两个输出](assets/dflash-draft-flow.svg)

- **输入**：新增 Target 特征、历史上下文 KV、`anchor + K 个 MASK`；anchor 是 Target 上次给出的最后一个 token。
- **输出**：一次前向得到 K 个候选及更新后的上下文 KV。只有 MASK 行生成候选，随后交给 Chunk 或 MTP Verify。
- **跨轮缓存**：首次加入 prompt 特征；之后加入上一轮旧 anchor 和接受前缀的 Target 特征。
  本轮 block 的临时 KV 不留到下一轮；零接受也加入旧 anchor 的特征。

<details>
<summary>算子与 OM 接口细节</summary>

同一份 FC/RMSNorm 结果供各层分别生成上下文 K/V；Query 只来自 anchor/MASK block。
每层内部为 RMSNorm → Q/K/V 投影（Q/K norm + RoPE）→ Attention → 输出投影与残差，
再经 RMSNorm → SwiGLU MLP 与残差。滑窗层按因果窗口计算，full-attention 层允许块内双向注意。

Chunk/MTP 共用选定精度的 Draft OM，把上下文更新和候选生成合在一次调用中。
三精度对比时，共享 Target 输出特征层的并集；每个 Draft 选取自己的特征输入。
发布的 W4/W8 checkpoint 为五层，FP16 为六层。量化路径保留压缩权重输入，默认使用 CANN
`WeightQuantBatchMatmulV2`：FP16 激活 × INT8 权重，group-128 scale，`inner_precise=0`；
W4 在调用前将 packed byte 无损展开为 INT8，不保留完整 FP16 权重。
`draft_quant_matmul=dequant` 可选显式解量化对照；实际 OM 时延和峰值显存须测量。
三种 Draft 分开运行，不同时驻留；[构建与对比命令](GDR_CHUNK_AIR_OM.md#统一测试)。
逻辑接口为 `(features, start_position, valid_rows, anchor, proposal_count, 历史 KV)`
→ `(候选 token, 更新后的 KV)`，位置和有效长度控制可见范围。
W4/W8 另带一维只读权重输入，加载一次后常驻设备，图内恢复形状并解量化。
OM 使用单个 Draft 的 16/64 两档上下文；特征缓冲区容量为 64 行，候选 block 始终为 16 行。
生成轮的新特征最多为“旧 anchor + 15 个接受 token”共 16 行；候选输出最多 15 个。
每个非末尾的 Target 64 行特征块调用一次 Draft 建缓存，准备阶段候选丢弃；
最后一块随首次正式生成处理。N 个输入 token 需 `ceil(N/64)-1` 次准备调用，1024 输入为 15 次。
首次 Draft 根据有效特征数选择 16 或 64 档，后续 Verify 最多提交 16 行，自动切回 16 档。
不复制特征切片，不增加第二个 Draft 或 KV 副本；CANN 档位控制缓冲区、权重布局和工作区计入显存实测。
生成时仍是每轮一次 Draft + 一次 Verify。

实现见 [Draft 模型](../models/dflash_v1/modeling_dflash.py)、
[DraftGraph / PackedDraftLayer](../framework/python/qwen35_dflash/ascend310p/incremental.py)。

OM 内每层将新增上下文和候选输入按行拼接，共用一次 K/V 投影；K/V 与 gate/up 权重在导出时打包替换。
KV 使用整行 `ScatterNdUpdate`，GQA 将 Query 分组并入行维，避免复制历史 K/V。
各 checkpoint 的层数与完整词表保持不变；每次加载一个 Draft OM，使用 current/next 缓存。
实际编译 workspace、峰值显存和浮点舍入需重新实测。

</details>

## Chunk 两遍与 MTP

| 项目 | `chunk`（默认） | `mtp` |
|---|---|---|
| 每个 GDN 层的验证 | 第一遍 Chunk 计算整块输出 | 一次 MTP 计算整块输出和逐行 state bank |
| 接受 a 个候选后 | 从初始状态重算 a+1 行 | 选择 bank 第 a 槽 |
| 跨轮 recurrent state | FP32 | FP32 |
| 第二遍 GDR | 有 | 无 |
| OM ABI | `qwen35-dflash-chunk-v4` | `qwen35-dflash-mtp-v2` |

GDN 状态压缩了历史，不能靠截短 KV 长度撤销拒绝的 token，因此需要重算或选取中间状态。
MTP 输入的旧状态选择值固定为 0，当前轮接受数用于选择输出 bank；二者含义不同。
普通 prefill/decode 使用 Chunk 算子，recurrent state 的初始化、存储、传递统一为 FP32。
Torch-NPU 和 AIR/OM 均采用这一规则；权重、conv 和 KV 的精度保持现有设置。

Torch-NPU 使用实际 `T=K+1` 行；OM 固定 16 行，通过有效长度限制接受和提交。
MTP bank 在 OM 内消费；原生提交复制所选状态，避免跨轮保留整块 bank。
缺少 MTP 算子会报错；`--verify-gdr mtp` 不会自动安装设备 kernel。

## OM 划分

| OM | 用途 | 加载模式 |
|---|---|---|
| `prefill.om` | 64 行物理块处理 prompt | 普通、DFlash |
| `decode.om` | 单行 greedy decode | 仅普通 |
| `draft.om` | 16/64 行特征投影、KV 追加、最多 15 个并行候选 | DFlash |
| `draft_w4a16.om` | W4A16 Draft，相同运行角色 | DFlash 选 W4A16 |
| `draft_w8a16.om` | W8A16 Draft，相同运行角色 | DFlash 选 W8A16 |
| `verify_chunk.om` | Chunk 两遍验证、接受判断、状态提交 | DFlash Chunk |
| `verify_mtp.om` | MTP 验证、接受判断、选择状态 | DFlash MTP |

普通模式加载 2 图；DFlash 加载 3 图；无独立 commit OM。
三种 Draft、两条 Verify 共 7 个 OM，全部位于同一个 `om/` 目录。
Prefill、Decode 为所有组合共用；每条 Verify 在三种 Draft 间共用，每个 Draft 在两条验证路线间共用。
运行时角色由清单映射到对应文件；紧凑执行只改变计算行数和调度，复用既有算子及模型权重。
C++ 在设备上维护 KV、conv、recurrent state；低显存模式分组加载，并共享串行 workspace。
runner 预先绑定 current/next 两组设备地址，提交后选择对应 dataset；热循环不再逐张量调用
`aclUpdateDataBuffer`。预绑定增加主机侧描述符，设备状态缓存仍只有 current/next 两份。
Chunk Verify 保留 24 份 FP32 discard state 输出（48 MiB），不回传 CPU、不进入缓存；
这是已有设备上避免单输出 GDR 性能退化的处理。

Recurrent state 不是 KV cache：batch=1、24 层 `[1,32,128,128]`，单份 FP32 共 48 MiB，
FP16 为 24 MiB，与序列长度无关。C++ 的 current/next 两份采用 FP32 比 FP16 多 48 MiB；
不会增大 KV cache。

## 加速如何衡量

接受率 = 接受候选数 / 提出候选数；每轮产出还包括修正或 bonus token。
整轮 Draft + Verify + 提交耗时低于普通模型生成同等 token 的耗时，才有加速。
零接受后仍持续投机，可能让低接受率 prompt 变慢。

[现有结果](DFLASH_CURRENT_USAGE_AND_RESULTS.md)保留速度、接受率和已知问题。
算子及张量 ABI 细节见 [算子清单](DFLASH_OPERATORS.md)与 [接口参考](QUANT_AIR_OM_FRAMEWORK.md)。
