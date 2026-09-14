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
| Target 特征 | 第 1、5、9、13、17、21、25、29 层输出（从 0 编号），拼成每 token 20480 维 |
| Draft | 官方 6 层模型；FC 20480 → 2560，norm，各层上下文 K/V 投影，6 层 attention/MLP，LM head |
| Draft 输入 | 新提交的 Target 特征 + anchor/MASK block；持久 KV 只保留已提交上下文 |
| 精度 | 原生 Target 可选 FP16/W8A8；当前 OM 为 W8A8 Target + FP16 Draft |

OM Draft 的 QK/PV 矩阵乘默认 FP16，缩放、Mask、Softmax 为 FP32；
原生 Torch-NPU Draft 使用 FP32 矩阵乘。两者不能视为完全相同的数值路径。

## Draft 内部流程

```mermaid
flowchart TD
    F["新提交的 Target 特征：8 × 2560"] --> FC["FC：20480 → 2560 → RMSNorm"]
    FC --> CK["各层独立投影 K/V；K 做 RMSNorm + RoPE"]
    CK --> CACHE["追加到各层的上下文 KV cache"]
    B["anchor + K 个 MASK，K ≤ 15"] --> EMB["Target 的 FP16 embedding"]
    EMB --> N
    subgraph L["Draft Decoder × 6 层"]
        N["RMSNorm → block Q/K/V；Q/K norm + RoPE"] --> A["Attention：Q 来自 block"]
        CACHE --> A
        A --> O["输出投影 + 残差"]
        O --> M["RMSNorm → SwiGLU MLP + 残差"]
    end
    M --> H["最终 RMSNorm → 取 MASK 行 → Target 的 FP16 LM head"]
    H --> TOP["全词表 Top1：一次并行提出 K 个候选"]
    TOP --> V["Target Verify：Chunk 或 MTP"]
    V --> C["仅提交旧 anchor + 接受前缀；修正 / bonus token 作新 anchor"]
    C -->|新增 Target 特征供下轮使用| F
    C -->|新 anchor| B
```

- **上下文分支**：首次使用 prompt 的 Target 特征，后续只追加新提交位置的特征。
  同一份 FC/RMSNorm 结果供 6 层分别生成 K/V；上下文不作为 Draft 的 Query。
- **候选分支**：每层 Attention 读取上下文 KV 和本轮 block 的临时 KV。
  滑窗层按因果窗口计算，full-attention 层允许块内双向注意；六层均处理整个 block。
  第 0 行是 anchor，只有后面的 MASK 行输出候选，一次 Draft 前向完成。
- **缓存更新**：block 的临时 KV 不作为下轮的已提交上下文。
  验证接受 a 个候选后，下轮用旧 anchor 加这 a 个候选的 Target 特征补入缓存；
  零接受也补入旧 anchor，新修正 token 留作下一轮 anchor。
- **OM 实现**：上述上下文更新和候选生成合在同一个 `draft.om`，Chunk/MTP 共用。
  特征输入物理 64 行、block 16 行，由有效长度屏蔽 padding；长 prompt 按 64 行逐块建缓存，
  中间块的 Draft 候选丢弃，其计算仍计入预填充成本。

实现见 [Draft 模型](../models/dflash_v1/modeling_dflash.py)、
[DraftContextGraph / DraftProposeGraph](../framework/python/qwen35_dflash/ascend310p/incremental.py)。

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
普通 prefill/decode 使用原有 Chunk 算子，recurrent state 的初始化、存储、传递统一为 FP32，取消 FP16 写回。
Torch-NPU 和 AIR/OM 均采用这一规则；权重、conv 和 KV 的精度保持现有设置。

Torch-NPU 使用实际 `T=K+1` 行；OM 固定 16 行，通过有效长度限制接受和提交。
MTP bank 在 OM 内消费；原生提交复制所选状态，避免跨轮保留整块 bank。
缺少 MTP 算子会报错；`--verify-gdr mtp` 不会自动安装设备 kernel。

## OM 划分

| OM | 用途 | 加载模式 |
|---|---|---|
| `prefill.om` | 64 行物理块处理 prompt | 普通、DFlash |
| `decode.om` | 单行 greedy decode | 仅普通 |
| `draft.om` | 特征投影、KV 追加、并行候选 | DFlash |
| `verify_chunk.om` | Chunk 两遍验证、接受判断、状态提交 | DFlash Chunk |
| `verify_mtp.om` | MTP 验证、接受判断、选择状态 | DFlash MTP |

普通模式加载 2 图，DFlash 加载 3 图；无独立 commit OM。
一个目录存 5 个 OM，两个路线清单直接引用同一份 Prefill、Decode、Draft。
运行时角色名仍为 `target_prefill`、`target_decode`、`draft`、`target_verify`，由清单映射到对应文件。
C++ 在设备上维护 KV、conv、recurrent state；低显存模式分组加载，并共享串行 workspace。
Chunk Verify 保留 24 份 FP32 discard state 输出（48 MiB），不回传 CPU、不进入缓存；
这是已有设备上避免单输出 GDR 性能退化的处理。

Recurrent state 不是 KV cache：batch=1、24 层 `[1,32,128,128]`，单份 FP32 共 48 MiB，
FP16 为 24 MiB，与序列长度无关。C++ 的 current/next 两份采用 FP32 比 FP16 多 48 MiB；
不会增大 KV cache。此前 FP16 写回尚无独立实测收益，新版 FP32 的精度与速度需重新测量。

## 加速如何衡量

接受率 = 接受候选数 / 提出候选数；每轮产出还包括修正或 bonus token。
整轮 Draft + Verify + 提交耗时低于普通模型生成同等 token 的耗时，才有加速。
零接受后仍持续投机，可能让低接受率 prompt 变慢。

[现有结果](DFLASH_CURRENT_USAGE_AND_RESULTS.md)保留速度、接受率和已知问题。
算子及张量 ABI 细节见 [算子清单](DFLASH_OPERATORS.md)与 [接口参考](QUANT_AIR_OM_FRAMEWORK.md)。
