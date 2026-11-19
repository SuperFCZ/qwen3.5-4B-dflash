# Native FP16 Head：已找到执行节点，实际权重读取量仍未知

同一 310P3 上的本轮 Raw 回传将两份编译图的 Head 都定位到 `MatMul → ArgMaxV2`。Head 的单任务时间长，适合单独测访存。现有证据还不足以决定修改哪一层缓存或怎样划分词表。

| 实际 Raw 记录 | C16 | C64 |
|---|---:|---:|
| 完整 OM bytes | 3090429227 | 3090516391 |
| 完整 OM SHA256 | `eeb1510326908a7890fd9d62ce7b2ec8801241abadf749e041cb16ce3d9fd17e` | `a120bc6e6b47ae8de9d0a771229fec363477f93bf6b0ba1e706987dee223075e` |
| Head MatMul / Top1，µs/Execute | 6779.029 / 45.318 | 6760.825 / 40.698 |
| 已关联 counters 的 timed tasks / 全部 timed tasks（全图） | 13086 / 14536 | 13095 / 14555 |
| 显式内存读取字段 / stall | 空 / UNAVAILABLE | 空 / UNAVAILABLE |
| 严格 Device Step / 当前口径 | UNRESOLVED / PROVISIONAL_HOST_ENCLOSURE | UNRESOLVED / PROVISIONAL_HOST_ENCLOSURE |

**身份。** 两档实际节点名均为 `MatMul`、`tvm_blockdim=8`，图中记录相同的以下两个标识。它们绑定图属性。若需证明某份 standalone ELF 与它相同，还需核对完整内嵌二进制及长度。

```text
_kernelname    te_matmul_b6c9ba31495399ba8fb6987c9aa07f78c462b0c7c7b796b00d74daaeab70845c__kernel0
_kernel_bin_id te_matmul_b6c9ba31495399ba8fb6987c9aa07f78c462b0c7c7b796b00d74daaeab70845ca12e87780e15684b818ee97e93a75220c7df52acbb494648c4c1691883bc73b0
```

原始证据根为 `/home/w00949577/qwen35-dflash-run/w8a16-deep-profile-acJHJJ`。见 `admit/{asset-index,head-om-evidence}.json`、`profile-paired-analysis/{raw-c16,raw-c64}/{head,counters,case-analysis}.json` 以及各 capture 报告绑定的 OM graph、op_summary、timeline。每档 op_summary 第 435 行为两个 FP16 FRACTAL_NZ 输入，形状分别为 `160,1,16,16` 和 `15520,160,16,16`，输出 FP16 ND `15,248320`。相邻转换名为 `trans_TransData_43`。仅凭邻接表不能认定它在权重输入端。需要核对输入 slot、生产节点、常量 payload/部署清单和运行 task。直接输出边连接 ArgMaxV2。

## 容量、分块、流量

| 问题 | 已有证据及边界 |
|---|---|
| 全词表权重 | 物理 15520×160×16×16×2 = **1,271,398,400 B**，逻辑 NK 为 [248320,2560]。输入已带 NZ 描述符。没有逐地址硬件 trace。 |
| M=15 权重复用 | activation 的物理 M 方向可容纳 16 行，逻辑只产生 15 行。一个 N,K 权重块可以被 15 行共享。是否只发生一次 GM→L1、L1→L0B，或跨 block 重读，同一形状描述符不能区分。 |
| N/K tile 与 native Cube | 15520 是权重的 N16 **存储块数**，160 是 K16 存储块数。8 个 block 可能按 N 分工，也可能采用其它 kernel 内循环。实际 baseM/N/K、归属、depth、L1/L0A/B/C 双缓冲、workspace/scratch 偏移、跨 tile 驻留均为 UNRESOLVED。不能套用 GroupQuant 或自定义 Head 的 N64/K128。整图 Workspace 即使已知也无法推出此 MatMul 的 workspace。 |
| MTE2 在搬什么 | 92.1% ratio、约 5840 µs/task 表明采样对应的搬运流水活动多。缺失的 PMU 关联数为全图 C16 1450、C64 1460，不可把这些行计为零。大权重、activation、重复分块搬运及缓存层次均可能贡献。计数没有地址/张量标签，也无 stall 类别。不可把 ratio 乘 Task Duration 反求 bytes 或 stall，分母和 core 聚合需按原 PMU 列定义解释。 |
| 输出与 Top1 | 单次完整 logits payload = 15×248320×2 = **7,449,600 B**。若经 GM 完整写一次、完整读一次，逻辑接口合计 14,899,200 B。15 个最终 ID 单独计。该数没有扣除缓存命中，也没有计入 cache line 或写分配。40.698–45.318 µs 只属于测到的 Top1 task，MatMul 自身的 logits 写尚未单独计时。 |
| 带宽/重复读 | 本轮两个 Raw `explicit_memory_read_fields=[]`。DRAM/HBM read bytes、权重读取次数和命中率均 **UNRESOLVED**。块内复用、不同 N 分片读不同权重、同一块因 K/M 或多核被重读，是不同假设。即便将来读到总 bytes，也只有在扣除/约束其他 traffic 后才可能区分。 |

## 下一个最小 Head Trace（独立于 Gate A/B）

1. 先对已有 OM/dump 离线复核 Head 的 launch attrs、输入 slot、固定全词表权重 payload、邻接转换、15 行真实 final-norm hidden 及其来源。若 OM 留有可解码 tiling，记录实际结构定义和原始字节。只有能验证长度/布局/对应 kernel 才报告 base、depth、DB 和 workspace。无法恢复时保留未知。
2. 首选 **一个 C16 样本**，只保留已绑定的 native MatMul 与 ArgMaxV2、同样 NZ 输入和 ND logits。新 trace 前先证明 native 专用 executable、实际 blockDim、tiling、权重和 15 行输入与完整图一致。若 standalone 导出改变了 kernel 或 tiling，回到原 Raw C16 OM 采一场限定窗口的 trace，仅提取该节点，或显式标记“独立实现”。不能用重新选型的 standalone 回答原 kernel 的字节数。原生 FP16 logits 若没有边界 Golden，需单独留参考证据。同一 hash 的输入/权重、15/15 IDs、两次一致、守卫与 pre-launch DSO 身份仍须验收。
3. 保存实际 CANN 9.0 CLI 能力与产品 PMU 字段表。第一选择为 task-based 的 **Memory** 组，同时采 AscendCL/runtime/task-time 供匹配。此组可能给带宽或传输层级指标。按原列名、单位、测量区间及 core 聚合解释。MemoryL0/MemoryUB 可另开一次交叉核验片内流量。文档对 Atlas inference product 的 L2Cache/MemoryAccess 有限制。不要造寄存器编号或假设 310P3 有 HBM 字节列。[CANN 9.0 msprof 指标和产品限制](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/devaids/Profiling/atlasprofiling_16_0011.html)
4. 权重与输入先驻设备，排除模型装载/H2D，独立进程、单设备单工作流。先取一小段重复执行（例如 2 次正确性、5 次预热、10 次采集、1 次 postcheck）。为每个使用的样本绑定 Device/Model/Stream/Task/启动时间、kernel、计数作用域。拒绝重复 ID 或跨窗口匹配。计数缺失保留为空。若只有设备级 DRAM 采样，6.78 ms 单次可能短于采样周期：另采足以覆盖多个周期的连续同形状批次和同长 idle 对照，记录采样周期及并发负载。结果只报告为批次/设备级区间，不能回填到 MatMul task。
5. 有明确定义的累计 DRAM read bytes 时，保存原始值，依定义处理芯片/每核计数与 wrap，勿重复汇总设备级行。若只有带宽率，只有在分母、时间、单位和采样覆盖均可核验时才积分，并标为估计。区分 GM 接口、L2、DDR/DRAM 层级。其他核流量、activation、可能的 weight-side 转换和中间量没有拆分前，`all_reads / logical_weight_bytes` 只是总读量与容量的比，不能报告为实际权重扫描次数。补充同一 kernel/tiling 的 resident 与受控 cache 初始状态对照。随意把 M=15 改为 M=1 会触发重新选型。
6. 只有在多读或布局开销被逐任务/地址或受控消融证明后，才提出单一的 N 分区或权重驻留变量。旧 C 的全层结果 C16 AC−A +9.3077 ms、C64 +9.3134 ms，已有回退证据。该路径的手工 N64 循环、partial 合并与跨核同步属于另一个实现。本设计保留全词表、FP16 舍入和 ArgMax tie 语义。[历史整图对照](../draft_ac/c-profile-analysis.md) · [现有 Head 评审](../draft_w8a16_deep_profile/head-optimization-review.md)
