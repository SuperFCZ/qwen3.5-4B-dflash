# 同核事件和 W16 世代

以下推导仅适用于直接、同核、无尾块 Gate。F(k) 是原 Kernel 276 行的 `PipeBarrier<PIPE_ALL>()`，D(k) 是 180–220 行的反量化。把 Iterate 返回当成一次发射。最坏情况下，SDK 已发出的 B 读可以一直占着 W bank 直到 F(k)。

```text
同一核、同一 N64 tile                           时间向右
                           已发 MTE3: W(k)→L1 → MTE1: L1→L0B → MMAD(k) ──────┐
D(k) → V_S → SetTensorA/B → Iterate(k) 成功返回                              │
raw DMA(k+1) → Set MTE2_S/V → 两个 Wait(k+1) → D(k+1) 只写另一块 W ─────────┤
                                                                      F(k) ◀─┘
下一轮                    F(k) → V_S(k+1) → rebind 到另一 W → Iterate(k+1)
原 W 的下一次写            F(k) → rebind 到另一 W → D(k+2)
```

| 时段 | 信号/资源所有权 | W、raw、scratch 的边界 |
|---|---|---|
| 初始化 | SDK 注册后应用通过同一 TPipe 分别 Alloc 一个 MTE2_S、MTE2_V。两者均合法后发首个 DMA。部分分配失败只归还成功取得的 ID。 | 先固定所有 bank 与 SDK scratch 的地址。broadcast 初始化一次、PIPE_V。 |
| N 头 / k=0 | 发 q0/s0，再 Set 两个方向。各 Wait 一次消耗本代信号，D0→W0。可以向另一 raw bank 发 raw1/Set。 | q0/s0 到齐才读。W1 未绑定。D0 保留原 V_S 交接。 |
| k 稳态 | SetTensorA、SetTensorB(W[k%2],true)、Iterate(k!=0)。成功后消费已发 raw(k+1) 的两个 Wait。V_S 为 Fetch 后紧邻 Set/Wait，期间不进入 SDK。 | 已绑定 W 只读。D(k+1) 只写另一 W。保留全部 PIPE_V，包括上次 Mul 对 broadcast 的最后读。此时不绑定下一代 A/B。 |
| K fence 与下轮 | 候选将原 F(k) 保留在 D(k+1) 后。下一轮开始才重绑 SDK。raw(k+2) 在 F(k) 后复用已消费的 raw(k)。 | 已发 W(k) 最后读、当前 MMAD 和本代 Vector 工作处于本核 fence 以内。保留下一轮 V_S，与原 Matmul 使用相同的交接合同。 |
| k=9 / N 尾 | 无 raw10、Set/Wait10 或 D10。成功的 Iterate(true) 后照常 F9。一次 GetTensorC，再发 281 行的 N 末全流水 fence。 | L0C 中十次升序 FP32 partial 完成，只在 GetTensorC 写 FP16。下一 N 从 raw0/W0 和 partial=false 开始。 |
| 最后 N | 最后一次 N 末 fence 后 Release 应用的两个 reserved ID，再 End。 | End 处理内部缓存和队列。应用的 DMA 排空发生在 End 之前。 |

**事件所有权。** 2002 上 EventToIndex 分方向定位池，Alloc 设置占用位，Fetch 寻找空位但不保留它，Release 清位且不等待。即使不同方向返回相同数值，事件身份仍由方向、ID 和本代 Set/Wait 确定。每代 raw 只允许一个未消费的信号对。应用不缓存 Fetch ID 跨库调用。SDK 自己管理 L1/L0 队列、M_MTE1 等事件。Brcb 的 B16 stride-one 分支以 Vector 指令和 PIPE_V 展开，不能套用其它 stride 分支的四个 Fetch。出现 SDK 内部事件耗尽或宏选择 client/service，立即阻断。

**消费者证明。** SetTensorB 保存地址及转置状态。Normal scheduler 在当前调用栈内完成 B LoadData 的发射，拿 L1 tensor 做 SplitB，用 L0 tensor 发 MMAD。L1 cache hit 可减少读。miss 走 CopyTileToCube。GetResult 只以 CO1、形状及目的 tensor 为输出拷贝参数。这里保护的是调用者 UB 的最后读，整个同核已发队列在 F(k) 处阻塞，下一次使用旧 W 已过该界线。无需借 SDK 内部队列 ID 标记 W 可写。[CANN 9.0 核内阻塞范围](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0271.html) · [队列顺序与事件语义](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_0008.html)

## 失败与证明边界

- Normal scheduler 的 MoveNext 返回 false 在本次 Compute 之前。若已发 raw(k+1)，只消费已经 Set 的两个 Wait，然后原失败 fence、Release、End、直接返回。**先检查 Iterate 结果，再发 D(k+1)**。尾 K 没有这两个 Wait。先前已完成的 N 保留，当前或剩余输出不得通过完整 Poison/Golden 检查。
- 资源分配、ID 合法性在启动 DMA 前检查。部分分配失败时不得 Release 未保留的 ID。正常路径中不增加有在途 DMA/Vector 写却直接 return 的分支。
- 设备异常、Wait 无法完成、同步超时时，软件没有成功排空证明。停止后续执行，保留 SDK 返回和已取证的日志、输出。不在该未恢复 Context 中重用 buffer 或发布成功报告。
- 如果服务核稍后读 W、后续 GetTensorC 延迟发起 B 读、或内部转换使用重叠固定 UB 地址，均推翻本证明。正式实验须绑定实际编译宏、ATC 选中的 ELF、地址范围与 Tiling。非 Vector ND2NZ 实例的 SetWorkspace 使用调用者独占的 LocalTensor。对齐、无 Bias、FP16 ND GM 输出的内部算法计入 CO2 16×64×2=2048 B，尾块 A/B 临时需求为零。这项偏移计算不能代替 Host 的 152832 B scratch、transLength 和系统 Workspace 检查。

## SDK 源码身份

本地只读检查 `/usr/local/Ascend/cann-9.0.0/aarch64-linux/asc/`。下列行号为各文件的实际行号，SHA256 覆盖整份文件。服务器可用末尾的 CPU 命令复核。生产 Kernel 的身份和行号见[设计](w16-double-buffer-revised-design.md)。

| SDK 相对路径 | 行号及用途 | SHA256 |
|---|---|---|
| `include/adv_api/matmul/matmul_intf.h` | 19–31、80–145 · 非 CPU 的直接实现与 client/service 的选择条件 | `32992a9d7a343f1deed1407e6a696e95b154299c3a339ea90378bba63035eece` |
| `include/adv_api/matmul/tiling.h` | 36–53、64–95、400 · CFG_NORM、INNER_PRODUCT、关闭 enVecND2NZ | `a9a55c80deb72b6b2b8bc1561ca17b13889dd91fa40950a0f70a02b37659da1f` |
| `impl/adv_api/detail/matmul/feature_trait/matmul_chip_cap.h` | 71–76、96–101 · arch 2002 的 feature 行 | `4274aec91fa3ff088aff38caa7a4f3922b51757f0b6710cd1a850769d9c727c8` |
| `impl/adv_api/detail/matmul/feature_trait/matmul_feature_trait.h` | 34–48、86–93 · IsNeedUB 及跨核 singleshape 条件 | `02813710c1eb143d42ff237debd95049cfc129f1dc0aef6d39dd2a6e0032991f` |
| `impl/adv_api/detail/matmul/matmul_impl_base.h` | 140–143、459–461、546–549、620–624、640–645 · SetLocalWorkspace/End/SetTensorB/GetTensorC/Iterate 分派 | `923527715a9baa2147538221ccf82afc14b5fd4f913b42363adbf00eb2de5006` |
| `impl/adv_api/detail/matmul/stage/copy_cube_in/base/copy_cube_in_base.h` | 44–54 · 绑定地址，重置输入缓存 | `ffc0476ab498f82527c432c17f2ce097ad395207c54a723d8b8c540ba7bdcb5e` |
| `impl/adv_api/detail/matmul/param/matmul_tensor_info.h` | 60–71 · 复制 LocalTensor 地址并设置转置 | `c43e590a7ce336b79a3b50bcc76bc0b597e953df8210f7ffb4db3b0e9107c1a5` |
| `impl/adv_api/detail/matmul/scheduler/base/scheduler_norm.h` | 32–38、93–123、164–206、228–255 · 当前策略同步发出 B 的 LoadData、SplitB、Mmad，循环后返回 | `bb565c78b53644f63cdf89cef18f4d622f11f4fca755ce03b989347f5f5083cd` |
| `impl/adv_api/detail/matmul/scheduler/base/scheduler_base.h` | 149–162、205–209、396–416、457–485 · GetResult 与 End 的操作对象 | `e37c5029455b8ffac29e2a51652592d02830504e77ce68b6d3fe7d4902d9a2a4` |
| `impl/adv_api/detail/matmul/stage/copy_cube_in/base/copy_cube_in_using_ub.h` | 61–74、77–124 · L1 Hit/Copy、入队/出队、释放 | `4f57a5faf40eca08996956691ffe91be90f90ab4500707bb28ce43b85dce1b8e` |
| `impl/adv_api/detail/matmul/stage/copy_cube_in/copy_tile_to_cube/copy_tile_to_cube_using_ub.h` | 40–55、61–89、140–165 · GM 与 UB、转置及非 INT8 特例 | `8fed124b939a1a2477622e1414362a95222c37dbfb24ad895494cacb58b33726` |
| `impl/adv_api/detail/matmul/stage/copy_cube_in/copy_tile_to_cube/data_copy_wrapper_using_ub_nd.h` | 137–175、176–250、253–280 · padding/Vector 尾块与对齐搬运 | `6d5315d222c1f4357f45adbc00ad76380ec0d7d6d3bcffd570ff6cd9867c18d6` |
| `impl/basic_api/dav_m200/kernel_operator_data_copy_impl.h` | 54–69、99–110 · GM→L1、UB→L1 实际拷贝 | `8a890aa80c39380519bcba461bfd19a67060e650253622d8b49b4b2c01ea3c8f` |
| `impl/adv_api/detail/matmul/resource/local_workspace/matmul_local_workspace.h` | 155–197、203–218、248–317 · 非 Vector ND2NZ 工作区域和容量 | `dbffdce3b74c3e18414c1c5e89e4a4260ec64654eeccc58f172ed89f7ae6c08b` |
| `impl/adv_api/detail/matmul/resource/l0_buffer/tbuf_pool_l0_v200.h` | 29–54 · SDK 持有 L0 ping/pong 的 M_MTE1 ID，其析构 Wait/Release | `dc547d6d1635d4456519ddca46b1b8f2b1607fe2cbec8b82b5aff11b76983f45` |
| `impl/basic_api/kernel_tpipe_impl.h` | 298–355、443–484 · InitBuffer 的物理 pool，Alloc/Fetch/Release | `7a91fe65d16c3775fa57d15ebb63a37b275c0c9ee1c269bca4d76355e2873c8d` |
| `impl/basic_api/kernel_event.h` | 227–237、361–382、890–915、940–942、1030–1119 · 同核事件池与方向 | `d974ecee61e39402dad6fa368e036b538bc1f69b05989b6b3a9fc9c946bd609c` |
| `impl/basic_api/kernel_operator_block_sync_intf_impl.h` | 97–101 · PipeBarrier 调用 PipeBarrierImpl | `7f0fa56ad683b86f2ded10b984645e698d05a04e541c04d25cedb69c4be70c6f` |
| `impl/basic_api/kernel_reg.h` | 65–108 · arch 2002 发出 pipe_barrier(pipe) | `36bcc76faf373f872ecbcd3a27eab7ec9d0578c87f2cac1b8abdcc08583f1d4a` |
| `impl/basic_api/dav_m200/kernel_operator_vec_brcb_impl.h` | 54–94、97–129 · B16 stride-one，临时区/尾部/三个 PIPE_V | `54943fdd95ab71ab731f003ed18e4aac01929d47922320636ac753f37a4a4176` |
| `impl/basic_api/kernel_utils_base.h` | 219–239 · 设备固定临时地址计算 | `f39fc8fc88a801829ffefef04baeb1e6b408510ec07b6369ab4b3a598f40aae0` |
| `impl/basic_api/utils/kernel_utils_constants.h` | 116–122 · 256 KiB UB、TMP_UB_OFFSET=248 KiB | `3ce93de8c25d198386e87656481f6bc9dfbeae86fe66a248f7e3d36c86cd32da` |
| `impl/basic_api/dav_m200/kernel_operator_vec_vconv_impl.h` | 30–40 · INT8→half 的 CAST_NONE 指令入口 | `3cd58234ff5fd626f0a7898a31e28ad6b1c8d6f775d02eb28a7045c1653f5126` |

## CPU 复核

从仓库根运行。第一条只需要 Python 标准库。第二条额外逐文件核对上表。

```bash
D=framework/custom_ops/draft_w8a16_overlap_review
python3 -B -m unittest discover -v -s "$D" -p 'test_review.py'
CANN_ASC_ROOT=/usr/local/Ascend/cann-9.0.0/aarch64-linux/asc \
  python3 -B -m unittest discover -v -s "$D" -p 'test_review.py'
```

CPU 模型让 SDK 读者一直存活到 fence，检查两种调度的同一组操作，覆盖 k=0…9 各轮的普通失败、缺失 Wait、两代误用同一 bank 和缺失 fence。模型验证依赖图的一致性。实际 TPipe 地址、设备发射时序与重叠程度需要各自的证据。

本轮在 macOS arm64 完成 17 项 CPU 检查，未配置 CANN 路径的 1 项 SDK 核对跳过。在 Linux aarch64 CANN 9.0 容器完成全部 18 项，其中 SDK 核对逐个匹配 23 份头文件的 SHA256。设备 Kernel 编译、ATC、NPU Smoke、Benchmark 和新 Profile 均为 NOT_RUN。
