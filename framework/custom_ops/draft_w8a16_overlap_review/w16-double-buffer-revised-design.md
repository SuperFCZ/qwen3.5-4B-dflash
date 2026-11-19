# Gate W16 双缓冲：在原 fence 之前只生产下一块权重

**设计结论**：对下述同核、直接 `MatmulImpl/CFG_NORM` 实例，原 K 末尾的 `PIPE_ALL` 可以作为 W16 **下一次复用**的消费者 fence。关键条件是先在另一块不相交的 W16 上写，并且在 fence 之后才重新绑定 Matmul。旧报告中“缺少逐 bank 的 Cube 完成票据”针对的是绕开 fence 的流水。这里无需申请这张票据。现阶段允许准备单 Gate 的独立对照设计。候选正式 Tiling、生成指令、实际地址、正确性和重叠时间均未验证，设备执行准入仍为 **BLOCKED**。代码定位、条件失效时的处理见[生命周期审计](w16-event-lifecycle-revised.md)。

## 已测对象

服务器实验 `/home/w00949577/qwen35-dflash-run/w8a16-deep-profile-acJHJJ` 回传状态为 `DEEP_PROFILE_PAIRED_ANALYSIS_PASS`。两档 Raw 的严格 Step 归属都是 `UNRESOLVED_DEVICE_ASSIGNMENT`。下表的整图值来自 `gate-raw-atc8-Np20yu/full-benchmark/report.json` 中无插桩的 90 个样本中位数。五个 Gate 值是正式 30 个插桩窗口内该角色的平均 Task Duration Sum，归属为 `PROVISIONAL_HOST_ENCLOSURE`。

| Gear | 整图 Off / Raw (ms) | 五个 Gate Off / Raw (ms/Execute) | Gate 差 (ms/Execute) |
|---|---:|---:|---:|
| C16 | 31.217700 / 30.569870 | 10.310968 / 9.641374 | 0.669594 |
| C64 | 32.091310 / 31.426295 | 10.300422 / 9.643620 | 0.656802 |

同期正式整图 Off−Raw 为 0.647830 / 0.665015 ms（C16/C64）。它和分类 task 差值分别测得。差得接近不足以证明剩余时间的因果归属。

后续对照使用 **Raw**。8 个 ATC block 各自完成 38 个 N tile。每一块输出仅一个 owner。每个 N64 tile 有 10 个 K256、20 个独立 Group128。已验收的生成头 SHA256 为 `26017103ea631e701088f9d7e44e8645737228b4ba869b062c5726b47bb543e6`。已测路径用 batched Cast、broadcast、loop DataCopy、两份 INT8/scale 和一份 W16。详见 `gate-raw-atc8-Np20yu/{source,full-export}/report.json` 及当前 Profile 的 `admit/asset-index.json`。本轮服务器数值和身份来自终端回传。

## 可验证的依赖

1. `SetTensorB` 保存地址及转置状态。直接实例的一次成功 `Iterate(k!=0)` 在返回前已调用 B 的 `LoadData`、L0 切分、CubeCompute 和内部释放。它返回迭代状态。已发出的搬运和计算仍可在途。读取 W 的指令是在该核发出的 UB→L1，后面的 MMAD 使用 L0B。
2. 对齐的 M16/N64/K256、无 Bias、FP16 ND、非转置 A 和转置 B，Normal 配置关闭 `enVecND2NZ`。A 为 GM 的 MTE2，B 为本地 MTE3，进入按 C0 的 strided CopyND2NZOnTheFly。宽为 256、两者高分别为 16/64，无 padding，无 tail Duplicate。在这些已限定的分支上没有输入转换的显式 Vector transpose。宽、高、模板或尾块一旦变化，SDK 可能用 UB scratch、Vector mask、Duplicate、MTE2_V、V_MTE3。须重审。外部 Brcb 自身会改 Vector mask 并使用固定 UB。
   对 B 的实际对齐分支，每次从 64 行取一块 32 B，跨行 gap 为 15 个 32 B 块。顺次做 16 个 K16 slice。1024 个 32 B 源区间恰覆盖本代 W 的 32768 B，最后一个区间含该 bank 最后一个 half。A 的行跨度为 2560 half，gap 为 159。CPU 枚举独立核对了 W0、W1 两代的读集。
3. 在 `Iterate(k)` 成功后，外部 Dequant(k+1) 只读已 Wait 的另一份 q/s，只写 W[(k+1)%2]、独占的 broadcast 区及 Brcb 固定临时区。此时保留 W[k%2] 及整个 SDK workspace。不存在同一 bank 在同一代被 SDK 读和外部 Vector 写。Cube 与 Vector 可独立排队，但 decoder、UB 端口、L1、内部队列等待和编译器插入的同步都可能缩短实际重叠窗口。
4. F(k) 覆盖该核在它之前发出的各流水指令。正常 SDK 分支已发完本次 W 读，下一次 Matmul 绑定的是另一代地址，GetTensorC 只输出累加器。**W(k+2) 的覆盖晚于 F(k)**。不能将 F(k) 用于一个在 F(k) 之后才发起读的异步服务，也不能用它补救在 F(k) 之前已发生的别名写。多核之间只分 N，不需要借别核的 fence。参见 [CANN 9.0 PipeBarrier](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0271.html) 和 [CANN 9.0 核内流水及事件](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_0008.html)。

> 实际依赖是 MTE2(k+1) 完成后 Vector 才能读这一代。可以尝试让这两段与 Cube(k) 的不同区间交叠。三者同时满载、以及 Iterate 返回时仍有可观的 Cube 工作，均无测量证明。

## 每核 UB 和 Cube

| 区域，字节半开区间 | 双 W 控制与候选 | 生命周期 |
|---|---|---|
| q0 / q1 | [0,16384) / [16384,32768) | DMA 发出至该代最后一个 Cast 读完 |
| s0 / s1 | [32768,33024) / [33024,33280) | DMA 发出至该代 Brcb 读完 |
| W0 / W1 | [33280,66048) / [66048,98816) | Cast 首写至 SDK 最后读完成，按代交替 |
| broadcast（含 256 B rear） | [98816,101120) | 单份，各组 Mul 之后才复用 |
| SDK `SetLocalWorkspace` | [101120,253952) | Matmul 生命周期内独占 |
| SDK 顶部保留 | [253952,262144) | 其中 B16 Brcb 临时地址从 254976 起，声明 1024 B |

这些是从 UB 基址 0 算出的**设计区间**。TPipe 按物理 pool 的当前高水位分配，长度向上对齐到 32 B。若注册/初始化消耗了额外 UB，不能仍按零偏移填报告。未来须查实际分配/静态地址证明，各区之间无别名，最高分配端点不超过 253952。W bank 相差 **16384 个 half**。仅将字节数翻倍而不改变传给 SetTensorB 的 view 起始位置，会使整个证明失效。

原 Raw user/scratch 为 76544/185600 B。两份 W 为 109312/152832 B。已冻结的本地 CANN Host Tiling 实际查询中，两次 GetTiling 均返回 0，200 B TCube 序列化完全相同，SHA256 为 `745b846a7e3b31de10827caa5e493026d3984e282dd698177d2da380b826359a`。base 16/64/256，depth 1/1，step M/N/Ka/Kb 各 1，DB 2/2/1，transLength 34816。L1、L0A、L0B、L0C 分别 40960、16384、65536、4096 B。第二块 W 在 UB。L0B 已满。设计不需要退 K128，也不允许通过退 K128 让对照过关。父 Tiling 中的 scratch 字段理应随预算变化，TCube 仍须逐字节相同。真实 Formal Host 必须重新核对 Workspace 2097152 B 和全部实际资源。[既有 SDK 查询及字节记录](../draft_w8a16_deep_profile/w16-ub-budget.json)

## 将来的独立候选改动位置

基于 `draft_quant/op_kernel/d_flash_group_quant_linear.cpp` SHA256 `a4861bb5f7e38aceced9e4d89fc6c2519cb4a80e0914d7dd418c50389e8f1a4f`。以下为独立实验的改动图，本次未写入设备实现。

| 行 | 将来拟改或保留的动作 |
|---|---|
| 66–104、112–120 | 限于该 Raw Gate。weightBuf 一次申请 65536 B，取两个有界的 [64,256] half view。按上述分账把 152832 B 整块给 SDK。沿用原初始化及 reserve。Host 对应 139–171 行重算 user/scratch 并执行 GetTiling。 |
| 126–156 | 两个已保留的 raw-ready ID 继续复用。每个 N 单独预热 raw0、Wait 两种 ready、生成 W0。只在 hasNext 时发另一个 raw bank 的一次预取。 |
| 180–220 | 将此块作为一项 Dequant 操作，显式传入本代 q/s/W view。Cast、FP16 Mul、两个 Group128、Brcb stride 与全部 PIPE_V 保留。不要把下一次 SetTensorA/B 一起提到 fence 前。 |
| 239–276 | 每次仍有 V_S，绑定 A 的原始 k 偏移与本代 W、调用 Iterate。成功才移动下一次 Dequant 到原 K fence 前。第 0 轮 partial=false，其余 true，同一个 FP32 L0C 按 0…9 归约。失败分支见事件表。 |
| 278–287 | 每个 N 仍只一次 GetTensorC、一次 N 末 PIPE_ALL，最后才 Release、End。每个 N 的第一次 Iterate 重新初始化 L0C。 |

第一项实验以相同双 W 资源的“fence 后 Dequant”作对照以隔离调度，另用冻结单 W Raw 检查实际净收益。若实际 ATC 选择跨核服务、有延迟的 UB 读、未知 scratch 别名、不同 Cube 或不同数学序列，W16 路线 **BLOCKED**，转为独立 Down 实验。[实验准入](next-npu-experiment.md)
