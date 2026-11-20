# double-w-overlap-narrow-fence 设计

**当前决定为 `BLOCKED_MACHINE_ADMISSION`。** 设计保留一个比较变量：成功迭代 k=0…8、D(k+1) 后那个 Fk 的同步协议。k=9、普通失败、N 尾三个位置继续用原 PIPE_ALL。本文的候选有 SDK 方向映射和源码级依赖推导。当前没有该候选的设备实现、专用 ATC ELF 或机器级批准。

## A：单流水 PIPE_MTE3

`PipeBarrier<PIPE_MTE3>` 为同一 MTE3 流水上前后指令提供顺序。已证明 W 的最后一次源读归 MTE3，但下一次覆盖 W 是 V 上的 Cast。单 MTE3 barrier 给不出“MTE3 源读退休 → V 首写”的跨队列边。V_S(k+1) 是 V 发给 S，不能独立确认旧 MTE3 的退休。它也不给出对 L1 下一次写入的保护。可能等待同一流水上其它 DMA，无新增事件 ID。本设计不选择该替换。编译器是否碰巧补齐所有缺边尚无实际指令证据。

## B：先核对两类不同存储的复用

**仅使用 MTE3_V。** 如果在 D(k+1) 后正确配对 Set/Wait MTE3_V，随后 V 上的 D(k+2) 会等 W(k) 读完；下一次已有的 V_S 也在该 V 等待之后。该边不能确认 MTE1 已经读完 A1。若该 ID 保留到 N 尾，还须与 SDK CopyOut 对同方向的 Fetch 共存。`copy_cube_out_utils.h` 181–192 和 `copy_cube_out_datacopy.h` 520–544 给出了实际调用。已有 SDK 用过这个方向并不能自动赋予应用某个 ID。

**深度为 1 的 A1 暴露另一条反依赖。** 本 normal 配置取 `CubeInBufferType::NORMAL`。单内部 K 块、stepM/N=1、depthA1/B1=1 时，缓存分支分配一个 L1 槽。下一次 SetInput::Reset 释放缓存头，LoadData 再分配。`cube_in_buffer_normal.h` 40–71、87–151；`copy_cube_in_base.h` 44–53。需要 `上一代 MTE1 读 A1 → 下一代 MTE2 写 A1`。核对模板可以看到：

1. `GetNdNzMask(NZ,ND)=1`。2002 的 `CubeInQueType` 对 A 是 GM→A1 的 TQueBind，对 B 是 VECOUT→B1 的 TQueBind。整数 mask 传入 nd2nz 位。`matmul_utils.h` 163–171，`cube_in_buffer_utils.h` 24–63、110–133，`kernel_utils_ceil_oom_que.h` 247–260。
2. 实例实际使用按 **Hardware** 实参的 GetQueEvt 重载。nd2nz=true 时，GM→L1 的前/反向队列标志是 MTE3_MTE1/MTE1_MTE3，UB→L1 也取这一对。FreeBuffer 发反向 Set，AllocBuffer 在相应流水上 Wait。`kernel_tpipe.h` 73–85，`kernel_event.h` 503–542，`kernel_tquebind_impl.h` 206–246、324–329、409–452、491–519。
3. 对齐 **A** 的实际数据写是 `copy_gm_to_cbuf`，属于 MTE2。`CopyND2NZOnTheFly` 确实在该批 A 写后显式 Set/Wait **MTE2→MTE1**，解决本代“先写再读”。在这个无尾分支看不到反向 **MTE1→MTE2** 对下一次 GM→A1 的保护。不能引用另一格式/尾分支中的同名事件。`data_copy_wrapper_using_ub_nd.h` 137–175、253–281；`kernel_operator_data_copy_intf_impl.h` 863–894。B1 的实际写在 MTE3，其队列反向等待落在实际写流水上。

一个可行的反例顺序是：旧 W 已搬完，V 已通过 MTE3_V，但旧 A1→L0A 仍在 MTE1；新的 GM→A1 已能在 MTE2 开始。经审核的显式边不能排除这个顺序。这是源码合同缺口，并非对已测机器发生竞争的断言。

## B 的保守协议

候选保留 double-w-overlap 的所有既有事件、每个 PIPE_V、SDK 的 PIPE_M 以及同一个 L0C。只在原成功非末 K 的位置使用以下两对有方向的事件。伪代码中的名字表示独占预留，具体 ID 必须来自该 block 的 TPipe。

```text
Iterate(k) 成功，wait R(k+1)，D(k+1) 已发
  Set  MTE1→V(load_done)     Wait MTE1→V(load_done)
  Set  V→MTE2(raw_ack)       Wait V→MTE2(raw_ack)
下一轮仍按既有次序发 R(k+2)（若有），再 V_S(k+1)，再绑定 A/B
```

| 保护对象 | 因果链 |
|---|---|
| W(k) 两轮后被覆盖 | 最后读 W 的 MTE3 DMA → B 的 MTE3→MTE1 交接 → 本代 SplitB 的 MTE1 读 → load_done 的 Set → V Wait → 下一次写这个 W 的 Cast。W 的 SDK 源读可以早于该交接完成；选较晚的 L1 读末点同时覆盖 A1/B1。 |
| A1 被下一代填充 | 本代 MTE1 最后读 A1 → load_done → V Wait → 在 V 上发 raw_ack → MTE2 Wait → 下一次真正的 GM→A1 写。B1 的旧 MTE1 读到新 MTE3 写还保留原 SDK 的反向队列。 |
| raw 的 bank 与代际 | D(k) 的 raw 源读由原 V_S(k) 交接。D(k+1) 和对应的 raw Vector Wait 排在 raw_ack 的 Set 前，因而下一代 raw DMA 与其新 Set 都须等 V 已消费前一标志。在模型中仅保留第一对交接，不能推出这个消费先后。 |
| 重新绑定 | 下一轮 V_S 中的 V Set 排在 load_done 的 V Wait 后。现有 S Wait 在调用 SetTensorA/B 前。旧 W 的末读与下一次绑定之间有实际的跨流水传递链。 |
| L0A/B、FP32 C | 原 MTE1→M 和 M→MTE1 ping/pong 继续负责 L0 输入槽；`mmad_compute.h` 的小块 PIPE_M 单独约束相邻 MMAD。这里没有用 W 的 handoff 为 Cube 完成背书。 |
| 避免环等 | 使本代 MTE1 完成所需的本代 A、B 和 L0 空闲信号均先于新 MTE2 Wait 发出；D(k+1) 所需的 raw 搬运和原双 Set 已发出。第二对只推迟后续 MTE2，包含下一预取。 |

`kernel_event.h` 在 2002 的实际 Set/Wait 分派分别为 928–930/1050–1052（MTE1_V）和 893–895/1033–1035（V_MTE2），291–327 对应该架构的 set_flag/wait_flag。两个方向的等待均有成本。它们可能等同核该流水上早先的无关指令，也会延迟下一次 raw 预取直至 Vector 完成 D(k+1)。保留在 M 流水上的 SDK 栅栏、L1 单深度和已有 V_S 还可能序列化大部分工作。这里没有预测可回收时间。

## 所有权、边界和停止条件

- 按选中分支核对池的交集：已有 raw 为 MTE2_S/MTE2_V，W 交接用 V_S，SDK 的 L0 为 M_MTE1/MTE1_M，ND 队列为 MTE3_MTE1/MTE1_MTE3，CopyOut 另用 M_V/V_M、MTE3_V/V_MTE3 和 MTE3_MTE2。它们与候选两个“方向、ID”对分池。SDK 的其他格式实现中确有 Fetch MTE1_V（`data_copy_wrapper_using_ub_nd.h` 436–438），不属于本例对齐 FP16 的 OnTheFly 分支；GM→UB 的队列释放也可用 V_MTE2，而这里的 raw 使用 TBuf 加显式事件。编译器自动插入的同方向同步、实际分配 ID 和峰值仍需在选中 ELF 中核对。
- 每 block 先完成已有资源与两方向 raw ID 的准入，再在任何新 payload 发射前为 MTE1_V、V_MTE2 各预留一个 ID。2002 上每方向有 8 个编号，按 HardEvent 查池。Fetch 查空位，Alloc 置占用位，Wait 消费标志，Release 清占用位。查容量与申请之间不得进入 SDK；每一步核对合法返回和已占用/尚需的 SDK headroom。检查初始保留、内部硬编码/Fetch/Alloc 和失败分支，不能预填实际编号。新协议不借用 SDK 的 M_MTE1 ping/pong。依据 `kernel_macros.h` 136–143、`kernel_event.h` 227–245、`kernel_tpipe_impl.h` 443–484。
- 若第二个申请失败，只归还已经取得的新增 ID，再归还原有应用 ID，执行原初始化失败清理，留存 Poison。若容量预检本身在该构建可能触发断言，须将断言作为失败，不能以分配后的越界号工作。
- k=0 保留原预热，只有成功返回且有 next 才能发两对窄事件。k=1…8 的每一对只允许一代未消费。上一次 load_done 的 V Wait 先于 raw_ack，再经 MTE2 和下一代 A 的 MTE2→MTE1 返回到 MTE1，之后才能再次 Set load_done。上一次 raw_ack 的 MTE2 Wait 先于下一预取和其 MTE2→V 交接，之后才能再次 Set raw_ack。分支、无额外内部异步读、同一 ID 的所有复用都须由实际机器审查证实。
- k=9 不发 raw10 或 D10，不发新窄事件。保留 F9、唯一 GetTensorC 和 FN，使下一 N 重用 W、raw、L0C 及输出 scratch 前已退休。原应用 ID 和新增 ID 均在最后全栅栏后归还。不能用 Release 或 End 表示等待。V_S 取号失败、MoveNext=false 均只排空已发的下一代 raw，保留原失败 PIPE_ALL，停止后续 N，不发布当前 N。硬件异常、超时或 Wait 无法完成时停止，不在该 context 声称排空。
- **共同的首代问题仍需严格证据。** 双 W 每个 N 都在 `wait R0 → D0` 之后、`V_S(0)` 之前发 R1，原 MTE2_V 使用相同预留 ID。源码证明上一 Wait 已发出。仅按公开的二值标志描述，模型不能独立证明 Wait 在 V 上已消费时 MTE2 才进行下一 Set。此处先于 F0，原双 W 和候选共有。需用 dav_m200 明确的同 ID 重置位硬件互锁合同，或对应实际 ELF 中能验证这一先后的指令边来闭合。以 CPU 模型不能认定发生过丢信号，也不能免除此项。即使历史 Smoke 通过，本轮也不改变 raw 协议以救活候选。

成功路径上，这个方案每个 N 新增 9×4 次 Set/Wait 调用，整个单 Gate 为 10944 次，再加每 block 的两份事件管理。2736 个应用全栅栏被替换。源码调用数与机器码指令数、实际停顿时间各有不同，需分别取证。

结论：B 的两段交接可表达新的稳态内存依赖。首代事件证据、实际目标代码的 CFG/地址/事件和机器级准入仍未闭合。C 继续作为既有全栅栏对照。`double-w-overlap-narrow-fence` 保持设计状态。
