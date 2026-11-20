# 单 Gate 的 W、L1 和 L0C 生命周期

审查对象是 Ascend310P3、CANN 9.0.0、同核 dav_2002 的 C16 Layer0。A 为 GM ND FP16，Matmul 的 B 为转置的 UB ND FP16，C 为 GM ND FP16。原始 INT8 NZ 权重在核内按 Group128 解量化。实际 ATC 为 8 block，304 个 N64 tile 按 block 循环分配，每个 block 38 个。每个 N 有十个 K256。CFG_NORM 关闭 Vector ND→NZ，L1 A/B depth=1/1，L0A/B/C DB=2/2/1，base M/N/K=16/64/256，transLength=34816，Workspace=2097152 B。完整 200 B Cube SHA256 为 `745b846a7e3b31de10827caa5e493026d3984e282dd698177d2da380b826359a`。

下列顺序属于**源码发射顺序**。不同流水可以延迟或重叠。文中的 SDK 文件名与行号在 [SDK 路径、行范围和整文件 SHA256](sdk-evidence.json) 中对应唯一条目。26 份沿用 YHsAKu admission 的绑定，另列出这次沿直接调用链核验的 17 份。CPU 模型和源码统计均不产生设备 PC 或时间戳。

## 三版逐 K 调度

`Rk` 表示 MTE2 搬运 8 个 K32 的 q 平面及 2 个 FP16 scale 切片，随后用已经保留的两个 ID 分别 Set MTE2→S、MTE2→V。`wait Rk` 依次 Wait 这两个方向。`Dk` 表示在 Vector 上解量化。`Ik` 展开为 Fetch V_S、Set V→S、Wait V→S、SetTensorA(x+k×256)、SetTensorB(W,true)、Iterate(k!=0)。双 W 先检查 Fetch 合法性。`Fk` 为 K 末 `PipeBarrier<PIPE_ALL>`，`FN` 为 GetTensorC 后的独立全流水栅栏。下图的箭头只给出发射次序，各跨流水完成关系见下节。

```text
single-w-raw                 每个 N 开头 R0
 k=0                         wait R0 → R1 → D0(W) → I0(false) → F0
 k=1…8                       wait Rk → R(k+1) → Dk(W) → Ik(true) → Fk
 k=9                         wait R9 → D9(W) → I9(true) → F9
 N 尾                        GetTensorC → FN

double-w-serial              每个 N 开头 R0 → wait R0 → D0(W0)
 k=0                         R1 → I0(false,W0) → wait R1 → F0 → D1(W1)
 k=1…8                       R(k+1) → Ik(W[k%2]) → wait R(k+1) → Fk → D(k+1)(W[(k+1)%2])
 k=9                         I9(true,W1) → F9
 N 尾                        GetTensorC → FN

double-w-overlap             每个 N 开头 R0 → wait R0 → D0(W0)
 k=0                         R1 → I0(false,W0) → wait R1 → D1(W1) → F0
 k=1…8                       R(k+1) → Ik(W[k%2]) → wait R(k+1) → D(k+1)(W[(k+1)%2]) → Fk
 k=9                         I9(true,W1) → F9
 N 尾                        GetTensorC → FN
```

三版的 k=1…9 均传 partial=true。单 W 的原实现位于 [设备源码](../draft_quant/op_kernel/d_flash_group_quant_linear.cpp) 第 33–63、112–156、180–220、239–287 行。双 W 的位置在 [共用循环](../draft_gate_w16_atc8/kernel/tile_schedule.h) 第 5–24 行及 [设备操作](../draft_gate_w16_atc8/kernel/double_w.inc) 第 23–68、78–133 行。已验收 Raw 生成头为 `26017103ea631e701088f9d7e44e8645737228b4ba869b062c5726b47bb543e6`。两套双 W 的生成源码 SHA 和所选 ELF 在 [历史对照](fence-comparison.md) 中列出。

| 位置 | 该形状实际选择的同步 |
|---|---|
| 每 block 初始化 | expanded 整块清零后一次 PIPE_V。应用保留 MTE2_S、MTE2_V 各一个 ID，双 W 还检查 TPipe 实际基址和资源失败分支。SDK L0 池先保留两个 M_MTE1 并各自 Set，使 ping/pong 首次可用。 |
| 每个 Dk | 8 Cast 后一次 PIPE_V。两个 Group128 各执行 Brcb → PIPE_V → Mul → PIPE_V。最后一次 PIPE_V。合计 6 个外层 PIPE_V。每次 B16 Brcb 的 stride-one 分支在 8 个 repeat 中各有 3 个内部 PIPE_V，两个 group 共 48 个。该分支不使用通用 stride 分支的四个 Fetch。依据 `kernel_operator_vec_brcb_impl.h` 54–129。 |
| 每个成功 Ik | V_S 成对交接。A 的对齐 GM→A1 搬运后显式 Fetch/Set/Wait MTE2_MTE1。A/B 的 L1 normal 队列在 miss 时 EnQue/DeQue，在 Reset/再分配时回传复用事件。对本 ND 配置队列前向 MTE3_MTE1、反向 MTE1_MTE3，A 实际写流水另见下节。L0 池 Allocate 等 M_MTE1，SplitA/B 后 Set/Wait MTE1_M，CubeCompute 后 Free 再 Set M_MTE1。每次本尺寸 MMAD 后还有一个 SDK PIPE_M。 |
| 每个成功 N | 显式 10 个 Fk、1 个 FN。原始 304 个 N tile 共 3344 个应用全流水栅栏。保留 F9 与 FN、仅研究 k=0…8 时，候选位置为 2736 处，成功路径还保留 608 个应用全栅栏。 |
| CopyOut 与跨 N | CO1 队列 EnQue/DeQue 提供 M→V 交接，Free/再分配有 V→M。SDK CopyOutEnQue 用 MTE3_V、CopyOutDeQue 用 V_MTE3，CO1 FP32 转 FP16 到 CO2，再由 MTE3 写 GM。对齐 OnTheFly 路径还 Alloc/Release MTE3_MTE2，其 tail 在这里为零，写出末尾另有一对 Fetch/Set/Wait MTE3_V。该分支针对 B=int8 的 PIPE_MTE3 不适用于 Matmul 的 FP16 B。FN 退休当次输出后才进入下一 N。 |
| 失败、结束 | 单 W 和双 W 的 Iterate=false 均消费已发的下一代 raw 双 Wait，再走原失败 PIPE_ALL，当前 N 不写出。k=9 无下一代可等。双 W 的 V_S 取号失败在重绑前走同一清理。块尾归还应用 ID，再 End。L0 池析构 Wait/Release 自己的两个 M_MTE1。全栅栏不清事件占用位。 |

表中按选中分支列出 API 层的显式同步。编译器可能合并或补入同步，不能从这些次数换算执行时间。

## 谁最后读 W

1. `Weight(k)` 将 `weights[(k%2)*16384]` 作为有界的 16384-half view。双 W 源在首次 payload 前检查每核物理起点 `33280` 和 `66048`。W0、W1 分别为 `[33280,66048)`、`[66048,98816)`，随后是 expanded `[98816,101120)` 和 SDK scratch `[101120,253952)`。Brcb 固定临时区 `[254976,256000)` 在顶端 reserve 内。单 W 只有一个 32768 B 权重 bank。双 W 的数值是源码检查的设备地址条件，历史实验没有记录逐指令地址。
2. `matmul_impl_base.h` 504–509、546–550，`copy_cube_in_base.h` 44–53 和 `matmul_tensor_info.h` 60–71 表明：传入的是 LocalTensor 的地址描述及转置标志；SetInput 更新描述、Reset L1 缓存，Scheduler::Reset 重置迭代游标。这里还没有把 W 搬入 L1。
3. `Iterate` 在当前核调用 `ScheduleOnce`。成功时在返回前发出 LoadData、Split、CubeCompute。B 的 L1 miss 经 `copy_cube_in_using_ub.h` 97–124、`copy_tile_to_cube_using_ub.h` 139–165，重新用保存的地址构造源 view，进入 `CopyND2NZOnTheFly`。64×256 对齐，`calcWidth=16`、`tail=0`、`srcGap=15`。16 次 DataCopy 各取 64 行、每行 32 B。最后一次是 stripe 15，第 r 行的源区间为 `[W+512r+480,W+512r+512)`。命中 L1 时该次可直接用 L1 view。每次外层重新 SetInput 的 normal 缓存 Reset 应清掉旧代；正式验证仍需确认所选路径。
4. `kernel_operator_data_copy_intf_impl.h` 941–957 实际分派到 `DataCopyUB2L1Impl`；`dav_m200/kernel_operator_data_copy_impl.h` 99–110 将它标注为 `__inout_pipe__(MTE3)`，以 UB 源地址调用 `copy_ubuf_to_cbuf`。W 的最后读完成，指该代最后一个相关 DMA 的源读已经退休。C++ 调用返回只能确立已发射。`scheduler_norm.h` 194–206、228–255 继续以 B1 为源、以 L0B 为目的 SplitB。FP16 转置路径见 `load_to_l0b_loadInstr.h` 45–57、81–99、117–133 和 `kernel_operator_mm_impl.h` 56–75。其后 `mad` 的操作数是 L0A/L0B 和 L0C。**MMAD 对应用 W 的直接读次数为零。**
5. 对本核 CANN 2002，`kernel_event.h` 239–245 列出 S、V、M、MTE1、MTE2、MTE3。`kernel_reg.h` 84–107 将这里的 barrier 下发为 `pipe_barrier(pipe)`。Fk 阻塞以上各流水中已发的工作。相对 W 释放，它还可能等 A 的 GM 搬运、raw 预取、D(k+1) 的 Vector、L1→L0 以及 M 上的累加等。多少工作在 F 处仍未完成，现有数据没有给出。作用域限于本核，不替其他核或尚未发出的调用排空。[CANN 9.0 PipeBarrier](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0271.html) · [单流水与跨流水同步](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0179.html)

## FP32 L0C 的独立顺序证明

- `GetMmDstType<half>` 在 2002 取 float（`matmul_utils.h` 136–151）。`singleCoreM=baseM=16`、`singleCoreN=baseN=64`，满足部分和的单输出块约束。A/B 每次重绑重设游标。`scheduler_norm.h` 93–121 在 MoveNext 成功后仅为 partial=false 分配 CO1，其余使用现有 C；`cube_out_buffer_no_unit_flag.h` 38–76 在 dbL0C=1 保留同一 cMatrix。SDK 的 `sync` 模板参数不能充当 W 的“读完”票据。[CANN 9.0 Iterate](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0638.html)
- 每个 Ik 只给一个 K256 切片。本 normal 路径在 kOuterIdx=0 设置 `sL0CInit=!enPartialSum`，无 Bias、IsNeedUB 分支传 `cmatrixSource=false` 和该初始化位。m200 `MmadCal` 实际传给 `mad`。K0 初始化，K1…K9 在相同 FP32 CO1 上继续。见 `scheduler_norm_base.h` 180–187、331–365，`mmad_compute.h` 32–65 和 `kernel_operator_mm_impl.h` 368–392。
- **相邻 MMAD 的完成边另有来源。** `mmad_compute.h` 65–75 在 `(mmadM/16)*(mmadN/16)<10` 时发 `PIPE_M`。本投影取 1×4=4，每次实际选择该 SDK 分支。其后的 M 管线指令受该栅栏约束。L0A/B 的 M_MTE1 ping/pong 控制输入槽位再次装载，不能用来替代相邻 K 对同一 L0C 的依赖。不同形状、尾块或未保留该 M 边的编译结果须重新审查。
- 后续 Iterate(true) 不重新分配或清零 CO1，源码只有零到九的升序。F9 后才有一次 GetTensorC，`scheduler_base.h` 396–416 发出 CopyOut，再调用 CO1 的 FreeTensor。FN 保证跨 N 复用前已发的输出操作退休。FP32→FP16 见 `kernel_operator_data_copy_impl.h` 522–539。失败的 N 不发布部分和。以上是所选 SDK 路径的证明；候选实际 ELF 的 M 栅栏、目标地址与分支仍须有机器证据。
