# W16 消费者与事件审计

**源码事实（本地 CANN 9，dav_2002）**。以下 SDK 路径均相对于 `/usr/local/Ascend/cann-9.0.0/aarch64-linux/asc/`。哈希标识读取的文件，不宣称服务器所有库和本地二进制一致。

| 位置 | 可支持的结论 | SHA256 |
|---|---|---|
| include/adv_api/matmul/matmul_intf.h:135–145 | 默认非 split 路径 Matmul 使用 CFG_NORM / MatmulImpl。不能套用其它架构的异步服务模型。 | 32992a9d7a343f1deed1407e6a696e95b154299c3a339ea90378bba63035eece |
| impl/adv_api/detail/matmul/matmul_impl_base.h:546,641 | Local SetTensorB 转交 CopyCubeInB.SetInput；Iterate<sync>(enPartialSum) 转交 ScheduleOnce(enPartialSum)，该函数体没有返回 B-bank 完成句柄。调用 Iterate(false) 的 false 是首次 partial-sum 参数，不是异步开关。 | 923527715a9baa2147538221ccf82afc14b5fd4f913b42363adbf00eb2de5006 |
| impl/adv_api/detail/matmul/stage/copy_cube_in/base/copy_cube_in_base.h:44–48；impl/adv_api/detail/matmul/param/matmul_tensor_info.h:60 | 保存 LocalTensor 地址/转置状态并 reset 内部 B buffer；没有在 SetTensorB 内复制整个 W16 或转移其所有权。 | ffc0476ab498f82527c432c17f2ce097ad395207c54a723d8b8c540ba7bdcb5e；c43e590a7ce336b79a3b50bcc76bc0b597e953df8210f7ffb4db3b0e9107c1a5 |
| impl/adv_api/detail/matmul/scheduler/base/scheduler_norm.h:93–161；impl/adv_api/detail/matmul/stage/copy_cube_in/base/copy_cube_in_using_ub.h:77–120 | Normal scheduler 载入 A1/B1、执行运算，内部 buffer 的 Hit/Alloc、EnQue/DeQue、ClearLoadData 是库内协议。L1 tensor 可消费与用户的 UB bank 可覆盖不是同一证据。 | bb565c78b53644f63cdf89cef18f4d622f11f4fca755ce03b989347f5f5083cd；4f57a5faf40eca08996956691ffe91be90f90ab4500707bb28ce43b85dce1b8e |
| impl/adv_api/detail/matmul/stage/copy_cube_in/copy_tile_to_cube/copy_tile_to_cube_using_ub.h:40–55,140–165；同目录 data_copy_wrapper_using_ub_nd.h:137–175,253–280 | B 是 FP16 local ND，不是非转置 INT8 特例。非 enVecND2NZ 路径以保存的 local 地址进入 CopyND2NZOnTheFly；对齐分支有 strided DataCopy。末尾 MTE2→MTE1 的显式配对受 GlobalTensor 条件控制，不能据此为 local W 签出释放证明。 | 8fed124b939a1a2477622e1414362a95222c37dbfb24ad895494cacb58b33726；6d5315d222c1f4357f45adbc00ad76380ec0d7d6d3bcffd570ff6cd9867c18d6 |
| impl/basic_api/kernel_tpipe_impl.h:443–484 | Alloc 按 EventToIndex 找可用位并置位；Fetch 找空位但不置位；Release 清占用位，并不会等待 DMA。不要持有 Fetch ID 跨库调用。 | 7a91fe65d16c3775fa57d15ebb63a37b275c0c9ee1c269bca4d76355e2873c8d |

**已验收应用边界**：draft_quant/op_kernel/d_flash_group_quant_linear.cpp（SHA a4861bb5f7e38aceced9e4d89fc6c2519cb4a80e0914d7dd418c50389e8f1a4f）126–156 为 MTE2_S、MTE2_V 各 Alloc 一个 ID；先 Wait 当前，再对另一 bank 发下一块并 Set；末块不再发出。186–220 的 Cast、Brcb、Mul 和 PIPE_V 保持原数值边界。239–247 的 V_S 立即 Set/Wait 交接 W16；249–281 以 SetTensorA/B、Iterate(kBegin!=0)、消费者 PIPE_ALL 保留同一 FP32 归约，最后 GetTensorC；283–287 Release、End。Iterate 返回 false 时 259–271 先 Wait 已发出的 lookahead，再 barrier、Release 和 End 后退出。不能把这条已验收的单 W 异常路径移植成还有第二个 Vector 写入挂起时的 drain。

**未闭合的边**：对 W0/W1 每个世代需要独立证明 (a) raw DMA 完成→Vector 首读，(b) Vector 最后写→库首读，(c) 所有读取该 W 的 UB→L1 搬运完成→两轮后 Vector 覆写，(d) SDK 自身 A/B 转换的 Vector/UB scratch、L1/L0 队列与下一次 rebind 不冲突，(e) L0C 仍按 K 顺序累加。独立 Cube、Vector、MTE 管线使 overlap 有工程可能，但库也可能在 A 侧使用 Vector/搬运资源；不能从 busy counter 或调用已返回证明实际 overlap。已审查接口不交付一个区分 W bank/世代的 B-consumed event；内部 Fetch + 成对 Set/Wait 不等于公开的完成票据。不能猜测 MTE3_V 或 M_V 的任一 ID 即可代替消费者 PIPE_ALL。

**决策 BLOCKED**：首先需要 SDK 维护者的可引用契约或精确到该实例化分支的最后读/硬件队列语义，以及普通失败时每项已发出操作的排空边界。事件必须先成功分配，未完成 Wait 前不得 Release；超时/设备错误不发布成功结果。尚不批准删同步、两次并行 Iterate 或启动新 Kernel。

| 拟议阶段 | 可能在途 | 必须保持有效，直至何时 |
|---|---|---|
| 填充/启动 | raw0 DMA、W0 的 Vector 写 | MTE2_S/V 等到 raw0 就绪；W0 就绪才交给 Matmul |
| 稳态（仅设计） | Cube(k)、Vector→W(k+1)、MTE2→raw(k+2) | 复用 raw(k%2) 前须完成 Vector 对 raw(k) 的最后读；复用 W(k%2) 前须确认 SDK 最后读完 W(k)，而不是只见到下一次 SetTensorB |
| 最后一个 K、N 边界 | 未完成的内部 copy、L0C 更新、GetTensorC/GM 输出 | 不发越界 k+1/k+2；按原累加顺序完成最后 K，再输出 FP16，buffer/事件不得提前释放 |
| Iterate 返回 false | 原 lookahead，加上假设方案中的 W 写和 SDK 消费 | 原有失败分支只支持原设计；不能证明所有在途消费者排空即停止，不用 return/End 伪造完成 |
