# Gate 双 W16 设计：资源可行，执行 BLOCKED

**对象**：W8A16，不是 W8A8。GM ND FP16 A 乘由 INT8 NZ × 原 FP16 Group128 Scale 在 UB 中得到的 FP16 B。Gate M16/K2560/N19456，304 个 N64 tile，ATC 8 block，每个 block 38 个 tile；每个 N tile 完整按升序执行 10 次 K256，每次内含两组独立反量化。不要将消费者“Cube”简化为一次直接从 UB 读取 W16 的 MMAD。

**本地实测**：用 CANN 9.0.0 ARM64 的 PlatformAscendCManager::GetInstance("Ascend310P3") 获得本地已安装的 SoC 平台配置。用真正的 MatmulApiTiling，复制已验收 Host 的类型（A GM ND FP16、B VECOUT ND FP16/transposed、C GM ND FP16、无 Bias）、SetShape(16,64,256)、SetOrgShape(16,19456,2560,256)、SetFixSplit(16,64,256) 和 SetBufferSpace(-1,-1,scratch)。未调用 ACL，没有 stream、device 或计算 Kernel。它不是实际 Formal TilingContext。

| 每核预算 / 查询 | Raw 单 W16 | 候选双 W16 |
|---|---:|---:|
| INT8 raw 两个 bank / GN scale 两个 bank | 32768 / 512 B | 相同 |
| W16 总大小 | 32768 B | 65536 B |
| Broadcast 临时区 / Brcb 顶部保留区 | 2304 / 8192 B | 相同 |
| user UB（含 reserve） | 76544 B | 109312 B |
| Matmul UB | 185600 B | 152832 B |
| SDK GetTiling rc | 0 | 0 |
| base M/N/K；depth A1/B1；DB L0A/B/C | 16/64/256；1/1；2/2/1 | 全部相同 |
| transLength / 200 B Cube SHA | 34816 / 745b846a… | 逐字节相同 |
| L1 / L0A / L0B / L0C 使用量 | 40960 / 16384 / 65536 / 4096 B | 相同 |

B 的 L0B tile 是 64×256×2 B，double-buffer 恰好 65536 B；额外的是 UB 中的 W bank，不是 L0B 空间。已核验 UB=262144 B、L0B=65536 B。本地 Cube 用历史 Off 外层四维字段与 202240 的前缀封装，其 SHA 也重现历史 ee2d5b1e…。记录和逻辑地址区间见 w16-ub-budget.json。逻辑布局按每个 bank 32 B 对齐，绝不借用顶端 reserve。实际 TPipe 地址、隐式资源和峰值仍须在正式 Host/设备验证。原 SDK Matmul buffer 仍跨整个 Matmul 对象生命周期，不可另作 staging。

**时序设计而非实现**：先填 raw0 → 反量化 W0 → 允许库读 W0；概念稳态为 Cube 消费第 k 块已入 L1/L0 的数据、Vector 生成 W[(k+1)%2]，同时 MTE2 在安全的另一个 raw bank 填后续数据。对同一 k+1 的 *同一批字节*，MTE2 必须先完成再 Vector 读取，并非三者可以无条件同时启动。尾块没有越界预取；N 边界必须 drain。两份 W 不代表两份 Matmul 状态机；不能提前重绑 A/B、复用局部 workspace 或隔着未完成的 K 迭代开始下一 N tile。

**结论**：本地资源检查没有要求退回 K128，也没有理由改变 depth/DB 或扩大 L0B。按设计可保留 0→9 的 FP32 L0C partial-sum 顺序和最后一次 GetTensorC 的 FP16 舍入；这不是新二进制的数学验收。未找到能闭合调用者 bank 复用的释放合同；不实现双 W16、不删除 PIPE_ALL。后续如要推进，必须以同一真实 ATC 回调核对自身 50 个 Cube 字段、全 UB/L1/L0/Workspace 与 8-block，不能沿用本地 PASS 当作设备验收。
