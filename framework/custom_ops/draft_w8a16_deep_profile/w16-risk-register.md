# W16 风险和解锁条件

| 风险 | 当前证据 | 下一门槛；失败处理 |
|---|---|---|
| W0 被下一 Vector 写覆盖，库仍在 UB→L1 读取 | 只有写就绪交接，没有已证明的读完信号 | 建立 bank/世代与最后一个消费者的事件合同；否则保留 PIPE_ALL、禁止设备实验 |
| raw 与 W 的 slot 周期不同，展开的 Scale scratch 被重用 | 两套数据量不同；原 Brcb/Mul/PIPE_V 不能删除 | 逐地址/生产者/消费者 CPU 排程与首/中/末 tile、跨 N 验证 |
| Fetch ID 与 SDK 内部事件竞争或 Release 太早 | Alloc 保留、Fetch 不保留 | 逐时刻证明事件池不耗尽、无 outstanding 信号；失败路径先 drain 后释放 |
| 误以为有两份 Matmul/L0B | L0B 已满；两份 UB W 才是本设计 | 保留单 Matmul、SDK scratch、原 depth/DB、固定 K256；任何实际资源漂移 BLOCKED |
| named platform ≠ formal TilingContext | 本地 SDK 200 B 相同；不是设备证明 | 后续单算子正式 Host 比较完整 parent/cube 和所有容量，单独证明物理地址/对齐 |
| 异步影响 FP32 L0C 或 FP16 舍入 | 有序设计，不是新候选数值证明 | 每层真实 A 的完整 FP16 bitwise、Poison、Readonly、Repeat、Guard 及失败中止 |
| 额外 UB、同步或 Vector 资源竞争抵消收益 | Raw Pipe 归因尚待采集 | 正确性通过才做同协议新配对，最终只看完整 Draft 未插桩结果 |
