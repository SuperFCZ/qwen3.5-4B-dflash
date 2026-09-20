# 算子验收要求

精度按 [算子需求](OPERATORS.md)，形状按 [workloads.json](workloads.json)。
W4/W8 分别以同 checkpoint 的原生 OM 为基线，FP16 Draft 另作性能对比。

## 数值与接口

- 固定源码、checkpoint、权重、输入、OM 哈希，以及 SoC、CANN/ATC、torch_npu、驱动、固件和编译参数。
- 覆盖所有投影和 M 档位；包含真实权重与捕获激活，基线先重复执行确认稳定性。
- W4 覆盖全部 256 个 byte，W8 覆盖全部 signed code；scale 随 N 和 group 变化，包含随机、稀疏、零值和极值输入。
- 检查行列、group、tile 边界、padding、输入只读和输出哨兵；若支持 tail，覆盖 M/N/K 尾部。
- Top-1 覆盖首末词表列、跨 tile tie、FP16 舍入后 tie、接近的前两名及 Inf/NaN 行为。
- 离线载体声明端序和版本，验证逆变换、填充、哈希；拒绝错误 shape、布局、长度和 checkpoint。

## 编译与集成

分别验证静态小图、动态 M16/64 小图、实际投影尺寸和完整 Draft。
检查 logical/origin/storage shape、dtype、format、字节数及上下游一致性。
编译后执行 OM 并比较数值；检查最终图中解包/格式转换是否按设计消除，是否增加 CPU 算子或额外 workspace。

同一 Target 特征、anchor、cursor 和 cache 轨迹下，比较原生与候选 Draft 的候选 ID、全部 KV 写入及下一轮输出。
单层输出实际传给下一层，覆盖短/1K 输入、63/64/65 边界、C16/64 切换、候选数 1/3/7/15、
接受数 0/1/K-1/K、EOS、拒绝续写、重复请求及 current/next 状态切换。
缓存检查包含有效区、物理尾区和未写入区；仅建缓存图需验证回到正常 Draft 后的输出。
同量化 Draft 等价与 ordinary greedy 一致性分别验收。

## 性能

固定设备、输入、流及缓存条件，baseline/candidate 各预热 3 次、测量 10 次，
保留原始时间、median/p95/CV、输出哈希与峰值设备内存。正式时延在无 profiler 下测量。

| 测量项 | 范围 |
|---|---|
| 算子 | 所需解包、scale、布局、GEMM/归约及同步；离线转换和加载另列 |
| Draft OM | 完整图及同步，C16/C64 分开 |
| Decode 循环 | Draft + Verify/commit + 调度，记录 token、EOS、轮数和接受/提出数 |
| 模型生成 | Prefill + Decode，对应 benchmark 加速比；两模式 token 数分别记录 |
| 完整请求 | 按服务边界记录分词、传输、重置、文本解码等，区分冷启动与热服务 |

阶段表和算子表有重叠，不相加。分别报告算子、Draft 和模型生成的收益。

## 交付

提供接口/布局版本、host/tiling、kernel、GE/torch 注册、隔离构建及 run/compare 入口；
附 native golden、数值与链式测试、静态/动态编译结果、设备性能、内存用量及回退条件。
编译、OM 数值、性能和完整模型一致性分别列出结论。
