# 算子验收要求

精度按 [算子需求](OPERATORS.md)，形状按 [workloads.json](workloads.json)。
W4/W8 分别以同 checkpoint 的原生 OM 为基线，FP16 Draft 另作性能对比。
当前 W8 性能基线采用 `gdr-lengths-mjhazgip` 的离线 NZ 包；旧 C64 profile 仅用于排序，
性能验收要在同一设备对当前 baseline/candidate 重新配对测量。

## 数值与接口

- 固定源码、checkpoint、权重、输入、OM 哈希，以及 SoC、CANN/ATC、torch_npu、驱动、固件和编译参数。
- 覆盖所有投影和 M 档位；包含真实权重与捕获激活，基线先重复执行确认稳定性。
- W4 覆盖全部 256 个 byte，W8 覆盖全部 signed code；scale 随 N 和 group 变化，包含随机、稀疏、零值和极值输入。
- 检查行列、group、tile 边界、padding、输入只读和输出哨兵；若支持 tail，覆盖 M/N/K 尾部。
- Top-1 覆盖首末词表列、跨 tile tie、FP16 舍入后 tie、接近的前两名及 Inf/NaN 行为。
- 离线载体声明端序和版本，验证逆变换、填充、哈希；拒绝错误 shape、布局、长度和 checkpoint。
- D/E 用 A 的实际输出做链式对照，覆盖 SiLU 正负饱和、±0、down 与 residual 抵消、FP16 舍入分界；不能只喂理想随机中间量。
- F 覆盖因果、双向、滑窗边界、非连续 mask、全 mask 行、不同 KV head 的哨兵；验证 GQA head 映射、两段 stride 和 PV 前 FP16 概率。
- G 比较全部五层 cache，再接正式 Draft；检查不能产生或提交临时 MASK/anchor KV。

## 编译与集成

分别验证静态小图、实际投影尺寸，以及当前静态 C16/C64 完整 Draft。
G 另测 context K/V 的 M64；仅当交付声明支持动态档位时才增加动态编译验证。
检查 logical/origin/storage shape、dtype、format、字节数及上下游一致性。
编译后执行 OM 并比较数值；检查最终图中解包/格式转换是否按设计消除，是否增加 CPU 算子或额外 workspace。
接入位置及生产输出应与 OPERATORS 中一致：A 替 WeightQuant，D/E 替对应组合子图，
C 仅替 Draft 的 head/top1，F 替分段 attention，G 由 Prefill 调度选择。
输入 readonly、current/next 不别名；纯函数接口不能通过隐式原地写入规避输出复制。

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
| Decode 加速比 | 普通 Decode 总耗时 / DFlash Decode 总耗时，不计 Prefill；两模式 token 数分别记录 |
| 完整请求 | 按服务边界记录分词、传输、重置、文本解码等，区分冷启动与热服务 |

阶段表和算子表有重叠，不相加。G 的收益仅计 Prefill；D/E 与 A 的 GEMM 收益不能重复计数。
逐项启用算子测量：A → A+D → A+D+E，C/F 独立对照，再测最终组合，保留较快且通过精度的方案。
每份性能报告同时记录动态有效行、物理行、调用数、接受/提出数；改变候选输出或接受轨迹时，
不得将次数减少解释为同一计算任务的 kernel 加速。

记录每个 tile 的 M/N/K、权重重读次数、GM 字节数、片上 buffer 生命周期和峰值占用。
硬件 roofline 只能使用与接收端 SoC/运行配置匹配的带宽、算力及容量证据；缺失时标记未建立，
不根据 Block Num 或跨芯片参数给出理论上限。

## 交付

提供接口/布局版本、host/tiling、kernel、GE/torch 注册、隔离构建及 run/compare 入口；
附 native golden、数值与链式测试、声明支持的编译档位结果、设备性能、内存用量及回退条件。
编译、OM 数值、性能和完整模型一致性分别列出结论。
初次交付以 A 的 gate/up、down 为一个可独立验收的范围；未满足数值门槛不自动放宽误差。
