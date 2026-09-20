# 开发与验收清单

本目录尚无目标 kernel 和设备通过证据。以下是交付工作，不是已完成清单。
复用原 native OM 作为回退，先运行其数值与稳定性对照，再测试候选。

## 1. 冻结输入与设备

保留 source/config/checkpoint/权重/输入/OM SHA256，精度参数、布局、CANN/ATC/torch_npu、
驱动/固件、具体 SoC、设备 ID 和功耗状态。W4、W8、FP16 分别记录，不能共用不同 checkpoint
的 golden。记录实际 M 档位、热/冷缓存条件以及每次是否加载模型、搬运输入或重置状态。

`hardware_reference_policy=use`；当前工作区仅有 simulation profile，未找到匹配接收端
硬件/时钟/计时方式的能力画像。能力/工作账本/规划结果哈希及理论延迟均为空。
本阶段仅编写需求；实现算子之前补匹配设备的能力资源，按 OPERATORS 的物理工作账本核算
Cube、Vector、MTE、FIX 和依赖链。不得套用其他芯片的 L0/UB 容量或吞吐，也不据空缺画像
宣称优化已达极限。CAModel 仅诊断，最终速度以设备为准。

## 2. 单算子数值

- 对 workloads.json 的所有投影、M 档位分别测 W4/W8；tiny 同时覆盖 M16 和 M64。
- 合成 case 覆盖 W4 的全部 256 个 byte、W8 的全部 256 个 signed code，N/K/group 边界
  `15/16/17`、`31/32/33`、`127/128/129` 的索引位置（不代表首版要支持这些尺寸）。
- scale 按 n、g 不同而变化，覆盖正负 X、随机非二进制整倍数的 FP16、稀疏/全零/极值 X、
  非均匀 group、不同 seed，不能只用统一 scale 或 dyadic 值掩盖分组/舍入错误。
- 定点哨兵区分每个行/列/物理 block；源输入、前后 guard 和其他 owner 输出必须保持正确。
  padding 不参与有效输出；若允许一般 tail，分别覆盖 M/N/K tail、最小/最大尺寸。
- 浮点基线先同输入重复执行，确认自身是否逐位稳定。若 native 有漂移，要保留漂移分布，
  不把某一次随机输出当作唯一可复现基线；此时严格无损替换验收尚未就绪，禁止自动改为宽松门槛。
- A/B/C 按 OPERATORS 的零差异门槛分别比较；输出 dtype/shape/layout 同样是验收项。
- C 覆盖最大值在首/末词表列、跨 tile 相等最大值、FP16 舍入后才出现的 tie、负 logits、
  大小相近的第一/第二名、±0、Inf/NaN 的原路径行为；记录 token 和 margin。

离线 converter 要支持逐字节逆变换、零填充/哈希检查、错误 layout/version/shape/文件长度
拒绝，以及截断文件、不同 checkpoint、错误端序输入拒绝。生成器应声明小端存储。

## 3. ATC 与动态接口

必须分别通过静态小图、动态 M16/64 小图、实际 gate/up/down/fc 尺寸和完整 Draft。
特殊布局 metadata 与 payload 分开核验：logical shape、origin、physical storage shape、dtype、
format、字节数和 producer/consumer 一致。不允许仅靠关闭 shape 检查让模型生成成功。

r37 的已观察状态：预打包 static16 PASS、runtime-dynamic PASS、预打包 dynamic FAIL。
失败节点为新增 NZ→ND TransData，不能跳过动态图门槛。
现有探针只测编译，**PASS 不含 OM 执行、数值或时延**。

ATC 后检查最终执行图，而不是仅检查 AIR：预期消失的权重转换/解包是否真的消失，是否增加
NZ→ND→NZ、CPU 算子、隐含稠密权重、跨档位复制或额外 workspace。

## 4. 实际消费链与模型

A 的单层输出必须实际喂给下一层，不能每层都用 golden 输入掩盖误差累积。
用同一条 Target 特征/anchor/cursor/cache 轨迹分别运行 native Draft 和候选 Draft，比较
候选 ID、全部更新 KV 及下一轮输出，再接回真实 Verify/commit 调度。

覆盖短输入和 1K 长输入、prompt 63/64/65 边界、C16/64 切换、proposal_count 1/3/7/15、
接受数 0/1/K-1/K、EOS、拒绝后的续写、重复请求和 current/next 状态切换。
state 检查包含有效区、被遮挡但实际写入的物理尾区、未写入区域哨兵，不能仅比较可见 token。
仅建缓存图还需逐层证明 context KV 无 block hidden 依赖，并测试最终一块回到正常 Draft。

保持两个独立结论：

1. 优化相对同量化 native Draft 是否保持行为和速度。
2. DFlash 相对 ordinary greedy 是否满足完整模型一致性。

目前用户结果是 `PASS_WITH_DIFFERENCES`；第一项通过仍不能把第二项写成 PASS，任务质量需
独立评测。正式模型性能推广要求先通过冻结的模型精度门槛；此前的速度可作为诊断保留。

## 5. 性能分层测量

固定设备、输入和流，先做 3 次预热，再对 baseline/candidate **各保留 10 次**测量，
报告全部时间、median/p95/CV、输出哈希、峰值设备内存。量化版与同量化基线成对测量；
FP16 仅作为另外的模型图参照。若 10 次波动无法分离差异，记录不确定，不能挑最好一条。

| 测量项 | 必须包含 | 单独列出/排除 |
|---|---|---|
| 算子/融合区域 | 完整所需解包、scale、layout、GEMM/归约、同步 | 离线转换、加载、H2D/D2H 另列；不能漏掉候选额外转换 |
| Draft OM | 整张图、格式转换、小算子、流同步 | 模型加载/重置/文本处理另列；C16、C64 分开 |
| Decode 循环 | Draft + Verify/commit + 调度，累计所有轮 | Prefill 另列；记录接受/提出、token、EOS、轮数 |
| 模型生成 | Target/DFlash Prefill + 完整 Decode 循环 | 对应现有 benchmark 的 speedup 口径 |
| 完整请求墙钟 | 按声明服务边界包括分词、传输、重置、文本解码等 | 冷启动加载与热服务分别报告 |

算子统计之和不等于完整请求时延；图阶段表与算子表有重叠，禁止再相加。
比较不同 EOS 输出长度时，同时给原始 token 数、模型时间比和吞吐比。
msprof 会影响测量，主速度分布在无 profiler 下测，profiler 用于确认热点和执行路径。

## 6. 交付目录和报告

实际开发完成后，每个算子子目录至少提交：

- 锁定接口/物理载体版本、host/tiling、kernel、GE/torch 注册、build/run/compare 入口。
- CPU 索引/数学 oracle、匹配 native OM golden 的生成方式、独立与链式 fixtures。
- source/工具链/输入/二进制哈希清单、三档编译证据（静态、动态、实际形状）、10 次设备数值及性能。
- 局部内存与事件生命周期账本、物理工作账本、设备 portrait 身份和 CAModel 热点证据。
- 图替换位置、回退分发、显存增长、尚未支持形状/特殊值、模型 token/EOS/状态门槛结果。

报告中将“host 通过”“ATC 通过”“OM 数值通过”“更快”“整网通过”分别列项，
以已验证事实为准；不把本目录需求文档或编译 PASS 当成算子交付。
