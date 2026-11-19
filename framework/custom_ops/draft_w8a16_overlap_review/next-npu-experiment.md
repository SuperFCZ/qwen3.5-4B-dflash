# 下一项实验：C16 第 0 层 Gate 的双 W16 排程对照

**选择。** 第一项拟议的设备实验为一个真实 Gate/Up，M16/K2560/N19456，使用已验收 ATC 的 **8 block**。比较两版同样使用双 W 的实现，一版在 F(k) 后生成 W(k+1)，另一版将相同的反量化提前到 F(k) 前。其余计算、搬运、事件及布局相同。旧 Raw 单 W16 是第三个独立参照，用于检验净收益与数值。当前只具备源码设计和 CPU 合同证据。正式 Host、候选二进制和设备证据尚未生成。**暂不启动设备执行。**

优先选择 Gate 的依据是可测机会：两档五个 Gate 在 Raw 正式插桩窗口约 9.64 ms，五个 Down 约 4.71 ms。此前只改变 Gate 原始输入预取已经在实际完整 OM 上观察到无插桩改善。一次 Gate 含 304 个 N tile，每个有 9 个可移动的边界，共 2736 处。这个实验直接检验 Vector 与 Cube 之间是否还有可利用的窗口。Down 是第二项独立候选。Head 没有真实读字节证据，继续[只读及最小 Trace 研究](head-bandwidth-review.md)。这些依据支持**实验优先级**，不能据此预测某个版本更快。按当前整图中位数，达到 20 ms 尚差 C16 10.569870 ms、C64 11.426295 ms。达到 13.33 ms 尚差 17.239870 / 18.096295 ms。差值采用无插桩基线，未减去任何分类 Task Duration。

## 冻结输入和身份

从 `/home/w00949577/qwen35-dflash-run/gate-raw-atc8-Np20yu/source/report.json` 的 `binding.cases["16"]["layer-0-gate_up"]` 解引用。逐份校验绝对路径、bytes、SHA256。通过其中的历史 Source 进一步绑定五层 checkpoint、A-only AIR、W8A16 离线 NZ 和 GN Group128。历史 OM/Runner/私有 OPP 只读，从同根的 export、smoke、benchmark 清单取实际产物。不能拿 C64 输入替换 C16。硬件信息取 host sidecar/ATC 图与专用 ELF，接口名或包里通用 .o 的文件名不足以证明选择了该实现。

| Layer 0 / C16 | 字节 | 服务器冻结 SHA256 |
|---|---:|---|
| x.bin，FP16 [16,2560] | 81920 | `44fed068f079c18e95ef303c0d02cf5401470db7277ec17d9d024fbf29079256` |
| w_nz.bin，INT8 NZ [80,1216,16,32] | 49807360 | `c898ef7b555f6e3bcf3b183ec7742b012f0a4e40ab4a46cd34184b77218b5a2c` |
| s_gn.bin，FP16 [20,19456] | 778240 | `1365af20b6f09734c6d02f2f3d3045fb20e4af1f53d414c9b4dd9011c7ced333` |
| Native Golden，FP16 [16,19456] | 622592 | `353839a7a41bdd44930e91f72ae750adc2f7f15e14f2b679d5f0799e1b5da76c` |

## 预注册的顺序与停止条件

1. **先看源码与实际代码生成。** 后续仅在独立私有工程里生成“双 W 串行”和“双 W 提前”两版，独立 OPP、独立静态单投影 OM、独立进程。将已安装服务器 SDK 文件与[审计表](w16-event-lifecycle-revised.md)逐字节核对。保存预处理/编译选项和 ATC 专用 ELF 的图名、长度、节表及 SHA256，检查调用确实内联到同核 MatmulImpl、无 client/service 或延迟 B 读。检查 B 的 UB→L1、每次 W view 偏移、下一代 D 在指令流中的位置、末 K 没有下一次 D、原全流水 fence 和两个 raw Wait 均在。比较两版专用反汇编。若编译器把 D 放回 fence 后或已在 Iterate 返回前完全串行化，报告没有可测试的重叠，停止计时。若证据不能判定消费者/地址安全，则 **BLOCKED_W16**，不发起 Smoke。
2. **正式 Host/ATC 资源。** 两个对照均用 loop、batched、broadcast、两 raw bank、两 W bank、core_limit=0。要求每个静态 Gate 回调与图中实际 blockDim=8，304 个 tile 由 b+8j 唯一覆盖。核对两版完整 200 B Cube 与冻结预算记录的 SHA256、UB 109312/152832 B、Brcb 顶部保留 8192 B、L1/L0A/L0B/L0C、算子系统 Workspace 2097152 B。另从各自 OM Runner 保留真实整模型 work/weight 分配并检查漂移。算子 workspace 不能当作整个 OM 的内存请求。两版的完整父 Tiling 相同。相对旧 Raw 仅允许已解释的资源字段差异。验证 TPipe/Scratch 的真实范围及 SDK 临时区，不能用计算出来的表冒充实测。静态地址不可证时先设计独立诊断，能证明范围后才允许算术试跑。禁止用运行时 ACLNN 7-block 的结果替代。
3. **未计时 Smoke。** 两版各两次独立 Poison 执行，复用现有 Runner 的 pre-launch DSO socket 与冻结身份门禁。核验真实 OM 执行且无 CPU fallback、Stream 同步、Workspace 一致、只读输入/权重/scale、已有 GM/Workspace Guard、Poison 和完整输出。UB 按已证明的实际范围验证，不声称存在额外硬件 canary。每次对 **311296 个 FP16 位型**逐一比较 Native Golden、已验证 Raw 输出、两个变体及 Repeat。保存实际二进制/哈希、首个差异、差异总数、ULP、非有限值及 Guard 变化。普通 Iterate 失败应保留未写 Poison。任一失败或缺失证据立即停止。
4. **单层无插桩配对。** 正确性过后，保持同一 Device 和运行环境，输入、权重常驻，独立进程，先将双 W 串行/提前配对。三轮顺序串行→提前、提前→串行、串行→提前，每轮每版 5 次 warmup + 30 次连续 Execute+Sync。保存每版全部 90 个样本、每轮和总体 Median/P95/Min/Max、各轮倍率。timed loop 仅含原模型 Execute+Sync，保留前/后 Guard、Readonly、Poison、Repeat、timed tail/postcheck。不在窗口内做文件 IO、H2D/D2H、读回比较或 msprof。双 W 控制把下一代 D 放在当前 F 后，raw 预热/受理时刻也要与提前版相同。该控制可能改变旧单 W 的预取收益。若该配对有改善，再按同协议用冻结单 W Raw 进程新采样，与双 W 提前做净收益配对。历史单点 median 只作背景，不能直接放进新样本。
5. **何时扩大。** 建议预先采用 1.03× 作为值得继续的信号：单层两组配对需三轮 median 同向、总体中位倍率 ≥1.03 且无总体 P95 回退。未达标即记录 NO_GO，不通过修改阈值或去掉 fence 继续。过单层后才扩至 C16/C64 各五层冻结投影，各层各档逐位正确、分别测两种顺序，不能将层间独立 median 相加作整图时间。沿现有五层门槛，两个顺序各层均改善、各层无 P95 regression、每档各顺序独立中位数之和倍率 ≥1.03。同时核验相对当前单 W Raw 的净改善。
6. **何时进入完整 Draft。** 五层和两档全部合格后才考虑同一 A-only AIR、独立私有 OPP/OM。每图 26 个 GroupQuant、五个 Gate 以外图结构及 21 个投影专用指令/资源不漂移，Head、权重、Scale 和 ABI 保持身份。对全部 15 个 token 与 10 个完整 KV buffer（含未更新区）做 bitwise、Guard、Poison、Readonly、Repeat，再进行同设备无插桩整图配对。是否接近 20 ms 或 13.33 ms 只能以该阶段测量回答。

**若 W16 不能满足安全条件，第一项替代设备实验为 C16 Layer0 Down 的二维 raw DMA。** 从 Scheduler Flow `a-core-scheduler-flow-F6ia0q/source/report.json → binding.cases["16/layer-0-down"]` 和当前 Raw full-export 绑定 [16,9728,2560]、原 x/w_nz/s_gn、Native Golden、真实 ATC blockDim=8。在单 W、当前 scratch、原 V_S/PIPE_ALL 下仅改 CopyRawTile 的两组传输描述符。八个 K32 平面和两个 scale 组变成两次等价传输，跨度、长度与全部原始字节的证明见[Scalar/Vector 审计](group-quant-scalar-vector-review.md)。执行相同的独立 Source/Host/Smoke/配对/扩层门槛，Down 每次完整比较 40960 个 FP16。该实验不与 W16 调度叠加。
