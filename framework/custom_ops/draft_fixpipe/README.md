# FixPipe Group128：独立源码与数值可行性审查

当前是证据审查工具，**不是新的 Kernel**。只读使用实际 CANN SDK，输出新报告目录；
不编译、不执行 NPU、不运行 Simulator、不测试性能、不改 W8 Kernel/Host/OPP、Golden 或 SOURCE_LOCK。
不改变 `draft_antiquant/` 原有 K128 候选及其审查门槛。

本地 SDK 和设备不可用，所以 `AntiQuant Input Path` 与 `FixPipe Output Path` 的实际兼容性均
`UNRESOLVED`。所有设备项目为 `NOT_RUN`。已知时延仅作为用户提供的后续目标：
W8 Off 2.542 ms、W8 Raw 2.332 ms、Custom FP16 1.449 ms。

## 两条路径不能合并成一个支持结论

| 路径 | 要解决的事情 | 当前结论 |
| --- | --- | --- |
| AntiQuant Input Path | 在 Cube 之前把 signed INT8 权重与当前组 scale 转为符合原边界的 FP16；确认跨组参数刷新和 FP32 partial sum | UNRESOLVED；即使已有 K128 源码审查，也不证明 20 组正确 |
| FixPipe Output Path | 产生未缩放的 K128 部分和后按 N64 scale 缩放，以 FP32 保留并跨 20 组累加 | 硬件兼容性 UNRESOLVED；一般情况下与 RN16 权重基线不 bitwise 等价 |

FixPipe 的输出缩放能力本身，不能解决 FP16 A 和原始 INT8 B 是否能直接进入 Cube 的问题。
实际实现若仍先转换 B（即使 scale=1），必须记录内部转换，不宣称消除了反量化。
若要求把 A 量化为 INT8，则是 **W8A8 的另一种数值方案**，不是现有 W8A16，不接入现有 Golden 验收。
将 INT8 B 显式 Cast 成未缩放 FP16 也仍有 Cast/搬运成本，且输出缩放改变了舍入位置。

## 各前端分别检查

工具不会 import SDK 的 Python 文件，只按文本读取，避免执行 TIK 注册或设备初始化。
每份源文件保留实际路径、SHA256、完整副本、行号和条件编译上下文。
`frontend_hints` 仅辅助搜索，不是支持判断；找不到接口也不自动推断芯片不支持。

1. **Ascend C 显式 Fixpipe**：核查实际 dav_2002/dav_m200 的 overload、模板实例化、参数结构、
   下层 intrinsic 和芯片条件；不能因为存在 `FixpipeParamsV220` 就认为 310P 支持该接口。
2. **TIK fixpipe**：核查 Python API 产品检查、dtype/格式参数检查及 lowering；
   TIK 支持不能作为 Ascend C 显式 API 的支持证据。
3. **Matmul SDK 内部随路转换**：从 `GetTensorC`、`SetQuantVector`、`SetDequantType`、
   `enAtomic` 等进入 CO1 搬出实现，核查真实源/目标类型与缩放模式；
   不与 `SetAntiQuantVector` 的输入路径混淆。

三种前端各自有十项 `UNRESOLVED` 检查：

- **芯片、版本、混合输入**：实际 CANN 9.0.0 分支，FP16/INT8 Cube 或内部转换路径。
- **dtype/buffer**：FP32 L0C 输入、scaled FP32 输出是否成立；CO1/L0C、FixPipe Buffer、A1、UB、GM
  的实际位置、分配、容量与生命周期。不因支持 FP32→FP16 就推断支持缩放后 FP32→FP32。
- **格式和 stride**：源 NZ、目标 ND/NZ、N16 对齐、M16×N64、stride 单位和上限，是否要求额外排布转换。
- **scale**：N64 向量支持还是仅 scalar；GM/A1/UB/FixPipe Buffer 存储位置、长度、地址对齐、
  量化参数 uint64/float 编码、位域、舍入与有效尾数。没有本机证据时不编写编码器。
- **逐组更新**：scale 更新与参数缓存、事件依赖，下一组能否安全覆盖参数。
- **FP32 跨组累加**：缩放后的部分和是否保持 FP32；新一组使用 L0C 时，旧的缩放累加器放在哪里；
  是否存在片上路径、额外 Vector 加法、GM 临时区或 Atomic 的顺序/确定性限制。
- **精度**：FP16 输出只发生一次；同时核查 overflow/subnormal/NaN/±0，不改变 Golden。

公开文档只是查找线索：
[TIK fixpipe 旧版说明](https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/apiref/aicpuopapi/atlastikapi_07_0093.html)
曾提示量化参数有效位型可能与普通 FP32 不同；
[Ascend C Fixpipe 文档](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0251.html)
按型号/重载区别能力。这些都不能代替本机正式 CANN 9.0.0 源码。
旧文档提到修改 golden 的做法**不适用于本实验**：任何差异都必须保留并报告。
FP16 scale 转 FP32 本身是精确的，但量化参数编码及实际乘法精度仍需单独证明。

## CPU 已能确认的舍入风险

当前基线为 `w16=RN16(q*s)`，然后 FP32 完整 K 累加，最后 RN16 输出。
理想化输出缩放为 `p_g=FP32_sum(X*q)`，`z_g=FP32(p_g*s_g)`，再 FP32 累加各组后 RN16。
两者实数代数可分配，但舍入位置、归约分组不同。

`fixpipe_math.py` 使用标准库二进制 FP16/FP32 舍入，给出三组可重复的**CPU 算术反例**：

| 风险 | 基线输出 bits | 替代输出 bits |
| --- | --- | --- |
| X=1.5、q=3、s=1.0009765625；把权重 RN16 延后到部分和缩放 | 0x4482 | 0x4481 |
| 部分和 65505 在每组输出时先转 FP16，再与另一组 −65504 累加 | 0x3c00 | 0x0000 |
| 即使 scale=1、始终 FP32，按 group 聚合改变加法顺序 | 0x3400 | 0x0000 |

这些稀疏条目可填入完整 K128 group，其余位置及其余 group 为零；测试验证填充到 20 组后反例不变。
第一例只有一个非零乘积，不依赖复杂的归约树。
第二、三例是明示顺序的 CPU 模型，不能据此声称真实 Cube 使用相同归约树。
某些输入会恰好一致；真实 Frozen Capture 上是否一致仍 `UNRESOLVED`。
不得为取得通过而把 1 ULP 当容差，或调整 Scale/Golden。

## 条件流量账本，不是性能预测

固定 M16/K2560/N19456，20 组、304 个 N64 tile；每个 M16×N64 FP32 部分和 4096 B。

| 假设调度 | 额外逻辑 GM 流量（整个 Gate/Up） |
| --- | ---: |
| 每组缩放结果写 GM 一次、读一次；跨组累加器留片上 | 写 24,903,680 B + 读 24,903,680 B = **47.5 MiB** |
| 上述部分和落 GM，同时累加器也落 GM：首组复制初始化，其余读改写，最后读出转 FP16 | **95 MiB** |
| 能将 scaled FP32 部分和直接返回片上，且另有长期存活的 FP32 累加器 | 部分和额外 GM 可为 0；**此路径硬件可行性未确认** |

这些额外数字不包含共同的最终 FP16 写出 622,592 B、X/权重读、scale 重载、padding、cache/coherence。
第一种额外读写量已等于一次完整 INT8 权重的逻辑字节数；不能再视为“免费输出缩放”。
另有 5,914,624 次逻辑 FP32 累加加法；它们不是测得的指令数、launch 数或 cycles。
逐 N tile 流式临时部分和可为 4096 B，片上累加器另需 4096 B；若保存 20 组 slab，则每 tile 81920 B，
保存整张 Gate 的 slab 为 24,903,680 B。

如果参数是 4/8 字节编码向量，全 Gate 20 组的参数容量分别为 1,556,480 / 3,112,960 B，
而原 FP16 GN 是 778,240 B；这里只列情景，不断言实际编码及重复读取量。
额外访存、累加与同步**有可能抵消**减少 Vector 反量化的收益；没有合法路径及独立设备测量前，
不把字节除以假定带宽、也不从已有 2.542/2.332/1.449 ms 直接推断收益。

## 服务器命令：只读审查

不需要加载模型、Capture 或 OPP。采集三类输出路径，并复用原 AntiQuant 工具采集输入路径证据；
仅可能运行工具 `--version`/`nm` 取身份，不进行编译或设备调用：

```bash
git pull --ff-only && \
python3 -B framework/custom_ops/draft_fixpipe/audit.py \
  --cann-root /usr/local/Ascend/cann-9.0.0 \
  --output-dir /home/w00949577/qwen35-dflash-run/fixpipe-group128-audit-$(date +%Y%m%d-%H%M%S)
```

已有 AntiQuant 报告时可加 `--antiquant-audit /实际目录/sdk-capability.json`，工具会重新校验 SDK、
快照、工具、库和版本；有已完成的源码审查可再加 `--antiquant-review /实际目录/review.json`。
即使有效，也最多标 `SOURCE_REVIEWED_K128_ONLY`，不继承任何 Group128×20 或 FixPipe 支持结论。
如果旧审查发现缺少 AntiQuant API，FixPipe 仍会独立采集；文件/版本/身份有误则审查失败。

回传 `group128-capability.json`、`group128-capability.md` 和相关 `fixpipe-sources/`；
输入路径新采集的证据在 `input-antiquant/`。源码审查模板 `fixpipe-review-template.json` 绑定完成的
报告 SHA256，需逐项记录具体源码行引用；它不是 PoC 编译授权文件，工具不会自动把模板升级为 SUPPORTED。

本地：`python3 -B -m pytest -q framework/custom_ops/draft_fixpipe/test`。
无需 NumPy、torch、torch_npu 或 CANN Python 包；测试只构造临时 SDK 文本和 CPU 算术情景。

## SDK 扫描范围与未完成状态

文件路径筛选发生在 resolve/stat/open 之前。保留全部 `.h/.hpp/.cpp/.cc/.py/.ini` 及版本文件的
源码扫描；JSON 只接纳明确的 FixPipe/Matmul/量化算子路径、算子注册表（如 `ops-info`、
`op_info_cfg`）和平台/产品支持配置。没有上述相关标识的普通 `fusion_strategy`、L2 Cache
策略及其他 JSON 不读取、不 stat、不计算哈希，因此无关大文件不会触发源码大小限制。
有明确相关算子标识的配置即使位于策略目录仍会检查，避免仅凭目录排除关键限制。

**16 MiB 限制保持不变**。相关候选超限则保留路径、大小、以有限缓冲区流式计算的 SHA256，
标记 `SOURCE_SIZE_LIMIT_EXCEEDED`，不解析或复制超限内容、不继续声明完整扫描。
正常源码的快照字节必须与生成摘要/行号时读取的字节及其 SHA256 一致。
权限错误、路径逃逸、文件变动等仍会阻断；失败前已采集的源码快照及身份不会从报告中丢失。

新增 `fixpipe-scan.json`，同时内嵌到 `group128-capability.json.scan`：

- 枚举、接纳、读取、快照和跳过的文件数量。
- 每个跳过原因的精确计数及最多 20 条路径示例；示例截断有单独标记。
- 超限/失败文件、扫描阶段、错误类型及未完成原因。
- `COMPLETE_WITHIN_DECLARED_SCOPE` 只表示声明范围扫描完成；异常为 `INCOMPLETE`，总报告仍为
  `BLOCKED_AUDIT`，不会生成可被误认为完整证据的审查模板。

先扫描 FixPipe 可保留其失败细节；最终成功仍必须经过原有 AntiQuant SDK 版本和身份校验。
本工具未证明任何芯片支持结论，CANN 编译、NPU、Simulator、性能仍全部 `NOT_RUN`。
