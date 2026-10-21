# 同有效权重 Gate：FP16 Direct / W8 Off / W8 Raw

本实验独立于历史 FP16 checkpoint、六层/五层完整 Draft 的比较。唯一输入来自已验收
C16 `layer-0-gate_up`：M16/K2560/N19456，signed INT8 offline-NZ，Group128 GN scale。
不修改原 Kernel/Host Tiling、OPP、Runner、Golden、Source Lock 或生产模型。
**当前本地只完成 CPU/构建准备，CANN 编译、NPU 数值/计时、msprof 和 FP16 Simulator
均 NOT_RUN。**

## 公平对照与局限

- FP16 使用 CANN 原生 `aclnnMm`，`cubeMathType=0`，两输入和输出均 FP16。
  权重为设备常驻、连续 ND `[K,N]`，X 为同一 `[16,2560]`。接口语义参考
  [CANN aclnnMm](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/API/aolapi/context/ops-nn/aclnnMm.md)；
  该公开页面不能代替 receiver CANN 9.0.0 验证。入口先审查实际 SDK version/header，
  只链接该 CANN 的 opapi/nnopbase/ascendcl；编译或执行不支持就明确 BLOCKED。
- W8 直接复用已验收的 Loop/K256/Broadcast Off/Raw 原 OPP 和原 C++ Runner；独立
  子进程选择对应 vendor，关闭 2D 搬运，不重编 W8、不改 N64/K256/Raw bank/同步。
- 三组均在计时前完成文件读取、设备分配与上传。使用运行时权重，避免将 FP16
  constant-OM 优化与 W8 runtime-weight 混为同一种绑定。FP16 ND 与 W8 NZ 的
  内部布局处理、原生 tiling、累加实现可能不同；`aclnnMm` 的耗时包括其内部转换。
  本轮没有“相同 tiling 的 FP16 自定义 Kernel”，也不声称测得纯反量化开销。
- `cubeMathType=0` 保持输入 dtype，但不据此声称原生与自定义的归约顺序相同。
  原生 Runner 检查真实 SoC 为 Ascend310P3。

## 权重准备与数值门槛

从冻结 `w_nz.bin` 按 `[K/32,N/16,16,32]` 逐 K32 逆映射，和冻结 `q_nk.bin`
的 signed INT8 逐字节比较。然后逐 group 计算：

`W_eff[n,k] = RN16(INT8(q[n,k]) * FP16(scale[k//128,n]))`

FP16 multiply 同时用独立 FP64 product → FP16 nearest-even 验证（INT8×FP16 有效位
足够在 FP32 精确表示后再舍入）。拒绝非有限 scale/weight，不偷偷限幅、替换特殊值。
记录 subnormal/±0 数量，不假定设备的 denormal 行为。生成 NK 和 KN 两份 FP16 文件，
逐位验证转置关系、布局、大小、SHA256；所有生成工作在计时前完成。

- 三组先各运行两次**无计时**执行，检查 Guard、输入只读和重复输出。
- W8 Off/Raw 必须与同 Capture 的冻结 Native W8 Golden bitwise 一致，bits=0、ULP=0，
  任一失败不得进入 timing。沿用既有 Host launch、Copy、Raw Prefetch 资源校验。
- FP16 对 Golden 报告 bits、ULP、max_abs、RMSE、p99_abs 和首次差异。有限但不一致
  标记 `FP16_DIFFERENCE_REQUIRES_REVIEW`，允许采集诊断时延，但**不声称数值验收 PASS**，
  不把所有差异归因于归约顺序。非有限、Guard/只读/repeat 失败均阻断。
- 数学操作数身份与实际 bitwise 等价分别记录。小范围 full-K FP64 CPU probe 仅为
  诊断，绝不替代 Native Golden 或设备执行。
- 每次正式运行都检查两个前置输出、timed-tail、postcheck；FP16 输出也必须跨进程/
  交叉顺序/profile 与其无计时结果 bitwise 一致。

## 计时与 profiling

自动执行三组的**全部六种排列**；每组各位置出现两次，每次均 warmup=5、reps=30。
每组总共 180 个原始 samples；保存每轮 median/p95、原始顺序样本和汇总 median/p95。
计时采用现有 `continuous-v1`：工作区查询单独记录，正式比较 execute+stream sync，
计时循环内不读回全部输入/输出。模型加载、权重生成/上传、只读校验不算进推理时间。

所有正式无 profiling 时延完成后，分别单独运行三次 msprof PipeUtilization。
原样保存 Task Duration 与 MTE2/VECTOR/SCALAR/CUBE 等列、任务名和原始 CSV 哈希；
显式 `Task Duration(us)` 才生成按文件/任务名的设备统计。可能包含 warmup、precheck、
postcheck 或内部转换任务，不能与 30 个正式 samples 混用，也不将不同 Pipe/任务时间
相加作为 Kernel 总时间。缺失的计数器保持 UNRESOLVED。

存储 payload 从 INT8+Scale 到 FP16 的字节差异只是容量事实，不是测得的 DDR 读取量
或有效带宽；真实带宽收益、纯反量化成本、Cube/同步的独立因果贡献保持 UNRESOLVED。
`comparison.json/md` 中的时延比只描述这些特定实现，不代表完整 Draft 收益。

## Simulator 与已有指令证据

可用 `--w8-bounded-report` 只读复用已验收 Off/Raw 单 Tile 指令产物，并校验源 Kernel、
X/NZ/scale/Golden 身份与本实验完全一致。保存 Trace、指令 CSV、源码热点与二进制的
路径和 SHA256，不重跑 Simulator。其统计仅覆盖一个 N64 / 完整 K2560，不冒充完整
Gate 指令数。

原生 `aclnnMm` 的内部 N 调度没有已验证的单 Tile 限制接口。改成 N64 的小形状会改变
原生 tiling/全局 stride，不能当作同一个完整 Gate Kernel 的 bounded 执行。因此本轮
**FP16 bounded Simulator=NOT_RUN**，不把 W8 专属 Simulator Runner 当作 FP16 支持。
若后续需要纯反量化指令开销，须另建并标注“同 tiling 的自定义 FP16 对照”，先验收
其数学、事件及内存行为；本轮不擅自增加第二个 Kernel 变量。

## 服务器入口

默认复用 A5.1 `a51-fallback-20261012-063342/suite.json` 和 Raw reverse
`gate-raw-prefetch-reverse-20261020-025853/suite.json`，由记录解析真实 Off/Raw 构建。
不猜随机 OPP 路径。所有输出必须在新目录。

```bash
CANN_ROOT=/usr/local/Ascend/cann-9.0.0 bash framework/custom_ops/draft_quant/fp16_compare/run.sh \
  --w8-bounded-report /home/w00949577/qwen35-dflash-run/gate-simulator-bounded-2g-racefix-20261020-103256/bounded-report.json \
  --output-dir "/home/w00949577/qwen35-dflash-run/gate-same-weight-fp16-w8-$(date +%Y%m%d-%H%M%S)"
```

`--prepare-only` 只做旧证据校验和 CPU 权重生成，不构建/执行 NPU。
请回传 `comparison.json/md`、`weights/weights.json`；如失败，附 phase 对应日志。
NPU Runner 和 CANN API 的兼容性必须由服务器编译/执行确认，CPU 声明桩语法检查
只排除 C++ 语法错误，不能记为 CANN 编译 PASS。
