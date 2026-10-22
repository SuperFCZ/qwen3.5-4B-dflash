# Ascend310P3 / CANN 9.0.0 Matmul AntiQuant：独立可行性实验

当前交付为 **SDK 证据采集器 + 受源码审查约束的 K128 候选**。本地没有服务器 SDK/NPU；
实际 SDK 兼容性 `UNRESOLVED`，CANN 编译、安装、K128 数值、性能均 `NOT_RUN`。
不因 Python/CPU 回归通过、头文件存在或导出符号存在而宣布设备支持。
原 W8 Off/Raw、Custom FP16、OPP、Golden、SOURCE_LOCK 和生产 Draft 均不改动。

已有设备数据由用户提供：Native FP16 0.782460 ms、W8 Off 2.541605 ms、
W8 Raw 2.332320 ms、Custom FP16 Tiled 1.449375 ms。约 1.092 ms 的差额是数据准备路径
及其访存/同步的净差异，不是纯反量化时间。本实验还没有时延数据。

## 先确认实际 SDK，不依据公开文档自动通过

`aq_sdk.py` 只读扫描选定 CANN 根目录的头文件、实现、版本和平台配置，保存：

- `SetAntiQuantVector`、AntiQuant 实现、Matmul 类型、partial-sum 的源码位置、行号、上下文与完整源文件副本。
- 原文件和副本的实际路径、字节数、SHA256；保留条件编译分支，不把其他芯片分支当作 310P 支持。
- 版本文本、`Ascend310P3.ini`（若有）、编译工具路径/版本、实验源码 SHA256。
- V3/Nz 相关源码/算子配置、`libopapi`/`libnnopbase` 身份及 `nm` 符号发现。符号存在不代表支持 FP16/INT8/Group128。

结果默认为 `SDK_REVIEW_REQUIRED`。没有接口为 `BLOCKED_MISSING_API`，扫描/版本错误为
`BLOCKED_AUDIT`。任何实际源码证实关键组合不支持，源码审查填 `UNSUPPORTED` 后，
构建阶段输出 `BLOCKED_UNSUPPORTED`，不会生成工程、安装或运行。未确定的组合同样不能构建。
不会自动替换布局、切换芯片宏或转到 V3/Nz 路线。

需逐项核查并在独立 `review.json` 引用实际 SDK 的完整原始行：

| 项目 | 本候选要求，不是已确认能力 |
| --- | --- |
| 芯片分支 | Ascend310P3 / dav_2002，对应实现可能标为 dav_m200；核查实际宏与平台配置 |
| Matmul 模板 | A=GM/ND/half，B=GM/NZ/int8_t/transpose=true，C=GM/ND/half |
| NZ 地址 | `((k/32 * (N/16) + n/16)*16 + n%16)*32 + k%32`；K128/N64 为 `[4,4,16,32]` |
| 累加 | 混合输入特化确实使用 FP32 L0C；不能仅凭 FP16 输出或 BiasType 推断 |
| offset/scale | `LocalTensor<half>`，VECIN、各 64 元素/128 B、32 B 对齐；必须确认实现不要求更长向量/不同位置 |
| 生命周期 | GM→UB 后 `PIPE_ALL`；参数存储在 `IterateAll`、末尾 fence 和 `End` 之前不改动/复用 |
| Host tiling | GM/NZ/DT_INT8 转置参数与 Kernel 一致；singleCore/base M/N/K 严格 16/64/128 |
| SDK UB 分配 | 两份用户参数合计 256 B，SDK 临时区 UB−256−8192，其他内部分配必须能装入剩余 8192 B |

另外单独记录以下调查，不用 K128 成功代替它们：

- 每 K128 刷新 scale 的时机、缓存/指针持有方式、`Iterate(enPartialSum)` 的 FP32 生命周期、
  内部 K 分块与舍入边界。当前 K256/K2560 **未实现、NOT_RUN**。
- `aclnnWeightQuantBatchMatmulV3/Nz` 的本机实际产品分支、dtype/layout、group size 支持。
  只找到声明/二进制时保持 `UNRESOLVED`，不另写设备路径。
- 如果 GM INT8 NZ 不被此 API 接受，应说明所需 ND/转置/其他 NZ 排列及读写字节数；
  本候选直接阻断，不暗中做另一种布局。提取紧凑 K128 切片不是全矩阵格式替换。

审查文件由 `review-template.json` 另存生成，绑定 `sdk-capability.json` 的 SHA256。
`claims` 的每项必须有 `SUPPORTED/UNSUPPORTED`、具体 finding，以及原文件 path/SHA256、
start/end 行号、逐字 quote；未决填 `UNRESOLVED`。全部必要项确证支持后才写
`decision=K128_SOURCE_REVIEWED` 和 reviewer。不能仅机械地把模板改成 SUPPORTED。
构建及设备阶段重新校验源码、快照、工具和库的哈希，审查报告不替代真实编译及数值验证。

公开 API 文档仅用于提出待检验组合：
[SetAntiQuantVector](https://asc.gitcode.com/api/SIMD-API/adv_api/cube_compute/Matmul_Kernel/SetAntiQuantVector.html)
描述 LocalTensor offset/scale 和 GM→L1 转换。
[CANN 9.0 beta V3 文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/API/aolapi/context/ops-nn/aclnnWeightQuantBatchMatmulV3.md)
包含产品/layout/group 限制，但不是本机正式 9.0.0 安装内容的兼容性证据。

## K128 PoC 的边界与数值验收

独立算子 `DFlashAntiQuantK128`，一个核，M16/K128/N64，**只有一组 scale**。
ABI 为 `x FP16[16,128]`、`w_nz INT8[4,4,16,32]`（逻辑 `[64,128]`）、
`scale FP16[1,64]`、全 +0 `offset FP16[1,64]`、`y FP16[16,64]`。

手写 Kernel 只有参数 DataCopy、fence、Matmul 设置/调用，不含 UB Cast/Brcb/Mul/Muls。
源代码检查会拒绝引入这些调用，但**内部 AntiQuant 仍执行反量化**，不能据此宣称 SDK 内部没有
Vector 指令；尚未获取指令证据。Host 记录实际 tiling 和 L1/L0A/L0B/L0C 容量，严格检查 baseK，
不放宽不兼容计划。UB 用户参数 256 B，SDK 临时区为 UB−256−8192（32 B 对齐），8192 B 保留。
内部分配/量化存储需求仍须由 SDK 实现和真实编译确认。

沿用已验收 A5.1/Off/Raw 的只读 evidence 身份验证，核对 Capture、NZ、Scale、Native OM、
完整 Golden、原始源码和 OPP。只提取 layer-0-gate_up 的 `X[:,0:128]`、`q[0:64,0:128]`、
`scale[0,0:64]`。新 NZ 切片的轴序不变，N stride 从全矩阵变为紧凑 N64；逐位逆映射验证。
CPU 先计算 RN16(signed q×FP16 scale)，独立用 FP64 乘积→FP16 交叉验证乘法舍入；
再保存 FP32 顺序累加参考及 FP64 求和参考，不假定其归约顺序与 Cube 相同。

Native FP16 `aclnnMm` 使用同一 X 和预先生成的 FP16 `[128,64]` 权重。
它与 Custom 分别链接、分别运行，Native 子进程不加载新 Custom OPP。
两者各执行两次，使用不同输出 poison，全缓冲 Guard、完整输入只读、重复稳定性均严格检查。
Custom 与 Native、两份 CPU 参考逐项记录 bits、max ULP、max_abs、首个差异。
任何不一致为 `FAIL_NUMERICAL`，即使只有 1 ULP；不得自动归因为归约顺序并放行。

完整 K2560 的 Frozen Golden 不是 K128 partial sum 的 Golden，不能截取其列冒充。
它只用于原始身份绑定。将来 K2560 才能直接要求与完整 Golden 对应列 bitwise 相等。
K128 全部相等最多标记 `K128_DIAGNOSTIC_BITWISE_PASS`，不包含 K256/K2560、性能、完整 Gate 或 Draft。
本轮入口没有 warmup/reps、msprof、Simulator 或完整 Gate 参数。

## 服务器分阶段命令

每次使用新的独立目录，不在旧 evidence 内创建新文件。先只执行 SDK 审查：

```bash
git pull --ff-only && \
CANN_ROOT=/usr/local/Ascend/cann-9.0.0 \
bash framework/custom_ops/draft_antiquant/run.sh audit \
  --output-dir /home/w00949577/qwen35-dflash-run/antiquant-sdk-$(date +%Y%m%d-%H%M%S)
```

先回传 `sdk-capability.json`、`sdk-capability.md` 与 `sources/` 中相关实现。
**在审查完成前不要运行下一条。** 本地无法读取服务器 SDK，因此未提供预填的支持结论。
源码确认后，把另存的、带引用的 `review.json` 放在该 audit 目录；随后分别编译和验收：

```bash
AUDIT=/home/w00949577/qwen35-dflash-run/antiquant-sdk-实际时间
BUILD=/home/w00949577/qwen35-dflash-run/antiquant-build-$(date +%Y%m%d-%H%M%S)
CANN_ROOT=/usr/local/Ascend/cann-9.0.0 \
bash framework/custom_ops/draft_antiquant/run.sh build \
  --audit "$AUDIT/sdk-capability.json" --review "$AUDIT/review.json" \
  --output-dir "$BUILD"
```

仅当 `BUILD/report.json` 为 `K128_BUILT_NPU_NOT_RUN` 后：

```bash
CANN_ROOT=/usr/local/Ascend/cann-9.0.0 \
bash framework/custom_ops/draft_antiquant/run.sh k128 \
  --audit "$AUDIT/sdk-capability.json" --review "$AUDIT/review.json" \
  --build-report "$BUILD/report.json" --device-id 0 \
  --output-dir /home/w00949577/qwen35-dflash-run/antiquant-k128-$(date +%Y%m%d-%H%M%S)
```

默认复用 `a51-fallback-20261012-063342/suite.json` 和
`gate-raw-prefetch-reverse-20261020-025853/suite.json`；可通过 `--a51-evidence`、
`--prefetch-evidence` 显式指定其他已验收报告，身份检查不变。
构建失败回传 `report.json`、`build.log`、`runner-build.log`（存在时）；设备阶段回传
`report.json`、`numerical.json`、`data/slice.json`、`native/custom` 下 `execution.json` 和
`runner.log`。中断时 RUNNING/FAILED 不能视为 PASS。

## 本地回归

```bash
python -B -m pytest -q framework/custom_ops/draft_antiquant/test
bash -n framework/custom_ops/draft_antiquant/run.sh
```

CPU 回归覆盖源码证据绑定、未决/不支持阻断、篡改拒绝、NZ 切片、group 边界、signed INT8、
FP16 舍入/subnormal、offset 零位型、严格差异报告及两个 Runner 的 C++ 语法检查。
声明 stub 只做主机 C++ 语法检查，**不是 CANN API ABI、AscendC 编译或 NPU 数值证明**。

本次本地执行：新实验 17 项及现有同有效权重/同 Tiling 对照 18 项，共 35 项 CPU 回归通过；
`bash -n` 通过。实际服务器 SDK 审查、CANN 编译、安装与 NPU 均 `NOT_RUN`。
