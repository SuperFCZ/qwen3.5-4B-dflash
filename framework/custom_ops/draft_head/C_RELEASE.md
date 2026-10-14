# C 算子正式配置冻结与验收

## 状态边界

用户已确认 Ascend310P3 / CANN 9.0.0 的历史实测：
`/home/w00949577/qwen35-dflash-run/head-top1-full-20261013-115730`。

- Vector Primitive：2191/2191 PASS。
- Scalar 和 Vector 完整 Head：各 41/41 PASS。
- M15：Native Top1 ≈23.26 ms，Custom Scalar ≈46.5 ms，Custom Vector ≈17.1 ms。
- Vector 相对 Scalar ≈2.72×、相对 Native ≈1.36×；M1/M8 也快于 Native。

这些是用户回传的真实设备结果，本地未访问原始服务器文件；近似数值不充当新验收样本，
未提供的历史 p95 不补造。验收脚本会读取历史 `vector-full/suite.json` 的实际样本。

本轮只冻结默认配置和构建/验收流程，算法源文件与 **18c59c5** 完全一致。
`release_contract.json` 固定 Host Tiling、Kernel、Better、转换和公开 ABI 文件 SHA256；
可选配置头单独记录，允许选择 Scalar 回退。A5.1、生产 Draft、Native golden 未修改。

**本轮新正式构建的 CANN 编译、安装、41-case、median/p95：NOT_RUN。**
历史 PASS 不代表新双算子包已通过；后续由新目录 release.json/acceptance.md 留证。
A/C 图接入、D/E/F、完整 Draft/Decode 性能均未开展。

## 正式默认与回退

正式默认无需实验环境变量：

| 配置 | 默认 |
|---|---|
| FP16 转换 | compare_select |
| Top1 | Vector，原始 FP16 NaN 分类 + 非 NaN max/EQ/最小 ID |
| 核数限制 | 0，按设备自动选择 |
| Profiling observer | none |
| Conversion diagnostics | 关闭 |
| Top1 Primitive Probe | 关闭 |
| OPP 注册算子 | 仅 DFlashDraftLmHeadTop1 / DFlashDraftLmHeadTop1Audit |

`HEAD_TOP1=scalar` 选择已验证的 Scalar 回退；其余数学和接口相同。
默认运行和 Scalar 回退均只构建两个算子。`HEAD_TOP1_PROBE=1` 才额外构建/执行 Primitive；
`HEAD_DIAGNOSTICS=1` 才额外构建转换 Probe。实验脚本 `test/top1_experiment.py` 已显式启用
需要的 Probe，保留 `--primitive-only`、配对 benchmark 和 profiling-only observer。
Vector 算法选择本身不再隐式引入 Probe。

build-config.json 记录 `build_kind`、精确 `operators`、top1_probe/diagnostics/profile。
正式 OPP preflight 除原路径安全、非空 .o/JSON 配对与解析之外，还拒绝额外 kernel/Probe，
检查恰好两个 Head ACLNN 头文件，记录头文件、动态库、全部 kernel/JSON 大小与 SHA256。
正式验收还要求实际配置是默认 Vector、compare_select、自动核数、无 observer/probe。

## 冻结接口（后续集成使用）

两算子输入相同，均无 attr/bias，不截断词表：

| 输入/输出 | Fused | Audit |
|---|---|---|
| hidden | FP16 ND [M,2560]，M=1..15 | 同左 |
| weight | FP16 ND [248320,2560]，NK，完整词表 | 同左 |
| token_id | INT64 ND [M] | 同左 |
| logits | 无 | FP16 ND [M,248320] |

典型生产 M=15。输入只读；只比较最终已舍入的 FP16 logits，NaN 优先、多 NaN/相等值
取最小全局 token ID，±0 等价。FP32 完整 K 累加、compare_select、最终一次 FP16 舍入
不变；Better、N64、Matmul tiling 与跨核 SyncAll 不变。

公共调用：

- `aclnnDFlashDraftLmHeadTop1GetWorkspaceSize(hidden, weight, token_id, &bytes, &executor)`
- `aclnnDFlashDraftLmHeadTop1(workspace, bytes, executor, stream)`
- Audit 对应 `aclnnDFlashDraftLmHeadTop1Audit*`，GetWorkspaceSize 额外传 logits tensor。

调用方按 ACLNN 返回的总 workspace 大小分配；不要只按自定义 partial 大小分配。
内部用户 workspace 仍为 `192×P` 字节（P 为实际参与核数），系统 workspace 由 SDK 返回。
默认流/内存生命周期遵循现有 ACLNN runner，不引入 CPU fallback。此文档只固定独立算子
接口，尚未实现 TorchAir/GE 或生产 Draft 的接入。

## 服务器正式一键验收

在已有 CANN 9.0.0/NPU 环境、仓库根目录执行。复用既有 capture 与 30 个 Native OM，
不重新 capture/export；新建独立 build/OPP/evidence，不覆盖旧结果：

```bash
git pull --ff-only
"${MODEL_PYTHON:-python3}" framework/custom_ops/draft_head/test/release_acceptance.py \
  --capture /home/w00949577/qwen35-dflash-run/draft-head-UCt9vR8q/capture/manifest.json \
  --native-manifest /home/w00949577/qwen35-dflash-run/head-native-gpJaP6vN/models/native.json \
  --baseline-dir /home/w00949577/qwen35-dflash-run/head-top1-full-20261013-115730 \
  --output-dir "/home/w00949577/qwen35-dflash-run/head-release-$(date +%Y%m%d-%H%M%S)" \
  --device-id "${DEVICE_ID:-0}"
```

该入口清除继承的实验开关，以**代码的真实默认值**构建，不通过强行设置 Vector 来掩盖
默认配置错误。流程：

1. 校验冻结源码、现有 capture/Native OM、历史 Vector 完整 41-case 及 comparison hashes。
2. 新建正式双算子 OPP，保留原双次 msopgen、CMake 审计、OPP/ACLNN headers 检查。
3. 使用新 runner 跑完整 41-case；不能仅导入历史 PASS，不能只测六个性能样例。
4. 严格保留完整 Audit FP16 logits bitwise、INT64 ID、NaN/Inf/Overflow/Tie、Guard、
   输入只读、repeat、row permutation、timed-tail/postcheck。任何失败都终止正式验收。
5. 提取 C16/C64 × M1/M8/M15 的 Custom、Native Head、Native Top1 median/p95 及全部样本，
   与历史 vector-full 的同输入样本比较；核对输入/golden SHA256、设备 ID、tiling、workspace。
6. 输出 release.json 和 acceptance.md，包括新源码提交、配置、安装产物 hashes 和证据路径。

协议固定 **continuous-v1，warmup=5，repetitions=30**，无 msprof 插桩。median/p95 从原始
samples 重算。默认性能复核阈值为相对旧 Vector **10%**：任一 median/p95 超出阈值、
当前 Custom median 慢于同轮 Native，或 Native median 相对历史明显漂移，都会标记
`REVIEW_REQUIRED`（exit 2），要求检查机器负载/频率和重复测量。阈值可由
`--max-regression-ratio` 显式设置，它仅是性能复核标记，绝不改变数值验收标准。

只有新 41-case PASS 且无性能复核项时，`new_release_acceptance=PASS`。
正确性 PASS 而性能需要复核时，两者分开记录，不能称正式验收完成。
后续在这里回传结果后才能确认新版本最终冻结，不自动开展 A/C 集成。

## 需要回传的证据

- 新目录 `release.json`、`acceptance.md`（运行成功或性能需复核时生成）。
- `data/suite.json`、`server.log`；失败 case 的 comparison.json/runner.log。
- `build-state.json`、`build.log`；build-state 的 run_root 下 server.log、build-config.json、
  opp-preflight.json、source-commit.txt 和生成 CMake 快照。
- release.json 包含历史 vector-full 路径/SHA、冻结源码清单、.o/.json/头文件/动态库 hashes。

这些产物可用于之后 A/C 图集成确认实际加载的算子版本；本轮只准备接口与证据，不接图。

## 本地验证边界

CPU 回归覆盖默认 Vector、Scalar 回退、独立 Probe 配置、源码冻结、双算子安装清单，
以及禁止历史 PASS 代替新验收、拒绝输入/tiling/golden 不一致、median/p95 回退提示。
还以假 msopgen/cmake 执行真实脚本的配置派发，默认及 Scalar 回退均只生成两个算子，
在 build 处主动退出；该测试不是 CANN 编译。原算法、NaN/位型、Guard 与 runner 回归继续跑。
本地没有 NPU；正式包的设备验收保持 NOT_RUN，直至服务器提供上述新产物。
