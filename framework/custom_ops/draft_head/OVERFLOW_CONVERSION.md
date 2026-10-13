# Head overflow：设备诊断与 opt-in Vector 限幅候选

## 实测状态（与本轮待验收项分开）

Ascend310P3 / CANN 9.0.0，用户回传的最新 evidence：
`/home/w00949577/z50058744/qwen3.5-4B-dflash/framework/custom_ops/draft_head/.runs/head-apVtckzt`。
服务器原始文件未在本地读取。候选 `a4df66b` 使用 FP32 GetTensorC + CAST_NONE，仍为
**40/41 PASS**：30 个 C16/C64 真实 case 全通过；NaN/Inf、signed zero、rounding 等
10 个 synthetic 通过；唯一失败 `synthetic-overflow_inf`。所有 Top1 ID 一致。

`hidden[:,0]=2; weight[63/64,0]=65504`：数学结果 131008，Native FP16 为 `0x7bff`，
Custom Audit 为 `0x7c00`，30 个元素不同，max_ulp=1。这个差异继续严格 FAIL。
此前直接 FP16 GetTensorC 的 `.runs/head-5UuIaB5v` 也出现相同差异。
**旧文档基于默认饱和描述作出的 Cast 修复推断已被设备结果否定。** 当前尚不知道
GetTensorC FP32 是否已经为 Inf，必须读取原始 FP32 位型，不能用 CPU 数学推定。

本轮诊断算子 CANN 编译/安装、Vector 语义、候选敏感用例/41-case、M15 性能均 **NOT_RUN**。
原 Native-only 41/41 PASS 保留，不单独重跑。A5.1、生产 Draft、Native OM/golden 均未修改。

## 固定 SDK 核查与候选边界

核查官方 `cann/asc-devkit` tag `v9.0.0`，commit
`c8c1b9df6bb9cd61c09751c7ef414b89e06af274`：

- [GetTensorC](https://gitcode.com/cann/asc-devkit/blob/v9.0.0/impl/adv_api/detail/matmul/matmul_impl_base.h)
  的 LocalTensor 接口没有独立 saturation 参数。
- [dav_m200 DataCopy](https://gitcode.com/cann/asc-devkit/blob/v9.0.0/impl/basic_api/dav_m200/kernel_operator_data_copy_impl.h)
  的 FP32 L0C→FP32 UB 使用 CRMODE_NONE；FP32→half 使用 CRMODE_F32toF16_NONE。
- [dav_m200 Cast](https://gitcode.com/cann/asc-devkit/blob/v9.0.0/impl/basic_api/dav_m200/kernel_operator_vec_vconv_impl.h)
  的 FP32→half CAST_NONE 使用 vconv_f322f16；显式 CAST_RINT 在此路径不支持。
  [CANN Cast 文档](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0073.html)
  的通用饱和措辞不能代替当前 310P3 的实测。
- [dav_m200 Mins/Maxs](https://gitcode.com/cann/asc-devkit/blob/v9.0.0/impl/basic_api/dav_m200/kernel_operator_vec_binary_scalar_impl.h)
  支持 float，分别使用 vmins/vmaxs；count 接口支持批量操作。候选逐 tile 原地执行
  `Mins(+65504) → PIPE_V → Maxs(-65504) → PIPE_V → CAST_NONE`。
- [dav_m200 Compare/Select](https://gitcode.com/cann/asc-devkit/blob/v9.0.0/impl/basic_api/dav_m200/kernel_operator_vec_cmpsel_impl.h)
  有 FP32 比较及 FP32 Select 路径。若 Mins/Maxs 实测破坏 NaN/zero，应考虑用 GT/LT
  mask 精确选择限幅值、其它位置保留原值。它需要额外 mask/临时 UB 和严格预算，且比较
  NaN 的设备行为也必须验证；本轮不在缺少设备证据时同时更换第二套转换策略。

默认 `HEAD_CONVERSION=cast` 保留原路径。只有显式 `HEAD_CONVERSION=minmax` 才启用候选，
且服务器入口强制要求 `HEAD_DIAGNOSTICS=1`。没有 FP16 Inf/NaN bit patch，没有新全词表
Scalar 修补。FP32 完整 K 归约、唯一 FP16 边界、Better()、跨核同步和 GM workspace 不变。
候选复用现有 FP32 tile，**不增加 Head UB 或 workspace**；p=7 的 user UB 88448 B、
Matmul UB（256 KiB 总量）173696 B、user GM workspace 1344 B 均保持原值。
Host launch / build-config / 比较器分别绑定 cast 与 minmax，避免把对照误标为候选。

## 两个独立诊断入口

诊断源码在 `test/probe/`，仅 `HEAD_DIAGNOSTICS=1` 时加入隔离 OPP。
普通构建仍为原 Fused/Audit 双算子。诊断构建额外检查两个 probe 的 CMake、ACLNN 头文件、
非空 `.o`/对应 JSON 和 SHA256，不能跳过安装产物检查。

1. **DFlashCastProbe**：单核，无 Matmul、无 Head 输入，只有 FP32 `[64]` 输入和小型输出。
   分别保存原输入、Mins 后、Mins/Maxs 后的 FP32，以及每阶段 CAST_NONE 的 FP16。
   覆盖正负 0、FP16/FP32 subnormal、RNE tie、65472/65488/相邻 FP32、65503/65504/
   65519/65520/65536/131008、对应负数、两种 NaN 位型、±Inf。
2. **DFlashHeadProbe**：调用实际共享 `RunHead`，相同 full-K tiling、UB 预算及归约。
   将词表 tile 0/1 在转换**之前**的 FP32 `[2,16,64]` 原样写入独立诊断输出（8192 B），
   不借用/修改原 FP16 Audit ABI 或 GM workspace。随后执行同一个 Vector 诊断，检验
   Head 上下文与独立 Vector 实验是否一致。额外 DMA 只在诊断模板生效，不能用于性能计时。

诊断 runner 对所有缓冲区检查具名 Guard，输入只读，三次交替 poison 后输出重复一致；
保留每次 `.bin`。`diagnostics.json` 保存环境/工具链版本、runner/输入/输出 SHA256、
Vector 输入数值与位型、全部阶段 FP32/FP16 位型及 15 行 token63/64 的原始 FP32。
NaN/Inf JSON 使用字符串表示，原始位型为准。此诊断不是 41-case correctness 的替代品。

安全门槛：

- overflow 全部 30 个位置必须证实为有限 `131008 / 0x47ffe000`。若为 `0x7f800000`
  或其它位型，保存证据并停止，不猜测 Inf 的来源。
- Mins/Maxs 只能限幅超过 ±65504 的位置；正常值、零符号、subnormal 不得意外改变。
  NaN 必须保留原 FP32 bits，最终 FP16 NaN 必须与 control Cast 相同；否则候选停止。
- ±Inf 的限幅是实验候选行为，并非已证明的 Native 通用规则。NaN/Inf Head 用例仍使用
  原始权重 `0x7e00/0x7c00`，并在随后的 Native/Audit 全词表比较中严格验收。
- 即使 Vector 门槛失败，也先保存 overflow/NaN Head 的原始数据，再返回失败。
- 设备诊断通过后才跑五个敏感用例，再跑原完整 41-case。subset 的 full acceptance
  明确为 NOT_RUN；只有 41 个用例全部完成才能标为 PASS，原比较器不增加任何容差。

## 服务器一键诊断及候选验收

复用旧 capture 与 30 个 Native OM，不重新 capture/export，不覆盖旧 OPP/evidence。
沿用服务器的 MODEL_PYTHON、DEVICE_ID，CANN_ROOT 必须指向 9.0.0 安装（脚本记录版本文件
及 npu-smi 输出；无法确认 9.0.0/310P 就失败）。

```bash
git pull --ff-only
export HEAD_CAPTURE=/home/w00949577/qwen35-dflash-run/draft-head-UCt9vR8q/capture/manifest.json
export HEAD_NATIVE_MANIFEST=/home/w00949577/qwen35-dflash-run/head-native-gpJaP6vN/models/native.json
HEAD_NATIVE_ONLY=0 HEAD_DIAGNOSTICS=1 HEAD_CONVERSION=minmax \
HEAD_WARMUP=5 HEAD_REPETITIONS=30 \
  bash framework/custom_ops/draft_head/run_server.sh
```

脚本依次：新目录构建/安装 → 独立 Vector → Head precast → 敏感 subset → 完整 41-case。
仍严格要求全部 Audit FP16 logits bitwise、Fused/Audit INT64 IDs 一致，以及 guards、readonly、
repeat、row permutation、timed-tail/postcheck；Native 执行是同输入 golden 对照。
`HEAD_CONVERSION=cast HEAD_DIAGNOSTICS=0` 可恢复原 CAST_NONE 对照构建，已知 overflow 会失败，
不得把它的预期失败改成通过。只有真实设备结果决定后续是否采用候选，当前不修改默认值。

回传新目录的 `server.log`、`build-config.json`、`opp-preflight.json`、
`diagnostics/diagnostics.json`、`sensitive/suite.json`、`data/suite.json`（未到达的阶段为 NOT_RUN），
以及失败项的 `runner.log` / `comparison.json`。特别保留 `diagnostics/overflow_inf/precast-*.bin`、
`diagnostics/cast/vector-fp32-*.bin` / `vector-fp16-*.bin`。

41-case PASS 后提取生产 M15 的 continuous-v1（warmup=5,repetitions=30）median/p95：

```bash
python - /path/to/NEW_RUN/data/suite.json <<'PY'
import json, sys
x=json.load(open(sys.argv[1]))
assert x['full_41_case_acceptance']=='PASS'
for row in x['timing_summary']:
    if row['name'] not in ('real-c16-m15','real-c64-m15'): continue
    print(row['name'])
    for mode in ('native-logits','native-top1','fused'):
        t=row['modes'][mode]
        print(mode, 'median_ms=', t['median_ms'], 'p95_ms=', t['p95_ms'])
PY
```

这给出候选相对 Native 的延迟；仅凭该对照不能把差值全部归因于两次 Vector 操作。
若要隔离新增操作成本，还需同输入/同协议的 cast 对照 M15 数据。诊断 Probe 的耗时不使用。
完整 Draft/Decode、A/C 图接入、D/E/F 均保持 NOT_RUN / 未开展。

## 本地验证边界

CPU 回归验证原位限幅的**目标**语义、正负 overflow、RNE ties、signed zero、subnormal、
NaN 防改写门槛、原始 FP32 留证、构建身份和诊断 runner 语法。CPU Cast 对照现在使用
会产生 Inf 的转换模型，避免旧模型的隐式饱和掩盖问题。两条路径均运行原 full-vocabulary
kernel 控制流模型；这些测试不是 CANN intrinsic 仿真，更不是 NPU PASS。
