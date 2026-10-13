# C Top1：Scalar 瓶颈诊断与第一版 Vector 候选

## 已确认的设备基线

用户报告：`2a9211f`、Ascend310P3 / CANN 9.0.0、
`framework/custom_ops/draft_head/.runs/head-DGwqjbT8`，compare_select + Scalar Top1
**完整 41/41 PASS**。原始 evidence 在服务器，本地未读取；提交与旧 OPP/evidence 保留。

用户回传的近似延迟（不是本轮重新测量，未提供 p95）：

| M | Native Top1 | 原始 Custom | compare_select + Scalar |
|---|---:|---:|---:|
| 1 | ≈23.2 ms | ≈14.85 ms | ≈15.96 ms |
| 8 | ≈23.2 ms | ≈28.90 ms | ≈30.27 ms |
| 15 | ≈23.2 ms | ≈45.15 ms | ≈46.53 ms |

Compare/Select 的增量约 1.1–1.4 ms，远小于 Custom 从 M1 到 M15 的约 30 ms 增长。
Matmul 固定 M16；Scalar Top1 的比较数为 `M×248320`，M15 为 3,724,800 次，且每次
包含 UB GetValue 和 Better。因此它是首要嫌疑。X 的实际复制量、归约与其它同步仍可能
贡献 M 相关开销，不能仅由这张表断言全部时间都来自 Scalar Top1。

本轮 **CANN 编译、Vector primitive NPU、Vector 41-case、paired 性能、msprof 均 NOT_RUN**。
目标是让 M15 接近或优于 Native ≈23.2 ms；这是目标，不是已取得的结果。
A5.1、Native golden、生产 Draft 和 `Better()` 数学规则不修改。

## 默认与开关

- 默认 `HEAD_CONVERSION=compare_select HEAD_TOP1=scalar`，选择设备已验收的路径。
  之前源码默认 cast 现改为显式选择已验收转换；compare_select 算法本身不变。
- `HEAD_TOP1=vector` 是 opt-in 候选，要求 compare_select，不改变 Matmul tiling。
- `HEAD_CONVERSION=cast HEAD_TOP1=scalar` 保留 CAST_NONE 对照。它仍有已知 overflow
  synthetic 失败，只对六个真实输入测量，绝不标成完整 41-case PASS。
- minmax 保留，仍需诊断；不拿它的 NaN payload 失败作为通过结果。
- `HEAD_PROFILE_STAGE=matmul|convert` 仅用于隔离 build-only 性能观察器，不能进入
  `run_suite.py` 正确性验收。默认 `none`，不会进入生产 Draft。

## Vector Top1 算法及资源

每个 N64 tile 保持原来的完整 K FP32 Matmul → compare_select → 唯一 FP16 舍入。
随后仅对**已经舍入的 FP16 logits**拓宽到 FP32，复用已不再需要的 c32 buffer。
绝不比较原始 FP32 Matmul 累加值。

对每个逻辑行的 64 个值：

1. `Compare(row,row,NE,64)` 检测 NaN；存在 NaN 时，从 mask 取第一个位置。
2. 无 NaN 时，`WholeReduceMax(..., ORDER_VALUE_INDEX)` 取最大值。
3. 忽略 SDK 返回的 index，用 `CompareScalar(EQ)` 找到所有并列最大值，再取 mask
   最低有效位。FP32 的 +0/-0 应相等；此行为必须由独立 NPU primitive gate 确认。
4. 最低位索引用最多六次整数寄存器测试求出，不扫描 64 个 logits。
5. 只读取获胜位置的原始 FP16 bits，调用原 `Better()` 做跨 tile 合并；跨核归约不变。

因此 NaN payload 不参与大小排序；多个 NaN 和普通并列值都取最小 token ID，获胜的
原始 FP16 位型不改写。NaN 优先与全局最小 ID 仍由原 Better 跨 tile/分区保持。

M15 的 tile 合并为 `3880×15=58,200` 次 Better，相比逐元素合并减少 64 倍；这只是
操作计数，不是速度承诺。Vector 指令和 V→Scalar 依赖仍有开销，需实际计时。

M≤15，c32 的 padded row 15 在拓宽后可作为 64 B mask/reduce scratch。未新增 UB：
`UserUbBytes(P)`、Matmul UB、`WorkspaceBytes(P)=192P`、sync offsets 均不变。
FP16 logits buffer 只读用于 Top1，Audit 输出仍保存全部原 FP16 logits。

固定 SDK：官方 `cann/asc-devkit` **v9.0.0**，commit
`c8c1b9df6bb9cd61c09751c7ef414b89e06af274`。核查路径：

- [dav_m200 vconv](https://gitcode.com/cann/asc-devkit/blob/v9.0.0/impl/basic_api/dav_m200/kernel_operator_vec_vconv_impl.h)：half→float widening。
- [dav_m200 cmpsel](https://gitcode.com/cann/asc-devkit/blob/v9.0.0/impl/basic_api/dav_m200/kernel_operator_vec_cmpsel_impl.h)：FP32 Compare/CompareScalar，NE/EQ。
- [dav_m200 reduce](https://gitcode.com/cann/asc-devkit/blob/v9.0.0/impl/basic_api/dav_m200/kernel_operator_vec_reduce_impl.h)：WholeReduceMax→vcmax；仅使用已有 ORDER_VALUE_INDEX。

不用其它芯片的 ONLY_VALUE/ONLY_INDEX 等支持假设，不依赖 vcmax 的 NaN 或 tie 行为。
无法仅凭源码保证 310P 的 widening/比较对所有特殊值都符合要求，设备 primitive gate
覆盖全部 65,536 种 FP16 位型、多 NaN、sNaN、±0、±Inf、正负 subnormal、边界和并列。
失败会保存具体行的全部原始 FP16 bits、期望/实际索引，并阻止进入完整 Head 验收。

## 性能对照：保证 Matmul 有可观察结果

五份构建各自拥有新 build/OPP/evidence，不覆盖旧版本：

| 构建 | Matmul | 转换 | Top1 | 结果资格 |
|---|---|---|---|---|
| matmul | 原 M16/N64/full-K | 无 | 无，输出指纹 | MEASURED_NOT_CORRECTNESS |
| convert | 同上 | 原 compare_select + Cast | 无，输出指纹 | MEASURED_NOT_CORRECTNESS |
| original | 同上 | CAST_NONE | Scalar | 六个真实 case，完整 41 NOT_RUN |
| baseline | 同上 | compare_select | Scalar | 完整严格验收 |
| vector | 同上 | compare_select | Vector | primitive + 完整严格验收 |

两个观察器仍对 **3880 个 tile** 执行 `Iterate(false)` 和 `GetTensorC`。每个 tile 的
真实 FP32/FP16 输出进入滚动指纹，经原 GM partial 槽和原 SyncAll 汇合，最后写出负数
INT64（明确不是 token ID）。不是空 kernel 或丢弃所有计算结果；原数据依赖使 Matmul
不能被当作死代码删除。Iterate 失败会输出非负错误标记，由 runner 拒绝。

观察器保持同样 tiling、UB 配额、核分工、GM workspace 与同步；**仅性能构建**的最终
汇合合并指纹而不是 Top1。指纹只采样每个 tile 的一个输出，不是数值正确性证明。
msprof 另行记录 Cube/MTE/Scalar 活动，需检查原始 counters，不能只看主机计时认定瓶颈。

观察器的 Guard、只读、repeat、timed-tail/postcheck 仍检查。它没有 Top1 语义，因此
不声称通过 Top1 的行置换/数值验收，execution/suite 不会写 correctness PASS；正式
baseline/vector 的行置换、完整 logits、INT64 IDs 和其它门槛全部保留。

`convert−matmul` 与 `baseline−convert` 的 median 差只作定位线索，包含指纹开销与
最终汇合差异；不能当作精确可加的阶段耗时，也不能相加各 pipe 的 utilization。
旧“原始 Custom”二进制的具体身份未提供；新 original 是同源 FP32+CAST_NONE 对照，
不擅自把历史近似数据当作本轮 paired 样本。

## 一键服务器实验

在现有 Ascend310P3/CANN 9.0.0 环境中，复用原 Head capture 和 30 个 Native OM：

```bash
git pull --ff-only
export HEAD_CAPTURE=/home/w00949577/qwen35-dflash-run/draft-head-UCt9vR8q/capture/manifest.json
export HEAD_NATIVE_MANIFEST=/home/w00949577/qwen35-dflash-run/head-native-gpJaP6vN/models/native.json
"${MODEL_PYTHON:-python3}" framework/custom_ops/draft_head/test/top1_experiment.py \
  --capture "$HEAD_CAPTURE" --native-manifest "$HEAD_NATIVE_MANIFEST" \
  --output-dir "/home/w00949577/qwen35-dflash-run/head-top1-$(date +%Y%m%d-%H%M%S)" \
  --device-id "${DEVICE_ID:-0}"
```

默认执行顺序：

1. matmul / convert 观察器：C16/C64 × M1/M8/M15，另跑 C16 M15 msprof。
2. 构建 original / baseline / vector，执行 Vector Top1 全位型 NPU primitive gate。
3. 三条正常路径分别跑同样六个真实 case，严格 Audit/ID 验收与配对计时。
4. baseline 和 vector 各完整 41-case，再对两者 C16 M15 做独立 msprof。

全程 warmup=5、repetitions=30、continuous-v1。未经插桩的 median/p95 和全部样本保存
到汇总；msprof 是单独调用，不混入 paired samples。每个分支都有自己的 Native Head/
Top1 对照；比较器核对 hidden/weight/capture/Native manifest SHA256、tiling、workspace
和协议一致。不同 OPP 在独立进程环境中运行。串行配对仍可能有温度/频率漂移，需同时
检查各份 Native 对照；若波动显著，应在同条件下重复实验，不挑最快样本。

若暂不运行 msprof，显式加 `--skip-msprof`，所有相应 profiling 状态记为 NOT_RUN。
报告 PASS 只代表严格验收和测量完成，不意味着速度目标达成；`candidate_adoption`
始终为 NOT_DECIDED，不自动切换默认实现。

若只验证候选、暂不跑整套性能对照：

```bash
HEAD_NATIVE_ONLY=0 HEAD_TOP1=vector HEAD_CONVERSION=compare_select \
HEAD_DIAGNOSTICS=0 HEAD_WARMUP=5 HEAD_REPETITIONS=30 \
  bash framework/custom_ops/draft_head/run_server.sh
```

同样先跑 raw-FP16 primitive，然后完整 41-case。不重新 capture/export。

## 需要回传与判定

- 总目录 `suite.json`：六组 paired median/p95、原始 samples、诊断差值及完整验收状态。
- `matmul/suite.json`、`convert/suite.json`：仅性能观察器，禁止作为 correctness PASS。
- `top1-probe/suite.json` 及失败 batch 的 `runner.log` / `logits.bin` / `ids-*.bin`。
- `baseline-full/suite.json`、`vector-full/suite.json`；失败 case 的 comparison.json/runner.log。
- 各 `*-build.json` 指向隔离构建与 server.log，保留 OPP binary/JSON hashes 和 build flags。
- `profile-{matmul,convert,baseline,vector}/profile.json`、profile.log、profiler 原始目录。

先确认 primitive 和完整 41-case 全 PASS，再看 M1/M8/M15 两种 gear 的 median/p95，
特别是 M15 相对 Scalar 基线和 Native 的表现。即使 M15 达标，也要报告 M1/M8 是否回退。
没有真实数据前不写“Scalar 瓶颈已证实”“性能优化成功”或“达到 23.2 ms”。
完整 Draft/Decode 性能和 A/C 图接入仍 NOT_RUN。

## 本地回归

本轮 37 项 CPU 回归通过，另完成 Python/构建脚本静态检查；CANN/NPU 项仍 NOT_RUN。

CPU 运行实际 Vector helper 对照原 Better，遍历全部 FP16 位型并覆盖随机行、NaN/tie/
subnormal。Reduce stub 故意返回最后一个并列索引，确保实现不依赖 SDK tie 索引。
Scalar/Vector、cast/minmax/compare_select、两种 observer 的 kernel 控制流模型继续
核对 full-K 访问、分区、UB/workspace 和失败核心同步。增加 build/launch 身份、观察器
拒绝进入正确性验收、配对输入/tiling/协议不一致拒绝、ACL runner 语法等 CPU 回归。
这些测试不模拟真实 CANN 指令、缓存或 pipeline，不能替代设备测试。
