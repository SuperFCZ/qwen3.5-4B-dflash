# DFlashGroupQuantLinear：A2 / A3 / A3.1 / A3.2 优化结果

本页记录用户在 **Ascend 310P3 / CANN 9.0.0** 服务器上反馈的数值、独立投影计时和 msprof
结果。A3 通过按 N64 tile 多核分工获得约 6.85×/6.56× 的单核→多核加速；A3.1 在相同
`continuous-v1` 协议下，通过批量反量化再获得约 3.87× 的 legacy→batched 加速。
gate/up 的 batched 耗时已略低于原生 WeightQuant OM，down 仍有约 24% 的耗时差距。
后续 A3.2 用户反馈 down serial≈1.91 ms、prefetch≈1.79 ms、native≈1.54 ms。
按这些近似数计算，prefetch 比 serial 耗时少约 6.3%，仍比 native 高约 16.2%。
用户决定暂停此路径的进一步优化，进入 [A4 全投影形状覆盖](A4.md)。本轮反馈的证据边界见第 8 节。

**完整 Draft / Decode 端到端性能仍为 `NOT_RUN`。本文的 isolated projection 加速比不能
解释为模型整体加速，也不能据此宣称达到完整 Draft 的优化预算。**

数据依据是用户贴出的服务器 PASS 日志、A3 逐层统计，以及本轮 A3.1/profiling 摘要；代码
核对基于 `b5b6277`。这不是本机 NPU 复测，也不是对远端全部 JSON/CSV 的重新审计。
带“约”的数值保留用户摘要精度；A3.1 实际目录已由用户补充，未读取的逐次样本、p95、
原始计数器明细和哈希不作补造。
实现说明和复现入口见 [A2](A2.md)、[A3](A3.md)、[A3.1](A3_1.md)。

## 1. 工作负载、配置与比较口径

两类投影均为固定 W8A16 checkpoint 的真实权重和真实冻结输入回放产生的激活，五层各一组。

| 投影 | M | K | N | group size | N64 tile 数 | 每个 N tile 的 K128 次数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gate/up | 16 | 2560 | 19456 | 128 | 304 | 20 |
| down | 16 | 9728 | 2560 | 128 | 40 | 76 |

X/Y 为 FP16 ND；权重为 signed INT8 offline-NZ，逻辑 `[N,K]`、物理
`[K/32,N/16,16,32]`；custom scale 为连续 FP16 GN `[K/128,N]`。
native 对照是相同 X/W/S 的独立 WeightQuant OM，使用 group128、`inner_precise=0`、
`must_keep_origin_dtype`、`deterministic=0`。不是以 FP16 dense Linear 替代 W8 原生对照。

| 阶段 | 比较对象 | 计时协议 | warmup / repetitions | 核数与模式 |
| --- | --- | --- | --- | --- |
| A2 correctness | custom ACLNN ↔ native WeightQuant OM | 数值验收为主 | custom/native 各两次正确性输出 | A2 单核路径 |
| A3 | custom 单核 ↔ custom 自动多核；另列 native OM | `checked-v1` | 脚本默认 3/10；该次实际字段需以原始报告核对 | core cap=1 / cap=0；自动实测 7 核 |
| A3.1 | 同源码 legacy ↔ batched；两边均另测 native OM | `continuous-v1` | **本次实测 5/30** | 相同自动多核配置，7 核；仅 dequant 模式变化 |

原生 OM 是本页局部算子的性能对照。完整 Decode 加速比仍须按
[OPTIMIZATION.md](../../../docs/OPTIMIZATION.md) 以 ordinary Decode 为基线，另行实测。

### 计时协议的区别

`checked-v1` 在每次执行后回读输入、检查 guards/输出，执行前写入 poison。这些检查不在
`execute_sync` 计时区间内，但可能改变后续调用的 cache 状态和执行间隔。

`continuous-v1` 先做两次带 poison 的完整检查，再连续预热和测量；测量循环内不做
poison、输入/输出 D2H 回读或文件写入。测量结束后检查最后一个 timed 输出，并追加一次
带 poison 的检查。中间每个 timed 输出没有逐次回读。custom 的 GetWorkspaceSize/准备
耗时单列，计入 `execute_sync` 的是执行入口到 stream synchronize 返回。

因此 A3.1 重新测了 legacy baseline，**3.87× 来自同协议的约 14.24→3.677 ms 和
7.40→1.91 ms，不是拿旧 A3 的时间直接与新协议结果相除**。也不把 A3 与 A3.1 的倍率相乘
作为已测总收益。原始样本、median/p95 保存在报告中；本页未获得 A3.1 全部原始样本。

## 2. Correctness：A2 建立真实投影基线

A2 已由用户确认：五层 gate/up + down，共 **10 个真实 projection**，custom 与相同输入
的 native WeightQuant OM **FP16 bitwise equal，bit mismatches=0、max ULP=0**。

验收不是仅看 signed-zero。真实权重、GN scale、NZ 布局和投影输出均受比较约束；还保留
输入只读、输出/workspace guards、重复结果一致性检查。A3 后续运行再次通过了 A1 九种
形状/81 个合成 case 和这十组真实投影，日志见第 6 节的已知服务器路径。

A1 `rounding_probe` 曾出现 custom↔CPU、native↔CPU 各 1 bit，而 custom↔native 为 0。
该 case 的 CPU FP64 累加只作诊断，不放宽 custom↔native 的逐位一致门槛。

A3.1 保留该门槛。当前比较工具还逐位复核 legacy/batched/native 的 correctness、timed
tail 和 postcheck 输出，检查输入/OM 身份、核数和计时协议。本轮 A3.1 计时与 profile 已由
用户反馈完成；其逐 case 数值报告和精确哈希由第 6 节的比较 JSON 定位。本文未在本地
重新读取该原始报告，不以计时数字本身充当 bitwise 验收证据。

## 3. A3：N tile 多核分工

核 b 按 cyclic 方式处理 `b, b+B, b+2B, ...` 的 N64 tile。每个输出 tile 只有一个 owner，
由该核完成完整 K 归约；没有 split-K、跨核原子累加或归约顺序变化。自动模式本次启动
**7 个核**，来自平台可用核数与 N tile 数的限制，不是硬编码为 7。

| 投影 | custom 单核 | custom 7 核 | 单核/多核 | native OM | 多核 custom/native |
| --- | ---: | ---: | ---: | ---: | ---: |
| gate/up | ≈98.1 ms | ≈14.31 ms | ≈6.85× | ≈4.05 ms | ≈3.54× 耗时 |
| down | ≈48.93 ms | ≈7.46 ms | ≈6.56× | ≈1.56 ms | ≈4.79× 耗时 |

这证明 N tile 并行分工有效，但该版本每核处理 tile 的成本仍高于原生实现。
在各 tile 成本相同的简化模型中，gate/up 最忙核处理 44 个 tile，分工加速比约
`304/44=6.91`；down 最忙核处理 6 个，约 `40/6=6.67`。实测接近这一分工模型，不能据此
把每核 Cube 的计算利用率也判断为接近峰值。

### 用户提供的 A3 逐层 median

以下保留先前终端输出的三位小数，单位 ms；倍率保留当时打印精度。

| Case | 单核 custom | 7 核 custom | 单核/多核 | native OM | custom/native |
| --- | ---: | ---: | ---: | ---: | ---: |
| layer-0-gate_up | 98.270 | 14.311 | 6.87× | 4.047 | 3.54× |
| layer-0-down | 48.934 | 7.463 | 6.56× | 1.554 | 4.80× |
| layer-1-gate_up | 98.038 | 14.311 | 6.85× | 4.047 | 3.54× |
| layer-1-down | 48.925 | 7.461 | 6.56× | 1.560 | 4.78× |
| layer-2-gate_up | 98.125 | 14.310 | 6.86× | 4.049 | 3.53× |
| layer-2-down | 48.927 | 7.464 | 6.56× | 1.557 | 4.79× |
| layer-3-gate_up | 98.058 | 14.316 | 6.85× | 4.050 | 3.54× |
| layer-3-down | 48.932 | 7.471 | 6.55× | 1.559 | 4.79× |
| layer-4-gate_up | 98.133 | 14.316 | 6.85× | 4.054 | 3.53× |
| layer-4-down | 48.952 | 7.472 | 6.55× | 1.558 | 4.79× |

down 的汇总约值可能写为 7.46 或 7.47 ms、6.55 或 6.56×；应以此逐层值及原始 JSON
为准，不能把约数的末位差异解释为新的性能变化。

## 4. A3.1：批量反量化与同步优化

A3.1 保留七核 N-tile 分工、N64/K128 tile 和完整 K 归约，只改变 tile 内反量化的组织方式。
每个 N64×K128 tile 的源码级调用如下：

| 操作 | legacy | batched |
| --- | ---: | ---: |
| Cast | 256 | 4（每次 64 个 repeat） |
| Muls | 256 | 64 |
| PIPE_V barrier | 512 | 2 |

这些是 API/barrier 的源码调用次数，不是芯片指令总数或单独的加速倍数。Cast 覆盖相同
signed INT8 元素，Muls 仍在同一 group 内使用同一个 FP16 scale。跨流水依赖及 MatMul
前后的必要屏障保留，UB 分配与 Cube tiling 不变，没有完整 FP16 权重落 GM。

量化计算边界保持为：

```text
signed INT8 q → 精确转换为 FP16
             → 按 group128 的 FP16 scale 做 FP16 乘法，得到 W16
X16 × W16    → 原有 K 次序的 FP32 累加
             → 完整 K 归约结束后，最终一次 FP16 输出舍入
```

实现见 [kernel](op_kernel/d_flash_group_quant_linear.cpp)。对照仍保留原指令序列，分别用
`DFLASH_DEQUANT_MODE=legacy/batched` 构建。

### 同一 continuous-v1 协议下的结果

warmup=5、repetitions=30，以下为用户提供的五层结果近似汇总；未补造逐层更多小数位。

| 投影 | legacy custom | batched custom | native OM | legacy/batched | batched/native |
| --- | ---: | ---: | ---: | ---: | ---: |
| gate/up | ≈14.24 ms | ≈3.677 ms | ≈4.03 ms | ≈3.87× | ≈0.91× 耗时 |
| down | ≈7.40 ms | ≈1.91 ms | ≈1.54 ms | ≈3.87× | ≈1.24× 耗时 |

gate/up 的代表原生值 4.034 ms 对应 `3.677/4.034≈0.91`：**custom 耗时少约 9%**，
等价的 native/custom 耗时比约 1.10×。down 的 `1.91/1.54≈1.24`，即 **custom 耗时仍高
约 24%**。这里明确百分比的分母，避免混用“耗时下降”和“速率提升”。

用户反馈五层结果接近、波动很小；本页没有原始 30 次样本及 p95，不进一步声称统计显著性
或跨设备/跨负载稳定性。比较 JSON 会保留两次 native 的时间，便于排查运行环境漂移。

代表值相加，一层 gate/up + down 的 custom 约 `3.677+1.91=5.59 ms`，原生约
`4.03+1.54=5.57 ms`，在这一粗略账面上基本持平。这只是**两个独立 projection 的 median
近似相加**，不是串联子图、完整 MLP 或 Draft 的测量；未包含 SwiGLU、residual 等阶段。

## 5. Profiling：观测与推断分开记录

以下是用户反馈的 **batched** msprof 摘要，沿用其指标名称；运行目录见第 6 节，原始 CSV
字段与采样明细未在本地读取。它们来自插桩运行，不能替代未插桩 `execute_sync` 的 median。

| Projection | Task Duration | Cube utilization | MTE2 | MTE3 | MTE1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| gate/up | ≈3.60 ms | ≈81.1% | ≈18.7%（18.6–18.8%） | ≈13.5%（13.5–13.6%） | ≈4% |
| down | ≈1.848 ms | ≈78.1% | ≈20.2% | ≈12.6% | ≈4.1% |

观测说明当前 Cube 活动较高，搬运流水也有可见活动。结合受控 legacy/batched A/B 中约
3.87× 的提升，结果支持把上一版的重要成本归因于**碎片化反量化调用及过多同步**，并把
下一轮优化重点转向 **Cube 计算、数据搬运及 MTE2→dequant→Cube 的重叠效率**。

这些结果尚不能独立证明唯一瓶颈或给出下一轮可获得的收益：

- 各 pipeline 的活动可以重叠，百分比不能相加后当作串行 wall-time 分解。
- Cube utilization 的 81%/78% 不直接等于达到理论峰值 FLOPS 的 81%/78%。
- 本页只有 batched 的 profiler 摘要；没有 legacy 的同口径计数器和完整流水/stall 时间线，
  不给“Vector 占比下降多少”或“可隐藏多少等待”编造数字。
- Task Duration 与 host launch+sync 来自不同运行和边界，不能直接相减当作精确 host 开销。

因此，“瓶颈转向 Cube、搬运和 pipeline overlap”在这里是**后续工程重点的判断**。
具体主导限制仍需结合时间线、等待及实际搬运数据确认；继续减少少量 Cast 调用不应预期
再次自动获得一个 4× 提升。

## 6. Evidence 路径、版本和待补录项

下面标为“已知”的绝对路径来自用户先前提供的**服务器**命令和日志，并不表示这些文件
已复制到本地开发机或提交到 Git。

| 证据 | 已知服务器路径 | 已有反馈 |
| --- | --- | --- |
| A3 总验收 | `/home/w00949577/z50058744/qwen3.5-4B-dflash/framework/custom_ops/draft_quant/.runs/a3-gKSilJg0/data/suite.json` | `PASS: A3 multi-core`；真实 projection 的 block_dim=7 |
| A3 下的 A1 回归 | `/home/w00949577/z50058744/qwen3.5-4B-dflash/framework/custom_ops/draft_quant/.runs/a3-gKSilJg0/data/a1/suite.json` | `PASS: 9 a1 workloads` |
| A3 下复用 A2 的真实投影对照 | `/home/w00949577/z50058744/qwen3.5-4B-dflash/framework/custom_ops/draft_quant/.runs/a3-gKSilJg0/data/real/suite.json` | 十组 custom↔native OM，bits=0、ULP=0 |
| A3 单核/自动多核比较 | `/home/w00949577/qwen35-dflash-run/a3-single-vs-auto.json` | `status=PASS`；第 3 节逐层 median/倍率来自用户打印结果 |
| A3.1 legacy 总验收 | `/home/w00949577/z50058744/qwen3.5-4B-dflash/framework/custom_ops/draft_quant/.runs/a31-jamr59qm/data/suite.json` | 用户确认的 legacy run；与 batched 使用相同 continuous-v1、5/30 配置 |
| A3.1 batched 总验收 | `/home/w00949577/z50058744/qwen3.5-4B-dflash/framework/custom_ops/draft_quant/.runs/a31-KEg03LMS/data/suite.json` | 用户确认的 batched run；第 4 节计时对应本轮摘要 |
| A3.1 legacy/batched 比较 | `/home/w00949577/qwen35-dflash-run/a31-legacy-vs-batched.json` | 配对输出比较、原始时间、median/p95、前后 native 和 summary 哈希的原始记录 |
| A3.1 gate/up msprof | `/home/w00949577/qwen35-dflash-run/a31-profile-gate-up` | 第 5 节 gate/up 指标；具体 case/层号和采集命令以目录内 `profile.json` 为准 |
| A3.1 down msprof | `/home/w00949577/qwen35-dflash-run/a31-profile-down` | 第 5 节 down 指标；具体 case/层号和采集命令以目录内 `profile.json` 为准 |

以下是**定位规则与待补录项，不是已确认的实际目录**：

| 待补录证据 | 路径/字段定位规则 | 用途 |
| --- | --- | --- |
| 原始 A2 独立验收 run | `.runs/a2-*/data/suite.json` 的具体路径待补 | 与 A3 内复跑的 A2 数值结果区分 |
| capture 与 native OM 根路径 | 实际 `A2_BUNDLE`、`A2_NATIVE_OM_MANIFEST`；或真实投影 `suite.json` 的 `bundle` / `native_om_manifest` | 核对同一 checkpoint、输入、scale、离线权重与原生 OM |
| A3 单核 run | `a3-single-vs-auto.json` 的 `single_summary_sha256` 对应的 `data/suite.json` | 恢复单核构建身份与完整 timing 字段；该比较 JSON 不直接记录原 summary 路径 |
| A3.1 原始字段明细 | 上表已知 JSON 及 profile 目录内的 CSV/采样文件，未在本地读取 | 核对逐次样本、p95、计数器表头、case/层号和精确哈希 |
| 版本与环境 | 各 run 的 `source-commit.txt`、`build-config.json`、`server.log`、`opp-preflight.json` | 实际源码/header/runner/OPP 身份；补充 driver、firmware 和工具版本 |

真实投影每个 case 的 `comparison.json`、`custom/runner.log`、`native_om/runner.log`、
`execution.json` 和 `actual-{0,1}.bin` 是数值与执行明细；A3.1 另有 `benchmark-last.bin`
和 `postcheck.bin`。原始产物继续留在服务器证据目录，本文不生成替代性的 PASS JSON。

在两次已知 A3.1 run 中，`data/a1/suite.json`、`data/real/suite.json` 分别是合成回归和
十组真实投影的子报告；`source-commit.txt`、`build-config.json` 等位于相应 run 根目录。
两份 profile 目录按采集入口保存 `profile.json`、`profile.log`、`profiler/` 和新的 `case/`。
这些子路径依据当前工具的输出约定定位，具体文件内容以服务器存档为准。

相关**实现提交**为 A2 `e603019`（随后 `1c0cf81` 修复 capture row update、`09f5c3c`
修复 FileConstant descriptor）、A3 `6875c3a`、A3.1 `b5b6277`。这些用于定位代码演进；
每次实测的确切提交/构建哈希仍以其 evidence 为准，不能用实现提交代替运行身份核对。

### 本次 A3.1 配置与复现入口

```bash
export A2_WARMUP=5
export A2_REPETITIONS=30
# A2_BUNDLE、A2_NATIVE_OM_MANIFEST、CANN_ROOT、MODEL_PYTHON、DEVICE_ID
# 沿用同一套已通过的数据和服务器环境；实际路径以 evidence 为准。

DFLASH_SUITE=a31 DFLASH_CORE_LIMIT=0 DFLASH_DEQUANT_MODE=legacy \
  bash framework/custom_ops/draft_quant/run_server.sh
DFLASH_SUITE=a31 DFLASH_CORE_LIMIT=0 DFLASH_DEQUANT_MODE=batched \
  bash framework/custom_ops/draft_quant/run_server.sh
```

`A2_WARMUP=5` / `A2_REPETITIONS=30` 是本次配置，区别于脚本默认的 3/10。
`profile_a31.py` 请求 `--ai-core=on --aic-mode=task-based --aic-metrics=PipeUtilization`，
profile 数据与未插桩 A/B 分目录保存；具体命令见 [A3.1 操作说明](A3_1.md)。

重新核对本次已知 A/B 产物时可使用以下命令。输出文件刻意使用新名称，避免覆盖原比较记录：

```bash
"$MODEL_PYTHON" framework/custom_ops/draft_quant/test/compare_a31.py \
  --baseline /home/w00949577/z50058744/qwen3.5-4B-dflash/framework/custom_ops/draft_quant/.runs/a31-jamr59qm/data/suite.json \
  --candidate /home/w00949577/z50058744/qwen3.5-4B-dflash/framework/custom_ops/draft_quant/.runs/a31-KEg03LMS/data/suite.json \
  --output /home/w00949577/qwen35-dflash-run/a31-legacy-vs-batched-recheck.json
```

上面是复核命令，不表示本地已经执行；`-recheck.json` 是新输出建议名称，不是本轮既有证据。

## 7. A3.1 时的阶段结论与后续方向

A3 解决了核间并行度问题；A3.1 显著降低了核内反量化碎片化和过多 barrier 的成本。在不
改变量化数学、K 归约顺序和 FP16 舍入边界的前提下，gate/up 降至约 3.68 ms，耗时比本次
原生 OM 少约 9%；down 降至约 1.91 ms，耗时仍高约 24%。代表值下 gate/up + down 的
独立计时之和与原生基本持平。

建议暂时冻结通过验证的 gate/up 作为回归与性能对照，优先继续分析 down。这里的“冻结”
是保存稳定工程基线，不是宣称达到 [完整 Draft 时延预算](../../../docs/OPTIMIZATION.md)。
下一步方向是 prefetch、double buffer、MTE2→dequant→Cube overlap，以及依据实际依赖
减少 K tile 间等待。先用 profile 确认可隐藏的等待，再逐项修改；不能直接删除必要屏障。
若调整 tile 或缓冲生命周期，继续通过 A1 与真实投影的 bitwise/只读/guard 回归。

| 验证层次 | 本轮可以记录的结论 | 仍未完成的范围 |
| --- | --- | --- |
| Correctness | A2 十组真实投影 bitwise 通过；A3 完整回归通过；A3.1 沿用相同严格门槛并已有比较报告路径 | 未覆盖全部 26 次投影和所有 M 档位；本页未重新审计远端逐 case 原始报告 |
| Isolated timing | A3 多核有效；A3.1 同协议 A/B 约 3.87×，gate/up 局部优于原生、down 仍慢 | 不能作为串联子图或完整图时间；不宣称已达性能预算 |
| Profiling | batched 的 Cube/MTE 活动观测支持转向计算、搬运及重叠效率的分析 | 原始计数器口径、完整时间线/stall 数据待补；不能推出唯一瓶颈或保证下一轮收益 |
| Full-model validation | **完整 Draft / Decode performance：`NOT_RUN`** | custom 接入后的候选/KV 轨迹、token/EOS/stop reason、C16/C64、峰值显存与端到端收益均需另测 |

## 8. A3.2 用户计时反馈与阶段冻结

用户随后提供的 down 独立调用近似结果为：

| 路径 | 耗时 | 比较 |
| --- | ---: | --- |
| serial | ≈1.91 ms | 本轮串行对照 |
| raw prefetch | ≈1.79 ms | serial/prefetch≈1.067×；耗时下降≈6.3% |
| native OM | ≈1.54 ms | prefetch/native≈1.162×；custom 耗时仍高≈16.2% |

这些数字按用户摘要保留，不补造逐层、p95、样本或置信区间。A3.2 的实现提交为 `345ccdd`，
运行入口默认 `continuous-v1`、warmup=5、repetitions=30；本次实际配置及提交应以服务器
原始报告核对。用户本轮未提供 `suite.json`、比较 JSON 或 profile 目录，也未贴完整数值门槛
日志，因此本页仅记录**用户报告的 isolated timing**，不新增一份原始 correctness PASS 或
profiling 结论，不宣称所有形状及完整模型已验证。

按用户要求，暂停 down 性能调优，保留 serial/prefetch 两条路径与现有默认 serial。
A4 只扩展 Q/O、FC M16/M64、KV M32/M80 的支持与独立原生 OM 验收；不继续修改反量化、
预取调度或 down 的 tile 参数。完整 Draft / Decode 仍为 `NOT_RUN`。
