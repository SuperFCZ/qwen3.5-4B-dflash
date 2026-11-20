# 下一步实验与门禁

**当前停在设计审查。** `double-w-overlap-narrow-fence` 的备选协议是在 double-w-overlap 的成功非尾 K 末用 `MTE1_V → V_MTE2` 两对 Set/Wait 替代一次 Fk。直接 MTE3 barrier 未闭合 W 的跨流水复用，单独 MTE3_V 未闭合 A1。此目录只提供只读取证和 CPU 合同测试，没有编译或启动设备的入口。设计、代码行与证据在 [生命周期审计](pipe-all-lifecycle.md)、[窄同步设计](narrow-fence-design.md)、[历史配对](fence-comparison.md)。

## 候选的唯一差异

依照 `draft_gate_w16_atc8/kernel/tile_schedule.h` 第 17–21 行，只在 `Iterate(k)` 成功、k<9、下一代 raw 双 Wait 及 D(k+1) 已发后替换原 `op.Fence()`。仅该处调用独立 fence 协议。不能把 `Operations::Fence` 全局换掉，它还用于失败、F9 和 FN。新方向的预留、等待、释放和失败回滚计入该协议的成本。其余运行模式、权重地址、SDK 构建配置、数学路径和原同步保持一致。

主对照用封存 double-w-overlap，再与 single-w-raw 独立配对检查净收益。两组均使用同一 310P3 device 0、同一 CANN 9.0、同一冻结 C16 Layer0 AIR 和真实常量。首次仅一个 N=19456 的 Gate 投影。复用原 Runner 和 DSO socket gate，候选使用独立私有包、缓存、ATC worker 和根目录。严禁将旧 overlap ELF、包内通用 .o 或旧功能 Smoke 用作新 Fence 的证明。

| C16 Layer0 冻结文件 | 用途与 SHA256 |
|---|---|
| `x.bin` | FP16 [16,2560]，`44fed068f079c18e95ef303c0d02cf5401470db7277ec17d9d024fbf29079256` |
| `w_nz.bin` | INT8 NZ 权重，`c898ef7b555f6e3bcf3b183ec7742b012f0a4e40ab4a46cd34184b77218b5a2c` |
| `s_gn.bin` | FP16 Group128 Scale，`1365af20b6f09734c6d02f2f3d3045fb20e4af1f53d414c9b4dd9011c7ced333` |
| Native Golden | FP16 [16,19456] 全部 311296 个元素，`353839a7a41bdd44930e91f72ae750adc2f7f15e14f2b679d5f0799e1b5da76c` |

## 七个独立阶段的交付合同

以下为下一轮的阶段设计。只有 Source 依据和 CPU 合同在本轮审查。候选的 Build/ATC/Smoke/Benchmark/Compare 均为 `NOT_RUN`，Static Admission 为 `BLOCKED`。每个实际阶段须单独启动，生成新目录的报告，绑定源码与所有前驱 SHA，发生失败即停。

| 阶段 | 下一轮必须产生和核验的证据 |
|---|---|
| Source | 重验 YHsAKu 的原决定和两个配对，绑定控制的源码、动态实现、头、Golden、私有 OPP、Runner、SDK。审查上表中的每个同步端点、应用与 SDK 的逐方向 ID 占用上界、共同首代 Wait 的契约。候选与 overlap 只允许有指定的成功分支及新增事件所有权差异。其他改动应阻断。 |
| Build | 仅在独立实验根生成新私有 OPP。绑定实际生成源码、工具与命令、ABI、私有库、设备对象、编译宏。单 W 和原 overlap 包只读。可保存相同召回链的预处理/IR 辅助审查，注明与最终二进制的关系。 |
| ATC | 只编译该层冻结 AIR，独立缓存。核对注册 vendor、8 block 循环归属、两份 W0/W1、109312 B user UB、152832 B scratch、最高 8192 B reserve、L1/L0A/L0B/L0C=40960/16384/65536/4096 B；核对 224 B parent、全部 50 字段、不变的 Cube SHA、2097152 B 单算子 Workspace。专用 ELF 必须通过图的符号、大小、偏移和整段 SHA 与真实 OM 字节绑定。不能用 ACLNN 的 7 block。 |
| Static Admission | 先用独立已知 dav_m200 ELF 校准解码工具；候选实际 .text 和其他非空可执行段要有 100% 的 PC、编码、助记符、操作数及可达 CFG、重定位审计。逐路径验证 W0/W1、A1/B1、L0A/B、CO1、scratch 的实际访存操作数。原 strict 验证器继续对应原 Fk，窄协议必须有单独、按实际 ELF 锁定的证明。核对两个新方向的 Set/Wait 及来回代数、初次预热及普通失败、尾 K/FN/End；独立证明小块 PIPE_M、初始化位和唯一最终 F32→F16。若给其他对照重新发起设备调用，也须满足本轮适用的机器级准入。旧 experimental PASS 没有此权限。 |
| Smoke | 只有 CPU 正确性和上一步机器级安全通过后，手动起独立进程。先对照、再候选，两个不同 Poison 的未计时执行，与冻结 Golden 全量 bitwise、跨变体、Repeat、Readonly、输入输出与 Workspace Guard、同步、无 fallback、前后 DSO/maps 一一核验。随后做多轮独立进程的未计时 stress。本阶段的设备正确性通过才可计时。 |
| Benchmark | 单设备、resident 输入、无 profiler。分别配对 single-w-raw/候选 和原 PIPE_ALL overlap/候选；每组按控制先、候选先、控制先交替，三轮，每次 5 warmup + 30 Execute+Sync。每组 180 个新样本，候选在两组独立测量。计时外进行加载、拷贝、读回、数值与 Guard 扫描。保留原始样本、每轮及总体 Median/P95/Min/Max、Workspace、设备/模型身份和插桩状态。 |
| Compare | 两个配对分开报告每轮方向、总体倍率、P95、候选跨组漂移。相对 single-W 按用户给出的延迟下降 ≥3% 定义，`candidate_median ≤ 0.97 × single_median`，同时给出倒数速度比，要求三轮都更快且总体 P95 无回退。相对旧 overlap，建议预注册原研究口径 `overlap_median / candidate_median ≥ 1.03`、三轮都更快、总体 P95 无回退，作为“明确收益”。这两项判据在采样前冻结，不能用改善的某一轮覆盖其余轮次。 |

YHsAKu 的 single-w-raw median=2.004735 ms，乘 0.97 是 1.94459295 ms，与历史 net 组 overlap 相差约 119.992 µs。这只是说明目标的量级；正式分母必须取新配对中同时测量的 single-W。上述门槛均通过后，才可另行提出五层验证。单投影收益不外推完整 Draft。

## 现在可做的复核

```bash
S=framework/custom_ops/draft_gate_w16_narrow_fence
python3 -B -m unittest discover -v -s "$S" -p 'test_*.py'
# 310P 服务器或无 NPU 的 CANN 9.0 ARM64 环境，用安装的 SDK 复核所有 SHA 和引用区间
FENCE_CANN_ROOT=/usr/local/Ascend/cann-9.0.0 python3 -B -m unittest discover -v -s "$S" -p 'test_*.py'
```

最先需要补齐的是可信 dav_m200 解码能力，以及原 raw 首代同 ID 重新 Set 的设备契约。此前独立 Head 校准和两个 ATC ELF 的报告在 `w16-isa-audit-20yqefdn/static-audit.json`。在这两个缺口关闭前，当前没有可放行的设备命令。原 W16 的 `W16_EXPERIMENTAL_NO_GO` 与既有报告继续有效。
