# 五层 Gate Scale Transpose B16 与完整 A-only Draft

本实验在 Ascend 310P3、CANN 9.0.0 上比较两种 Group128 Scale 展开。两版都使用 Scheduler A、ATC 8 block、Raw Prefetch、K256 batched Cast、单 W16。对照 `raw` 使用 Brcb，候选 `raw-transpose-b16` 使用已验收的 TransDataTo5HD B16。完整 Draft 的“关闭”指关闭 Scale 优化，Raw Prefetch 始终开启。

冻结的 C16 Layer0 Pilot 每版已有 90 个计时样本。Brcb Median/P95 为 2.004000/2.009120 ms，Transpose 为 1.638825/1.645760 ms。C16/C64 完整 Raw 的既有 Median 为 30.569870/31.426295 ms。新的完整 Draft 性能由两档新配对决定。

## 在服务器开始

在同一终端保留 `R`。每次调用执行一个阶段，读完状态后再选择下一条。每个阶段的输出目录限用一次。BLOCKED 报告、已生成的产物和旧实验留在原处。每个设备阶段都要求显式的 `--allow-device`。

```bash
cd /home/w00949577/z50058744/qwen3.5-4B-dflash
git pull --ff-only origin feature/ascend310p-custom-ops
BASE=/home/w00949577/qwen35-dflash-run
PY="$BASE/venv-hf513-probe/bin/python"
S="$PWD/framework/custom_ops/draft_gate_scale_expansion"
R=$(mktemp -d "$BASE/gate-scale-expansion-XXXXXX")
echo "R=$R"
expand_scale() { TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PY" -B "$S/run_expansion.py" "$@" --root "$R"; }
expand_scale pilot-lock
```

第一步预期 `GATE_SCALE_PILOT_LOCK_PASS`。它在 CPU 上复算已封存的计时并复核 Golden、Runner 产物、设备记录和实际 ELF，需要读取历史目录。回传终端状态或 `$R/pilot-lock/traceback.txt` 即可定位阻断。

| 顺序 | 单独执行的命令 | 输出及准入 |
|---|---|---|
| 1 | `expand_scale c16-source` | `five-gate-c16/source`。逐层绑定真实输入、Native Golden、单 Gate AIR 和完整 Raw 图的节点及常量。 |
| 2 | `expand_scale c16-build` | 复核冻结的两个独立 OPP、生成源、Header、工具、API、Runner 及 DSO manifest。|
| 3 | `expand_scale c16-atc` | 保留 Pilot 单 Gate OM，为 Layer1–4 分别从冻结 AIR 编译候选并检查五个对应的 Raw OM。|
| 4 | `expand_scale c16-review` | 校验资源及 CANN CPU 模型对每层全部 Scale 字的展开。至此没有设备调用。 |
| 5 | `expand_scale c16-smoke --allow-device` | 对其余四层运行独立的两版 Smoke。C16 Layer0 绑定 Pilot。 |
| 6 | `expand_scale c16-benchmark --allow-device` | 对其余四层逐层采集新鲜配对样本。 |
| 7 | `expand_scale c16-compare` | 五层决策须为 `C16_FIVE_GATE_GO`。 |
| 8 | `expand_scale c64-source` | `five-gate-c64/source`。校验该 gear 的五套输入、权重、Scale 和 Golden。 |
| 9 | `expand_scale c64-build` | 复核同一组已验收 OPP。 |
| 10 | `expand_scale c64-atc` | 在每层权重/Scale、single ABI 与静态 Shape 相符后，复核可复用 OM 内的实际 ELF、部署、编译及回调。 |
| 11 | `expand_scale c64-review` | 用自身 Scale 身份绑定同层逐字 CPU 结果。 |
| 12 | `expand_scale c64-smoke --allow-device` | 在 C64 自身 Capture 上验收全部五层，含 Layer0。 |
| 13 | `expand_scale c64-benchmark --allow-device` | 五层分别采集本档数据。 |
| 14 | `expand_scale c64-compare` | 全图入口要求 `C64_FIVE_GATE_GO` 且 C16 已通过。 |
| 15 | `expand_scale full-draft-export` | `full-draft-export/`。逐档克隆实际 Raw 完整 AIR，隔离编译两个候选，与原始两个 Raw OM 验证。 |
| 16 | `expand_scale full-draft-smoke --allow-device` | `full-draft-smoke/`。两档、两版的 Token/KV 与安全检查。|
| 17 | `expand_scale full-draft-benchmark --allow-device` | `full-draft-benchmark/`。两档、两版的全图新配对。|
| 18 | `expand_scale final-comparison` | `final-comparison/`。从实际运行报告的 360 个新计时重算、逐档给出结论。|

每个阶段的文件是表中目录下的 `report.json`。C16/C64 每层阶段位于 `five-gate-c{16,64}/<stage>/report.json`。退出码 0 表示通过，1 表示 BLOCKED，3 表示有完整测量但决策为 NO_GO。此前门禁失败将阻断设备执行和后续验收。若需重开，创建全新根目录。

## Pilot 与样本身份

- `gate-scale-broadcast-ABnQDE` 提供 Source、Build、ATC。原失败 Review 保持 BLOCKED，并用于验证已发布的源码续接。
- `gate-scale-reviewfix-bXR45o` 提供 Review、Smoke、Benchmark、Compare。代码固定七份报告的 SHA256，逐段核对绝对路径、前驱、Git 源码合同及实物，再调用原 Pilot 的只读验证器。完整 Raw 的归档来自 `gate-raw-atc8-Np20yu`。
- 两档均按源证据里的 NZ/Scale SHA、逻辑权重身份、V3 投影和 GE 节点交叉绑定五层，依次对应 `DFlashGroupQuantLinear_4/_9/_14/_19/_24`。每层 M/K/N 为 16/2560/19456，X 81,920 B、NZ 权重 49,807,360 B、GN Scale 778,240 B、输出 622,592 B。C16 与 C64 同层共享 checkpoint 常量，每次设备调用使用当前 gear 的 Activation 和 Native Golden。
- C16 Layer0 唯一使用 `FROZEN_PILOT` 标签，其余九个单投影位置使用 `FRESH`。每层规模为 389,120 个 GN 字及 6,225,920 个展开字，复用已验证的 65,536 位型 CANN CPU 测试。若同层 Scale 字节不同，跨 gear 复用拒绝。
- Build 指向已实际编译、封存的两份 OPP。两个历史 Runner 分别用于单 Gate 和完整 Draft。设备包、生成配置、数学和 SDK 不再重复构建。ATC 每次独立进程、独立缓存、独立输出。

## 完整图与编译资源

完整图复制 Raw 部署所指的实际 AIR manifest 和全部 payload，重新检查文件字节、SHA256 与相同 ATC 选项。每档独占一个新私有 OPP。它保留 Raw 安装树、通用静态设备对象及非目标源，加入 Pilot 审查过的 Gate 源码和同一生成头，并在私有动态编译 wrapper 的源码选择点记录实选源。分派使用已绑定的编译 key 和输入输出形状；动态描述导致 Gate/非 Gate 歧义时阻断。共享编译结果可对应多个节点，必须和最终 OM 实际符号关联。

导出门禁逐项核对：

- 26 A、0 C、0 C-Audit。5 个目标节点的实际嵌入 ELF 与可执行段都发生变化，其余 21 个的完整 ELF SHA256、符号及可执行段逐一相同。Head 和其他原生可绑定的设备代码、图边、描述符、端口顺序受保护。
- 六种真实 ATC 回调的 Scheduler A、8 block、Core 分配、Workspace、资源和实际 Tiling 等同控制。Gate 特别核验 224 字节 parent、其中 200 字节 Cube 序列化及全部 50 个字段。Gate 使用两份 Raw bank、单份 32,768 B W16；User UB 76,544 B，SDK scratch 185,600 B，算子 Workspace 2,097,152 B。Runner 单独记录 OM 实际分配。
- 从两份真实 IMOD 逐块计算整个权重分区 SHA256，核对分区起止、无间隙和整个文件长度，同时比较未删减 Const 记录的逐节点摘要（含内联张量数据及顺序敏感的维度）。适用范围限定在服务器回传的 256 字节头、64 位对齐五分区布局。类型 1 的含义也可参照公开的 OM 读取实现。[Netron OM reader](https://github.com/lutzroeder/netron/blob/main/source/om.js)
- Worker Report、实际环境、私有 OPP、编译器、部署 Manifest、原始回调、分派日志、图 dump 与内嵌偏移都有身份记录。任一选择无法绑定、非目标 ELF 改变、实物或资源不符，均在 Smoke 前停止。

## 数值、安全与决策

每个新测位置，两轮 Smoke 均以独立进程分别运行两版，每个进程使用两个不同 Poison。Runner 的实时 DSO socket 检查通过后才允许执行，保留前后 maps、输入 Readonly、GM/Workspace Guard、Repeat、Stream Sync 和 CPU fallback 检查。单投影对 311,296 个 FP16 全量 bitwise 比较。全图检查 15 个 int64 Token，及每层 Key/Value 共 10 个 FP16 `[1,8,2112,128]` 缓冲，每份 4,325,376 B，包含未更新位置。每个实际快照与其完整 Native Golden 逐字节核对。

正式计时为 Brcb→Transpose、Transpose→Brcb、Brcb→Transpose。每版每轮 5 次 Warmup 和 30 次连续 Execute+Sync，记录 90 个原始样本、每轮倍率、Median 和最近秩 P95。模型加载、上传、读回、Golden 比较、DSO 扫描及 Profiler 均在正式计时之外。计时后检查 Timed Tail 和 Postcheck。

每档五层以 `1 − Σ候选层 Median / Σ控制层 Median ≥ 0.03` 为门槛，同时每层三轮方向一致，所有层 P95 均无回退。此聚合只评价独立投影。完整 Draft 用新鲜样本独立决策；本实验预注册为两档各自也满足 3% 延迟降低、三轮改善、P95 无回退方记总体 GO。退出决策只用于研究，生产配置的变更需要另外决定。

## 报告字段

| 阶段 | 主要字段 |
|---|---|
| 全部 | `phase/status/sources/predecessors/root/folder/atc/device_execution/benchmark/error/integrity_error/active_case` |
| Pilot | `lock.reports`、跨根续接与历史失败身份、冻结样本和判定 |
| Source / Build | `binding.cases`、完整 ABI、Raw AIR/OM、逐层常量与 `builds` 的 OPP、Source、API、Runner、DSO |
| ATC / Review | `models/evidence` 的节点、嵌入 ELF、实测资源及 `scale_words` 的 CPU 日志、输入和实际执行身份 |
| Smoke / Benchmark | `runs` 含每次 argv/文件、输入和 Golden、实际快照、allocation、maps、同步和日志。计时样本位于 `validation.timing.samples_ms`，每层统计位于 `statistics` |
| Full export | `packages/models/evidence` 按 gear，包括 `dispatch/compile/constants/unchanged_projections/native_images` |
| Compare | `result.layers` 或 `result.gears` 保存原始样本、Median/P95、轮次、来源和决策。完整 Draft 使用 `gears` |

本地 CPU 合同可用 `python3 -B -m unittest discover -s framework/custom_ops/draft_gate_scale_expansion/test -v`。在已有 CANN ARM64 容器中设置 `SCALE_ROUTER_BUILD` 为本地 SDK 包的编译报告，可另外核验真实 wrapper 的代码选择点和两份源码。CPU 测试与服务器真实 ATC、数值及性能结果各自出具证据。
