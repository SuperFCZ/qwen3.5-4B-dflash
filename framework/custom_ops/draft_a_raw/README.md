# 独立五层 A-only Gate Raw Prefetch OM 实验

目标：测量 **完整五层 W8A16 Draft OM** 的 Raw Prefetch 收益。不是 W8A8，
不是五个独立 Gate/Up median 的和，不修改生产 deployment/导出路径。
本目录首次实现后的服务器 Precheck、私有 OPP 编译、OM 导出、NPU Smoke、Benchmark
均为 **NOT_RUN**。本地证据见 [LOCAL_VALIDATION.md](LOCAL_VALIDATION.md)。

## H / B / R 与不可变数据

| 身份 | 来源 | 用途 |
|---|---|---|
| H | `ac-draft-full-20261016-071408` 原 A-only C16/C64 OM、原 OPP | 与冻结 Native Golden 重新桥接；不参与新倍率分母 |
| B | 已验收五层 Off Child 的生成 Header + 当前已绑定源码 | 新导出/编译的完整 A-only 控制 |
| R | 已验收五层 Raw Child 的生成 Header + 同一源码 | 仅 Gate `gate_prefetch_mode=raw` 的候选 |

`accepted_reports.json` 的根报告 SHA 来自用户提供的服务器终端。Precheck 必须在服务器
重新读取全部引用，核实 Child、生成 Host/Kernel Header、安装 ELF/JSON/API、真实 SDK
文件、全部原始比较输出。没有从文件名猜测服务器 ELF SHA，也没有使用 tracked 默认 Header。

默认两轮五层证据：

- `/home/w00949577/qwen35-dflash-run/gate-raw-prefetch-five-20261116-015738`
- `/home/w00949577/qwen35-dflash-run/gate-raw-prefetch-five-reverse-20261116-022749`

历史完整根：`/home/w00949577/qwen35-dflash-run/ac-draft-full-20261016-071408`。
H 的 `draft_only=PASS` 是完整 Draft-only 图验收，顶层 `full_draft=NOT_RUN` 不予改写。
H 未声明的旧 Raw 字段记录为 `NOT_DECLARED_LEGACY`，不补填伪造配置。

保留原 Native Base 的严格 Source Lock、五层 checkpoint、原 FP16 激活、INT8 NZ 权重、
Group128 FP16 Scale、原 FP32 累加及舍入、Native head、全部其余算子。Raw 只有
INT8/Scale 两个 raw bank，W16 仍一份，Iterate 后 PIPE_ALL 保留；不宣称 Vector/Cube 完整重叠。
B/R Gate 的 UB 资源差异必须与已验收 Raw 设计一致，不能把 Off Tiling 与 Raw Kernel 混配。

## 实现和身份门控

- `raw_evidence.py`：只读 H、正反五层、A5.1、C16/C64 原始 Capture/Native Golden；
  重算旧比较（输出到新目录），不再次执行历史 Gate 实验。不加载 Torch/ACL。
- `build.py`：复用现有联合包构建器，在两个新目录分别构建 B/R。
  A 使用实际已验收 Header，C/C-Audit 是相同且冻结的注册库存，**不得进入图**。
- `raw_probes.py`：新私有包的每档 26 个投影无计时验收，共每版 52 项；复用原 Capture/
  Native Golden，不重新采集激活。验证 C64 KV M80 等非 Gate 路径未受影响。
- `worker.py`：单独进程、严格 Native Base import boundary，调用原 Draft-only factory 和
  `integration_spec('a')`；不执行 Target/Verify，不改旧代码。逐项绑定 26 份权重/Scale 和 Head。
- 四份新 OM 分别导出并 ATC；严格图审计 26 A、0 C、0 C-Audit、0 剩余 WeightQuant；
  15 输入/11 输出 ABI 与 H 一致。B/R 逻辑 IR 仅允许本次输出根路径归一化，其余差异拒绝。
- `raw_resources.py`：只解析真实 ATC Host callback 的六行记录（LAUNCH/PREFETCH/COPY/
  SCALE/K_TILE/CUBE），覆盖六类真实投影形状；不从 Standalone 或预期数值填充。
  交错、缺失或无法归属的日志 BLOCKED。Gate 计划须与已验收 Child 完全一致；其他投影
  B/R 资源必须相同。每个 Op 请求 Workspace 与模型实际 Workspace 分别记录。
- `runner/runner.cpp`：新 ACL OM Runner，复用已验证 guarded allocator 和 continuous-v1；
  独立 CMake 只链接所选 CANN ascendcl。测试 fake ACL 不参与实际 CMake。
- `run_experiment.py`：每次一个阶段，无自动重试、修补、发布或 `all` 入口。
  每次重查源码、前序报告、输入/输出、SDK、OM/OPP/ELF/Header/Runner 哈希。

每个子进程在启动前选择唯一 H/B/R OPP，移除历史 A/C 路径，保留已绑定 receiver/ADN
依赖。不在同进程叠放 B/R，不给旧 H OM 换新 OPP。每个新进程/模型编译在独立目录；
实际 Raw Host 日志是配置选择证明之一，缺失时不允许仅凭环境变量认定生效。

## 正确性与计时

完整 OM Smoke 每档按 H → B → R 执行，每项两次独立 Poison：

1. 校验实际模型全部 15 输入/11 输出描述符、分配大小、设备；加载不计时。
2. 全部 15 个 INT64 Token ID 和十个完整 FP16 KV buffer 逐字节比冻结 Native Golden；
   包括未更新 KV 区域，不使用容差。相等关系同时证明 B/H、R/B 相等。
3. 输入全字节 Readonly（包含物理 padding）、输入/输出/模型 Workspace/权重分配外侧
   Prefix/Suffix Guard、Poison 初始回读、两次 Repeat、同步成功。
4. 继承原数学 Kernel，不新增/删除其内部 Guard 或同步；不虚报检测到模型内部每个临时区越界。

任一失败保留当前日志和已生成输出，`BLOCKED`；后续阶段拒绝读取失败报告。
数值失败在进程内发现，不能先完成 Benchmark 再由 Python 事后判错。

每档 Benchmark 三轮，B/R 独立进程：

| Round | C16 | C64 |
|---|---|---|
| 0 | B → R | R → B |
| 1 | R → B | B → R |
| 2 | B → R | R → B |

每个进程两次 Poison 前检 → 5 Warmup → 30 Execute+Sync → Timed Tail → 独立 Poison
Postcheck。每档每版 90 新样本，两档共 360 个；不混入历史数据。正式区间只有执行、同步、
计时及记录至预留 Host 样本容器，没有上传/回读、比较、Guard Host 扫描或文件输出。
内核本身原有工作不删。异常 Host 日志在区间外检查后使整轮不准入。
模型加载/上传在计时外；完整 Draft Execute+Sync 包含图内全部实际工作。

`comparison.json` 分 C16/C64 保存每轮和整体 Median/P95/Min/Max、原始样本、B/R 倍率及
P95 回退。模型 Workspace/weight allocation 须与 Smoke 及各轮一致。
没有设备时间戳，纯设备时延标记 UNAVAILABLE；没有调用分布，不造跨档加权倍率。
不做 msprof、不做完整 Decode；性能成功也不自动生产采用。

## 服务器：现在只执行 Precheck

在已配置 CANN9.0.0、历史实验使用的 Python/Torch 环境内，拉取新提交后：

```bash
cd /home/w00949577/z50058744/qwen3.5-4B-dflash
S=framework/custom_ops/draft_a_raw
R=$(mktemp -d /home/w00949577/qwen35-dflash-run/a-only-raw-full-XXXXXX)
printf 'EXPERIMENT=%s\n' "$R"
bash "$S/server_stage.sh" precheck "$R"
```

预期 **`A_RAW_IDENTITIES_VERIFIED_NO_DEVICE`**。此步只读旧文件、核对哈希/协议/布局和
检查工具文件身份；不运行编译器、Torch-NPU、ACL、NPU、Benchmark。新目录中只写新的证据报告。
原始数据较大，哈希和已冻结输出重核需要时间。

回传 `show_report.py` 的终端输出即可。失败时：

```bash
python3 -B "$S/show_report.py" "$R/precheck/report.json"
```

先提供 status/error/integrity_error 和具体失败路径，无需上传权重/Golden/SDK 原文。
不存在的历史文件保持 BLOCKED，不能用新的 Golden 或默认配置补齐。

## 后续手动阶段（本轮不要自动执行）

每条都单独执行、验收后才进入下一条；脚本会自动传入同一 R 下的准确前序报告路径，
不自动运行前序阶段。默认 device0，与 H 一致；若不同需独立重做环境验收。

| 命令 | 精确成功状态 | 实际动作 |
|---|---|---|
| `bash "$S/server_stage.sh" build "$R"` | `A_RAW_PRIVATE_PACKAGES_BUILT` | B/R 私有 OPP + 独立 Runner；无 NPU执行 |
| `bash "$S/server_stage.sh" probes "$R"` | `A_RAW_52_PROJECTIONS_PER_VARIANT_MATCH` | 两档26投影/版，全部原始 Golden；不计时 |
| `bash "$S/server_stage.sh" export "$R"` | `A_RAW_C16_C64_OMS_COMPILED` | 四份完整 A-only OM；Torch设备Tensor准备可能使用原生设备操作，未执行完整 Draft |
| `bash "$S/server_stage.sh" smoke "$R"` | `A_RAW_ALL_TOKEN_KV_MATCH` | 两档 H/B/R 全输出+安全验收 |
| `bash "$S/server_stage.sh" benchmark "$R"` | `A_RAW_FULL_DRAFT_PAIRED_MEASURED` | 新三轮配对、每档180样本 |

每阶段 `report.json` 保留命令、日志和引用身份；build 下 B/R/report.json 保留实际 OPP。
export 下 B/R/c16,c64 保留 AIR/OM、编译日志和计划；smoke/benchmark 每进程保留全部11输出。
失败时最少回传本阶段 report 的 status/phase/error/integrity_error，以及错误指定日志末80行。
ATC Host 日志不完整/跨线程交错时提供其原始相关行，不手工拼接生成资源证明。

回滚无需改生产：停止该实验，继续使用原 H。任何修复源码都会改变 source inventory，
必须新建 R 从 Precheck 开始；不得覆写旧报告、续用旧 Source Hash 链或原地重装旧 OPP。
