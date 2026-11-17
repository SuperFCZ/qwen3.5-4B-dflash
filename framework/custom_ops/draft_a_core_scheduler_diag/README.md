# 单FC Core Scheduler Host-only A/B

范围：Ascend310P3 / CANN9.0.0 / M16 K12800 N2560。
API/版本/7-Core解释、资源与环境边界见 [REVIEW.md](REVIEW.md)，源码身份见 [sdk-review.json](sdk-review.json)。
不修改生产Host/Kernel、历史7-Core证据、Golden、Scale或任何旧OPP。

A沿用GetCoreNum和原LaunchBlockCount；B在实际dav_2002耦合1:1配置下使用真实Aic/Aiv getter，
以受容量限制的worker切分数调用实际SDK的CalcTschBlockDim。没有硬编码8，没有虚构平台资源。
它是1:1换算的Host候选，不是分离架构的Cube/Vector新组合，也不是设备8-Core准入证明。

## 门控

| 阶段 | 精确成功状态 | 执行范围 |
|---|---|---|
| Source | CORE_SCHEDULER_SOURCE_REVIEWED | 重核已通过Platform Diagnostic五阶段、Host DSO 1ef6d60c…、冻结输入/Kernel/SDK；记录Kernel JSON类型字段和前缀目录 |
| Build | CORE_SCHEDULER_HOST_OPPS_BUILT | 独立A/B Host DSO与Runner；原Kernel/API逐字复制，不编译Kernel |
| ATC | CORE_SCHEDULER_ATC_HOST_OBSERVED | 每版编译一次单FC冻结AIR，绝不生成四份完整Draft OM |
| ACLNN | CORE_SCHEDULER_ACLNN_HOST_OBSERVED | A、B各独立进程，各两次GetWorkspaceSize；Execute=0 |
| Compare | CORE_SCHEDULER_HOST_COMPARISON_RECORDED | 重核原始sidecar、DSO、Workspace及完整Tiling字节；不执行设备 |

任何阶段失败保留报告、日志和原始输出，不进入下一阶段。
状态中的Host成功不等同NPU数值或性能成功。当前所有新服务器阶段NOT_RUN。

## 服务器命令

使用已通过诊断的同一解释器和设备环境；Loader环境仍由子进程最小环境策略构造，不手动清空全局变量。

```bash
cd /home/w00949577/z50058744/qwen3.5-4B-dflash
git pull --ff-only origin feature/ascend310p-custom-ops
PY=/home/w00949577/qwen35-dflash-run/venv-hf513-probe/bin/python
S=$PWD/framework/custom_ops/draft_a_core_scheduler_diag
BASE=/home/w00949577/qwen35-dflash-run/a-platform-diag-pinned-opp-XYztZ2
NEW=$(mktemp -d /home/w00949577/qwen35-dflash-run/a-core-scheduler-host-XXXXXX)
echo "NEW=$NEW"
```

先只读Source：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PY" -B "$S/run_scheduler.py" source --baseline-root "$BASE" --output-dir "$NEW/source"
```

成功后独立Build：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PY" -B "$S/run_scheduler.py" build --source-report "$NEW/source/report.json" --output-dir "$NEW/build"
```

随后单Shape ATC：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PY" -B "$S/run_scheduler.py" atc --source-report "$NEW/source/report.json" --build-report "$NEW/build/report.json" --output-dir "$NEW/atc"
```

随后ACLNN Host-only：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PY" -B "$S/run_scheduler.py" aclnn --source-report "$NEW/source/report.json" --build-report "$NEW/build/report.json" --output-dir "$NEW/aclnn"
```

最后比较：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PY" -B "$S/run_scheduler.py" compare --source-report "$NEW/source/report.json" --build-report "$NEW/build/report.json" --atc-report "$NEW/atc/report.json" --aclnn-report "$NEW/aclnn/report.json" --output-dir "$NEW/compare"
```

回传阶段终端状态与Compare；失败则回传error/integrity_error、traceback及所指日志末尾80行。
每个查询输出独立 `A/fc`、`B/fc`：`platform.jsonl`、`scheduling.jsonl`、完整stdout/stderr及身份。
ACLNN另存execution/journal/maps、Readonly/Poison/GM Guard结果。原Raw/UB Guard Kernel没有被执行。

## 实现身份

`scheduler_derive.py`审核完整Host差异，允许的改变仅为：一个固定Shape门槛、blockDim策略选择、诊断输出。
所有原资源/数学检查不动。`scheduler_variant.h`只在新的生成项目中明确写0/1，没有默认策略。
两版共享同一Host源和SDK；Kernel、API、配置与冻结B包逐字比对。
新增头文件只依赖已经声明的CANN接口，源数据始终来自真实TilingContext。
B的Core来源改变，因此不会调用旧7-Core launch_evidence为B授予PASS；独立校验器检查真实Aic/Aiv、
SDK返回值、实际Context blockDim和完整覆盖，结果仅为Host候选记录，原生产准入规则不改。

依赖旧Platform Diagnostic的DSO清单、Source Lock、最小OPP环境与Runner协议；这些旧文件不改。
新目录使用独立模块名，避免改变历史sys.path解析或污染同名验证器。
ATC新增调度证据路径只加入本地绑定副本，不改共享环境策略。

## 本地证据

macOS与Docker ARM64各11项CPU回归通过，包括40 Tile精确覆盖、1:1门槛、错误SDK返回、
资源溢出、序列化变化、实际blockDim不符、两种C++编译策略、CLI门控及路径别名分析。
CPU模型不是硬件调度或资源配额证明。

Docker真实CANN9.0.0编译/链接：A、B的Host tiling/proto DSO、API wrapper及Host-only Runner均通过，
ldd -r无未解析符号；未执行ATC/ACLNN/Kernel。
本地记录：`/work/build/core-scheduler-7tnyl8u6/report.json`（A）、
`/work/build/core-scheduler-tkv046dd/report.json`（B）。全部标记LOCAL_ONLY_NOT_SERVER_VERIFIED。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s framework/custom_ops/draft_a_core_scheduler_diag/test -v
# Docker内，复用已经存在的生成模板，只编译Host：
python3 -B framework/custom_ops/draft_a_core_scheduler_diag/local_compile.py --variant A --template /work/build/a-raw-sidecar-host-le2oo1y_/CustomOp --output-parent /work/build
python3 -B framework/custom_ops/draft_a_core_scheduler_diag/local_compile.py --variant B --template /work/build/a-raw-sidecar-host-le2oo1y_/CustomOp --output-parent /work/build
```

未实现Smoke/Benchmark入口。只有服务器Host结果全部通过后，才单独评审设备Smoke与其Guard/数值验收。
