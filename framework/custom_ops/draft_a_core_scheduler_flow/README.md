# 五层 A-only Draft Scheduler A/B（310P3 / CANN 9.0.0）

设备 0；这是独立实验，不切换生产调度。前提是同一设备、挂载命名空间和已验收解释器中，单 FC Host Compare 和真实单 FC Smoke Compare 均可复核。**本项目的服务器阶段全部 NOT_RUN。**

A 使用原 `LaunchBlockCount(n, GetCoreNum(), coreLimit)`；B 在 dav_2002 耦合平台、auto core、AIC=AIV>0 时，用实际 `sliceNum=min(n/64,GetCoreNumAiv())` 调用 SDK `CalcTschBlockDim`。每次检查真实返回、两种循环分配及实际 SetBlockDim；拒绝分离架构与资源漂移。两版均以同一个冻结的 Off OPP 为基础，只替换 Host DSO，使用同一份原 Kernel/API。**没有固定 7/8，没有启用 Raw Prefetch。**

52 个是投影实例，并非 52 种不同几何形状。C16/C64 各包含一个 FC 和五层 KV/Q/O/Gate-Up/Down，分别绑定各自的 X、NZ INT8 Weight、Group128 Scale 和 Native Golden。相同几何不免测；KV 的 M 为 32/80，FC 的 M 为 16/64。Host Query 每版/实例一次 GetWorkspaceSize，总计 104 次、不调用 Execute。Projection Smoke 按版本、实例、重复号启动隔离进程：共 208 次未计时 Execute。每次 GetWorkspaceSize 后先复核本次 DSO、Tiling 序列化、分配和 Workspace，才经 socket 放行。每次同步后保留 actual.bin、原数值比较器结果、Readonly、输入/输出/Workspace 两侧 Guard、Poison 及 maps；要求 finite、bitwise、Repeat、A/B 全部通过。

Export 从历史已验收的两个 A-only AIR 及全部 payload 复制到新目录，分别用 A/B 新私有 OPP 编译四份 OM。不重新量化、导出权重或重建 Head；检查冻结 AIR、部署引用、实际 ATC 命令（只允许模型/输出路径迁移）、26 A / 0 C / 0 C-Audit、图审计及公共 ABI。每个 gear 的两条 ATC sidecar 按实际 M/K/N 归组，核对所有形状；允许 ATC 合并重复 shape 的回调，不用回调数冒充 GE 节点数。独立比较 ATC 中的 A/B 计划，跨路线只要求相同的容量、Matmul Tiling 字节和资源使用，不将 ACLNN blockDim 代入静态 OM。

Full Smoke / Benchmark 使用独立编译的 guarded ACL OM Runner；加载模型后检查 maps，通过父进程许可才执行。由 Runner 在进程内逐字节比较全部 15 个 INT64 token 与五层 key/value 全部逻辑字节（包括未更新区域），复核 15 份输入只读、所有 Guard、Poison、Repeat 和同步。所有 DSO 必须符合 Source/Build 封存身份；完整 OM 可能加载单 FC 未出现的库，未知库直接 BLOCKED，须独立审查身份后开新实验，不允许把整个目录加入白名单。

Benchmark 只使用本实验通过 Full Smoke 的 OM。按 C16/C64 分别交错 A/B、B/A、A/B（C64 反序）；每进程 2 次前检、5 warmup、30 连续 ExecuteAsync+Stream Sync、Timed Tail、带 Poison Postcheck。模型加载、数据上传、读回、数值、文件 I/O、maps 校验均不计入正式延迟；每个 gear、每版 90 个原始样本，共 360 个。报告每轮 A/B Median 比、总体 Median/P95/Min/Max。若报告为 `ACTUAL_ATC_SCHEDULES_IDENTICAL`，两版静态图没有观测到 blockDim 差异；其延迟比**不能解释为 7→8 核加速**。即使 blockDim 不同，也没有对每个物理核驻留作证明。

## 服务器：一次只执行一段

沿用已通过单 FC 的 shell、设备和解释器。预留多份完整 AIR/payload、OM 和全部原始输出所需磁盘空间；证据验证会重新读取大文件。所有命令从同一新根目录开始，已有阶段目录绝不覆盖。

```bash
cd /home/w00949577/z50058744/qwen3.5-4B-dflash
git pull --ff-only origin feature/ascend310p-custom-ops
PY=/home/w00949577/qwen35-dflash-run/venv-hf513-probe/bin/python
S=$PWD/framework/custom_ops/draft_a_core_scheduler_flow
R=$(mktemp -d /home/w00949577/qwen35-dflash-run/a-core-scheduler-flow-XXXXXX)
echo "R=$R"
flow() { TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PY" -B "$S/run_flow.py" "$1" --root "$R"; }
flow source
```

默认只读引用 `a-core-scheduler-host-MzJtLl` 和 `a-core-scheduler-fc-smoke-bWW88T`。之后逐阶段手工执行，前一阶段退出 0 且 report 的状态准确才继续：

| 单条命令 | 精确成功状态 | 阶段证据 |
|---|---|---|
| `flow source` | FLOW_SOURCE_PASS | binding：两条历史证据链、52 case、两个冻结完整 AIR |
| `flow build` | FLOW_BUILD_PASS | variants.A/B：OPP、Host 构建、DSO manifest、两个 Runner 与 ldd |
| `flow host-query` | FLOW_HOST_QUERY_PASS | plans/runs 各 104 项；真实回调与 Workspace；Execute=0 |
| `flow projection-smoke` | FLOW_PROJECTION_SMOKE_PASS | 每项两次 Execute，原始输出、数值/保护证据 |
| `flow export` | FLOW_EXPORT_PASS | models 四份，atc_pairs 按 gear/shape；attribution |
| `flow full-smoke` | FLOW_FULL_SMOKE_PASS | runs 四项，全部 Token/KV 文件和安全验收 |
| `flow benchmark` | FLOW_BENCHMARK_PASS | runs 十二项；statistics 分 C16/C64；保留原始样本 |

每阶段 `$R/<phase>/report.json` 的共用与专用字段见 `report.schema.json`；实际前驱链、SHA、状态、样本数与数值门槛在 Python 中强制执行。失败返回非零，保留 BLOCKED 报告、traceback 和当时已生成的日志/输出。先回传 `phase/status/error/integrity_error`、traceback 和失败子进程的 runner.log 或 worker.log；不要重写旧报告。这个实验不测端到端 Decode，不做 msprof。

## 本地复核

```bash
python3 -B -m unittest discover -s framework/custom_ops/draft_a_core_scheduler_flow/test -v
# 在已有 CANN Docker 中，source set_env.sh 后执行；使用本地历史 CPU 编译夹具
bash /src/framework/custom_ops/draft_a_core_scheduler_flow/local_check.sh \
  /work/build/core-scheduler-7tnyl8u6/CustomOp /work/build/core-scheduler-7tnyl8u6/opp
```

本地脚本仅配置、编译 Host tiling/proto 和 Runner 并做 CPU 测试，不编译设备 Kernel、不执行 ACLNN/OM。
