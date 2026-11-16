# A-only Host Evidence Sidecar：第一阶段 B-C16

目标状态：`A_RAW_BC16_HOST_EVIDENCE_SIDECAR_MATCH`。
本阶段不执行Projection/NPU、完整Draft Smoke、Benchmark或生产部署。

## 独立 Host 输出通道

用户最新实测说明CANN的GMM Warning来自stdout；Python分离stdout/stderr并不能隔离它。
本目录为新私有OPP派生Host源码，原 `draft_quant/op_host`、Kernel、旧OPP和日志全部只读。
`audit_host()` 对整份Host源码做精确变换审计，仅允许：

1. 引入 `host_evidence.h`。
2. 六个机器记录改为 `Batch.Record`，原格式串和参数逐字保留。
3. 一次Commit，失败返回 `ge::GRAPH_FAILED`。

没有改数学、Shape、GetTiling、Core Policy、Workspace、缓冲或任何设备同步。
普通Host说明和所有CANN输出仍进原stdout/stderr。

ATC子进程环境变量：`DFLASH_GROUP_QUANT_EVIDENCE_PATH`。
Runner为每次调用新建绝对路径的 `host-evidence.jsonl`，mode0600、O_EXCL；只在子进程env
设置，不污染父进程。新Host指定该变量时，六条机器记录不写stdout。

每行是独立JSON，例如 `{"record":"LAUNCH","payload":{...原字段...}}`。
Host先在CPU内存中格式化六条，然后每个callback独立open，使用
`O_APPEND | O_CLOEXEC | O_NOFOLLOW`。在LOCK_EX保护下写完整批次，处理短写/EINTR，
fsync后释放锁。**锁覆盖六条记录**，防止不同callback交错；没有共享单个已加锁fd。
[flock接口](https://man7.org/linux/man-pages/man2/flock.2.html) 与
[O_APPEND接口](https://man7.org/linux/man-pages/man2/open.2.html) 是该POSIX合同的依据。

不存在env时，Record调用原vfprintf格式与参数，保持旧stdout协议；env存在但为空、
相对路径、文件不存在、非regular文件、锁/写入/sync失败都失败，不回退stdout。
Host打印明确EVIDENCE_ERROR并返回GRAPH_FAILED，Runner也拒绝该错误标记。
不创建占位旧路径、不修复损坏JSON、不丢Warning。

## Parser与三份文件身份

新专用入口 `collect_host_evidence()` 替代该工作流中的stdout资源读取；旧
`collect_stdout()`和旧实验代码不改。只读取sidecar：

- 恰好36行、6个callback，顺序LAUNCH/PREFETCH/COPY/SCALE/K_TILE/CUBE。
- 六种记录的全部字段及类型、无重复JSON key、六种不同shape。
- 将已验证JSON值无损映射成原格式私有视图，再调用原collect/launch_evidence。
  该视图是格式适配，不是清洗stdout，也没有补造资源字段。
- 保留原Gate Raw与非Gate资源比较合同；B-C16的六类实际计划须逐项匹配旧B的C16
  Projection计划。旧同shape计划存在冲突时也拒绝。

ATC Capture、Worker Report、外层报告均保留：
`stdout/atc_log`、`stderr/atc_stderr`、`host_evidence`，每份都有path/bytes/SHA256。
选中sidecar时stdout再出现机器记录也BLOCKED，防止实际加载了旧Host库。
截断、缺失、次序错乱、重复shape都失败；stdout的189条Warning无需删改。

## 新 Host OPP 与身份边界

build阶段只在新目录操作：

- 从已验收B的生成项目复制配置与源码，核对原生成Header、A源码和历史Build身份。
- 仅改上述Host日志代码；独立构建项目移除Kernel子目录，只编译
  `cust_optiling`、`cust_op_proto` 两个CPU Host目标。
- 不调用旧build.sh（它会先编译binary）；不重新编译设备Kernel。
- 完整复制旧私有OPP，保留内部相对别名；只替换新Host tiling/proto DSO。
- Kernel `.o`、对应JSON、ACLNN API/头文件以及其余所有文件必须逐字节相同；
  完整库存和别名也保持。任一非Host文件差异BLOCKED。
- 比较新旧Host DSO的ldd依赖身份，记录实际compile_commands、CMakeCache及源差异。

这是一份**新Host身份的私有OPP**，不是把旧Build报告改成新PASS。
旧104 Probes仍只证明旧OPP，**不能转移为新OPP的NPU PASS**。
本阶段只验证B-C16 Host资源，不需要先执行新的NPU Probe。

进入后续完整Draft正确性/性能前，必须：

1. 对新B和新R各完成Host Build/ABI/Kernel字节检查与C16/C64资源验收。
2. 重新执行各52项Projection，即新包104项NPU验收，保留原Golden/安全协议。
3. 再执行完整Token/KV Smoke及后续新的配对计时。

现有Precheck、输入、Golden、H和已完成Probe文件可以只读复用作基线；新的Host Source
与OPP身份另行封存，不假称原Source Lock覆盖新Host。原工具目录的源码集合未变。

## 当前冻结资产

旧实验：`/home/w00949577/qwen35-dflash-run/a-only-raw-import-life-duXlSB`。
当前B-C16 OM应为3090431312 B，SHA256
`f4ca093e62230bba0bb5e6c3a5865189d32c783e72abfbd8ad8184d0257b9587`。
Source准入会核对该Worker/OM/日志和旧Admission。旧AIR、Manifest、OM、stdout/stderr及
失败外层报告不修改。ATC使用已冻结AIR的逐字节私有副本，沿用原精度及编译参数。

## 服务器命令：严格分步

```bash
cd /home/w00949577/z50058744/qwen3.5-4B-dflash
git pull --ff-only origin feature/ascend310p-custom-ops
PY=/home/w00949577/qwen35-dflash-run/venv-hf513-probe/bin/python
S=framework/custom_ops/draft_a_raw_sidecar
OLD=/home/w00949577/qwen35-dflash-run/a-only-raw-import-life-duXlSB
NEW=$(mktemp -d /home/w00949577/qwen35-dflash-run/a-only-raw-sidecar-XXXXXX)
printf 'NEW=%s\n' "$NEW"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PY" -B "$S/run_sidecar.py" admit \
  --admission-report "$OLD/admit/report.json" \
  --previous-worker "$OLD/resume/B/c16/report.json" --output-dir "$NEW/source"
```

预期 `A_RAW_SIDECAR_SOURCE_ADMITTED`，只读。若实际旧报告路径不同，传其真实路径，
不复制或伪造报告。

仅准入后构建B的新私有Host包：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PY" -B "$S/run_sidecar.py" build \
  --source-report "$NEW/source/report.json" --output-dir "$NEW/build"
```

预期 `A_RAW_SIDECAR_HOST_OPP_BUILT_NPU_NOT_RUN`。旧生成项目需仍存在；缺失就BLOCKED。
需检查build报告中 `B.opp_comparison.all_kernel_bytes_equal=true` 及允许的Host DSO差异。

仅Build通过后执行B-C16 ATC Host资源验证：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PY" -B "$S/run_sidecar.py" bc16 \
  --source-report "$NEW/source/report.json" --build-report "$NEW/build/report.json" \
  --output-dir "$NEW/bc16"
```

目标 `A_RAW_BC16_HOST_EVIDENCE_SIDECAR_MATCH`。报告包含三份日志身份、36/6/6计数、
六类实际资源与新旧OM比较。完整Draft执行、Smoke、Benchmark、R包均NOT_RUN。
此阶段通过后再规划剩余三份OM，不自动继续。

失败回传本阶段report的error/integrity_error，build/configure日志或
`bc16/B/c16/atc-streams/call-0/{stdout.log,stderr.log,host-evidence.jsonl}`。
不要只贴stdout或把损坏旧日志当成sidecar。本地证据见LOCAL_VALIDATION.md；真实服务器
新Host包及B-C16 sidecar验收仍为NOT_RUN。
