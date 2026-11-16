# A-only 专用 ATC stdout/stderr 隔离与冻结 AIR Resume

仅本目录新增。原 `draft_a_raw`、`draft_a_raw_env`、`draft_ac/test/run.py`、共享 compiler、
Host/Kernel、Header、OPP、Golden、Native Source Lock 与历史报告全部不修改。
本地不具备服务器资产或NPU；新 ATC/日志分离的服务器验收仍 **NOT_RUN**。

## Admit 的接口相对路径修复

`a-only-raw-atc-split-M0FLNV` 的失败栈确认：外层Export Report扫描相对身份为0，
报错实际来自 `old_worker()` 中的 `c.verify_tree(model)`，即
`.../export/B/c16/report.json` 的接口记录，不是 `c.verify_tree(outer)`。

| Worker JSON 字段 | 声明 | 真实基准 |
|---|---|---|
| `$.interfaces.source_lock` | `source-lock.json`，72988 B，SHA `a60acc77d90f953e417d723443b54a23a311ea6eb8f1b29c273d0d66297ff6c2` | Worker目录；服务器回传已确认匹配 |
| `$.interfaces.native_air` | `air-manifest-w8a16-chunk.json`，1390805 B，SHA `77a072e7e95b4ed8171bbfbf74d313fed2f55a63779ddab094923fec3427fc02` | 已绑定Native Deployment的AIR目录，须再次按声明验证 |

`draft_ac/test/draft_only.py:77–86` 分别将 source audit 相对于接口报告目录、Native AIR
相对于AIR自己的目录记录。前者是诊断审计报告，不能当作Native Base的`SOURCE_LOCK.json`。

新增 `raw_atc_interface_identity.py` 只处理这两个已知字段。在内存深拷贝中：

- Worker报告位置、request.output、model.output_root必须指向同一根；冲突直接BLOCKED。
- Native AIR必须同时符合H已绑定身份与Native Deployment中的air_manifest；不同路径
  即使同bytes/SHA，也不挑选其中一个，直接BLOCKED。
- 保留原声明bytes/SHA，使用明确owner核验真实文件；不从CWD或其它候选目录查找。
- 与未修改的`draft-interface.json`生产者内容交叉核对，再记录绝对身份和上下文绑定。
- 原始Worker/Interface/Source Lock/AIR文件不写入，不创建替代文件或软链接。
- 未知相对身份继续由未修改的`raw_common.verify/verify_tree`拒绝。

独立新worker也在写入自己的新报告前应用同一内存规范化，避免后续三个新Export重复
触发同一问题；原`draft_a_raw/worker.py`及共享接口生产者不修改，原接口检查不跳过。
该修复改变ATC辅助目录的源码清单，因此旧BLOCKED Admit保留，必须新目录重新Admit。
原Precheck/Build/104 Probes依旧只按原身份规则重核，不重新执行。

## 原因与接口选择

- Host 的 CUBE 记录由 `draft_quant/op_host/d_flash_group_quant_linear.cpp:288` 写 stdout。
- 共享 `compiler.py::_default_runner` 合并 stderr→stdout，Warning 可以插入长 JSON。
- `draft_ac_export_pipeline.compile_air_bundle(manifest, **kwargs)` 保留 A/C audit 并透传
  `runner`；原 `draft_ac/test/run.py::export_one` 没有 runner 参数。

本目录 `split_export.py` 是该函数的有界副本：只增加专用 Runner 参数和显式
`validated_runtime_input_abi(..., allow_test_double=False)`。共享编译器收到自定义runner时
原有测试辅助开关不能成为真实导出的绕过入口。原 dtype/precision/deterministic/
GE/Const/ABI审计不变。

独立 `worker.py` 由原Raw worker做确定的源码变换；`worker_derivation.py`逐字核对两份
派生源码，允许日志分离、身份记录/验证，不允许数学、权重、导出配置或Interface删改。
没有运行时 monkeypatch 共享模块，没有修改全局ATC行为。

## 日志与严格资源合同

`SplitAtcRunner` 使用两个独立二进制文件描述符运行真实ATC，不合并stdout/stderr。
返回给共享编译器的CompletedProcess.stdout只含原stdout。两个原始文件均记录path、
bytes、SHA256；失败也保留。stderr不用于Host资源解析，任何Warning都不删除或过滤。
编译器自己的stdout副本须逐字节匹配原stdout。

Worker Report中：

- `atc_log`：原始纯stdout，唯一资源输入。
- `atc_stderr`：完整原始stderr。
- `atc_captures`：命令、cwd、ATC二进制身份、returncode与两份日志身份。
- `compiler_stdout_log`：共享编译器保存的stdout副本。

外层报告的 `worker_evidence` 同样保留日志和子报告身份，包括Worker/资源验证失败时。
最终 `models.B/R.16/64` 均含两份日志和严格资源证据。

`collect_stdout` **先严格检查，再调用未修改的 raw_resources.collect**：

- 六个完整callback，顺序严格为LAUNCH→GATE_PREFETCH→GATE_COPY→SCALE→K_TILE→CUBE。
- 六条CUBE JSON字段齐全、类型正确、没有重复key。
- 只允许已知Host记录；未知DFLASH记录、截断/交错/重复shape全部BLOCKED。
- 原六种shape覆盖、launch_evidence、资源容量和已验收Gate计划全部保留。
- B/R Gate差异只允许原Raw合同；非Gate计划必须相同。

不修补JSON、不删除Warning后重试、不忽略CUBE、不用默认资源。若CANN把Warning写进
stdout本身，或多个Host回调在stdout交错，本入口仍BLOCKED，不能把分离FD视为自动PASS。

## 身份续接和旧资产

新目录不属于旧 `raw_common.sources()` 或环境辅助目录的源码清单。准入使用原
`prior()`、`verify_build()`、104项Projection证据验证器，要求旧源清单逐文件一致、
准确PASS、前序链一致、文件/SDK/OPP/ELF/Runner/Header/输入/Golden仍有效。
新增源码独立记录为 `continuation_sources`，没有Source Hash豁免。
因此**无需无条件重跑Precheck、Build或104 Probes**；若实际旧源身份不一致，仍停止，
不能修改报告或把本文中的Git基线名字当作字节身份。当前报告失败的Source需先具体审查，
再决定重跑；本工具不自动续接未知源码变更。

Admission只支持“原Export外层BLOCKED、B-C16 Worker已编译成功”的情况：

1. 重核原三阶段证据、B-C16请求的source/evidence/device/variant/gear。
2. 要求Worker状态、import boundary、interfaces PASS；核对ABI、26 A/0 C/0 Audit、
   逻辑IR、AIR/Deployment相互绑定及全部Payload SHA，OM身份和选定私有B环境。
3. 原ATC命令必须与Deployment一致，指向该AIR/OM；ATC可执行文件与既有tool身份一致，
   保留原shape、SoC、precision与deterministic选项。
4. 原外层在collect抛异常前未保存Worker SHA，所以本次Admission首次封存该Worker
   文件哈希，并核对其原有artifact引用；不伪称已有父报告提供了该哈希。终端打印封存值，
   后续Resume逐项重核。若已有独立保存的Worker SHA，应与本次打印值核对。

原Worker、OM、AIR、Manifest、混合日志和失败报告不改变，不改旧BLOCKED状态。

## ATC-only Resume 的具体行为

将冻结AIR Manifest和其Payload逐文件**复制**到新私有目录，大小/哈希完全相同：
不硬链接、不软链接、不重新TorchAir导出、不实例化模型。额外磁盘占用由Payload大小决定。
原FileConstant绝对目标继续保留原位置及哈希约束；相对文件也有同布局的字节镜像。

使用原 compile_air_bundle 审计并重新ATC。Runner执行前要求新旧完整命令只差
`--model`的字节镜像路径和`--output`的新目录，其他参数完全一致。
ATC工作目录是新AIR目录，避免在旧AIR目录写临时文件。
ATC版本文本复用旧Deployment已记录值，并重新核对实际ATC二进制SHA；不额外运行一个
混流的`atc --version`。新Deployment写到新目录，旧Manifest不覆盖。

输出新旧OM的bytes/SHA比较：同哈希只说明整文件相同；不同哈希记录
`DIFFERENT_BINARY_UNATTRIBUTED`。路径/编译非确定性可能影响文件，不能据此推断指令差异、
数值等价或性能收益。后续真实Smoke仍必要，本轮不运行。

B-C16得到六条可信CUBE并通过资源准入后，才独立导出/编译B-C64、R-C16、R-C64。
这些新进程使用专用Runner，仍执行原Native Source Lock、Interface与权重/GE审计。

## 服务器分阶段命令

使用H真实成功的venv及现有CANN shell。以下不安装组件，也不重建OPP。

```bash
cd /home/w00949577/z50058744/qwen3.5-4B-dflash
git pull --ff-only origin feature/ascend310p-custom-ops
PY=/home/w00949577/qwen35-dflash-run/venv-hf513-probe/bin/python
X=/home/w00949577/qwen35-dflash-run/a-only-raw-export-hf513-Df3swI
NEW=$(mktemp -d /home/w00949577/qwen35-dflash-run/a-only-raw-atc-split-XXXXXX)
S=framework/custom_ops/draft_a_raw_atc
printf 'NEW=%s\n' "$NEW"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PY" -B "$S/run_atc.py" admit \
  --failed-export-root "$X" --output-dir "$NEW/admit"
```

预期 `A_RAW_ATC_RESUME_ADMITTED`，该步只读证据，不执行ATC或设备代码。
自动定位只查看根目录及export/resume/export等四个明确报告位置；若不唯一，使用
`--failed-export-report /实际原Export/report.json`替代root参数，不扩大搜索或猜测。

准入后执行B-C16 ATC-only：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PY" -B "$S/run_atc.py" resume \
  --admission-report "$NEW/admit/report.json" --output-dir "$NEW/resume"
```

预期 `A_RAW_BC16_STDOUT_RESOURCES_MATCH`。先运行import-only环境检查，再执行ATC；
完整Draft不执行。检查 `$NEW/resume/B/c16/atc-streams/call-0/stdout.log`、`stderr.log`，
及外层 `report.model.host_resources.stdout_schema` 和 `report.model.om_comparison`。

仅通过后补齐另外三图：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PY" -B "$S/run_atc.py" complete \
  --admission-report "$NEW/admit/report.json" --resume-report "$NEW/resume/report.json" \
  --output-dir "$NEW/complete"
```

预期 `A_RAW_FOUR_STDOUT_OMS_COMPILED_DRAFT_NOT_RUN`。最终报告明确绑定全部四份OM、
两路日志、资源比较、旧三阶段身份和新增源码；这是一份独立续接报告，不能直接冒充旧
Export报告或跳过下一阶段Smoke准入。没有自动Smoke/Benchmark/生产部署入口。

失败时保留目录，回传report的status/error/integrity_error、对应Worker report及
**两份**ATC日志末80行。不要只贴合并Worker日志。stdout损坏仍BLOCKED。
当前真实服务器的新日志分离、ATC Resume和四图完成状态均为 **NOT_RUN**。
