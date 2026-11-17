# 正式独立 B8/R8 A-only Draft 实验

目标：完整五层W8A16 Draft OM的8-Core配对延迟。只新增本目录，历史7-Core、Sidecar、
Kernel、Golden、Native Source Lock和生产部署不修改。所有服务器新阶段仍NOT_RUN。

## 新资源合同（不与7-Core做Resource Equality）

- 平台实际 `available_cores=8`、实际 `block_dim=8`、`core_limit=0`，每个shape都强制核对。
  Kernel/Host不硬编码8；平台返回7/9或Block不等于8立即BLOCKED。
- 复用canonical pipeline/scale/K Tile/UB/L1/L0验证器，独立生成当前8-Core准入。
  不调用旧 `accepted_gate()`，也不删除Core字段后与历史计划比较。
- Gate：B8 off/loop/raw_banks1，R8 raw/loop/raw_banks2；原Raw规定的UB差异保留。
  非Gate保持原pipeline行为（例如Down原有raw双缓冲，两边都保留），不强行把全部B8
  shape改成单bank。每个实际计划先通过launch_evidence及canonical检查。
- B8/R8除了Gate规定的Raw资源变化，其余资源/Workspace一致；非Gate所有资源一致。
- 按每行记录的起点/步长/数量展开assignments，严格证明每个N64 Tile恰好覆盖一次。
  Gate 304 Tile，8核每核38个；其它shape同样展开核对。该证明是Host计划及已绑定Kernel
  调度合同，不能冒充设备时间线。

历史7-Core报告只提供冻结输入、Golden、源身份和独立历史事实。它们不是新8-Core
Resource Equality或NPU正确性参考，新性能统计也不含其样本。

## B8-C16直接复用

默认原Sidecar根：
`/home/w00949577/qwen35-dflash-run/a-only-raw-sidecar-kV7ZFn`

新准入核对Source/Build链、新Host OPP、Kernel字节相同证明、Worker Report、AIR、全部
Payload、Deployment、ATC命令/工具身份、三份日志、26 A/0 C/0残留WeightQuant与15/11 ABI。
再从原Sidecar重新计算全部六类8-Core资源，**不用旧7-Core equality结果**。

用户冻结身份：

- OM：3090431253 B，`f67c41cda787fc552a6c35c5ed7dfde26a98f49e8b8cc8d40b52f383f03e19af`。
- Sidecar：7810 B，`2e021c01a46db3c41864cef5acc3b06893695ae81d6abefa4836a925c36a66e4`。

原 `bc16/report.json` 必须继续保持BLOCKED。新准入通过仅封存新B8身份，不改旧报告。
B8-C16和B8私有OPP直接引用原文件，不复制替换、不重新ATC。若文件/配置身份变化，先
BLOCKED并明确差异；不自动接受当前文件或偷偷重编译后冒充复用。

## 构建与新的104项Probe

Build复用已验收Sidecar Host派生器，只构建尚缺的R8私有Host包，保留原R设备Kernel字节。
核对B8/R8生成Header只差Raw宏，Host源码相同，所有其余配置相同。两套ACLNN Probe Runner
分别链接自己的私有OPP；完整OM Runner沿用已验收的原严格Golden Runner及其依赖身份。

Probes是新NPU阶段：B8/R8 × C16/C64 × 26投影 = **104项**，每项两次未计时执行。

- 复用原Frozen A4输入、权重/Scale及Native Golden，全部输出逐位比较。
- 原Readonly、GM Guard、Poison、Repeat、同步、无CPU fallback检查保留。
- 每次单独进程及独立sidecar。两个GetWorkspaceSize允许一份缓存Host回调或两份完全
  相同回调；缺失/交错/漂移拒绝。转换有效JSON为原日志视图后使用原严格数值/安全验证器。
- 每个Probe的实际Core、Block和所有资源严格验证；平台变化不静默降级。
- B8/R8实际Workspace也要相等。同shape不同layer计划不一致拒绝。
- 复用B8-C16 OM的资源必须与新B8-C16六类Probe资源相同。

旧104项只作只读溯源，不被改标为8-Core PASS。新失败会立即阻止后续阶段。
本次只运行受影响的Core8 CPU合同测试，不重跑无关历史CPU套件。

## 四图、整图Smoke与正式计时

104项通过后：复用B8-C16，仅新导出B8-C64、R8-C16、R8-C64。新worker由既有导出worker
严格派生，只改用Sidecar ATC transport；不改Interface/数学/权重/导出策略。
四图资源必须分别与新8-Core Probe计划相等，再做B8/R8同gear资源与逻辑IR/权重比较。

Smoke：两档B8/R8各两次独立Poison检查，进程内逐位核对全部15 Token IDs和10个完整KV，
含未更新区域；Readonly/Guard/Repeat/同步检查保留。B8/R8模型实际Workspace须一致。

Benchmark：三轮交错，C16为B8→R8、R8→B8、B8→R8；C64反序平衡。
每进程2次前检、5 warmup、30正式Execute+Sync、Timed Tail、独立Poison Postcheck。
每档每版90新样本，两档共360样本。加载/上传/回读/比较/Host Guard扫描/文件写入不计时，
内核原有工作全部保留。每档报告Median/P95/Min/Max、每轮倍率、原始样本、实际Workspace。
七核历史样本不进入新倍率分母；无调用混合比例不造跨档总体倍率。Profile/Decode NOT_RUN。

## 手动分阶段服务器命令

```bash
cd /home/w00949577/z50058744/qwen3.5-4B-dflash
git pull --ff-only origin feature/ascend310p-custom-ops
export MODEL_PYTHON=/home/w00949577/qwen35-dflash-run/venv-hf513-probe/bin/python
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
S=framework/custom_ops/draft_a_raw_core8_flow
R=$(mktemp -d /home/w00949577/qwen35-dflash-run/a-only-raw-core8-XXXXXX)
printf 'CORE8_ROOT=%s\n' "$R"
bash "$S/server_stage.sh" admit "$R"
```

先回传admit/report.json状态；此步只读，无编译、无NPU。
随后每条都单独手动执行，上一条准确PASS后才继续；没有自动all或生产发布入口。

| 命令 | 精确成功状态 |
|---|---|
| `bash "$S/server_stage.sh" admit "$R"` | A_RAW_CORE8_ADMITTED_BC16_REUSED |
| `bash "$S/server_stage.sh" build "$R"` | A_RAW_CORE8_PACKAGES_RUNNERS_READY |
| `bash "$S/server_stage.sh" probes "$R"` | A_RAW_CORE8_104_PROBES_MATCH |
| `bash "$S/server_stage.sh" export "$R"` | A_RAW_CORE8_FOUR_OMS_COMPILED |
| `bash "$S/server_stage.sh" smoke "$R"` | A_RAW_CORE8_TOKEN_KV_MATCH |
| `bash "$S/server_stage.sh" benchmark "$R"` | A_RAW_CORE8_PAIRED_MEASURED |

Probes/Smoke/Benchmark会执行真实NPU；Export会做原有设备Tensor准备/ATC，不执行完整Draft。
本地不会替用户运行这些阶段。任一步失败保留新报告、日志、实际输出并停止，不覆盖原资产。
错误回传本阶段status/error/integrity_error及错误指向日志末80行；不重建Golden。
所有Core8文件另有Source Hash清单，前序链/模型/包/Runner/实际结果每阶段重核。

工作区中已有的未跟踪 `draft_a_raw_core8/` 草稿未使用、未修改；本目录不依赖其7→8比较逻辑。
