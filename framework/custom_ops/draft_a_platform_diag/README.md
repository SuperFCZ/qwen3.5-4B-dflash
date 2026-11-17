# ATC / ACLNN PlatformInfo 单 Shape 诊断

范围仅为 `DFlashGroupQuantLinear`，M16/K12800/N2560，Device 0。
这是独立 Host 诊断包，不是 8-Core 准入修复，不执行 Matmul，不重跑 104 Probes，
不重建四份完整 Draft OM。历史 Kernel、OPP、报告和 Golden 全部只读。

## 当前判断

用户回传：同一 Shape 在 ATC 为 8/8/0，在 ACLNN 为 7/7/0
（available/block/limit），ACLNN 实际计算及 Guard/Readonly 已完成。
这不是本工具的新设备结果。新诊断 ATC、ACLNN Host Query、NPU均 **NOT_RUN**。

当前不能确定差异来自 Runtime 平台配置、设备资源视图或虚拟化。
不能将“没有运行进程”“产品标称 8 核”或 `GetCoreNumAic()==8` 当作
Runtime `GetCoreNum()==7` 的修正依据，也没有证据支持“坏了一核”或“固定预留一核”。

**现阶段不得继续将 ATC8 与 ACLNN7 视为同一正式 Core 合同。**
不改变 ACLNN 已验证的实际 7-Core 调度；暂停新的统一 8-Core 数值/性能准入。
最终采用哪条正式主线，要等相同诊断 Host DSO 的两条平台视图和设备映射证据返回。
即使两条视图相同，本诊断也不授予新包 NPU 正确性 PASS。

## SDK 与初始化证据

完整路径、SHA256、行范围和服务器核验级别见 [sdk_evidence.json](sdk_evidence.json)。
本地根目录 `/usr/local/Ascend/cann-9.0.0`（Docker Linux ARM64）。

| 文件相对路径 | 源码依据 | 能证明 / 不能证明 |
|---|---|---|
| `aarch64-linux/asc/include/utils/tiling/platform/platform_ascendc.h` | 96 构造参数 `fe::PlatFormInfos*`；102/107/112 三种 Core getter；133 内存查询；140/141 SoC/Arch | 三种接口真实存在；不能仅凭声明推出 GetCoreNum 的内部计算公式 |
| `aarch64-linux/include/platform/platform_infos_def.h` | 40–47 资源字典、GetCoreNum/WithLock；60–70 setters 与独立 `core_num_` | 聚合 Core 字段与资源字典可以分别观察；不调用任何 setter |
| `aarch64-linux/include/platform/platform_info.h` | 45–64 通用、按设备、Runtime API；155–163 各缓存 | SDK具备不同资源视图；不是两次具体调用获得不同值的实证 |
| `aarch64-linux/include/acl/acl_rt.h` | 1194/1266/1335/2245 Context/SetDevice/GetDevice/CreateStream | 可观察原初始化路径，无需改变 Context 选择 |

platform_ascendc.h 的 SHA 已在历史服务器核验；另外三个文件当前为
`LOCAL_ONLY_NOT_SERVER_VERIFIED`。source 阶段须在服务器逐个匹配，失配直接 BLOCKED。
不上传 SDK 源码。`GetCoreNum` 等实现位于链接库，声明和局部字段不能代替运行时证据。
服务器新旧 Host ldd、ATC ldd、Runner ldd、实际映射和 SHA 分别记录，不把本地库当成服务器库。

原 `draft_quant/test/main.cpp:116–118` 使用
`aclInit → aclrtSetDevice → aclrtCreateStream`，没有显式 CreateContext。
新 Runner 保留该顺序，增加 `aclrtGetDevice`、`aclrtGetCurrentContext`：
SetDevice 后 Context 必须存在，CreateStream 后仍为同一 Context，Device 必须为0。
两次 GetWorkspaceSize 后再次核对，保存每个返回码；不通过改变初始化流程“修正”核数。

官方接口索引可辅助定位：[CANN 9.0 Platform API](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_00059.html)。
本实现的类型/签名依据仍是上述本地文件及服务器 hash gate。

## 独立包及同源比较

1. source 重新验证原 ATC Admission、Precheck/Build/104历史报告的身份，**不执行这些实验**。
2. 读取 H 的 `a-probes.json` 中已验收 `c16-fc`，绑定单 FC AIR、Deployment、旧 OM、
   ATC 二进制/命令、输入和权重/Scale hash。只接受 1 A/0 C/0 Audit 的固定 Shape。
3. build 从验收 B 生成配置复制项目，仅重建两个 CPU Host DSO，新增诊断记录；
   原 `GetCoreNum`、`LaunchBlockCount`、`SetBlockDim`、资源计算和数学保持逐字变换审计。
   不改历史源码。私有 OPP 中 Kernel .o/.json、ACLNN API/头文件和非 Host 文件必须字节相同。
   Host ldd 依赖须与原包一致。新 Runner 只链接私有 API，不调用 Execute。
4. atc 从冻结 **单 FC** AIR 的字节相同副本重新编译一个诊断 OM；不导出模型，
   不重建完整 Draft，不执行 OM。新旧命令只允许 model/output 路径迁移。
5. aclnn 同一包/同一 Host 源码，固定输入、两次 GetWorkspaceSize，检查 Device/Context、
   Workspace稳定、输入/输出GM Guard、Readonly、Poison未改变；同步空 Stream。
   **这是 Host query，不是 UB Guard、Kernel 或数值验证。**
6. compare 验证两报告绑定同一 Source/Build，逐项列出 getter、容量与原始资源字典差异。

诊断回调写入独立 `platform.jsonl`，记录 PlatformInfo 非空、原始 aggregate/locked Core、
GetCoreNum/Aic/Aiv、SoC enum、Arch、五类容量、实际 available/BlockDim/CoreLimit、Host workspace、
完整资源字典、实际 Host DSO (`dladdr`)。文件使用 O_APPEND/flock，每行一次完整提交；
写失败使 callback失败，无 stdout 回退。没有 `DFLASH_PLATFORM_DIAG_PATH` 时不写诊断。
旧六类 Host stdout 仍原样保留，但本实验不从被 Warning 污染的 stdout 推导平台信息。

每条观测必须来自已绑定新私有 Host DSO，两条路径实际加载的 Host DSO 文件身份必须相同。
ATC/ACLNN 的库搜索路径及设备选择环境在 Build 时冻结，后续漂移拒绝。Runner 保存 `/proc/self/maps` 和实际库的
alias、符号链接链、realpath、bytes/SHA；该记录是映射路径及当前文件身份，**不是映射页内容哈希**。
不将 maps dev 与 overlay stat dev 强行比较。额外私有 OPP、另一 Ascend 安装、deleted DSO被拒绝。
Driver库独立归类，不混入 Toolkit 身份。原始 stdout/stderr、报告、CMake、ABI、Kernel等身份保留。

## vNPU / 整卡直通证据

先只读 `npu-smi info -l`，确认逻辑 Device0 对应物理 `npu_id/chip_id`。
`device_readonly.py` 不猜测映射，显式接收两个值；捕获 info、help、
`info -t info-vnpu`、返回码、工具身份、可见设备节点与 mdev目录、设备选择环境变量。
官方有此[只读 vNPU 查询](https://www.hiascend.com/developer/techArticles/20251212-1)，
但本机 24.1.rc3 是否支持以实际返回码为准；不支持保留原始错误，状态仍 UNRESOLVED。
在容器内看到“无vNPU条目”也不足以独立证明宿主整卡直通；如仍不明确，下一步仅需管理员
确认该逻辑/物理映射及设备分配方式，禁止修改虚拟化配置来尝试消除差异。

## 服务器分阶段命令

必须使用新目录，每步 returncode=0 且 status 为下表值才进入下一步；不自动执行全流程。
保持既有通过导入检查的 hf513 解释器与 CANN 环境。不重跑完整导出/Probes。

```bash
cd /home/w00949577/z50058744/qwen3.5-4B-dflash
git pull --ff-only origin feature/ascend310p-custom-ops
export CANN_ROOT=/usr/local/Ascend/cann-9.0.0
source "$CANN_ROOT/set_env.sh"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
PY=/home/w00949577/qwen35-dflash-run/venv-hf513-probe/bin/python
S=$PWD/framework/custom_ops/draft_a_platform_diag
R=$(mktemp -d /home/w00949577/qwen35-dflash-run/a-platform-diag-XXXXXX)
echo "$R"
SIDE=/home/w00949577/qwen35-dflash-run/a-only-raw-sidecar-kV7ZFn
ADMIT=$("$PY" -B -c 'import json,sys,hashlib; from pathlib import Path; r=json.loads(Path(sys.argv[1]).read_text()); assert r["status"]=="A_RAW_SIDECAR_SOURCE_ADMITTED"; x=r["admission_report"]; p=Path(x["path"]); b=p.read_bytes(); assert p.is_absolute() and len(b)==x["bytes"] and hashlib.sha256(b).hexdigest()==x["sha256"]; print(p)' "$SIDE/source/report.json")
```

只读身份阶段：

```bash
"$PY" -B "$S/run_diag.py" source --admission-report "$ADMIT" --output-dir "$R/source"
```

仅构建新 Host DSO 与 Host-only Runner，原 Kernel .o复制并核对：

```bash
"$PY" -B "$S/run_diag.py" build --source-report "$R/source/report.json" --output-dir "$R/build"
```

两个独立进程，先单 FC ATC，再 ACLNN Host query（不执行计算）：

```bash
"$PY" -B "$S/run_diag.py" atc --source-report "$R/source/report.json" --build-report "$R/build/report.json" --output-dir "$R/atc"
```

```bash
"$PY" -B "$S/run_diag.py" aclnn --source-report "$R/source/report.json" --build-report "$R/build/report.json" --output-dir "$R/aclnn"
```

```bash
"$PY" -B "$S/run_diag.py" compare --source-report "$R/source/report.json" --build-report "$R/build/report.json" --atc-report "$R/atc/report.json" --aclnn-report "$R/aclnn/report.json" --output-dir "$R/compare"
```

设备映射单独收集；以下先查看列表，再将实际物理ID填入（不假设0/0）：

```bash
npu-smi info -l
# 确认对应关系后，替换 PHYSICAL_NPU_ID / PHYSICAL_CHIP_ID：
"$PY" -B "$S/device_readonly.py" --npu-id PHYSICAL_NPU_ID --chip-id PHYSICAL_CHIP_ID --output-dir "$R/device"
```

| 阶段 | 精确成功状态 | 当前实际状态 |
|---|---|---|
| Source | PLATFORM_DIAG_SOURCE_BOUND | Server NOT_RUN |
| Build | PLATFORM_DIAG_OPP_RUNNER_BUILT | Server NOT_RUN |
| 单 FC ATC | PLATFORM_DIAG_ATC_OBSERVED | NOT_RUN |
| ACLNN query | PLATFORM_DIAG_ACLNN_HOST_OBSERVED | NOT_RUN |
| 比较 | PLATFORM_DIAG_COMPARISON_RECORDED | NOT_RUN |
| Kernel/NPU/完整Draft/Benchmark | 没有执行入口 | NOT_RUN |

若失败，只回传该阶段 `report.json` 的 status/error/integrity_error、`traceback.txt` 和
错误所指日志末尾80行。成功时回传 compare终端输出、ACLNN execution.json、device查询结果；
原始资源字典可保存在服务器，不需上传SDK源码。任何失败不修改原报告，改用新的阶段目录。

## 本地验证

macOS及Docker ARM64：14项CPU合同回归通过；包含7/8分离、未授权Core8、重复callback漂移、
缺失/截断JSON、Device/Context/Workspace失败、DSO漂移/别名、CLI门控、C++并发日志及写失败。
这不是硬件同步或NPU正确性证明。

Docker真实CANN9.0.0：两个Host DSO +生成的API wrapper +新Runner编译/链接通过，
`ldd -r`无未解析符号；未编译设备Kernel，未运行ATC/ACLNN。
本地报告 `/work/build/platform-diag-zv8ddi85/report.json`，本地库级别
`LOCAL_ONLY_NOT_SERVER_VERIFIED`。SDK接口编译证明和真实设备资源观察严格分开。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s framework/custom_ops/draft_a_platform_diag/test -v
# Docker已存在的生成项目仅作Host编译模板；不提交这些SDK生成文件：
python3 -B framework/custom_ops/draft_a_platform_diag/local_compile.py \
  --template /work/build/a-raw-sidecar-host-le2oo1y_/CustomOp --output-parent /work/build
```

## Source ATC wrapper 依赖检查修复

服务器 `a-platform-diag-LnLfd1/source` 原失败保留为 BLOCKED。
原因是对 Shell wrapper 执行了 `ldd`，并非已经证明 ATC ELF 缺库。
2026-11-17 用户只读回传确认：

| 文件 | 真实路径 | Bytes | SHA256 | Mode |
|---|---|---:|---|---|
| ATC wrapper | `/usr/local/Ascend/cann-9.0.0/aarch64-linux/bin/atc` | 1233 | `90aa69b5e18a19e65d1f934f1cb68089f9d8092a4dc00e2fb1152ca2b7c15f35` | 0555 |
| ATC ELF | `/usr/local/Ascend/cann-9.0.0/aarch64-linux/bin/atc.bin` | 670800 | `a32df12914369e4f41a60deff26e8b641c877c070f02ecebfacb121aa5703170` | 0555 |

wrapper别名 `/usr/local/Ascend/cann-9.0.0/bin/atc` 与真实路径、符号链接链都保存。
该 wrapper 与本地 SDK 文件完整SHA一致，因此本地读取的控制流可作为此服务器脚本的依据：

1. 第1行解释器 `/bin/bash`。
2. 第13行通过 PATH 调用 `realpath "$0"`。
3. 第14行通过 PATH 调用 `dirname`；第14–15行 `cd`、`pwd` 为 Bash builtin。
4. 第20–21行分别在 LD_LIBRARY_PATH、PYTHONPATH 前加入实际 package目录的 lib64、python/site-packages。
5. 第23行调用 `${PKG_PATH}/bin/atc.bin "$@"`。

新 `atc_dependency_identity.py` 只识别上述完整文件 SHA 对应的已审查流程，
不实现猜测性的通用Shell解析器，也不执行wrapper进行探测。未知脚本、动态选择目标、
非ELF helper或无法完整确定的调用链一律 `dependency_audit=INCOMPLETE` / BLOCKED。
Bash解释器、realpath、dirname、atc.bin四个ELF分别执行ldd；前三者使用继承的库环境，
atc.bin使用wrapper真实导出的库环境。不会对1233B脚本本身执行ldd。
直接ELF的入口仍按一个ELF审计。

[Bash官方启动合同](https://www.gnu.org/software/bash/manual/html_node/Bash-Startup-Files.html)
允许 BASH_ENV 引入额外代码；因此存在相关启动/导出函数/loader覆盖或依赖CWD的库路径时，
本工具不宣称静态链完整，不静默清除环境变量后继续。
[ldd的范围](https://man7.org/linux/man-pages/man1/ldd.1.html)是共享对象依赖；本报告不证明
ATC程序内部的动态启动行为、Python导入或按需dlopen完整集合，也不执行编译器来补齐它们。

每个ELF保存角色、原alias、realpath、bytes/SHA、mode、可执行性、符号链接链、
ldd命令及returncode、原始stdout/stderr身份和所有解析到的库身份。
任何 `not found`、非零返回、非空stderr或未识别依赖行均BLOCKED，不跳过错误行。
后继报告读取及Source finally均重新核验这些文件身份、权限、链接和完整调用链库存。
原 `raw_common.dependencies/verify/verify_tree` 保持不变。

新证据：

- `source/report.json`: `atc_dependencies` 与 `atc_dependency_report`。
- `source/atc-dependencies/report.json`: 成功状态 `ATC_STATIC_ELF_DEPENDENCIES_BOUND`。
- `source/atc-dependencies/{0-bash_interpreter,1-realpath,2-dirname,3-atc.bin}.{stdout,stderr}.log`。

原 Source 成功状态仍为 `PLATFORM_DIAG_SOURCE_BOUND`。
本修复改变了本目录Source Hash，因此必须新目录重跑Source；历史Precheck/Build/104 Probes
仅按原身份只读复核，不需要重新执行。本轮在Source之后停止，不运行Build/ATC/ACLNN。

```bash
cd /home/w00949577/z50058744/qwen3.5-4B-dflash
git pull --ff-only origin feature/ascend310p-custom-ops
export CANN_ROOT=/usr/local/Ascend/cann-9.0.0
source "$CANN_ROOT/set_env.sh"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
PY=/home/w00949577/qwen35-dflash-run/venv-hf513-probe/bin/python
S=$PWD/framework/custom_ops/draft_a_platform_diag
OLD=/home/w00949577/qwen35-dflash-run/a-platform-diag-LnLfd1
ADMIT=$("$PY" -B -c 'import json,sys,hashlib; from pathlib import Path; r=json.loads(Path(sys.argv[1]).read_text()); x=r["admission_report"]; p=Path(x["path"]); b=p.read_bytes(); assert p.is_absolute() and len(b)==x["bytes"] and hashlib.sha256(b).hexdigest()==x["sha256"]; print(p)' "$OLD/source/report.json")
R=$(mktemp -d /home/w00949577/qwen35-dflash-run/a-platform-diag-atcfix-XXXXXX)
echo "R=$R"
"$PY" -B "$S/run_diag.py" source --admission-report "$ADMIT" --output-dir "$R/source"
```

回传终端状态即可；若失败，再回传 `source/traceback.txt` 和
`source/atc-dependencies/report.json` 的 error及对应ldd日志。
不要修改或补齐旧文件。

修复回归：macOS / Docker ARM64均33项CPU测试（原14项 + 新19项）。
覆盖直接ELF、wrapper、四ELF库存、未知脚本、helper缺失、目标/库SHA漂移、mode与别名漂移、
缺库、非零ldd、stderr、wrapper原SHA不变。源码/资源/Kernel合同未改。
本地实际 `atc.bin` 为724048 B、SHA `4d2535f8f27d239d7b26f748aaa14eda392933f2fe55751fb2c4805f2e7a3c4e`，
与服务器不同；服务器pin拒绝本地替代属于预期结果，不安装其他SDK或放宽pin。
本地真实ELF的ldd检查只标记 `LOCAL_ONLY_NOT_SERVER_VERIFIED`；ATC、ACLNN、NPU、Benchmark均NOT_RUN。
