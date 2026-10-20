# Gate/Up Simulator：隔离最小验证入口

本地只有 CPU/源码准备，**真实 CANN 仿真编译、Simulator 执行、热点解析均 NOT_RUN**。
不沿用历史 NPU PASS 宣称新仿真已通过；所有输出保留 `npu_correctness`、
`hardware_performance`、`full_draft_token_kv=NOT_RUN`。

## 已确认的接入信息与选择

用户提供的 receiver：Linux aarch64 Docker，CANN 9.0.0，MindStudio Ops Profiler
26.0.0 独立安装。`msopprof simulator --help` 已成功：支持 soc-version、output、
kernel-name、launch-count、core-id、timeout、PipeUtilization。`--dump` 明确只适用
A2/A3，本入口不使用。工具默认路径：

- `/home/w00949577/tools/msopprof-26-isolated/tools/msopprof/bin/msopprof`
- `/usr/local/Ascend/cann-9.0.0/tools/simulator/Ascend310P3/lib`
  实际解析为 `.../aarch64-linux/simulator/dav_2002/lib`。
- 实际文件 `libruntime_camodel.so`、`libnpu_drv_camodel.so`（另有 pvmodel，不使用）。

既有 C++ Runner 是 AscendCL/ACLNN 路径，链接 ascendcl/cust_opapi/acl_op_compiler/
nnopbase；它不是已验证的 simulator executable。原 `.o` 是否带调试段不能从历史
NPU PASS 推断。本入口先读取两个已验收构建的 ELF、原始 JSON、生成 CMake、SDK
版本与库架构，记录 SHA256，再创建独立工程。

选择继续通过 ACLNN 和原 Host Tiling 进入真实 Kernel，不手工编造 TCubeTiling 或
Matmul 的 system workspace，也不直接用 `--config` 猜测裸 `.o` 参数布局。
CMake 在新 Runner 中显式链接实际存在的 runtime_camodel、npu_drv_camodel 完整路径，
使用 NO_DEFAULT_PATH 和 no-as-needed；不搜索真实 npu_drv，不创建仿真目录内的假别名。
采用官方使用场景中指定的 camodel preload 路径，LD_LIBRARY_PATH 先放 Simulator 与
新 OPP；旧 OPP 不作为运行时 fallback。

参考：[官方使用场景](https://github.com/Ascend/msopprof/blob/master/docs/en/user_guide/msopprof_usage.md)、
[Simulator 指南](https://github.com/Ascend/msopprof/blob/master/docs/en/user_guide/msopprof_simulator_user_guide.md)、
[26.0.0 参数参考](https://github.com/mindstudio-docs/26.0.0/blob/main/msopprof/docs/zh/user_guide/msopprof_user_guide.md)。
上游 master 可能更新，实际 argv 以 receiver 26.0.0 保存的 help 检查为准，不能由其他芯片
示例推断当前设备已兼容。

## 隔离构建与兼容性关卡

- 绑定用户给出的已通过 Off/Raw pair：
  `/home/w00949577/qwen35-dflash-run/gate-raw-prefetch-reverse-20261020-025853/suite.json`。
  检查 stage/config/header/原 OPP/runner、layer-0 bitwise 与同一 A5.1 Capture/Native
  身份。原 Kernel/Host 源码必须与已验收 config 内 SHA256 一致。
- 输出必须是新目录；默认 `umask 077`。不写原 OPP、原 Runner、Golden、msopprof 安装
  或 Simulator 目录。接受的源文件、二进制和元数据在退出时再次校验。
- msopgen 仍以 `ai_core-Ascend310P3` 生成工程；由 SDK 选择对应设备 ISA，不把
  `dav_2002` 目录名猜写成另一个版本的编译器选项。
- 将未改动的原 Kernel/Host 文件复制到新工程，保留 Loop/K256/Broadcast、原 core cap
  和 Off/Raw 配置。仅在生成的 op_kernel CMake 注入 `-g`，保留原优化等级，不使用
  `-O0`。仅接受 documented add_ops_compile_options / npu_op_kernel_options hook；
  不存在就阻断。安装后 ELF 必须确实含 `.debug_info` 和 `.debug_line`。
- 从原 main.cpp/runner_common.h 生成**私有单次调用副本**：原 ABI、guard、输入只读、
  workspace 查询不变；只把两次调用改为一次，禁用 warmup/计时参数，并改成明确的
  SIMULATOR_COMPLETED 标签。原文件逐字节不变。
- Runner 的 `--link-check` 不调用 ACL；先检查 `/proc/self/maps` 中两个模型库，核对
  rtMalloc/rtMemcpy/rtStreamSynchronize/rtKernelLaunch 的来源。只接受 camodel 或已指定
  profiler 安装根下的拦截器；普通硬件 runtime/未知库直接阻断。实际运行再次检查，
  保存 process-maps 与 runtime-symbols。该检查不是全部 SDK API 兼容性的证明。
- 编译、链接或仿真失败/超时均保存 phase、argv、环境、日志，状态 BLOCKED。超时只
  终止此命令自己的进程组，不切换芯片、驱动、CPU 替代算法或重试另一套不明配置。

## 三个显式阶段

### `--stage prepare`

只做 help/SDK/ELF/CMake/evidence 只读审查，不构建、不初始化 ACL、不运行 Simulator。
报告 PREPARED_NOT_EXECUTED 只说明检查完成，不是可执行兼容性 PASS。

### `--stage smoke`（默认、首轮推荐）

新建 Off debug OPP 与 simulator Runner，运行 M16/K256/N64 的单次工具链样本。
样本是新目录中的 signed INT8 NZ、正有限 GN FP16 scale 和 basis X，包含 K32/K128/N16
边界；与原 Gate Capture 无关，不截断、重排或覆盖原权重。输出比对仅为 Simulator
对这个明确 CPU basis 结果的诊断，不是 NPU 数值验收。

此形状只运行一个 N64/K256 tile，Raw Prefetch 精确 Gate predicate **不触发**。
因此 smoke 成功不能证明 Raw Prefetch 或完整 Gate 的仿真正确性。

```bash
bash framework/custom_ops/draft_quant/simulator/run.sh --stage smoke --output-dir "/home/w00949577/qwen35-dflash-run/gate-simulator-smoke-$(date +%Y%m%d-%H%M%S)"
```

默认工具/Simulator/A5.1/prefetch evidence 路径就是以上用户确认路径，可用对应参数显式
覆盖。首轮请回传 simulator-report.json、simulator-analysis.md 与失败 phase 的日志。
`--timeout-minutes` 默认 5；这是仿真运行上限，不是 Kernel 耗时目标。

### `--stage gate`（smoke 真实完成后）

在另一个新目录构建 Off/Raw 两个 debug OPP；各自先重新通过 smoke，然后使用原始
C16 layer-0 的完整 X、INT8 NZ、Group128 Scale 拷贝（SHA256 必须与原文件相同），
完整执行 M16/K2560/N19456 一次，比对同一冻结 Native 输出。

`--core-id=0` **仅限制解析核心**；不修改 blockDim、不限制 Kernel 的 N/K 循环，不缩小
shape、不增删 PIPE_ALL/Event，不修改数学或原始输入。完整 Gate 仿真可能很慢，可能
超过默认上限；不会用残缺 dump 当作完整完成。

## 输出与关键 Tile 分析边界

每个成功阶段要求工具实际生成且可解析的：

- `profile/**/trace.json`：指令流水图；
- `profile/**/*code_exe*.csv`：代码热点；
- `profile/**/*instr_exe*.csv`：指令明细。

同时核对 profiler 保存的 `.o` 是否匹配新 debug OPP 的 SHA256。缺少匹配二进制、
trace、热点或指令文件均标为不完整并阻断，不生成空占位文件冒充成功。
保存 SDK/原与新二进制/CMake/源码/私有 Runner、输入与期望值的 hash，实际 launch plan、
guards/readonly、Simulator bitwise match 和全部命令。源码热点关注列表在
`source-map.json`：DataCopy、Cast、Brcb、Mul、Matmul、PIPE_ALL、Event 调用行。

本轮**不根据出现次数猜 Tile ID**。当前仅提供 core0 的真实完整执行 trace 和精确源码
行焦点；first/middle/last K tile 或某个 N tile 的动态归属在拿到实际指令 CSV/PC 信息
之前明确为 `UNRESOLVED_TILE_MAPPING`。后续只对真实 dump 做离线选择，不截断原 Kernel
来伪称“完整 Gate 的几个关键 Tile”。

Simulator cycles、工具换算时间及 CPU 运行秒数均不是 Ascend310P3 真机 wall-time。
本报告不计算 NPU speedup、不替代既有 bitwise/5-30 真机验收、不改变 adoption 或
full_draft_token_kv。`SIMULATOR_DIAGNOSTIC_ONLY` 是工具链诊断状态，绝不是性能或
硬件正确性 PASS。

## Bounded Gate：只执行一个原始 N64 Tile（独立入口）

`44afc78` 的服务器历史结果（用户回传）：Off debug OPP 和 smoke 已完成，smoke
bitwise=True / Kernel Identity MATCHED / Artifacts COLLECTED；完整 Off Gate 在 20 分钟
后中止，execution.json=RUNNING，Raw 未执行。部分 core0 trace 约 171 MB，目录约
6.3 GB。这是 **BLOCKED / 未完成** 的历史仿真，不是完整 Gate 正确性结果。
历史目录 `gate-simulator-full-20261020-084557` 完全只读保留。

新入口 `run_bounded.sh` 不调用旧的全量 Gate 路径，也不增加超时：

1. 首先只读检查历史报告、smoke 状态、未完成 Gate 的 execution.json、trace/CSV，
   记录每个文件大小、关键文件 SHA256、CSV 表头与样本、trace 解析结果。截断 JSON
   允许记录为 UNPARSED_PARTIAL，不变成“完整 trace”。检查结果写到新目录
   `partial-inspection.json`；源码身份不匹配就禁止构建。Codex 本地无法读取服务器
   产物，因此尚未确认历史 trace 的实际 PC/指令内容。
2. 复用既有 Off/Raw 的 Capture/Golden/OPP 配置身份验证，独立构建两个新 debug OPP。
   **只对新工程的 Kernel 副本插入两段代码**：在 TPipe/事件分配前检查原始 Gate
   shape 并只放行 `n_tile % blockNum` 的核；在原 cyclic N 循环开头跳过其他 N tile。
   不修改原 blockDim、不修改 Host、不缩小输入张量或权重、不改变 stride/NZ 寻址。
3. 原始逻辑 shape 始终 M16/K2560/N19456，N64/K256，20 个 group128。
   选中 N tile 的 10 次 K256 循环、FP32 累加、最终一次 FP16 写回、Raw 双 bank、
   首/末预取、Matmul/PIPE_ALL/Event 与异常清理逐字保留；删除两段插入后源码必须
   恢复原 SHA256。非 owner 核不创建 Matmul，不执行有效计算。
4. Off/Raw 使用相同完整冻结输入和 Golden 的逐字节拷贝。只比较选中列
   `[n_tile*64, (n_tile+1)*64)` 的全部 16 行（1,024 个 FP16 位型）；另检查未选列仍
   为原 Runner 单次调用前写入的 `0x7f7f`，以及全部原 guard、输入只读。
   任一项失败就阻断，不继续下一模式。
5. 新 trace 解析所有启动核，完成后从真实 Host Tiling 记录计算 owner。没有用
   `--core-id` 限制执行；`--launch-count=1` 仍只控制采集次数。真正减少工作量的是
   审计过的源代码范围限制。每个模式分别保存 trace/热点/指令 CSV；必须匹配新
   bounded 二进制 SHA256。`bounded-source.diff/json`、`source-map.json`、DWARF
   与热点文件保留真实源码行号。源代码限定了一个 N tile，但不从调用次数猜测
   每条指令对应的具体 K tile。
6. 默认 5 分钟；不自动延长或切换其他芯片。新输出总预算 1 GiB，单文件 256 MiB。
   子进程继承 RLIMIT_FSIZE；每 0.25 秒检查新目录总量（逻辑大小/分配块取较大值），
   达到超出条件就停止自己的进程组。总量监测不是硬文件系统 quota，可能短暂超额，
   报告记录实际大小；不自动删除旧/新证据。超时/限额/缺失产物始终 BLOCKED。

先只检查已有部分产物（不运行 CANN 编译或 Simulator），可用同一入口的
`--inspect-only`，输出仍要求新目录；也可直接 `python3 simulator/bounded.py` 执行该
纯离线检查，不要求初始化 CANN。

最小服务器命令（默认 N tile0，默认绑定用户提供的历史目录）：

```bash
CANN_ROOT=/usr/local/Ascend/cann-9.0.0 bash framework/custom_ops/draft_quant/simulator/run_bounded.sh \
  --n-tile 0 \
  --output-dir "/home/w00949577/qwen35-dflash-run/gate-simulator-bounded-$(date +%Y%m%d-%H%M%S)"
```

可显式指定 `--n-tile 0..303`；Off/Raw 同时使用该值。核数仍由原 Host 决定。
`--partial-report` 可指定旧的 simulator-report.json；`--max-output-mib` 和
`--max-file-mib` 明确记录磁盘预算。原始 Kernel、OPP、Runner、Golden 均不写入。

请回传 `bounded-report.json`、`bounded-analysis.md`、`partial-inspection.json` 和
失败阶段日志（若失败）。成功后每个模式的 `bounded/result.json` 记录局部数值、
完整性、真实 trace 大小及 artifact 路径；`build-report.json` 同时保存原始来源和
bounded 副本身份，不能当作原始算法二进制。

状态最多为 **BOUNDED_SIMULATOR_DIAGNOSTIC_ONLY**。完整 Gate、NPU、Draft correctness
和 hardware performance 全部保持 NOT_RUN。现阶段代码/CPU 回归完成，bounded
真实编译、仿真、局部 bitwise 验收均 NOT_RUN；不以 cycles 推算真机或整网加速。

### tmp_dump 监控竞态处理

预算扫描不再先 `is_file()` 再 `stat()`。每轮使用 `scandir` 枚举、`lstat` 分类，在同一
扫描内收集逻辑大小、分配块数和最大单文件大小。只允许已枚举子项/子目录的 ENOENT
被记录后跳过；输出根目录消失、PermissionError、EIO、ENOTDIR 等仍阻断。符号链接
不计入总量、不遍历其目录，但继续保留原先对链接文件目标的单文件上限检查。

每条命令的 `.budget.json` 保存累计 `scan_diagnostics`（扫描次数、ENOENT 次数和
最多 20 条路径/操作样本）。子进程正常退出后仍做最终预算检查，报告复用该次扫描
结果。0.25 秒轮询、RLIMIT_FSIZE、超时和数值/产物验收不变；消失记录不是完整采集
证明，`COMMAND_COMPLETED` 也不等于 Simulator 数值或 Trace 完整性 PASS。

## 只读指令瓶颈分析（不运行 Simulator）

用户回传 `5f43f65` 的配对目录
`/home/w00949577/qwen35-dflash-run/gate-simulator-bounded-2g-racefix-20261020-103256`：
Off/Raw 均完成局部 1024 个 FP16 bitwise、未选列 poison、Kernel identity、采集和预算
检查。该状态仅是 BOUNDED_SIMULATOR_DIAGNOSTIC_ONLY；完整 Gate/NPU/Draft 不因此升级。

`bottleneck.py` 只读该报告及其绑定文件，不调用 Simulator、msprof、Runner、编译器或
模型。新输出必须在历史 evidence 目录之外。它重新验证报告嵌套记录、输出/Golden 的
局部位型及 poison、Guard/readonly 状态、输入 SHA256、Off/Raw 唯一变量、预算结束
状态、原始/隔离源码 SHA256、可逆 bounded patch 和 diff、安装/采集 ELF SHA256、
DWARF 段审计，以及报告中的所有 Trace/CSV 大小/哈希。分析结束再次检查读取的证据。

已由用户提供的 26.0.0 receiver 样本确认：

- CSV `code,call_count,cycles,running_time(us)` 和
  `instr,addr,pipe,call_count,cycles,running_time(us),detail`。
- CSV `addr` 十进制；Trace `args.pc_addr` 十六进制。关联时要求 PC、指令名、
  process_name 所标 Pipe 和动态指令次数一致；不根据序列顺序猜 K Tile，也不猜 PC 基址。
- 只使用报告中绑定的 `core0/core0_code_exe.csv`、`core0/core0_instr_exe.csv` 和同目录
  `trace.json`；不混入汇总 Trace 或非 owner core。当前入口要求 owner=0。
- Trace 的 Scalar `dur` 可能为 0，但 CSV cycles 非零。因此零宽事件不是零成本，
  `displayTimeUnit` 也不能单独证明 Simulator cycle 时钟。

CSV 字段语义参考[官方性能数据说明](https://github.com/Ascend/msopprof/blob/master/docs/en/user_guide/msopprof_simulator_performance_data.md)。
上游可能更新；不符合已确认完整表头的 CSV 明确标为 UNSUPPORTED，不猜字段别名。

源码归属使用本机已有 `llvm-addr2line` 或 `addr2line`（也可显式 `--addr2line PATH`）
只读解析已匹配的 ELF，保存工具/ELF SHA、命令和原始输出。只传实际 PC，不推断加载
偏移；工具不支持该 ELF 或没有行信息时保留 UNMAPPED。只有绝对源码路径、行号与已
校验的 bounded 源码匹配且操作类别唯一，才能标记 DataCopy/Cast/Brcb/Mul/Matmul/
PIPE_ALL/Event。SDK 内联代码能通过 DWARF 调用栈关联回已绑定调用行；无此证据不猜。
Scalar、MTE2、CUBE 等 Pipe 指标单独按原始 CSV/Trace 元数据记录，不强行转成源码类别。

输出包括：

- `bottleneck-analysis.md/json`：Off/Raw 指令 call_count/cycles、按 Pipe 和已证实源码
  操作分类；未映射部分明确 UNMAPPED。call_count 是动态指令次数，不是 C++ API 次数。
- 每模式 `instructions.json`、`code-hotspots.json`、`timeline-intervals.json`：保留 PC、
  原字段、DWARF 栈、源码关联及完整区间。Code 热点可能含内联调用者归属，不与指令
  表叠加，也不宣称各行 cycles 互斥。
- Trace 中报告区间并集、同 Pipe 重叠、MTE2 与其他 Pipe 的交集及零宽事件数。数值
  使用原始 ts/dur 单位；不是 cycles，更不是 NPU ms。交集不能再相加解释总耗时。
- 如 DWARF 明确映射到 `CopyRawTile(qBanks[(1-bank)...` 的下一 K 预取调用点，记录该
  调用点的 MTE2 指令区间与计算 Pipe 的重叠；不据此猜每个动态 K ID 或宣称因果收益。
- `format-evidence.json`：真实表头/样本、顶层字段值、process/thread 元数据、完整
  指令样本 args、事件名称计数及等待相关名称样本。名称匹配本身不是等待依赖证据。

**边界**：CSV instruction cycles 之和仅是执行周期统计，不能当 Kernel elapsed
cycles。未获得独立 cycle clock/end-marker 时，Kernel elapsed cycles 差异保持
UNRESOLVED；可独立比较各 Pipe 的 instruction cycles 差值。关键路径、同步等待
占比、暴露的 MTE2 等待减少量，以及 PIPE_ALL 是否造成关键路径停顿，缺少可验证
依赖时均 UNRESOLVED。不能把带等待语义的指令的全部 dur/cycles 当作纯等待。

服务器命令（纯离线；不需重新 source CANN 或执行 NPU）：

```bash
python3 -B framework/custom_ops/draft_quant/simulator/bottleneck.py \
  --output-dir "/home/w00949577/qwen35-dflash-run/gate-bottleneck-offline-$(date +%Y%m%d-%H%M%S)"
```

可加 `--inspect-only` 只提取精简格式证据，连 DWARF 工具也不执行。当前本地 CPU 回归
不代表已读取服务器 Trace。默认分析状态为 OFFLINE_ANALYSIS_PARTIAL，数值结果必须
等待服务器离线执行；不会输出新的 Simulator/NPU 或完整 Gate PASS。

### CACHEMISS、源码级热点与 ELF 地址空间补充

用户提供的 `950dac0` 服务器离线结果：Off 1914 个 PC / Raw 1966 个 PC 的独立
DWARF 解析均为 0；未关联 X 事件分别有 106/118 条，属于 CACHEMISS，普通指令没有
count mismatch。附件显示两份 AI Core ELF 均为 ET_EXEC、Machine `0x1029`，`.text`
虚拟地址为 0（Off 大小 `0x3e6c`，Raw `0x3e90`），含 `.debug_info/.debug_line`
及重定位段；运行时 PC 在另一地址区间。**这些字段不能证明应减去 `0x13dcd000`。**

本入口新增以下只读分析，不改原 Trace 或 Kernel：

- 按明确、无冲突的 `process_name=CACHEMISS` 元数据识别独立 iCache-miss 通道。
  单独保存 `cachemiss-events.json`（完整原事件、args、起止区间和分类依据），报告
  数量、区间总和/并集以及与普通指令区间的交集。不加入普通指令 CSV 次数/占用统计。
  仅名称含 cachemiss、缺少通道证据或元数据冲突的事件仍未解决，不能静默排除。
  依据：[官方通道定义](https://github.com/Ascend/msopprof/blob/master/docs/en/user_guide/msopprof_simulator_user_guide.md)。
- `code_exe.csv` 的绝对文件路径和行号可独立关联到已校验 SHA256 的 bounded Kernel。
  输出对应源码文本、行号、类别与**该行原始** call_count/cycles。此状态为
  SOURCE_ROWS_CLASSIFIED，独立于指令 PC 的 UNRESOLVED。包装源文件/SDK 头文件只有
  路径，没有历史哈希绑定时保持 UNMAPPED；不会仅凭文件名包含 matmul 就归类。
- 源码行与内联调用者/被调用者的成本可能重叠。源码热点仅逐行展示，**不汇总类别
  cycles、不与 instr_exe 叠加**。Off/Raw 只有 bounded 源码 SHA256 相同且行唯一时
  才逐行比较差值。源码分类不转化成 PC 映射、不猜 K Tile 或同步依赖。
- 直接读取匹配 ELF 的 ELF64 header、section table 与 LOAD 段，保存 Machine、
  `.text` 地址/大小、可执行区间和 DWARF 段到 `elf-diagnostic.json`。不执行 ELF。
  原样 PC 不在可执行节与可执行 LOAD 内时，记录 UNRESOLVED_ADDRESS_SPACE；不调整
  PC、不搜索“看起来能解析”的偏移，也不反复调用已知无法映射的 host addr2line。
- 检查 PATH，或经历史 SHA256 核对的 msopprof 安装包目录内已有的 `llvm-symbolizer`；
  可显式用 `--llvm-symbolizer PATH` 选择已有程序。多个候选不擅自选择。保存工具
  SHA、版本、调用参数、JSON/错误输出，不下载工具或通过 debuginfod 获取调试文件。
  只用 ELF 自身 `.text` 的首地址及 +4 做设备格式解码能力探针：探针成功也**不证明
  运行时 PC 对应关系**，不能将其结果用作运行时指令归属。
  参考：[LLVM 官方接口](https://llvm.org/docs/CommandGuide/llvm-symbolizer.html)。
- `visualize_data.bin` 如存在，仅记录当前路径、大小、SHA256，标记未解码、未在历史
  manifest 中绑定；没有验证其二进制 schema 前不猜内容，不运行 msopprof --export。

机器可读报告保留 `source_hotspots`、`elf`、独立 CACHEMISS 统计及按相同源码行比较的
`comparison.source_hotspot_rows`。核心指令流水统计与源码热点始终是两套统计。
同步等待占比、关键路径、Raw Prefetch 具体因果链和 Kernel elapsed cycles 仍可为
UNRESOLVED；更多 MTE2/VECTOR 交集本身不证明暴露等待减少。

服务器仍执行上一节的同一只读命令，并使用新输出目录。若找不到已有 LLVM 工具，
源码级热点仍可交付，不以此冒充 PC/DWARF 已解决。本轮本地 CPU 回归不等同于真实
服务器报告已经重新生成。
