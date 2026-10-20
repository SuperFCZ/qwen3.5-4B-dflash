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
