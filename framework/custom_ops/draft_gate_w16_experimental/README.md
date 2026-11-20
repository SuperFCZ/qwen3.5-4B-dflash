# C16 Layer0 Gate W16 实验准入

此流程只研究真实 M16/K2560/N19456 单投影。阶段报告与每次 Runner 记录均带有 `EXPERIMENTAL_ONLY`、`MACHINE_PROOF_UNRESOLVED`、`PRODUCTION_NOT_APPROVED`。原 EqxSwk 的 Source、Build、ATC 通过，严格 static 保持 `BLOCKED_STATIC_ADMISSION`。独立 Head 和两个 W16 ELF 的解码覆盖仍为零。

实验准入以 CANN 9.0 按 Ascend C/SDK 语义编译已确认的源码为前提，核对选择的源码、私有编译环境、实际 ATC 结果和作用域受限的安全条件。设备是否实现重叠、机器指令顺序和真实 UB 访问仍待证。没有人工勾选 PASS、忽略失败或冒充正式 static 的接口。

## 先做两个离线阶段

在服务器仓库根目录、原 CANN 环境中逐行执行。每次只运行选中的阶段。

```bash
cd /home/w00949577/z50058744/qwen3.5-4B-dflash
git pull --ff-only origin feature/ascend310p-custom-ops
BASE=/home/w00949577/qwen35-dflash-run
PY="$BASE/venv-hf513-probe/bin/python"
S="$PWD/framework/custom_ops/draft_gate_w16_experimental"
R=$(mktemp -d "$BASE/gate-w16-experimental-XXXXXX")
echo "$R"
wexp() { TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PY" -B "$S/run_experimental.py" "$@" --root "$R"; }
wexp compiler-audit
```

核对 `W16_EXPERIMENTAL_COMPILER_AUDIT_PASS` 后单独执行：

```bash
wexp admit
```

查看 `$R/admit/report.json` 的 `status`、`safety`、`cpu_witness`、`error`、`integrity_error`。准入状态为 `W16_EXPERIMENTAL_ADMISSION_PASS` 或 `W16_EXPERIMENTAL_ADMISSION_BLOCKED`。默认封存根为 `$BASE/gate-w16-atc8-EqxSwk`，离线反汇编报告根为 `$BASE/w16-isa-audit-20yqefdn`。可通过 `--archive` 和 `--isa-audit` 显式选择；原报告内的文件身份和绝对路径仍须匹配。所有阶段沿用同一实验根。目录锁阻止在同一根并行运行，阶段独占新目录，失败报告继续保留。

## 编译器证据

- 冻结 Raw 源重新渲染后的两份 Gate C++ SHA256 分别为 `144ab97b82cf91d7a3ce8cf3d20018f8394fc180e2bb41a06f727656fc17a13b` 和 `4fd99adb3fb57b8946ca1ecebec4d1bf3100946632e05bd6c97e66f6a62972d5`。同时核验安装的动态源码、生成头、Host 源和 OPP 清单。两个 ATC 入口复用经审查的动态编译脚本 `2fdf2abe2a51d8cd6faf09e3c70749093d69f1c80dd9be0b1de3bd27f2f4c4c7`，实际取用各自私有包的实现。实验检查脚本的源码查找顺序、无覆盖变量、实际子进程环境和独立 cache。
- 保留真实 CMake/OPC recipe、实际源与编译选项、工具 SHA256、ATC wrapper 到 atc.bin 的身份、原始 argv、stdout/stderr、AIR 与常量。核对 mode-1 图的私有 vendor、节点、blockDim、唯一嵌入 ELF 的偏移、长度、哈希。包内通用二进制与最终专用对象分开记录。serial/overlap 选中 ELF 的 SHA256 分别为 `7de3096e7697b9056366321c2585d3e77079651957a9f9f44611c4d4a7e0f323` 和 `195d076c225e54db57d50dda7a550df65a5c6fcf3842b4f256ecaa0b1cf90eb2`。
- `compiler-audit/report.json` 同时保存构建和 ATC 目录中 .ccei/.ii/.i、.ll/.bc、.s/.asm/.lst 的身份。邻近文件标为 `COLOCATED_ONLY`，空列表表示未发现留存文件。历史 ATC 外层记录没有其内部毕昇完整 argv。中间件没有可重验的编译输入→输出链时不用于证明最终 ELF。
- 此次在相同 SHA 的毕昇 `b87dd4f943eb6450d9e0f8476eb69a43af1ac3da4b581c4386b88ce76825c737` 上，对既存独立 dav-m200 样例实际使用 `-x cce --cce-aicore-only --cce-aicore-arch=dav-m200 -std=c++17 -O2`。`-E` 生成 175405 B 预处理文本；`-emit-llvm -c` 生成 2524 B bitcode，SHA256 为 `ddde3e6fea29d9cbd1bc49aa88184fa9f5eaa10e54dc606b3bf8e64a160cde49`。LLVM 读取器验证 IR，target triple 为 `hiipu64-hisilicon-cce`。`-emit-llvm -S` 和设备 `-S` 均返回 1。早先 `-save-temps -c` 保留过 .ccei。新测试日志在本地 `/work/build/w16-intermediate-probe-_tmg1tjb/report.json`，SHA256 为 `93d837017d2fe9827a012184e3c10b8482e2c6956ebc11aafa318aec7d186921`。它们只说明编译器能力，不能追回已丢失的 W16 中间产物。

## 必须闭合的安全条件

| 对象 | 实际核验 |
|---|---|
| 来源、调度、Cube | 根据封存链、隔离环境、全量 SDK 文件 SHA 和实际回调核验 dav_2002、A 调度、8 block、core_limit=0，304 个 N tile 各有唯一 owner。224 B parent 内的 200 B Cube 解码全部 50 字段并重建规范 SHA `745b846a7e3b31de10827caa5e493026d3984e282dd698177d2da380b826359a`。B/C 的完整 plan、父序列化和资源相同。Host Workspace 为 2097152 B。 |
| UB 范围 | 每核字节半开区间：q0/q1 [0,16384)/[16384,32768)，s0/s1 [32768,33024)/[33024,33280)，W0/W1 [33280,66048)/[66048,98816)，expanded [98816,101120)，Matmul scratch [101120,253952)，reserve [253952,262144)。Brcb 固定临时区 [254976,256000) 属于 reserve。所有起点以 32 B 对齐。源中实际 TPipe 地址检查支配 payload 访问；审查器校验两代 W 的 Cast、Mul 和对齐 UB→L1 读集。UB 地址未作设备观测。 |
| 同核 SDK 生命周期 | 绑定原 23 份 SDK 审查文件和 Matmul 声明、注册宏、LocalTensor 地址实现 3 份附加文件。生成源拒绝 split-core/client 宏，并静态检查 CFG_NORM。在本对齐、无尾块的直接 MatmulImpl 路径，Iterate 在调用栈内发出 UB→L1、SplitB 和 MMAD。SDK 可持有 W 直至原 PIPE_ALL；重用前该核已发操作须完成。SDK 读 W，外部 Vector 写另一 W 及其独立 expanded/Brcb 临时区。 |
| 事件、循环与失败 | 两个方向 MTE2_S/MTE2_V 的 ID 先保留，每代均 Set/Wait 完毕再复用；V_S 紧邻 Set/Wait 后才 SetTensorA/B。serial 的 D(k+1) 在 F(k) 后，overlap 源的 D(k+1) 在 F(k) 前。下一轮重绑在 F(k) 后。k=9 无下一 K，N 末 GetTensorC 后 fence。资源/地址失败发生在 payload 前。V_S 取 ID 失败或 MoveNext=false 时，消费已发 raw 的两个 Wait，fence 后退出，不发布本 N。硬件异常无成功排空声明。 |
| 数学与 CPU | k=0 partial=false，k=1…9 partial=true，同一 FP32 累加器，每 N 只写出一次 FP16。对 x、NZ 权重、GN Scale、Native Golden 验证冻结 SHA。admit 仅用旧 Build 的主机 C++ 编译器实例化实际 Gate 循环模板，记录两种排程的成功及每一 K 普通失败，共 22 条 CPU 轨迹。CPU 轨迹验证源码顺序。 |

原 SDK 各文件与行范围见同仓库 `draft_w8a16_overlap_review/w16-event-lifecycle-revised.md`。核内流水与事件的定义见 [CANN 9.0 Ascend C 架构说明](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_0008.html)。

## 设备阶段，仅在实验准入成功后手动执行

每次只运行一行，等待该阶段 PASS。三个正确性阶段均不计时。

```bash
wexp smoke-serial --allow-device
```

```bash
wexp smoke-overlap --allow-device
```

```bash
wexp stress --allow-device
```

两版各在新进程完成两种不同 Poison 的 Execute+Sync。stress 先对 single-w-raw 做一次新鲜未计时对照，再按交替顺序独立复测 B/C 各三轮，每进程两次。每次逐位验证 311296 个 FP16、原始 Native Golden、Repeat、Readonly、Poison、GM/OM 工作区 Guard 和同步。权重、Scale 为 OM 常量，绑定 AIR payload 和 OM 身份。复用已验收的隔离 Runner、PID/FD 凭据、独立 OPP、实时前置 DSO socket 放行和后置 maps。由本流程检查实验前驱和每次输出。严格 W16 Smoke 入口保持独立。

```bash
wexp schedule-benchmark --allow-device
```

```bash
wexp net-benchmark --allow-device
```

```bash
wexp compare
```

schedule 配对 serial/overlap；net 配对 single-w-raw/overlap。每对独立做三轮，顺序为控制先、候选先、控制先，每轮每版 5 warmup、30 次连续无插桩 resident Execute+Sync。每组 180 个新鲜样本，两组共 360 个，overlap 独立测量两次。数值和保护检查在计时区间前后执行。记录所有样本、逐轮及总体 Median/P95/Min/Max、倍率、Workspace 和 overlap 跨组漂移。两组均需每轮 median 改善、总体倍率 ≥1.03、总体 P95 无回退，才可讨论五层扩展。只有单投影研究决定，本模块没有五层或完整 Draft 执行入口。

身份、资源、数值、保护、同步或采样任一失败即停止。报告保留活动变体、可得输出、Runner 错误和原始日志；无法确认同步时标记 `NOT_CONFIRMED`。退出 1 为阻断，比较结果 NO_GO 退出 3。保留失败根用于诊断。生产批准和机器级结论保持原状态。

## CPU 回归

```bash
python3 -B -m unittest discover -v -s framework/custom_ops/draft_gate_w16_experimental/test
# 已有冻结 package fixture 的本地 ARM64 容器
CANN_ASC_ROOT=/usr/local/Ascend/cann-9.0.0/aarch64-linux/asc \
W16_LOCAL_BUILD_REPORT=/work/build/gate-w16-px0mgc_g/report.json \
python3 -B -m unittest discover -v -s framework/custom_ops/draft_gate_w16_experimental/test
```
