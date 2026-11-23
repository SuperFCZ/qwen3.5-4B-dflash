# 310P3 W8A16 Gate/Up 带宽实验

从服务器仓库根目录执行。命令会复核已有 tile-ND 对照，生成候选的私有单算子 OM，校验输出后测量完整投影的 Execute+Sync。

```bash
git pull --ff-only origin feature/ascend310p-custom-ops && bash framework/custom_ops/draft_gate_bandwidth/run.sh --reuse-build gate-tile-nd-ZHIxuE
```

省略 `--reuse-build` 可从 Git 和已安装的 CANN 重新构建对照。默认在独立 Conda 环境 `gate310p` 使用与 QyzcNC 相同种子的合成数据。可选 `--panels 512` 单测一个候选，`--compile-only` 停在 ATC，`--layer0` 使用新根下四个身份匹配的真实张量。终端打印每一对的 Median、P95、倍率和最终目录。原始样本保存在 `/home/w00949577/z50058744/qwen35-dflash-run/runs/gate-bandwidth-*/experiment/result.json`。

### 以已测 K2560 为对照，只测 W16 双缓冲

```bash
git pull --ff-only origin feature/ascend310p-custom-ops && bash framework/custom_ops/draft_gate_bandwidth/run.sh --w16-pair gate-bandwidth-VL4yZa
```

此入口只编译 `panel-k2560-w2`。它复核指定报告中单 W16 的 GO、源文件、Host 回调、私有 OPP、内嵌 ELF 和两份常量，复用已验收 K2560 的 AIR、OM 和 runner；两边重新做完整 Smoke 与三轮配对。两版使用同一份原始 AIR 字节，ATC 产生的 200 B Cube Tiling 必须相等。两份 W16 各占 80 KiB，应用增加 81,920 B UB，SDK scratch 从 128,512 B 降到 46,592 B，raw 区、A1/B1 和 GM Workspace 均保持相同。

每个 N64 有四个 N16 分片。双缓冲按 0、1、0、1 交替写 W16。分片 1 可以在 bank1 上 Cast/Mul，同时分片 0 由 MTE3 从 bank0 写 B1。分片 1 的最后一次 Mul 后，先等待分片 0 的存储，再交接分片 1；因此分片 2 可以覆写 bank0。分片 2 同样等分片 1 的存储，保护分片 3 对 bank1 的覆写。最后的全流水栅栏确保 B1 四片到齐才启动 Cube。CPU 回归对仍在读取的 UB 区间延迟拷贝，并注入缺失这两处等待的错误。

## 数据流

X 是 FP16 [16,2560]，Y 是 FP16 [16,19456]。对照为已经测得 1.324100 ms 的 tile-ND/B16。候选保持 Scheduler A 的循环 N64 分工。离线把 INT8 权重排列成 [N64 tile,K panel,4,K within panel,16 channels]，把原 FP16 GN scale 排列成 [N64 tile,20,64]。两种变换都对每个元素做逆映射验证，参数总字节与对照相同。在线 Cast 后的 W16 自然是逻辑 B[K,N] 的 FP16 NZ 布局，交给非转置的 `SetTensorB`。在 310P3 的 SDK 分支，局部 NZ 的 `CopyNZ2NZImpl` 可用一次有步长的 DataCopy 搬运一个 Cube B 子块。

每条 Mul 处理一个 Group128 的 16 个通道。第三个源使用一个 32 B 的只读 scale 块，块步长和重复步长均为零，128 个 K 行共享这 16 个通道 scale。相邻 Mul 写不同的 W16 区间。Cast 后的 PIPE_V 处理 Cast → Mul 依赖；一次 V_S 交接完整 W16。这样消除了单独的 scale 展开及其共享临时区的读写冲突。

| 版本 | 外层 K 步长 | 每个 N64 的 Matmul 调用 | 原始数据及同步 |
|---|---:|---:|---|
| tile-ND/B16 对照 | 256 | 10 | 原 Raw 双 bank、B16 展开、逐 K 交接 |
| panel-k256 | 256 | 10 | 每个 panel 各一次连续权重/scale DMA，直接得到 Cube NZ |
| panel-k512 | 512 | 5 | 同上，每次 SDK 调用覆盖 2 个 baseK256 |
| panel-k1280 | 1280 | 2 | 单 W16 panel，单 40 KiB raw 区分两批处理，每次 SDK 调用覆盖 5 个 baseK256 |
| panel-k2560 | 2560 | 1 | 激活在每个 block 的 A1 常驻；权重在线反量化后在 B1 拼出完整 K，每次 SDK 调用覆盖 10 个 baseK256 |

K256/K512 的两个 raw bank 在块内交替，首次搬入后有一次 PIPE_ALL。之后现有每个 Iterate 末尾的 PIPE_ALL 同时完成下一 bank 的 DMA 和当前 SDK 消费，下一轮可直接使用。最后一个 panel 按实际 blockNum 预取本核的下一 N64，不越过本核最后的 tile。K1280 在第一批 Cast 后用 V→MTE2 保护同一 raw 区的覆写，MTE2→V 保护第二批首次读取；第一批的 Mul 可以与第二批搬运重叠。在前三个候选中，单份 W16、Matmul scratch 与上一轮 L0C 归约仍由 Iterate 后的 PIPE_ALL 保护；GetTensorC 后保留跨 N 的 PIPE_ALL。返回 false 的 Iterate 先排空再退出，让不完整输出触发验收。

K2560 每个 block 从 GM 把运行时 X 打包为 L1 中的 [160,16,16]，81,920 B。每个 N64 的四份 N16 权重分片各占 40 KiB INT8，依次在 UB 使用单份 80 KiB W16 完成 FP16 反量化，再连续写入 B1 中互不重叠的区域。V→MTE2 保护 raw 区覆写，V→MTE3 保护 UB 到 B1 的首次读取，MTE3→V 保护 W16 的下一次覆写。这条读取指令由应用显式发出。全部分片完成后用 PIPE_ALL 向 SDK 交接 A1/B1。B1 的 327,680 B 及 A1 常驻区都从 Host 可供 SDK 使用的 L1 中扣除。仍在完整 K 归约后写回一次 FP16。使用 `--panels 2560` 可只和这个版本配对。

每个 panel 都从真实 Host 回调取回完整 Tiling bytes、硬件容量及实际资源。准入要求实际 8 block、304 个 N tile 恰覆盖一次、Cube base=16×64×256、FP32 部分和、原输出 stride 和每个 N64 一次 FP16 写回。SDK 必须接受该版本的 UB 预算及所有 L1/L0 配置。检查最终 OM 中的两份常量、所选专用 ELF 和私有 Host DSO 身份。

## 数据、计时与参考线

每个版本用同一份稠密输入和独立参考校验全部 311296 个 FP16 输出，再做一次带符号的双项输入校验，索引跨越 K16/K32、Group128、K256/K512/K1280 边界。两次 Smoke 都检查两种 Poison、Readonly、Guard、Repeat 与同步。单个候选构建或 ATC 审查失败时会保留原因并排除该版本。设备启动、数值或安全失败立即终止设备测量。

每个候选分别和同次重新计时的 tile-ND 对照配对，3 轮交错，每轮每版 5 预热 + 30 正式，记录每个版本 90 个样本。只有三轮都快、总体 Median 至少下降 3%、P95 无回退的候选才记 GO。性能比较不含离线打包、编译、校验、H2D 和模型加载。

矩阵等效吞吐用 1,593,835,520 FLOP/次计算。表中有效 GB/s 用单遍逻辑输入和输出 51,290,112 B 除以耗时。若单芯片可用 204.8 GB/s，理想单遍耗时为 0.25044 ms；之前同卡只读权重和 scale 的实测为 0.55971 ms。完整 OM 还包含反量化、片上读写、Cube 累加和同步。本实验没有采集 DDR 事务或实际流水重叠率。[华为 Atlas 300I Pro 公开规格](https://e.huawei.com/cn/products/computing/ascend/atlas-300-ai)

CPU 回归可用 `python3 -B -m unittest discover -s framework/custom_ops/draft_gate_bandwidth/test`。设置 `GATE_TEST_GE_DESCRIPTOR` 到由 CANN 的 ge_ir.proto 生成的 descriptor set，可附加 AIR 往返验证。CPU 生命周期测试还分别注入遗漏 V→MTE2、MTE3→V 的故障，并检查会在 raw 或 W16 被提前覆写时拒绝。

## 服务器实测

在独立 Conda 环境的 310P3 上，前三个候选通过稠密合成输入及跨边界输入的全输出位型、Guard、Readonly、Poison、Repeat 检查。下表单位均为 ms，每组每版 90 个新样本。实验由服务器回传，路径为 `/home/w00949577/z50058744/qwen35-dflash-run/runs/gate-bandwidth-L9C3fI/experiment/result.json`。

| K | 对照 Median | 候选 Median | 对照 P95 | 候选 P95 | 倍率 |
|---:|---:|---:|---:|---:|---:|
| 256 | 1.323200 | 1.160330 | 1.356490 | 1.231820 | 1.1404× |
| 512 | 1.321860 | 0.977770 | 1.351610 | 1.022070 | 1.3519× |
| 1280 | 1.323035 | 0.762700 | 1.348720 | 0.818070 | 1.7347× |
| 2560 | 1.319765 | 0.505715 | 1.334920 | 0.524270 | 2.6097× |

K2560 由另一次独立配对测得，报告为同一运行根下的 `gate-bandwidth-VL4yZa/experiment/result.json`。两组输入、两种 Poison 及全部输出位型检查通过。等效矩阵吞吐 3.152 TFLOP/s，单遍逻辑字节除以耗时为 101.42 GB/s（条件峰值 204.8 GB/s 的 49.5%）。0.55971 ms 的独立 copy 微基准包含同步和观测写回，不能据此限定完整算子的最短时间。

### 相同完整 K 的 W16 直接配对

服务器运行 `gate-bandwidth-6BCgLO/experiment/result.json` 复用了 VL4yZa 的单 W16 对照 OM（SHA256 `ad5e7562f8b131982bbd028ff6033483534cde686181062cf187956b39249a92`）。两版的 8-block、baseK256、完整 Cube Tiling 及每个 N64 的一次 Matmul 均通过准入。两组输入、每组两种 Poison 的 311,296 个 FP16 输出及 Guard 验证通过。

| 版本 | Median (ms) | P95 (ms) | 本次样本数 |
|---|---:|---:|---:|
| K2560 单 W16 | 0.505895 | 0.515930 | 90 |
| K2560 双 W16 | **0.469815** | **0.481390** | 90 |

三轮同向，Median 下降 7.13%，倍率 1.0768×，判定 **GO**。最快版本的等效矩阵吞吐为 3.392 TFLOP/s，逻辑吞吐为 109.17 GB/s。在单遍 204.8 GB/s 模型下对应约 53.3%。这些数据来自合成 C16 单 Gate/Up 的未插桩 Execute+Sync；本次没有采集外存事务或硬件流水时间线。
