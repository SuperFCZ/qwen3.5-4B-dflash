# Ascend 310P3 · 完整 Draft 的入口 FC 投影

在 310P3 的仓库根目录运行：

```bash
git pull --ff-only origin feature/ascend310p-custom-ops && bash framework/custom_ops/draft_fc_bandwidth/run.sh
```

默认两档。终端打印每档的直接配对和独立的设备任务时间。报告位于 `/home/w00949577/z50058744/qwen35-dflash-run/runs/fc-bandwidth-*/result.json`，使用已有的独立 `gate310p` 环境。可加 `--gear 16`、`--gear 64`、`--compile-only`、`--no-profile` 或 `--resume fc-bandwidth-XXXXXX`。编译产品和失败日志按阶段留在运行目录，生产部署使用既有配置。

## 对照与数据流

从 `full-draft-panels-UFqlys` 取真实 W8A16 checkpoint 的 `proj_fc`。C16 和 C64 分别为 [16,12800,2560] 和 [64,12800,2560]。两档共用 NZ 权重（SHA256 `6f316939…`）和 GN scale（`cab921d5…`），各自绑定原始 features、valid_rows、ATC 执行段及父 tiling。对照单 FC 必须命中原完整 OM 的相同执行段和资源。按原 AIR 的行可见性运算还原 FP16 features，逐字核对 mask 和 reshape；UFqlys 的两档均使用全有效行。本次输入来自真实 checkpoint、固定种子的整图回放。

| 版本 | 每个 N64 的 K 归约 | 激活 | 权重 |
|---|---|---|---|
| `raw-nz-brcb` | 50 × K256 | 原 FC 路径 | 原 NZ + Group128 |
| `fc-a1-k2560-next` | 5 × K2560 | C16 每核缓存完整 A1，C64 每个 K 面板将当前 64 行从 GM scatter 到 A1 | K2560 的 N16 分片，两个 80 KiB W16 |
| `fc-gm-k6400-next` | 2 × K6400 | SDK 从 GM 按原始行跨距 12800 取 A | 每个 N16 分为五个 K1280，两个 40 KiB W16 |

两版都离线无损重排 INT8 和 FP16 scale。每个 N64 的同一份 W16 供整个 M16/M64 计算，C64 避免按四个 M16 重新反量化。每核负责五个循环分配的 N64。按 K 递增调用 Iterate，首段初始化，中间段保留 FP32 L0C，最后一次 GetTensorC 写 FP16。当前 A/B 完全就绪后，在独立的 UB 中预备下一 K 面板（或本核下一个 N64）的首段，移动的有效性以设备测量为准。SDK scratch、原有全流水屏障及 W16/原始码的双向事件各自保留。

真实 ATC 回调核对 8 个 block、全部 N64 恰好覆盖、baseK256、A/B 深度、L1/L0/UB 容量、2 MiB workspace、OM 内执行段及实际常量字节。A1/K2560 的 C64 方案使用完整的每核 1 MiB L1，其通过条件包括实际资源合同。

## 验收和数字口径

1. 在原 FC OM 上采集原始 features 和稠密变换输入的参考；另有跨 Group128、K2560/K6400 首尾边界的独立两项参考。每份 OM、每组输入各用两种输出 poison，检查全部 40,960 / 163,840 个 FP16 输出、输入只读、Guard、同步和实际加载 DSO。
2. 三轮交错，每轮每边 5 次预热、30 次正式 Execute+Sync。原 FC 与通过准入的 A1/K2560 直接配对。按 gear 独立选择，门槛为 Median 下降至少 3%、三轮同向且 P95 无回退。正式原始样本保存在 `benchmark-c16/report.json` 和 `benchmark-c64/report.json`。
3. msprof 在独立进程中测原版及通过准入的候选的设备 FC 任务，使用 38 次 Execute/Sync 中下标 7–36 的 30 个窗口。输出实际归属、设备 Median/P95、逻辑 GB/s、等效矩阵 TFLOP/s、可核对的流水计数。正式配对先于 profiling 落盘。
4. INT8 权重有 32,768,000 B，scale 有 512,000 B。按权重、scale、X、Y 各经过一次外存及 204.8 GB/s 的条件峰值，C16 为 33,771,520 B、**164.90 µs**，C64 为 35,246,080 B、**172.10 µs**。FC 权重与 scale 大于 SDK 所列的 16 MiB L2。逻辑吞吐分子为上述单遍字节；profile 中的 MAC/MTE 等活动时间可重叠，真实 DDR 事务需要单独计数。

CPU 合同测试可用 `python -B -m unittest discover -s framework/custom_ops/draft_fc_bandwidth/test -v`，设置 `DFLASH_GE_DESCRIPTOR` 可同时验证原 GE 输入选择。延迟 DMA 模型运行两套实际 kernel 源码的 C16/C64 分支，覆盖每个输出的数值、写入次数和核归属，每个 K 面板的 FP32 接续、首尾失败清理及遗漏 W16 消费者 fence 的故障注入。

## qR8ETP 正确性记录

服务器在 C16 上完成原 FC、A1/K2560 的三组逐位检查。GM/K6400 在 features 输入的第一个快照有 5,868 / 40,960 项不同，Guard 完整。差异集中在第一个循环波次的 N64 编号 0、1、3、4、5、6、7，后 32 个 N64 全部相等。错误形态不能单独确定具体硬件原因。该版本退出默认设备路径，现有失败快照只读保留。

从这个已经建好的目录继续做两档 A1 对照：

```bash
bash framework/custom_ops/draft_fc_bandwidth/run.sh --resume fc-bandwidth-qR8ETP
```

续跑复核原包和已通过的 ATC 收据，沿用 C16 的设备参考，重新封存只含原 FC 和 A1/K2560 的 smoke。C64 单独获得并核对自己的设备参考。

## C16/C64 第一轮设备选型

`fc-bandwidth-6bPlhz/result.json` 是无插桩的独立配对及单独的设备采集，每边各 90 个正式样本。

| Gear | 原版 → A1/K2560 Execute+Sync Median (ms) | P95 (ms) | 加速比 | 原版 → A1 设备任务 Median (µs) | A1 有效 GB/s |
|---|---:|---:|---:|---:|---:|
| C16 | 1.387965 → 0.295910 | 1.393980 → 0.309310 | 4.6905× | 1322.73 → 224.84 | 150.20 |
| C64 | 1.497355 → 0.730650 | 1.517220 → 0.759410 | 2.0493× | 1431.69 → 657.67 | 53.59 |

两档 GO，全部位型通过。设备任务采用 `PROVISIONAL_HOST_ENCLOSURE`。C16 距 164.90 µs 的条件参照为 1.364 倍。C64 为 3.821 倍，MTE2 活动达 444.25 µs，Vector 为 100.95 µs。C16 对应为 93.35 / 100.75 µs。

### C64 激活搬运的直接对照

```bash
bash framework/custom_ops/draft_fc_bandwidth/run.sh --packed-a fc-bandwidth-6bPlhz
```

该入口复核并复用已通过的 C64 A1/K2560 OM、Runner 和三组真实参考。公开输入仍是原 FP16 ND `[64,12800]`。两个候选在模型内执行原始位型的 reshape、transpose，从 `[M,K/16,16]` 得到 `[K/16,M,16]`，打包后的每个 A1 面板只需一条连续拷贝。模型加载后的正式 Execute+Sync 包含这项激活准备。

- N64 版本与原 A1 版本有相同的权重和 scale 字节、完整 Cube tiling、全部分配和 8-block 循环所有权，分离激活访问模式的影响。
- N320 版本给每个实际 block 分配独立的 320 个连续输出列，全 K 分 10 个 K1280 面板，A1/B1 常驻当前面板，SDK 基块为 64×320×32、stepKa/Kb 为 2/1。真实 ATC 单独记录 N320、8 个 tile 与对应所有权。每核只需扫描一次整份激活，8 个核合计向 GM 请求 13.11 MB A；N64 对照请求 65.54 MB。两者的权重都只需反量化一次。
- 每种布局与同一正确对照重新做三轮直接配对及完整输出校验；若两者都达标，额外直接比较两种布局。可用 `--width 64` 或 `--width 320` 限定。每个已完成的配对独立保存。单独 profiling 同时报告 FC 任务、按已绑定图节点归属的布局任务、从布局启动到 FC 结束的设备区间；未能逐调用验证时保留 UNRESOLVED。整算子的选型依据是含布局准备的正式时延。

## SS7hQ7 与 C64 尾延迟

`fc-bandwidth-SS7hQ7/result.json` 记录了含在线激活重排的真实配对。

| C64 直接对照 | Execute+Sync Median (ms) | P95 (ms) | 判定 |
|---|---:|---:|---|
| A1 → packed N64 | 0.730535 → 0.396705 | 0.737500 → 0.409190 | 1.8415× GO |
| A1 → packed N320 | 0.730145 → 0.381420 | 0.761080 → 0.386870 | 1.9143× GO |
| packed N64 → packed N320 | 0.397230 → 0.380315 | 0.415580 → 0.434500 | 1.0445× NO_GO |

保留 N64。设备端打包启动至 FC 完成的 Median/P95 为 N64 322.56/326.80 µs 和 N320 307.30/310.50 µs，范围为带插桩的设备区间（`PROVISIONAL_HOST_ENCLOSURE`）。N64 的 FC 内 MTE2 活动由 444.77 降为 112.74 µs。N320 虽降低了中位数，直接配对的正式 P95 超过对照。

```bash
bash framework/custom_ops/draft_fc_bandwidth/run.sh --n160 fc-bandwidth-SS7hQ7
```

该实验复核 SS7hQ7 的选型，复用已测 N64 OM、输入、参考和 Runner。N160 有 16 个列组，每核恰好两个。首先使用 K1280 单 A1/B1、SDK baseK80。另一版 K640 为 A1/B1 各分配两个不相交的完整面板，一条 DMA 取得当前面板的全部 INT8。Cube 提交后可以在另一 B1 中准备完整的下一面板，跨调用的 FP32 L0C 与应用 W16 独立；板端效果以直接配对验证。应用通过 TPipe 预留并释放三个方向的实际事件 ID，原有全流水屏障保护每次缓冲区切换和最终输出。两版分别检查实际资源、完整位型、失败退出和图内重排成本。CPU 延迟 DMA 测试会将写入故意重定向到 Cube 正在消费的 B1，确认能拦住错误复用。
