# Ascend 310P3 的 W8A16 Layer0 KV 投影

在服务器仓库根目录运行：

```bash
git pull --ff-only origin feature/ascend310p-custom-ops && bash framework/custom_ops/draft_kv_bandwidth/run.sh
```

默认在独立的 `gate310p` 环境完成 C16 和 C64，终端打印每组无插桩 Median/P95、各档胜者、设备端 Median/P95 和匹配的流水计数。每次使用新目录 `/home/w00949577/z50058744/qwen35-dflash-run/runs/kv-bandwidth-*/`，总表为 `result.json`。可用 `--gear 16` 或 `--gear 64` 单独测量，`--compile-only` 停在编译，`--no-profile` 只做正式计时。中断后加 `--resume kv-bandwidth-XXXXXX` 可复用合同和 SHA 均匹配的成功阶段。

## 测量的投影

基线来自 `full-draft-panels-UFqlys` 实际使用的 `proj_layer0_kv`。每行有 2560 个 FP16 输入，输出的 2048 列为 K 和 V 各 1024 列。C16 的 32 行与 C64 的 80 行均由已归一化的上下文行和 16 行提议 token 拼接而成。两档绑定同一份 Layer0 权重和 Group128 scale，分别核验原始 OM 的执行段、实际 8-block 调度与完整 tiling。

参考 OM 从原 AIR 截取 FC、归一化、拼接、原版 KV 和第一层缓存更新。实测输入读取封存的 features 和实际 token 的 FP16 embedding；回放产生投影输入和输出，并逐位核对完整首层 key/value 缓存与 UFqlys。独立测速以单 KV OM 的公开 ND 输入为起点，以 FP16 K/V 投影输出为终点。缓存更新和注意力各有独立的整图节点。

## 参与配对的实现

每个优化版将 N 分成 32 个 N64 tile，8 个 block 各负责 4 个。INT8 权重、GN scale 离线按 N64、N16 stripe 打包，检查逆变换的每个字节。每个 block 缓存一份完整 A1，每个 N64 的 W 只反量化一次并供全部 32 或 80 行使用。沿 K 顺序在两份 W16 中执行 Cast 和 Group128 FP16 Mul；完整 B1 就绪后一次 MatMul Iterate 覆盖 K2560，并在独立 UB 中准备下一 N64 的首段。沿用已测试的 Q 跨 N 调度、完整屏障和 FP32 累加/FP16 输出。

| 版本 | 输入搬运 |
|---|---|
| `raw-nz-brcb` | 完整 Draft 的原版 NZ+Brcb；C16 每个 N64 有 10 次 Iterate，C64 有 20 次 |
| `kv-k2560-next-n` | ND 激活按行从 GM 直接分块写入 A1 |
| `kv-k2560-ub-stage` | 在第一次反量化之前借用两份 W16 的空间，GM 连续读入最多 32 行，再按 NZ 坐标写入 A1。每批有写完和读完两道屏障，M80 最后一批为 16 行 |
| `kv-k2560-packed-a` | 单 OM 内按 `[M,K/16,16] → [K/16,M,16]` 重排，一次连续 DMA 搬入 A1；正式计时包含图内重排 |

CANN 9.0 主机 tiling 查询的结果如下。板上 ATC 会再次读取回调与 OM 内核，要求三候选的权重和 scale 字节、完整序列化 tiling 相同，实际执行段有别。各候选均用 215,552 B 用户 UB 和 46,592 B MatMul scratch。

| Gear | Cube 基块 | stepKa / stepKb | A1 / B1 深度 | 本核 L1 总用量 |
|---|---|---|---|---:|
| C16 | 32×64×256 | 10 / 10 | 10 / 10 | 983,040 B |
| C64 | 80×64×160 | 4 / 2 | 8 / 4 | 1,024,000 B |

## 时延与带宽口径

| Gear | 输出 FP16 数 | 单遍逻辑字节 | 204.8 GB/s 单遍参照 |
|---|---:|---:|---:|
| C16 | 65,536 | 5,619,712 B | 27.44 µs |
| C64 | 163,840 | 6,062,080 B | 29.60 µs |

逻辑字节为 INT8 权重、FP16 scale、公开 FP16 输入和输出各一遍。两档的权重与 scale 合计 5,324,800 B。实际外存流量取决于缓存命中、跨核重复读取和重排。报告中的逻辑 GB/s 便于比较同一工作的完成速率。msprof 的设备区间从每次已匹配的首个重排开始直到 KV 结束；原生 ND 两版只含一个 KV 任务。独立报告严格、临时或未能核验的归属状态。

每种 OM 对测量输入、稠密压力输入、跨 Group128 和 Cube 边界的稀疏输入分别施加两种 poison，逐位比较两个输出半区，检查 Guard、Readonly、Repeat、Sync 和 DSO。每组配对三轮正反序，每轮每边 5 次预热和 30 次正式计时。候选必须在与原版的新鲜配对中达到 Median 降幅至少 3%、三轮同向且 P95 无回退。多个候选通过时，逐对与当前胜者直接复测；每个比较使用自己的对照数据。msprof 单独运行，覆盖 30 个正式调用。

CPU 合同测试可用 `python -B -m unittest discover -s framework/custom_ops/draft_kv_bandwidth/test -v`。提供 `DFLASH_GE_DESCRIPTOR`、Torch 和 Transformers 后还会将两档截取的参考图与五层原图的中间值及缓存逐位比较。`test/sdk_plan.cpp` 在无 NPU 环境调用本机 SDK 的主机 tiling API。

## a6xFlL 首轮设备结果

真实 Layer0 上，两档均选择 `kv-k2560-next-n`。所有参与者在各自 gear 的 3 组输入和 2 种 poison 中通过全量 bitwise、Guard、Readonly、Repeat 和 Sync。原始报告位于服务器 `/home/w00949577/z50058744/qwen35-dflash-run/runs/kv-bandwidth-a6xFlL/result.json`。

| Gear | 原版 → N64 正式 Median / P95（ms，各 90 个样本） | 加速比 | N64 设备 Median / P95 | 与单遍参照之比 |
|---|---|---:|---:|---:|
| C16 | 0.287850 / 0.301360 → 0.113165 / 0.116960 | 2.5436× | 47.37 / 47.92 µs | 1.726× |
| C64 | 1.024485 / 1.551510 → 0.168640 / 0.245160 | 6.0750× | 102.94 / 106.07 µs | 3.478× |

上表为各自原版与胜者的直接配对。C64 的该批 P95 波动较大，其余候选对 N64 的两个直接配对中，N64 的 P95 分别为 0.183590 和 0.172090 ms。C16 片上 UB 暂存相对 N64 的同期中位数改善未达 3%。C64 图内预打包的 KV 本体为 84.80 µs，计入打包的区间为 109.95 µs。首轮胜者的设备归属均为 `PROVISIONAL_HOST_ENCLOSURE`。

## 宽输出列的独立对照

```bash
bash framework/custom_ops/draft_kv_bandwidth/run.sh --wide kv-bandwidth-a6xFlL
```

此入口重新核对既有胜者的实际 ELF、原始配对、三份输入和参考，直接复用 N64 OM 与 Runner。只编译新增的 N128/K1280 和 N256/K1280。在相同公开 ND 输入、无在线重排的条件下，每核输出 tile 写回次数从 4 降到 2 或 1。两份 K1280 均按原序累加，第一个初始化 FP32 L0C、第二个 partial，最后一次写 FP16。N128 每核缓存完整 A1 给两个列组复用，N256 的每核只负责一个列组并按 K 面板搬运 A。每个完整 N64/128/256 输出列组内的权重反量化供全部 M 行使用。实测决定减少写回与改变 Cube 分块的净影响；计数器本身无法定位关键路径。

两种宽度各与同一冻结 N64 用新样本配对。若均通过，则两种宽度之间另做新鲜直接配对。中位数门槛、P95 和正确性合同与首轮一致。

宽列胜出后再与封存的原版 Raw OM 补一组新鲜直接配对，用于报告最终相对原版的加速比。

## nafZjO 宽列设备结果

C16 保留 N64。N128 和 N256 对其直接配对分别为 1.0284× 和 1.0019×，均未过线。C64 选中 N128/K1280；在同轮试验中，N64 → N128 的正式 Median/P95 为 0.166750/0.174280 → 0.147870/0.154940 ms，1.1277×。N256 对 N128 的直接比值为 0.9799×。C64 原版与 N128 的新鲜直接配对为 **1.032330/1.050300 → 0.148085/0.167990 ms，6.9712×**。两档所有宽列候选均通过完整位型检查。报告在 `/home/w00949577/z50058744/qwen35-dflash-run/runs/kv-bandwidth-nafZjO/result.json`。

在这次独立采集中，C16 N64 为 47.27 µs（1.723× 单遍参照），C64 N128 为 85.53 µs（2.890×）。C64 的 N64 / N128 / N256 分别有 27.78 / 34.34 / 43.64 µs 的 MTE2 活动和 41.87 / 23.81 / 19.87 µs 的 MTE3 活动。各流水可重叠，计数没有给出 DDR 读字节。

## C64 N128 的单变量激活搬运试验

```bash
bash framework/custom_ops/draft_kv_bandwidth/run.sh --fused-a kv-bandwidth-nafZjO
```

复核两轮封存的判定，使用上一轮 N128 的原始 AIR 字节，仅改变私有 Kernel 的激活装入方式。原始公开 ND 输入、两段 K 归约、权重与 Scale、UB、MatMul scratch 和序列化 Cube tiling 保持一致。C64 激活沿 K 切成 10 个 256 元素宽的块。每块从 80 行各读连续 512 B，放入第一份尚空闲的 W16。两条 128-lane 的 uint16 `Or(src,src)` 按块和 repeat 步长重排至第二份 W16，一条连续搬运将 40,960 B 写入 A1。本机 dav_m200 SDK 的通用 Vector Copy 报 unsupported；`Or<uint16_t>` 使用 `vor` 实现，可保留有符号零和 NaN payload。四个方向的同步分别保护输入就绪、输入复用、输出就绪和输出复用，最终全流水屏障后才开始常规权重反量化。

CPU 合同覆盖全部 65,536 种 uint16 位型、每个 A1 坐标、四种删除 handoff 的失败注入，以及实际 Kernel 的全尺寸累加。服务器单独复核已归档的 Runner 和真实设备样本。两组正式配对均将候选的片上准备计入 Execute+Sync；通过 N128 对照后才追加与原版的配对。
