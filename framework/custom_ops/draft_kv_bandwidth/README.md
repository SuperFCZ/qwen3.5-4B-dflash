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
