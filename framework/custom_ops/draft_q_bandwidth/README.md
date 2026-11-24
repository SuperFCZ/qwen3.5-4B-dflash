# Ascend 310P3 · C16 Layer0 Q 投影

从仓库根目录运行：

```bash
git pull --ff-only origin feature/ascend310p-custom-ops && bash framework/custom_ops/draft_q_bandwidth/run.sh
```

默认使用 `full-draft-panels-UFqlys` 的真实 W8A16 投影 `proj_layer0_q`。日志、OM、冻结输入、原始样本及 profiler 导出均写入 `/home/w00949577/z50058744/qwen35-dflash-run/runs/q-bandwidth-*`。使用已有 `gate310p` 环境。终端打印分阶段状态、失败日志及 Median/P95。可选 `--compile-only`、`--no-profile` 或 `--resume q-bandwidth-XXXXXX`。

## 数据与对照

M16/K2560/N4096，8 个 block 共分 64 个 N64。对照复用已通过整图测量的私有 OPP。Source 核对原阶段报告、实际 OM 的 Q ELF、父 tiling、checkpoint NZ 权重（SHA256 `da1c1106…`）及 GN scale（`5a4edd4a…`）。单 Q 对照须命中整图相同的 14,516 B 执行段（`e8f0beab…`）、相同实际 tiling（`91cf1d40…`）和常量。

原整图保存了模型输入和最终 Token/KV。为了取得中间 Q 张量，程序从已绑定 AIR 截出 Layer0 的 embedding 后算子、原始 Norm 和 Q。按原图的 INT64 索引，从已校验的完整 FP16 embedding 表读取 anchor 及 15 个 mask 行，作为这份小图的输入。小图在设备上返回归一化的 Q 输入和原 Q 输出，两种 poison 的全部位必须一致。这是从同一计算图截出的设备回放参考。第二组使用同表 16 个不同 token，第三组使用跨 Group128、baseK256 和 K 首尾边界的稀疏输入及独立 CPU 金标准。计时使用第一组。

| 单 Q OM | INT8 原始 bank | W16 bank | Matmul scratch | 数据流 |
|---|---:|---:|---:|---|
| `raw-nz-brcb` | 1 | 1 | 202,240 B | 整图原 Q，NZ、Group128、十次 K256 Iterate |
| `q-k2560-w2` | 1 × 40 KiB | 2 × 80 KiB | 46,592 B | 当前分片 Cast 后搬下一片 |
| `q-k2560-w2-prefetch` | 2 × 40 KiB | 2 × 80 KiB | 5,632 B | 当前分片 Cast 前，将下一片搬入另一 raw bank |

两个 panel 具有相同的公开 ABI、打包后的权重及 scale 字节、完整 K 归约和 W16 生命周期。每核用 16 次按行 scatter 将激活装入 A1，四个 N16 分片反量化为 B1 的 Cube NZ，每个 N64 进行一次 K2560 Iterate 并最终舍入为 FP16。MTE2→V 保护 raw 的首次读，V→MTE2 保护覆写；V→MTE3 和 MTE3→V 保护 W16 的提交及复用。B1/L0C 切换保留原全流水栅栏。ATC 复核 SDK 实际选择的 8-block、两版一致的 Cube tiling、UB/L1/L0 用量、Workspace 和各个独立机器映像。

## 看结果

- 三份 OM 各通过三组输入、两种 poison，逐位检查全部 65,536 个 FP16 输出，并检查 Repeat、输入 Readonly、Guard、同步及实际加载的 DSO。数值差异显示首个位置并保留快照。
- 三轮交错，分别比较两份 panel 与原 Q，再直接比较预取与单 raw bank。每轮每边 5 次 warmup、30 次正式 Execute+Sync。GO 要求 Median 至少下降 3%、三轮同向且 P95 无回退。直接配对决定额外 raw bank 是否值得保留，正式测量写在 `benchmark/report.json`。
- 默认另外采集原 Q 和选中版本的 **device Q task Median/P95**，报告仅取 38 次 Execute 中下标 7–36 的 30 个窗口，每个窗口恰好有一次 Q 任务，终端同时给出真实归属级别。正式延迟和采集数据各自保存。若采集失败，可在同版本下用 `--resume` 复用成功的 benchmark。
- 一次 Q 的单遍逻辑字节为 10,862,592，矩阵乘为 335,544,320 FLOP。按 204.8 GB/s 的条件峰值，单遍时间为 **53.04 µs**。用设备 Q 任务同它比较，逻辑 GB/s 的分子使用同一组字节；当前协议没有测实际 DDR 事务或缓存命中率。整图已测 Q 的约 0.442 ms 属于另一次采集。

CPU 回归可用 `python -B -m unittest discover -s framework/custom_ops/draft_q_bandwidth/test -v`。设置 `DFLASH_GE_DESCRIPTOR` 指向一次运行生成的 `cann-ge.desc` 可以一并验证原始 GE 格式的参考前缀。延迟 DMA 测试执行实际候选源文件，覆盖所有输出和 tile 所有者、失败的 Iterate、故意遗漏 raw 或 W16 消费者 fence。正式设备测试由服务器命令手动启动。
