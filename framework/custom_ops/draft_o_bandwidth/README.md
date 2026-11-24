# C16 Layer0 O 输出投影 · Ascend 310P3

在服务器仓库根目录运行：

```bash
git pull --ff-only origin feature/ascend310p-custom-ops && bash framework/custom_ops/draft_o_bandwidth/run.sh
```

默认使用真实 checkpoint 测量归档 `full-draft-panels-UFqlys`。日志、私有 OPP、独立 Runner、原始样本、msprof 和报告保存在 `/home/w00949577/z50058744/qwen35-dflash-run/runs/o-bandwidth-*`。脚本使用已有 `gate310p`，直接显示错误。可用 `--compile-only` 停在 ATC，`--no-profile` 只看正式配对，或 `--resume o-bandwidth-XXXXXX` 在新目录续接成功阶段。

## 对照和候选

真实 Layer0 O 为 `proj_layer0_o`，M16/K4096/N2560。原私有 OPP、O 的 INT8 NZ 权重（SHA256 `48a22ef0…`）、GN scale（`dbbb820a…`）、原始 tiling（`4920dd17…`）以及 14,516 B ATC 执行段（`07f2874c…`）均来自完整图。单 O 对照和设备参考前缀须选择相同执行段。

候选直接用已验收 Q 的 `op_kernel/next_n.cpp` 生成专用源码。保留其 A1 整块激活、两份 W16、B1 消费者、下一 N64 的首片预备和全部 Cube 屏障。专用化只涉及入口、形状、分片数与 scale 下标：每个 N16 的 K4096 分成两个 K2048 向量分片，共八片；第 `p` 片的通道偏移是 `16*(p/2)`，起始 Group128 是 `16*(p%2)`。两片依次写到同一完整 B1，Cube 在一个 Iterate 中完成当前 N64 的 K4096 归约，末尾一次 FP16 写出。5 个 N64/核，ATC 独立验证真实 blockDim=8。

| 项目 | 候选每核用量 |
|---|---:|
| INT8 bank / GN scale | 32 / 4 KiB |
| W16 | 2 × 64 KiB |
| 显式 UB 预算（含 8 KiB 预留）/ SDK scratch | 172 / 84 KiB |
| L1（含 SDK）/ L0B | 832 / 64 KiB |
| Cube baseM/N/K，stepKa/Kb | 16/64/256，16/1 |

## 数值和测量口径

- 从封存的 C16 优化 AIR 截取四个原始投影 FC、Layer0 KV、Q、O 及其间所有原始算子和常量。按原图 token 索引读取真实 Target embedding 行，并回放封存的 features、起始位置、有效行、proposal 数及历史 KV。前缀在 NPU 输出真实 Attention 后、O 投影前的 FP16 张量和原 O 输出。第一组还逐位比对完整的 d0_key、d0_value 与整图参考。第二组改用 16 个不同真实 token 的 embedding 做稠密压力检查，第三组用跨 K0/K4095、K2047/K2048 等边界的独立数值参考。
- 每份 OM 每组使用两种 poison，检查 40,960 个 FP16 输出、Readonly、Guard、Repeat、Sync、加载 DSO 和 pre-execute socket。正式基准使用封存输入对应的 Attention 激活。三轮正反序，每边每轮 5 次预热、30 次 Execute+Sync，各 90 个正式样本。GO 条件为 Median 下降至少 3%、三轮同向且 P95 无回退。
- msprof 另起进程，分别给出原 O 和候选的单设备任务 Median/P95、任务归属和逻辑 GB/s。使用 38 个窗口中的 30 个正式窗口，每窗须有唯一 O 任务。单遍逻辑流量为 10,862,592 B，矩阵乘为 335,544,320 FLOP。在 204.8 GB/s 条件下，单遍参照为 **53.04 µs**。同一 O 的权重和 scale 合计 10.16 MiB，小于 SDK 所列的 16 MiB L2，重复执行可能命中缓存；本次用设备任务时间衡量距离，实际 DDR 事务数另需硬件计数。
- CPU 回归使用 `python -B -m unittest discover -s framework/custom_ops/draft_o_bandwidth/test -v`。设置 `DFLASH_GE_DESCRIPTOR` 后，有 CPU Torch 和 transformers 的环境还会逐位对照截取前后的五层前向。CPU 延迟 DMA 模型执行真实候选源文件，包括两处缺失事件的故障注入。
