# W8A16 单投影迁移实验

在 Ascend 310P3 上逐一比较 [迁移审计](../draft_all_w8/TRANSFER_AUDIT.md) 中可独立测量的方案。默认测六类投影的八条实际路径，分别以该形状上次实测选出的实现作为对照。Gate、Down、Q、O 在 C16/C64 都是同一 M16 形状，本轮取真实 C16 Layer0。FC 与 KV 分别测两档。每次正式计时只运行对应的单投影 OM。

在服务器仓库根目录运行

```bash
git pull --ff-only origin feature/ascend310p-custom-ops && bash framework/custom_ops/draft_w8_transfer/run.sh
```

实验使用独立的 `gate310p` 环境。每次新建 `/home/w00949577/z50058744/qwen35-dflash-run/runs/w8-transfer-*`。报错与简表直接在终端显示。`--only down gate` 可选择路径，`--resume w8-transfer-XXXXXX` 在新目录复核并复用之前合同相同的阶段。默认包含 msprof。`--no-profile` 仅测无插桩延迟，`--compile-only` 停在 ATC 与资源检查。

| 路径 | 对照 | 候选及要区分的问题 |
|---|---|---|
| Down | 单 W16，N64/K4864，循环装权重 | 单 W16 后继首片前移、两个分片合并一次 raw DMA、二者组合、双 W16 配合首片前移、按 K 面板装 A 的 N160/K2432。共 5 个 |
| Gate/Up | 已通过的 N64/K2560 Next-N | 单改 A1 按行装载、N128 的 K2560/K1280 分别配原 gather 或按行装载、原宽度合并两片 raw。共 6 个 |
| O | N64/K4096，向量 K2048，Next-N | 整条 N16 一次搬入、N160/K2048、在宽列上合并 raw。共 3 个 |
| Q | N64/K2560 Next-N | 全 K 的 N128、原宽度合并 raw。共 2 个 |
| FC C16 | N64/K2560，整 A 常驻 | N160/K1280、原宽度合并 raw、宽列合并 raw。共 3 个 |
| KV C64 | N128/K1280，按行装 A | 图内打包 A、仅合并 raw、二者组合。共 3 个 |
| KV C16 | N64/K2560 Next-N | 合并两片连续 raw |
| FC C64 | N160/K1280，图内打包 A | 保留公开 ND 输入与打包，合并两片连续 raw |

“合并 raw”让一笔 DMA 在同一缓冲区连续搬两片，每片仍分别 Cast 和 FP16 Mul。驱动须等最后一片的 Vector 读者结束才允许回填。改变 N/K 几何时同时改变离线权重打包及 Cube 分块，结果代表整个候选组合。每份 ATC 报告核对实际 8-block、归属、常量字节、UB/L1/L0、完整 Cube 与机器实现。相同 Cube 几何的候选要求 Cube 序列化相同。

Down 的真实激活来自封存的 Gate 输出，经原图的 split 与 SwiGLU 回放。参考 Down 同时产出 FP16 Golden，计时只覆盖 Down。单算子对照的机器执行段和 tiling 须与 TglfhB 中已部署的 Down 一致。其他路径复用各自获胜 OM、冻结常量和验收过的参考。每条路径测试真实回放、独立稠密压力输入以及覆盖 Group128、首尾 K、向量分片和 K 面板的边界输入；两种 poison 下全输出逐位匹配，检查 Guard、Readonly、Repeat 和 Sync。数值不符的候选单独记录，设备、同步和内存检查异常时停止。

每个通过正确性的候选与原胜者做新鲜配对，每边三轮，每轮 5 次预热和 30 次计时，顺序交替。**Median 延迟下降至少 3%、三轮同向且 P95 无回退**才晋级。多个候选晋级时按表内预登记顺序做新的直接配对，按同一标准选型。公开 ND 输入需要的图内重排全部计入 Execute+Sync。

独立的 msprof 采样记录 30 个正式窗口的投影设备任务、准备任务到投影结束的逐次区间、任务归属、MAC、Vector、Scalar 和 MTE1/2/3。计数器仅输出唯一关联的 30 个任务集的中位数。流水活动可以相互重叠。逻辑字节与实际外存流量须分别解释。

终端最后的 `Single-OM Execute+Sync` 表列出每个候选的 Median、P95 与判定。根目录的 `summary.csv` 包含候选配对，`profiles.csv` 含对照与候选的设备中位数、P95 和流水统计，`result.json` 另有晋级者直接配对、每轮比值和原始报告位置。完整 Draft 延迟需另行整网配对测量。
