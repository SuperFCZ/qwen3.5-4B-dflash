# Ascend 310P3 单个 W8A16 Down 投影

在服务器仓库根目录运行。脚本使用独立 Conda 前缀 `gate310p`，从同一批输入构建三个私有单算子 OM，校验输出后交错计时。

```bash
git pull --ff-only origin feature/ascend310p-custom-ops && bash framework/custom_ops/draft_down_bandwidth/run.sh
```

默认测 C16 形状的稠密合成数据，M=16、K=9728、N=2560，输出 40,960 个 FP16。终端直接显示错误或每组的 Median、P95 和倍率。开头的 `RUN` 是本次目录，终端副本在 `terminal.log`，原始数据及样本索引在 `experiment/result.json`。默认根为 `/home/w00949577/z50058744/qwen35-dflash-run/runs/`。

仅构建可加 `--compile-only`。若原始冻结材料仍可读，可用 `--gate8-source /absolute/path/to/gate-raw-atc8-Np20yu/source/report.json` 检查已知 Gate8 Source 摘要，再经绑定的 Flow 源报告加载 `16/layer-0-down` 的 X、NZ 权重、GN Scale、Native Golden 四份原始文件。合成模式自行生成精确的整数累加参考，报告记录数据来源。

## 计算和存储

| 版本 | 权重路径 | 每个 N64 的外层 Matmul | SDK UB scratch |
|---|---|---:|---:|
| raw-nz-brcb | NZ K32 平面循环搬运、K256 双 raw bank、Brcb + FP16 Mul | 38 | 185600 B |
| l1-k4864-w1 | A1 缓存全部 X、B1 拼一个 K4864 面板、单 W16 | 2 | 132352 B |
| l1-k4864-w2 | 同一面板和相同 AIR，两份 W16 交替使用 | 2 | 54528 B |

对照使用项目已验收的配置。在 Down 形状上，Gate 专用的 B16 分支关闭，实际父 Tiling 必须匹配此前 Down 的 SHA256 `39dd1427cafee3d85bd8736498e19b5f167f746399bd08912695ab0c5b9570d6`。对照的输入传输使用 loop。

优化版离线将 INT8 权重无损排列为 [40,2,4,4864,16]，私有 ND 端口的等效尺寸为 [40,2,311296]。FP16 scale 紧凑排列为 [40,76,64]。逐字节检查逆映射和 OM 内的两份常量。每个 block 承担五个 N64；运行时用 16 条按行的 GM→A1 scatter 将 [16,9728] 的激活排成 NZ，每个 block 只缓存一次。

整份 N64 × K9728 的 FP16 权重需要 1,245,184 B。内核分两个 K4864 面板，A1 用 311,296 B，B1 使用 622,592 B。在 Host 查询 SDK Tiling 前从 L1 扣除两者，随后再验证 SDK 的 L1/L0 资源及实际 UB。每个面板的八个分片各含 16 个通道、2432 个连续 K、19 组 scale。每个分片读一次 INT8，Cast 后按 Group128 执行 FP16 Mul，直接将产生的 NZ 分片写入 B1。双 W16 时 Vector 可在另一 bank 工作，覆写同一 bank 前须完成上次 UB→B1 搬运。单、双 W16 的实际 200 B Cube Tiling、A1/B1、Workspace 和 AIR 必须一致。

Cube baseK=256。第一个 K4864 面板初始化 FP32 L0C，第二个面板继续累加。完整 K 后只写回一次 FP16。单、双 W16 都保留 SDK 消费 B1 的完成栅栏；失败的 Iterate 直接退出，未完成的输出接受 Poison 检查。

## 测量与验收

每版先在稠密输入和跨边界输入上逐位检查全部输出。后者覆盖 Group128、K256、K2432、K4864 及最后一项 K9727。保留原 runner 的两种 Poison、Guard、Readonly、Repeat、Stream Sync 和实际进程 DSO socket 准入。任一数值或安全检查失败即停止计时。

分别测量 Raw↔W1、Raw↔W2 和 W1↔W2，每组 3 轮正反序配对，每轮每版 5 次预热、30 次正式 Execute+Sync。GO 需要每轮同向、总体 Median 下降至少 3%、P95 无回退。若单 W16 相对 Raw 已通过，双 W16 还须在直接配对中通过才能取代它。每版每对均有 90 个新样本。打包、加载和读回在计时外。表中吞吐按 796,917,760 FLOP 和单遍 25,686,016 个逻辑字节折算。实测范围限于单 Down 投影。

CPU 回归可用 `python3 -B -m unittest discover -s framework/custom_ops/draft_down_bandwidth/test`。可设置 `GATE_TEST_GE_DESCRIPTOR` 运行基于 CANN 实际 schema 的图描述和权重检查。
