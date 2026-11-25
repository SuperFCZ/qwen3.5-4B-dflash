# Ascend 310P3 完整 Draft 的 26 个 W8A16 投影

310P3 已完成 C16 和 C64 两档的整图配对与模块采集。全部 26 个 W8A16 投影集成后，正式 Median/P95 分别为 **13.601305/13.660620 ms** 和 **13.964895/14.013520 ms**。[整图、各模块和五层的实测结果](MEASURED.md)

关于六类投影已共用的优化、Down 的跨面板机会和其他候选的 SDK 资源校核，见[投影间优化迁移审计](TRANSFER_AUDIT.md)。

在服务器仓库根目录可用一条命令复现。

```bash
git pull --ff-only origin feature/ascend310p-custom-ops && bash framework/custom_ops/draft_all_w8/run.sh
```

每次生成新目录 `/home/w00949577/z50058744/qwen35-dflash-run/runs/full-all-w8-XXXXXX/`。使用独立 `gate310p` 环境和 CANN 9.0，不需要再次下载 checkpoint。可用 `--gear 16` 或 `--gear 64` 限定档位，`--compile-only` 停在两档 OM 审计，`--no-profile` 只做正确性与无插桩计时。若中断，给同一命令添加 `--resume full-all-w8-XXXXXX`，以相同的档位选择在新目录中复用已完成的阶段。ATC 与编译后审计各自保存收据。失败会在终端、`terminal.log`、`traceback.txt` 和 `result.json` 显示。

| 类型 | 每次调用数 | C16 | C64 |
|---|---:|---|---|
| Gate/Up | 5 | panel-k2560-w2 | 同 C16 |
| Down | 5 | l1-k4864-w1 | 同 C16 |
| Q | 5 | q-k2560-w2-next-n | 同 C16 |
| O | 5 | o-k4096-v2048-next-n | 同 C16 |
| KV | 5 | kv-k2560-next-n（M32） | kv-n128-k1280（M80） |
| FC | 1 | fc-a1-k2560-next | fc-pack-a-n160，含图内激活重排 |

对照是 `full-draft-panels-UFqlys` 已验收的 Gate/Up + Down 优化完整 OM。新候选从该 OM 对应的 AIR 增加 16 个获胜实现。权重使用每一层自己的 INT8 张量与 GN scale，并逐一验证无损离线重排。C64 FC 在原有激活边插入 reshape 和 transpose。整个模型保留 15 个公开输入、15 个 token ID、十份完整 KV 输出以及原生完整词表 FP16 Head。两档分别构建私有 OPP，核对每个实际投影的源码来源、ATC 内嵌 ELF 执行段、完整 Cube tiling、workspace、8-block 分配和权重端口。生成的私有包和 OM 均在本次实验目录内。

对照、候选使用同一已封存的 checkpoint 回放输入和参考，分别通过两种输出 poison、Guard、Readonly、Repeat、设备同步、DSO socket 准入及 11 个完整输出的逐位检查。所有正式数字来自加载完成后的整图 Execute+Stream Sync，每档三轮交错正反序，每轮每边 5 次预热、30 次正式计时。UFqlys 的历史输入属于真实权重与固定种子回放。

完成正式配对后，另开进程用 msprof 测优化 OM，只分析 38 次 Execute 中间的 30 次正式调用。首选完整的 device-step 验证，无法严格归属时按唯一完整 Host 窗口保留 `PROVISIONAL_HOST_ENCLOSURE`。跨模块融合、未归属以及 Notify 等待单独显示。各模块指标先逐调用累加对应任务的持续时间，再计算 Median/P95。并发任务可能重叠。W8 小计仅在每个正式窗口都完整观测到 26 个预期投影时报告。C64 的 FC 输入准备单列，若编译器与其他模块融合，详见融合行及其覆盖。

| 文件 | 内容 |
|---|---|
| `latency.json`、`benchmark-c16/report.json`、`benchmark-c64/report.json` | 两档整图原始样本、Median、P95 和逐轮倍率。内部 baseline 指 Gate/Down 对照，optimized 指全部 26 个投影 |
| `audit-c*/inventory.json` | 26 个投影的实际节点、实现、ELF、tiling 和 52 个常量的身份 |
| `components-c*/summary.md`、`components.csv`、`component-calls.csv` | 类别、层和逐调用的设备时间，以及无法完整归属的状态 |
| `modules-c*/module-latency.csv`、`layers.csv`、`module-calls.csv` | 逐层逐模块的完整分布，包括 FC、Attention、RoPE、Norm、Cache 更新、SwiGLU 和 Head |
| `modules-c*/timed-tasks.csv`、`task-ownership.csv` 和 `profile-c*/profiler/` | 原始时间线、设备任务及归属依据 |

CPU 合同测试可用 `python -B -m unittest discover -s framework/custom_ops/draft_all_w8/test -v`。设置 `DFLASH_GE_DESCRIPTOR` 并安装 CPU Torch、Transformers 后，两档用例会逐位比较缩小宽度的完整五层原图、Gate/Down 对照和全部投影集成图的 11 项输出。
