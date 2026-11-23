# 五层 W8A16 Draft 整图测速 · Ascend 310P3

从真实 W8A16 Draft 和 Target checkpoint 重建 C16、C64。每档编译同一组输入、相同权重及 Head 的对照和优化 OM。命令在服务器上手动启动，终端打印正式延迟、模块表、运行目录和错误。

```bash
cd /home/w00949577/z50058744/qwen3.5-4B-dflash
git pull --ff-only origin feature/ascend310p-custom-ops
bash framework/custom_ops/draft_full_panels/run.sh
```

默认设备为 0，输出在 `/home/w00949577/z50058744/qwen35-dflash-run/runs/full-draft-panels-*`。环境、临时文件、构建、缓存、OM、终端日志和原始计时均在新工作区。首次安装会从当前 pip 索引获取 ARM64 CPU PyTorch 及读取 checkpoint 的依赖，已有离线 wheel 可以通过 `DFLASH_WHEELHOUSE` 指定。ATC 和 Runner 使用独立的 `gate310p`，导出器单独使用其上的 `envs/full-draft-frontend` CPU 环境。首次完整构建要存储四份约 3 GB 的 OM、两档 AIR、权重副本和测量快照，请给它们留足磁盘空间。

| 选择 | 入口参数 |
|---|---|
| 只测一档 | `--gear 16` 或 `--gear 64` |
| 只导出 AIR，使用 CPU | `--prepare-only` |
| 到 ATC / OM 检查为止 | `--compile-only` |
| 测整图正式延迟 | `--no-profile` |
| 继续已完成的阶段 | `--resume full-draft-panels-XXXXXX`，使用原档位和输入参数 |
| 使用自己的 Draft replay | `--inputs-dir /绝对路径/输入` |

续跑创建新目录，通过前次 `stages.json` 引用各阶段产物。可先用 `--no-profile` 测完整延迟，再用 `--resume` 加上同样的档位和输入参数做模块测量。变更了输入合同的阶段会在新目录重建。ATC 子进程使用同一工作区里较短的私有临时目录，避免 Python SyncManager 超过 Linux Unix socket 地址长度；每次 ATC 启动前都用同一环境验证真实 Manager 连接。错误的阶段保留现场，终端和 `terminal.log` 均能看到报错。

## 模型与比较范围

默认从 `/home/w00949577/model_weights/qwen3.5-4b-dflash-w8a16`、`/home/w00949577/model_weights/Qwen3.5-4B` 加载，可用 `DRAFT_W8A16_DIR` 和 `TARGET_DIR` 指定。量化权重和配置经仓库 checkpoint 锁核实。Target 的语言 embedding、完整词表 FP16 Head 按 safetensors 索引加载，显式绑权时复用同一表。两个 OM 使用仓库的 `DraftGraph.forward`、五层权重、15 个 public inputs 和 11 个 outputs。它们共享 FP16 attention matmul、FP32 RMS 归一化、FP16 scale 乘法和整行 KV 更新。RMS 采用标准 GE 算子实现；当前机器可从空的自定义 OPP 环境启动。KV 更新采用复制后写整行的 functional 语义。[TorchAir 的对应转换](https://github.com/Ascend/torchair/blob/master/python/torchair/_ge_concrete_graph/ge_converter/custom/npu_scatter_nd_update.py)

| 投影 | 对照 | 优化 |
|---|---|---|
| 5 × Gate/Up | Raw NZ + B16 Scale | `DFlashGatePanel`，K2560，双 W16 |
| 5 × Down | Raw NZ + Brcb，Loop Copy | `DFlashDownPanel`，K4864，单 W16 |
| 其余 16 个 FC、KV、Q、O | Raw Group128 | 同一实现 |

前端逐字节验证十处重排权重及 Scale 的可逆性，比较每个 AIR 节点的序列化内容。ATC 检查真实六种 Shape 的回调、完整投影清单、8-block 调度、实际 Gate/Down 资源和 OM 内嵌的 AI Core ELF，以及其余 16 处的执行段。生产部署独立于本实验。

## 输入和正确性

默认标记 `REAL_CHECKPOINT_SEEDED_INPUTS`。权重取自真实模型，features 和初始 KV 使用可重复的非零输入。C16/C64 的有效 context 分别是 16/64 行，默认固定起始位置 64、anchor 42，取满 15 个预测。对照 OM 先在本次输入上生成参考，检查两种 poison、重复执行、Guard、Readonly、Token 范围、整张 KV 的有限性和写入区外的字节。随后两版独立进程均与参考逐位比较全部 15 个 ID 和十份 `[1,8,2112,128]` KV。所有 Execute 由核验过的 AscendCL 进程执行，并保留 pre-launch DSO socket、pre/post maps、同步和 Guard 检查。参考来源在报告中区分为本轮对照捕获，历史 Golden 和真实 prompt 回放需要另有文件。

自带 replay 的目录为 `c16/`、`c64/`，每档含 `features.bin`、`start_position.bin`、`valid_rows.bin`、`anchor.bin`、`proposal_count.bin` 和 `d0_key.bin` 至 `d4_value.bin`。按每档 `checkpoint-graphs/graphs.json` 的 ABI 使用小端、连续的原始二进制。

## 如何读结果

- `latency.json` 与 `benchmark-c{16,64}/report.json` 保存正式的完整 Draft Execute+Stream Sync。每档三轮交错，每轮每版 5 次预热、30 次正式计时，输出 90 个原始样本、Median、P95 及各轮倍率。边界从已加载 OM 和已就绪设备输入开始，到整图输出流同步为止。
- `profile-c{16,64}/` 另起优化 OM 进程做 msprof。每进程 38 个 Execute/Sync，只使用下标 7 到 36 的 30 次。正式 benchmark 在采集前已落盘。
- `modules-c{16,64}/module-latency.csv` 为设备任务按模块、按调用聚合的耗时，单位 ms，含 Median、P95、任务数、出现调用数及区间并集。覆盖各层 Gate/Up、Down、Q、O、KV、Attention、Norm、RoPE、SwiGLU、缓存更新，以及 FC、embedding、Head MatMul 与 Top1。`layers.csv` 汇总可定位到单层的任务，`module-calls.csv`、`timed-tasks.csv`、`task-ownership.csv` 保留逐调用、逐任务及归属依据。
- 优先使用校验通过的 device steps。若 step 证据不足，则仅将完整落入唯一 Host Execute/Sync 窗口的任务标记为 `PROVISIONAL_HOST_ENCLOSURE`。跨窗口/重复任务保留状态。跨模块 fusion 单独列出，无法识别的作为 `Unmapped`；某模块未覆盖全部 30 次时，其整段 Median/P95 留空，已观测的逐次耗时仍在明细中。并发会让任务区间重叠，因此每行任务耗时之和不能直接代表整图墙钟时间。

CPU 回归可在装有 CPU Torch、transformers、NumPy、protobuf 的 Python 中执行。将 `DFLASH_GE_DESCRIPTOR` 设为一次新运行生成的 `cann-ge.desc` 后运行 `python -B -m unittest discover -s framework/custom_ops/draft_full_panels/test -v`。完整五层缩小宽度的 CPU 用例在两档检查 26 个投影，逐位比较 GE 算子语义与独立 PyTorch 前向的全部 11 个输出。
