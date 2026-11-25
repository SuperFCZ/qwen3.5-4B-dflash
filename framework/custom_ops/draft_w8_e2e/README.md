# 最优 W8A16 Draft 的完整投机生成

实验使用 310P3 设备 0、W8A8 Target、五层 W8A16 Draft、Chunk Verify/commit。正式生成取与历史离线 NZ 实验相同的 20 份 prompt token（8 短、12 长）、thinking on、EOS 248044、最多输出 128 个 token、每轮最多 15 个提议。每题预热 1 次、正式 3 次，单个已加载的 C++ 进程运行所有题。ordinary 与原生 W8 的逐题比较来自封存的 `qwen35-w8nz-repro/gdr-lengths-w55diia2`；另列 [docs 的历史设备表](../../../docs/RESULTS.md) 中 `mjhazgip` 的舍入结果。新进程以 `--mode dflash` 运行，加载 Prefill、Verify 和两档 Draft，仅执行候选完整推理。

```bash
git pull --ff-only origin feature/ascend310p-custom-ops && bash framework/custom_ops/draft_w8_e2e/run.sh
```

所有本轮构建、缓存与报告写到 `/home/w00949577/z50058744/qwen35-dflash-run/`。可先单独用 `prepare_target.sh` 重建 Target，随后传入 `run.sh --target-run e2e-target-XXXXXX`。若已完成候选 Draft 或某阶段中断，`run.sh --resume w8-e2e-XXXXXX` 在新目录复用通过且身份一致的阶段。两种参数可以同时使用。`--draft-only` 单独完成两档新 Draft 的构建、校验与计时，输出范围为单次 Draft 调用。

## 实际模型及衔接

- Gate/Up 使用 `panel-k2560-w2-next-n`，Down 使用 `down-next-w2`，Q 使用 `q-n128-full`，C64 FC 使用 `fc64-raw2`。其余路径采用各自已测胜者。每档编译后核对 26 个节点的实际执行段、52 份常量、8-block 和 Cube tiling。
- Target 输出的 11 层特征按 `[1,5,8,9,13,15,17,21,22,25,29]` 拼接。新 Draft 在图内按槽位 `[0,2,5,8,10]` 取 checkpoint 指定的五层，公开 ND 输入宽 28,160，FC 输入宽 12,800。两档共享逻辑容量 2,048 和物理 KV 容量 2,112。使用旧单 Draft 的冻结输入构造等价宽输入，对 15 个 ID 和十份完整 KV 做逐位验收，含两种 poison、Guard、Readonly、Repeat 和同步。
- Target 优先识别历史 OM 的 SHA。也可用 `DFLASH_E2E_TARGET_OM_DIR` 指定原始 `prefill.om` 与 `verify_chunk.om`。缺少原件时，默认从两份按 SHA 锁定的 W8A8 receiver AIR 直接恢复 GE 图并重新 ATC。每图逐文件核验 701 个 FileConstant，对其中八个注意力页表从 9 页扩展至 33 页。其余 693 个文件绑定、249 个量化矩阵乘、105 个 AdnRmsNorm、原始 CacheUpdate 链和带有效长度的 ChunkGatedDeltaRule 均保留。以原图的八个层端连接校验全部 32 层的顺序，再接出所需的 11 层。FP32 递归输入与已提交输出直接连接原始 GDR，对 Verify 分别核对第一遍的 discard 输出及第二遍的接受前缀状态。两个重建图都在进入 ATC 前校验实际端口。
- 默认 GE 路线使用 `gate310p`。有历史原生导出前端时，可显式使用 `DFLASH_TARGET_BUILD_ROUTE=torchair` 和 `DFLASH_TARGET_PYTHON`；`DFLASH_TARGET_OPP_PATH` 可指定接收机配套 vendor。每个新 Target OM 都记录与历史二进制的 SHA 对照，完整生成另报告相同前缀与提议上的 Target 预测。
- `w8-e2e-AsGU1u` 已恢复两张 Target AIR。该次编译缺少 Tensor 索引版本的 CacheUpdate，第一次设备错误指向 stream 7、task 6 的 AICPU `libaicpu_kernels.so / CacheUpdate`，该次计时无效。续接使用 `run.sh --resume w8-e2e-AsGU1u`，从 `/home/l00886453/custom_ops_llm/vendors/customize_scatter`（可用 `DFLASH_CACHE_UPDATE_OPP` 指定）绑定四输入 310P 包。编译后核验 Prefill 的 16 个及 Verify 的 256 个 CacheUpdate 实际使用 AI Core。一条独立候选请求先走通 Prefill、提议与 Verify；它的样本排除在正式 20 题统计之外。

## 结果

2026-11-25 的 `w8-e2e-c6OVnI` 已完成公开 28,160 维输入的新版 Draft 测量。C16 Median/P95 为 **13.255985 / 13.302990 ms**，C64 为 **13.592400 / 13.669090 ms**，各 90 个新样本。两档均验证了 26/26 个投影、52 份常量、8-block、15 个 ID 和十份完整 KV。续接这两份模型使用 `bash framework/custom_ops/draft_w8_e2e/run.sh --resume w8-e2e-c6OVnI`。

根目录 `result.json` 保存实际模型、Runner、样本及报告位置。`drafts.json` 保存两档单 Draft 的候选 Median/P95。`comparison/comparison.md` 按全体、短题、长题列出接受率、token/轮、Generation tok/s、相对历史 ordinary 的 Decode 加速比、Prefill/Decode ms/次和 Target Prefill、Draft、Verify ms/call。`comparison/cases.csv` 给出逐题对照；逐次 token、停止原因和接受轨迹保存在 runner 的 case JSON。整段 Generation tok/s 以实际输出数除以 Prefill+Decode 总时长，Decode 加速比按循环时长总和计算。重建 Target 和跨时间取样的差异随报告一起记录。
