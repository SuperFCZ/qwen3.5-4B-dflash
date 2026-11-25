# 最优 W8A16 Draft 的完整投机生成

实验使用 310P3 设备 0、W8A8 Target、五层 W8A16 Draft、Chunk Verify/commit。正式生成取与历史离线 NZ 实验相同的 20 份 prompt token（8 短、12 长）、thinking on、EOS 248044、最多输出 128 个 token、每轮最多 15 个提议。每题预热 1 次、正式 3 次，单个已加载的 C++ 进程运行所有题。ordinary 与原生 W8 的逐题比较来自封存的 `qwen35-w8nz-repro/gdr-lengths-w55diia2`；另列 [docs 的历史设备表](../../../docs/RESULTS.md) 中 `mjhazgip` 的舍入结果。新进程以 `--mode dflash` 运行，加载 Prefill、Verify 和两档 Draft，仅执行候选完整推理。

```bash
git pull --ff-only origin feature/ascend310p-custom-ops && bash framework/custom_ops/draft_w8_e2e/run.sh
```

所有本轮构建、缓存与报告写到 `/home/w00949577/z50058744/qwen35-dflash-run/`。可先单独用 `prepare_target.sh` 重建 Target，随后传入 `run.sh --target-run e2e-target-XXXXXX`。若已完成候选 Draft 或某阶段中断，`run.sh --resume w8-e2e-XXXXXX` 在新目录复用通过且身份一致的阶段。两种参数可以同时使用。`--draft-only` 单独完成两档新 Draft 的构建、校验与计时，输出范围为单次 Draft 调用。

## 实际模型及衔接

- Gate/Up 使用 `panel-k2560-w2-next-n`，Down 使用 `down-next-w2`，Q 使用 `q-n128-full`，C64 FC 使用 `fc64-raw2`。其余路径采用各自已测胜者。每档编译后核对 26 个节点的实际执行段、52 份常量、8-block 和 Cube tiling。
- Target 输出的 11 层特征按 `[1,5,8,9,13,15,17,21,22,25,29]` 拼接。新 Draft 在图内按槽位 `[0,2,5,8,10]` 取 checkpoint 指定的五层，公开 ND 输入宽 28,160，FC 输入宽 12,800。两档共享逻辑容量 2,048 和物理 KV 容量 2,112。使用旧单 Draft 的冻结输入构造等价宽输入，对 15 个 ID 和十份完整 KV 做逐位验收，含两种 poison、Guard、Readonly、Repeat 和同步。
- 若找回 `prefill.om`、`verify_chunk.om` 原文件，可用 `DFLASH_E2E_TARGET_OM_DIR` 指定，两份 SHA 必须匹配历史计划。重建使用对应发布期的锁定源码、真实 W8A8 权重、原始 FP32 递归状态与 16 行二遍 Chunk Verify。Target 绑定以实际生成的 ABI、层序、OM 身份和归档生成序列对照。512 容量的旧包仅供恢复配置和包位置。
- Target 导出使用独立 `envs/target-export310p`，默认从 `/root/anaconda3/envs/torch_npu_py3.11/bin/python` 读取 NPU 扩展。导出必须有真实 CacheUpdate、支持 effective_length 的 ChunkGatedDeltaRule、ADN attention 等前端和相配的 310P GE 包。`DFLASH_TARGET_PYTHON` 可指定受支持的 Python，`DFLASH_TARGET_OPP_PATH` 可显式指定配套 vendor。普通 CANN 编译和回放使用 `gate310p`。

## 结果

根目录 `result.json` 保存实际模型、Runner、样本及报告位置。`drafts.json` 保存两档单 Draft 的候选 Median/P95。`comparison/comparison.md` 按全体、短题、长题列出接受率、token/轮、Generation tok/s、相对历史 ordinary 的 Decode 加速比、Prefill/Decode ms/次和 Target Prefill、Draft、Verify ms/call。`comparison/cases.csv` 给出逐题对照；逐次 token、停止原因和接受轨迹保存在 runner 的 case JSON。整段 Generation tok/s 以实际输出数除以 Prefill+Decode 总时长，Decode 加速比按循环时长总和计算。重建 Target 和跨时间取样的差异随报告一起记录。
