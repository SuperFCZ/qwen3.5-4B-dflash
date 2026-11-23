# C16 Layer0 Gate/Up：NZ/B16 与 tile-ND/B16

在 Ascend310P3 服务器仓库根目录运行：

```bash
bash framework/custom_ops/draft_gate_tile_nd/run.sh
```

脚本自动使用已验收的真实 Layer0 权重、激活、GN Scale、Native Golden 和 NZ/B16 单算子 OM。它离线打包相同的 INT8 权重，编译独立的 `DFlashGateTileNd` 私有 OPP 与单算子 OM，随后直接在设备 0 打印两版 Median、P95、三轮倍率、样本数和独立打包耗时。编译和 Runner 错误直接显示在终端；`RUN` 指向这次的新目录，`result.json` 保留原始样本。

**本次测量的是完整投影。** 输入 FP16 `[16,2560]`，输出 FP16 `[16,19456]`，包含权重反量化、Cube Matmul、完整 K 的 FP32 累加与 FP16 写出。控制为 `gate-scale-reviewfix-bXR45o` 已通过 `SCALE_GATE_GO` 的 NZ + Raw Prefetch + Scale Transpose B16 单算子 OM。两版均由同一个冻结 ACL OM Runner 新测。

| 每个 N64/K256 tile | NZ/B16 控制 | tile-ND/B16 候选 |
|---|---|---|
| 权重物理布局 | INT8 NZ `[80,1216,16,32]` | INT8 ND `[304,10,64,256]` |
| INT8 权重 GM → UB | 8 次 × 2 KiB | 1 次 × 16 KiB |
| INT8 → FP16 Cast | 8 次，mask 32、64 repeat | 1 次，mask 128、128 repeat |
| Scale / Cube 输入 | 两个 Group128、B16 展开、FP16 Mul，单份 `[64,256]` W16 | 相同 |

打包使用 `NZ[kt*8+plane, nt*4+row/16, row%16, lane] → tile[nt,kt,row,plane*32+lane]`，每个元素做逆映射验证。新 GE 端口 `w_tiles` 在 Const、输入描述和 Host 中统一使用四维 ND；Host 只接收上述固定形状。检查实际编译 OM 中的权重字节、唯一计算节点、真实 blockDim、Cube Tiling、UB 和 Workspace。双 Raw bank、原始 Event、PIPE_V/V_S/PIPE_ALL 以及十次 K256 累加来自冻结 Gate 源码。新项目在独立目录构建、安装和运行。

计时前，两版分别用两种 Poison 核验全部输出与同一 Native Golden，保留原 Runner 的 Guard、Readonly、Repeat、流同步和 DSO socket。随后按 NZ→tile、tile→NZ、NZ→tile 三轮，每轮每版 5 次预热、30 次连续 Execute+Sync。打包、模型加载、传输和读回都在正式计时之外。测量范围止于一个 Gate/Up 投影。

默认使用 `/home/w00949577/qwen35-dflash-run` 下已有实验和 `venv-hf513-probe/bin/python`，以及 `/usr/local/Ascend/cann-9.0.0`。有需要可指定 `DFLASH_RUN_ROOT`、`DFLASH_PYTHON`、`CANN_ROOT`。激活、权重、Scale、Golden 与对照 OM 按封存报告中的路径和 SHA256 恢复。
