# 310P3 单个 Gate/Up 的 NZ 与 tile-ND 测速

在服务器仓库根目录执行。默认启动全新的 C16 Gate/Up 实验，M=16、K=2560、N=19456，设备 0。首次运行通过 Conda 创建独立的 Python 3.11 环境，安装 NumPy、CANN 编译所需 Python 依赖及 grpcio-tools。GE protobuf 从本机 CANN 9.0.0 的 schema 生成。服务器还需已有 CANN、Driver、C++、Make 和 CMake。无需恢复旧报告、checkpoint 或完整 Draft。

```bash
git pull --ff-only origin feature/ascend310p-custom-ops && bash framework/custom_ops/draft_gate_tile_nd/run.sh --synthetic
```

两版共享固定种子生成的稠密 FP16 激活、signed INT8 权重、Group128 Scale 和同一份独立 CPU 参考。参考用精确整数点积累加，按二进制比例缩放，最后舍入至 FP16。程序验证两份实际 OM 各自的 311,296 个输出，之后报告三轮交错配对的 Execute+Sync Median、P95、倍率和样本数。结果标记为 `SYNTHETIC_SEEDED_C16_GATE`。它衡量该形状的完整投影，包括反量化、Cube MatMul 和输出写回。真实模型输入和端到端 Draft 性能需另测。

构建时从仓库源码生成 NZ/B16 与 tile-ND/B16 的两个私有 OPP、AIR、单算子 OM 和同一个 ACL runner。实际 ATC 回调必须报告 8 个 block、相同的资源与 Cube Tiling。独立进程运行各个版本，保留预启动 socket、DSO 检查、两种 Poison、Readonly、Guard、Repeat 和逐位输出验证。每轮有 5 次预热及 30 次正式计时；离线权重打包、编译、模型加载与校验均在计时外。

默认把终端副本、编译日志、两个 OM 和结果写入 `/home/w00949577/z50058744/qwen35-dflash-run/runs/gate-tile-nd-*/`。开头的 `RUN` 是本次路径，完整样本在 `ab/result.json`。Conda 前缀为同一根下的 `envs/gate310p`，编译器子进程也使用这个前缀。可直接从当前 Shell 启动，无须激活环境。只准备环境和验证 SDK 依赖可用 `--env-only`。每次测量都创建新目录。只想编译两份 OM 可加 `--compile-only`。

| 每个 N64/K256 tile | NZ/B16 控制 | tile-ND/B16 候选 |
|---|---|---|
| 权重物理布局 | INT8 NZ `[80,1216,16,32]` | INT8 ND `[304,10,64,256]` |
| INT8 权重 GM → UB | 8 次 × 2 KiB | 1 次 × 16 KiB |
| INT8 → FP16 Cast | 8 次，mask 32、64 repeat | 1 次，mask 128、128 repeat |
| Scale / Cube 输入 | 两个 Group128、B16 展开、FP16 Mul，单份 `[64,256]` W16 | 相同 |

打包使用 `NZ[kt*8+plane, nt*4+row/16, row%16, lane] → tile[nt,kt,row,plane*32+lane]`，每个元素做逆映射验证。新 GE 端口 `w_tiles` 在 Const、输入描述和 Host 中统一使用四维 ND；Host 只接收上述固定形状。检查实际编译 OM 中的权重字节、唯一计算节点、真实 blockDim、Cube Tiling、UB 和 Workspace。双 Raw bank、原始 Event、PIPE_V/V_S/PIPE_ALL 以及十次 K256 累加来自冻结 Gate 源码。新项目在独立目录构建、安装和运行。

## 已有真实输入时

- `--layer0` 使用新根 `inputs/c16-layer0-gate/` 中的 `x.bin`、`w_nz.bin`、`s_gn.bin`、`golden.bin`。四份历史 SHA256 全部匹配后，新建两份 OM 并测量。
- `--archive` 可复用完整原始封存链及原始对照 OM。历史链的默认读取位置是 `/home/w00949577/qwen35-dflash-run`。所有新输出仍在新根。
- 可用 `DFLASH_RUN_ROOT` 指定新运行根。首次创建环境时可用 `CONDA_EXE` 指定 Conda 管理器。离线 pip wheels 可通过 `DFLASH_WHEELHOUSE` 指定。`--archive` 才使用 `DFLASH_PYTHON` 指定的旧流程解释器。
