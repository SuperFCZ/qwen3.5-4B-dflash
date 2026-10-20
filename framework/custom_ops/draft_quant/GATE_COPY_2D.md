# Gate/Up 二维批量搬运候选

eb67337 的 layer-0 真机正确性/性能结果见文末追加，结论为暂不采用。
最新源码的 CANN 编译与 NPU 复验 **NOT_RUN**；完整 Draft Token/KV 始终 **NOT_RUN**。
现有 A5.1 和历史 OPP/Evidence 保留。此次仅改变 Gate/Up 的原始数据搬运指令组织；
没有引入 Raw Prefetch、W 双缓冲、N tile 改动、计算重排或 PIPE_ALL 改动。

## 精确范围与参数推导

独立开关 `DFLASH_GATE_COPY_MODE=loop|2d`（默认 loop），仅由 `DFLASH_SUITE=gate-copy`
使用候选。编译宏 `DFLASH_GROUP_QUANT_GATE_COPY_MODE` 默认为 0；候选要求现有 batched /
broadcast / K256。运行时同时限定 **M16 K2560 N19456、tileK256**。Down/Q/O/KV/FC
仍走原 CopyRawTile 循环，包括已通过的 Down prefetch 和 KV M80 A-UB；Head 不变。

固定 N64/K256：304 个 N tile × 10 个 K tile = 3,040 次 tile 迭代（所有核合计）。
单 tile 原 INT8 是 `[8,64,32]`，每个 K32 plane 内 2,048 B 连续，plane 间 GM pitch
为 `19456×32 = 622592 B`。GN Scale 每个 group 连续 64 个 FP16（128 B），group
间 GM pitch 为 `19456×2 = 38912 B`。候选参数如下：

| 数据 | blockCount | blockLen（32B） | srcStride（32B，尾到下块头） | dstStride |
| --- | ---: | ---: | ---: | ---: |
| INT8 NZ | 8 | 64 | `(622592-2048)/32 = 19392` | 0 |
| FP16 GN scale | 2 | 4 | `(38912-128)/32 = 1212` | 0 |

GM 首地址仍使用原 kBegin/nBegin 公式，UB 首地址、大小及顺序不变。所有起点与长度
满足 32B 对齐，blockCount 不超过 4095，块长/间隔不超过 uint16。最后一个 N/K tile
恰好结束于原 payload 尾部。原每 tile 8+2 次 raw DataCopy 变为 1+1，逻辑搬运字节
仍是 16,384+256 B，不能把指令减少比例当作时延加速比。

CANN9.0 [Basic Data Transfer](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0103.html)
列明 Atlas inference AI Core 的 GM→VECIN 支持同类型 INT8/half、分块 DataCopy 及
32B 的块长/间隔单位。本候选采用该普通 DataCopy 四参数重载；**不是** DataCopyPad
的字节 stride，也不使用其他芯片的 NDDMA 接口。仓库已验收的 KV M80 A-UB 使用相同
DataCopyParams 路径，但不能据此替代此次大 stride/INT8 路径的实际设备验收。

`gate_copy_sdk.py` 在构建前保存 receiver CANN 9.0.0 version 文件、DataCopyParams
头文件及 dav_m200 的 `copy_gm_to_ubuf` 实现 SHA256。找不到对应版本/架构则阻断，
不切换另一架构。此步为 SOURCE_INSPECTED，真正支持仍须目标 CANN 编译和 NPU 位型验收。

## 最小 A/B 验收

默认只测 **C16 layer-0-gate_up**。从已通过的 A5.1 pair suite 自动读取 K256 的
staged/full A4 报告，严格核对旧配置/头文件/捕获/Native OM/原生输出身份。
不猜随机 build 目录，不重新导出 Native，原生 reference 与 candidate 使用同一 Capture。

1. 两次独立 run_server 生成新的 loop / 2d OPP、Runner 和证据目录。两者保留相同
   accepted core cap、pipeline mode、KV M80 模式；Gate 的 selected pipeline 必须为
   serial-v1、raw bank=1。源码一致，唯一实验配置差异是 gate_copy_mode。
2. 每个构建先用 Custom Runner 的无计时入口执行两次，将全部 FP16 输出逐 bit
   对比已冻结 Native OM 的重复/tail/postcheck 一致输出。保持 guards、输入只读、
   full-M/K256 契约和非有限值拒绝。失败则该构建不运行性能阶段。
3. 通过后复用 run_a2 的 continuous-v1，Native 与 Custom 各 warmup=5、repetitions=30。
   保持输入 SHA256、原生离线 NZ Const 审计、重复、guards、readonly、timed-tail、
   postcheck；新阶段双方所有输出仍须等于冻结 Native reference。
4. 配对比较重新验证两份结果与全部输出位型，记录 Native-before/after、loop、2d
   原始 samples、median/p95、ratio 和回退标志。完整 Host launch/Cube/UB/tiling 配置
   必须完全一致；额外记录 Gate copy log，防止候选实际误跑 baseline。

```bash
bash framework/custom_ops/draft_quant/run_gate_copy.sh --a51-evidence /home/w00949577/qwen35-dflash-run/a51-fallback-20261012-063342/suite.json --output-dir "/home/w00949577/qwen35-dflash-run/gate-copy-2d-$(date +%Y%m%d-%H%M%S)"
```

沿用服务器已验收环境；可显式设置 `MODEL_PYTHON`、`CANN_ROOT`。默认设备 0。
通过首个投影后，可在**新目录**加 `--layers 0 1 2 3 4` 检查五层真实 Gate/Up；
可加 `--order candidate-first` 控制先后顺序。单次差异未超过噪声不认为有明确收益。

回传新目录的 `suite.json`、`comparison.json`、`loop.log`、`2d.log`；两份 run 指针
内包含 build-config、SDK、OPP 二进制/JSON/hash、Host launch、Runner 和分阶段原始输出。
外层 status PASS 仅代表这些选中投影的严格数值门槛，**不是性能 adoption**。
Native 是 immutable-weight OM，Custom 为 ACLNN runtime buffers；配对输入相同，但
该诊断仍不能直接推算 immutable 完整 Draft 图收益。

**完整 Draft C16 的 Token IDs 和全部 KV bitwise 门槛保留，不在本轮执行或标为 PASS。**
只有单算子正确性通过且 median/p95 显示可复现收益后，才另建 opt-in 完整 Draft 候选，
重新验证同一 W8 frozen input 的 Token/KV 与图级性能。A4 全形状 NPU 回归也是 NOT_RUN；
本轮不接入 C，不开展新的融合/数学优化。

## 本地验证范围

- CPU API 模型直接执行真实 CopyRawTile，遍历全部 3,040 个 Gate tile；按独立 NZ
  索引检查每个字节、两个 GN group、首末边界、哨兵、输入只读，以及 10→2 API 次数。
- 真实 Kernel 的 CPU 控制流模型覆盖 Gate 和原有其余形状、多核 owner、完整 FP32 K
  累加、最终 FP16 输出、缓冲区字节量、Cast/Mul 次数、barrier/event 生命周期。
- 配置默认/opt-in、错误 SDK/架构、配置/hash/log 不匹配、无计时 pre-gate、失败不进入
  timing、两构建隔离与 Token/KV NOT_RUN 等 CPU 回归。
- CPU 模型没有模拟真实 MTE2/Cube 流水、带宽或 SDK 指令实现，不能据此声称 NPU PASS。

## eb67337 真机结果（用户回传，2026-10-20 记录）

Ascend310P3 的 C16 layer-0-gate_up bitwise PASS。以下为用户提供的实测摘要，
原始 evidence 目录尚未提供，本地未读取原始 samples；不虚构路径/哈希。

| 实现 | median ms | p95 ms |
| --- | ---: | ---: |
| Loop | 2.536365 | 2.546840 |
| 2D | 2.527310 | 2.544700 |

Median ratio 1.00358×，约减少 0.36%；Native-before/after median 分别为
4.01461 / 4.02760 ms。收益接近基准波动，**暂不采用**。原报告的
`adoption=NOT_DECIDED` 与完整 Draft Token/KV NOT_RUN 保留；2D 仍为独立 opt-in。
后续 Gate Raw Prefetch 实验基于 Loop，强制关闭 2D，不叠加两个变量。
此历史结果仅属于 eb67337，不冒充后续提交的 NPU 验收。
