# C：DFlashDraftLmHeadTop1 正确性基础版

Native Top1 的 M 非 4 倍数 Guard 失败已通过 NPU 定位为输出尾部 32B 写回问题：见
[具名 Guard、分配尺寸与复用产物复测说明](NATIVE_GUARD_DIAGNOSTICS.md)。
Native Top1 对齐与 `synthetic-nan_first` 实测语义修复后，用户已确认 Native-only **41/41 PASS**。
不再单独重复 Native-only；Custom 的最新验收状态与后续实验见下文。

目标 Ascend310P3 / CANN 9.0.0，遵循 [OPTIMIZATION.md 的 C 契约](../../../docs/OPTIMIZATION.md)。
**正确性基线已确认：`2a9211f` / `.runs/head-DGwqjbT8`，compare_select + Scalar Top1
在 Ascend310P3/CANN 9.0.0 完整 41/41 PASS。** 历史溢出诊断见
[OVERFLOW_CONVERSION.md](OVERFLOW_CONVERSION.md)。当前默认选择该已验收转换及 Scalar Top1，
保留旧提交、cast/minmax 路径与隔离 OPP/evidence。
`ca58d9f` Vector Probe 在全 NaN 行返回 -1，未通过 Primitive。本轮改用原始 FP16
位型分类，增加 NE/EQ/widening/mask-layout 留证，默认 Scalar 基线不变。见
[TOP1_VECTOR_EXPERIMENT.md](TOP1_VECTOR_EXPERIMENT.md) 的一键 paired 测试入口。
**新路径 CANN 编译、primitive NPU、完整 41-case、性能均为 NOT_RUN**；默认不启用 Vector。
本地 CPU 模型/声明桩通过不能替代设备验收。
本目录不改变 Qwen/DFlash 默认路径。单个 gate/up 接入目前只有
[opt-in 设计](../draft_quant/GATE_UP_OPT_IN.md)，也为 `NOT_RUN`。

## 安装后 Head OPP 检查修复

用户提供的构建 evidence：
`/home/w00949577/z50058744/qwen3.5-4B-dflash/framework/custom_ops/draft_head/.runs/head-9D7ZDIYK`；
build 为同仓库 `.build/head-SXxhKx62`。两个 Host/Kernel 均已成功编译，OPP 打包安装输出
SUCCESS；随后误调用 A 的专用检查入口，因找不到 `d_flash_group_quant_linear` 而失败。
这不是 Head kernel 缺失的证据，也不是 NPU correctness 失败；当时尚未执行 Custom 验收。

现在安装阶段之后进入 `installed_opp_preflight`，调用 `test/head_opp_preflight.py`。
它复用 A 检查器的 `check_root` / `check_binary_path`，不修改 A 的默认名称或行为，
分别检查 `d_flash_draft_lm_head_top1` 与 `d_flash_draft_lm_head_top1_audit` 目录：
每个目录至少有一个非空 `.o`，所有 `.o` 均须有按 CANN 9.0 FIRST `.o`/`.json` 规则
正确导出的同名 JSON，JSON 必须可解析为 object。路径检查包括原路径和解析后的真实路径。
报告 `opp-preflight.json.operators` 保存每个算子的目录、全部 binary/JSON 路径、字节数、SHA256。
任何缺失、空对象、无效 JSON 或路径前缀冲突均使脚本失败，不跳过预检查。
之后仍检查两个 ACLNN 头文件存在且非空，再编译 Custom runner。

检查器的 PASS 仅表示文件系统预检查，报告中的 `npu_status` 始终为 `NOT_RUN`。
本次新检查器针对服务器实际安装目录的运行结果待回传，不能以 CPU 假文件回归替代。

## 接口、算法与范围

| Tensor | shape | dtype/layout |
| --- | --- | --- |
| hidden | `[M,2560]`，M=1..15，生产 M=15 | 连续 FP16 ND，已经过 final norm |
| weight | `[248320,2560]` | 完整 FP16 NK，stride `[2560,1]` |
| token_id | `[M]` | INT64，原始词表 ID |

没有 bias、softcap、温度、词表截断、INT8 激活或 CPU fallback。输入只读。无效候选尾部
仍由调用方遮挡，本算子为全部提供的 M 行计算结果。权重仍为 FP16 NK，**不套用 INT8 NZ**。

每核拥有 cyclic N64 tile（共 3880 个）。M 在私有 UB 补齐到 16，补齐行不会输出；每份
权重 tile 复用全部实际 M 行。Matmul 使用完整 K=2560 的单输出块 Iterate，L0C 为 FP32，
GetTensorC 输出 FP32 `[16,64]` tile，再通过 310P 支持的 Vector Cast CAST_NONE 形成
唯一的 FP16 输出边界，然后进行 Top1。不是逐 token 扫完整 head，
也不对未舍入的 FP32 累加器求 argmax。SDK 可选择实际 baseK，必须正数、16 对齐且不超过
2560；singleCoreK 必须是完整 2560，baseM/baseN 必须是 16/64，不做 split-K。

Top1 的整数比较器只读取已舍入 FP16 bit pattern：数值相等取最小 ID，+0/-0 视为相等。
`Better()` 针对实际 FP16 logits 的 NaN 候选仍按“NaN 优先、最小 NaN ID”实现。
这不意味着输入 NaN 必须传播为输出 NaN。本次 Native OM 在指定用例输出有限值，不能
据此推断 Custom Matmul 的行为相同，也不修改 `Better()`。不一致仍必须失败并定位。
输入有限的溢出到 ±Inf、NaN、signed zero、舍入后并列都在设备用例中覆盖。

每分区写一个 `(FP16 max, INT64 id)` 对每行。等待全部分区完成后，core 0 做最终归约，
仍采用相同比较规则。任何 Matmul 失败的核心仍参与同步，写无效 partial ID，让最终输出
`-1` 并触发严格比较失败，而非提前退出导致其它核心永久等待。

## Workspace 与同步

逻辑 workspace 为 partial_max `[P,M]` FP16、partial_id `[P,M]` INT64，其未对齐 payload
为 `10*P*M` 字节。实现按物理行 stride=16 对齐，填充行不参与归约：

| 区域 | offset / 大小 |
| --- | --- |
| partial_max | 0 / `32*P` B |
| partial_id | `32*P` / `128*P` B |
| software barrier flags | `160*P` / `32*P` B |
| 用户 workspace 合计 | `192*P` B（P=7 时 1344 B） |

另保留平台 `GetLibApiWorkSpaceSize()` 的系统 workspace。msopgen 算子工程中 kernel 的
`workspace` 参数已指向用户空间，系统空间通过 `GetSysWorkSpacePtr()` 交给 Matmul，
不再次调用 GetUserWorkspace 叠加偏移，依据
[CANN 9.0 工程说明](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_best_practices_10_00016.html)。

310P 使用带 GM/UB 参数的 software `SyncAll`，不调用该平台不支持的无参 hardware SyncAll。
**每个核心在每次调用开始都清零全部 P×32 B flags**，不依赖调用者提供全零 workspace。
Host 设 `SetScheduleMode(1)`，核数不超过实际可用核数，避免多轮调度造成 barrier 等待。
参考 [v9.0.0 SyncAll 约束](https://gitcode.com/cann/asc-devkit/blob/v9.0.0/docs/api/context/SyncAll.md)
及 [dav_m200 实现](https://gitcode.com/cann/asc-devkit/blob/v9.0.0/impl/basic_api/dav_m200/kernel_operator_sync_impl.h)。
跨核心时序/缓存仍需 NPU 验证，CPU 模型只验证“全部 partial 完成后再归约”的抽象依赖。

每核显式用户 UB 为 X padded `[16,2560]` 81920 B、FP32 tile 4096 B、FP16 logit tile 2048 B、running max/ID
32+128 B、sync UB `32*P` B，剩余空间给 Matmul。最终归约复用已经结束的 logit tile
作为输入 partial scratch。INT64 `[M]` 用精确长度 scalar store 并清理对应 cache line，
不按 32 B 向上写越界；M=1 和 15 都检查 guards。

## 正式算子与审计算子

- `DFlashDraftLmHeadTop1`：仅返回 token ID，生产候选时延测它。
- `DFlashDraftLmHeadTop1Audit`：同一 kernel body，额外输出完整 FP16 `[M,248320]` logits。
  该路径只用于 correctness，计时标为 `NOT_RUN`，不能用它代替正式算子性能。

审计版会物化 logits；正式版只保留 tile 和 partials。不以“ID 相同”代替全部 logit 位型
一致。也不声称去掉约 14.90 MB logits 写读即可解决仍需处理的约 1.27 GB FP16 head 权重。
本版 scalar Top1/同步以正确性为先，不保证接近 6 ms 工程预算。

## 真实 capture 和 native baseline

`test/capture.py` 复用已验证 W8 checkpoint、真实 C16/C64 frozen replay、生产 DraftGraph、
feature layer 顺序与 `torch.ops.npu.npu_scatter_nd_update.default`。只在最终 `graph.norm`
输出捕获 `[1,16,2560]`，去掉 anchor 行得到 `[15,2560]`，两次回放必须相同、冻结输入只读。
在调用词表 head 前终止回放，placeholder head 不执行计算。

完整 head 从原 Target safetensors 分块读取并保值转换到生产 FP16 dtype，保存全部
`[248320,2560]` bytes、tensor key、原 shard/config 哈希。拒绝 integer codes 和非有限真实
checkpoint 权重。只在 checkpoint 明确 tied embeddings 时允许使用对应 embedding；否则
必须是实际 lm_head，不以 embedding 代替未找到的 head。

`test/export_native.py` 生成 M1..15 两类静态 native OM：

- `head_logits`：FP16 F.linear，返回完整 FP16 logits。
- `head_top1`：FP16 F.linear → argmax，返回 INT64 ID。

两者都使用 C 的 **hidden/weight 双输入 ABI**，完整权重是只读 runtime ND 输入，避免
为 30 个独立 OM 重复嵌入约 1.27 GB 常量；没有改变 W8 既有的 immutable/offline-NZ 导出。
TorchAir 的 public input storage/shape 审计与 C++ OM IO 检查确保两个输入真实存在且完整。
这些是独立接口的 native baseline，**生产 constant-head parity 为 `NOT_RUN`**，不能当成
完整生产 Draft head 的测量。两类 OM 可能采用不同编译计划，其时差不直接等于纯 Argmax 开销。

## 一键服务器验收

沿用已成功的 A4 环境、原 W8 Draft/Target checkpoint、真实 C16/C64 replay report。
所有模型/导出产物放在已存在且位于仓库外的 `AI_RUN_DIR`；`FEATURE_LAYERS` 使用原导出的顺序。

```bash
git pull --ff-only
export MODEL_PYTHON=/path/to/existing/model/python
export CANN_ROOT=/usr/local/Ascend/ascend-toolkit/latest
export DEVICE_ID=0
export AI_RUN_DIR=/path/to/existing/run
export DRAFT_DIR=/path/to/pinned/W8A16-Draft
export TARGET_DIR=/path/to/original/Target
export C16_REPLAY_REPORT=/path/to/actual-c16/private.json
export C64_REPLAY_REPORT=/path/to/actual-c64/private.json
export FEATURE_LAYERS=1,5,8,9,13,15,17,21,22,25,29  # 仅在与原导出相符时使用此例

DFLASH_HEAD_CORE_LIMIT=0 HEAD_WARMUP=5 HEAD_REPETITIONS=30 \
  bash framework/custom_ops/draft_head/run_server.sh
```

有多个 lm_head key 时设置 `HEAD_WEIGHT_KEY` 为实际 key；默认只在唯一无歧义时自动选择。
若原 embedding key 非默认值，设置 `HEAD_EMBEDDING_KEY`，沿用成功 A4 的设置。
脚本执行 capture → 两类 native 导出 → 独立 native runner → msopgen/build/package/隔离安装
两个 custom op → 全词表验收。不会修改生产 OPP。失败由 `server.log` 标记具体 phase。

只先跑 native Head/Top1 benchmark：

```bash
HEAD_NATIVE_ONLY=1 bash framework/custom_ops/draft_head/run_server.sh
```

此时 custom correctness 明确为 `NOT_RUN`。已有产物可复用，避免重做 capture/ATC：

```bash
export HEAD_CAPTURE=/path/to/capture/manifest.json
export HEAD_NATIVE_MANIFEST=/path/to/native/models/native.json
bash framework/custom_ops/draft_head/run_server.sh
```

每次 `.runs/head-XXXXXXXX/` 的 `capture-path.txt` 和 `native-path.txt` 保存真实路径。
可另用 `DFLASH_HEAD_CORE_LIMIT=1` 或 3 创建新 run 验证单核/不同分区。采用同设备、相同
输入和 protocol 比较时延，不跨协议或拿不同核心数直接归因于融合。

## 严格门槛和报告

默认共 41 个设备 case：真实 C16/C64 hidden 各 M1..15 的前 M 行，共 30 个；另有 11 个
明确标为 synthetic 的完整 N/K 用例：零、N64 边界 tie、跨分区 tie、最后词表 ID、FP16
舍入后 tie/相邻值、每行不同 ID、全负 logits、溢出 Inf、NaN、signed-zero/subnormal。
合成权重使用**完整逻辑尺寸**的 sparse 文件，NPU 仍扫描全部词表，不缩 N/K。

每个 case 必须满足：

1. native logits 两次相同；其 FP16 argmax 与独立 native Head+Top1 ID 一致。若不一致，先判
   native references inconsistent，不能错误归因或降低 custom 门槛。
2. audit 的**全部** FP16 logits 与 native logits bitwise 相同，包含零符号及 NaN payload；
   有限项报告 ULP，不以近似误差容忍代替 bits=0。
3. fused/audit/native 的全部 INT64 ID 相同；audit logits 的 argmax 也须等于 audit ID。
4. 重复输出、输入只读和全部 guards 通过。timed tail/post-poison 不得漂移。M>1 另在同一
   分配上置换 hidden 行并恢复原输入，检查相应输出行置换和恢复，防止旧 partial 跨请求残留。

CPU argmax 仅是验收 oracle；输出文件始终来自 NPU runner，未提供 CPU fallback。
输入特殊值的验收仅绑定已观察的当前 Native OM 行为；其它设备/路径不能自动沿用这一结论。

报告 `.runs/head-XXXXXXXX/data/suite.json` 保存输入/权重/OM/runner/build 身份、每 case 的
launch、workspace、完整输出文件哈希及比较。`timing_summary` 分列 native-logits、native-top1、
fused 的所有 samples/median/p95。计时为共同 `continuous-v1`，H2D、poison、回读、权重只读
检查和文件 IO 均在计时外；workspace-query 单列。audit 没有正式性能成绩。
检查完整 head 的只读状态会产生较大 D2H IO，完整套件可能耗时，不能把其墙钟时间当成 kernel 时延。

只有完整 suite PASS 才能记录 C 的独立 NPU 验收通过。native-only、capture PASS、编译 PASS
或单个 case PASS 都不是 custom 验收。完整 Draft/Decode、普通解码对照、图集成和峰值显存
另行验证，当前分别标为 `NOT_RUN` / `NOT_MEASURED`。


## NaN/Inf 设备观察与最小修复（2026-10-13）

用户提供真实 Ascend310P3 / CANN 9.0.0 evidence：
`/home/w00949577/z50058744/qwen3.5-4B-dflash/framework/custom_ops/draft_head/.runs/head-9dbiz0h7`。
原始产物在服务器，本地未读取。C16/C64 M1..15 均 PASS，synthetic 10/11 PASS；
Native 重复输出和 logits argmax/Top1 一致。唯一失败为 `synthetic_expected_ids` 的旧
NaN 优先假设（原预期 63，实际全部 64）。32B 输出对齐修复已通过这一轮真实 M1..15 检查。

保留 case 名 `synthetic-nan_first` 和完整 FP16 非有限输入，采用固定策略
`ascend310p-cann9-native-nan-inf-v1`，不根据本次执行结果动态生成期望：

| 检查 | 固定要求 |
| --- | --- |
| 原始权重 `[63,0]` / `[64,0]` | FP16 bits `0x7e00` / `0x7c00`（真实 NaN / +Inf） |
| hidden 全部 15 行 | 第 0 列 `0x3c00`，其余列 `0x0000`，原有 basis 输入 |
| Native logits 全部 15 行 token 63 / 64 | 两次输出均为 `0x0000` / `0x7bff` |
| Native Top1 | 全部 15 行 ID=64；重复及 logits argmax 一致 |
| Custom Audit / Fused | 所有 logits bitwise equal / 所有 ID exact equal，仍须实际 NPU 验收 |

仅 ID=64 不足以 PASS；任一行对应 logit 位型不同都会失败。input preflight 读取原始
文件并报告错误坐标/位型，现有整个输入只读与文件 hash 检查继续执行。
`comparison.json` 新增 `native_observed_logits_0/1`，并为位型差异提供
`first_difference.index=[row,token]`、`expected_bits`、`actual_bits`。
Custom 全词表比较也提供该诊断，包括 NaN payload、signed zero，绝不只比较两个 token。

该现象的发生位置（Matmul 内部、加载路径或输出转换）尚未定位；只记录当前设备、
CANN 版本和既有 Native OM 的观察，不推广为通用浮点/所有昇腾 Matmul 规则。
用户核对了第 0/14 行没有非有限输出；本修复固定检查全部 15 行的两个目标 token，
没有为未检查的其它位置臆造额外位型预期。其它现有严格门槛全部保留。
CPU kernel 模型仍覆盖 NaN 优先比较逻辑，它不是 Native OM 非有限值处理的模拟。
后续用户已确认此修复的 Native-only **41/41 PASS**；Custom Fused/Audit NPU correctness
与后续性能验收仍为 **NOT_RUN**。

复用已提供的两份产物（不 capture、不重新导出）：

```bash
git pull --ff-only
export HEAD_CAPTURE=/home/w00949577/qwen35-dflash-run/draft-head-UCt9vR8q/capture/manifest.json
export HEAD_NATIVE_MANIFEST=/home/w00949577/qwen35-dflash-run/head-native-gpJaP6vN/models/native.json
test -f "$HEAD_CAPTURE" && test -f "$HEAD_NATIVE_MANIFEST" && \
HEAD_NATIVE_ONLY=1 HEAD_WARMUP=5 HEAD_REPETITIONS=30 \
  bash framework/custom_ops/draft_head/run_server.sh
```

保留原 MODEL_PYTHON/CANN_ROOT/DEVICE_ID 环境。脚本使用全新 `.runs/head-*` 目录；新的
synthetic 文件仍包含同样的 NaN/Inf，不修改旧 evidence 或真实 capture/OM。
Native-only 全部 41/41 PASS 后，保留上述 HEAD_* 环境执行：

```bash
HEAD_NATIVE_ONLY=0 HEAD_WARMUP=5 HEAD_REPETITIONS=30 \
  bash framework/custom_ops/draft_head/run_server.sh
```

如希望自动按退出码串行验收，可将两条 `HEAD_NATIVE_ONLY=... bash ...` 命令用 `&&`
连接，Native-only FAIL 时不会启动 Custom。暂不进入性能优化或图接入。
回传新目录的 `data/suite.json`、`server.log`；失败时附对应 case 的 `comparison.json`、
`<mode>/runner.log`、`execution.json`、`output-allocation.json`。权重 preflight 失败时，
错误位置/位型直接位于 suite.json 的 case.error 和 server.log，可能尚无 runner.log。

## 本地验证

```bash
python3 -B -m unittest discover -s framework/custom_ops/draft_head/test -p 'test_*.py' -v
bash -n framework/custom_ops/draft_head/run_server.sh
```

完整字节比较测试需要 numpy，kernel 模型需要支持 C++17/_Float16 的编译器；本轮使用已
安装的 numpy 环境运行，没有把 skip 算作通过。CPU 模型执行正式/audit 实际 kernel body，
使用 sparse 虚拟权重覆盖全部 248320 列、M1..15、1/3/7 分区、tie、最后 ID、K 尾部、
FP16 舍入、负值、Inf/NaN、哨兵和故障核心；每种转换配置共 66 个正常运行加 1 个失败同步检查，并覆盖正负饱和与最大有限值附近的舍入边界。
它不模拟真实 Cube 累加、缓存或 CANN software SyncAll 的硬件时序。
