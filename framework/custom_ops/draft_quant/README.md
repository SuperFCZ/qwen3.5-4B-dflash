# DFlashGroupQuantLinear A1：N/K 分块正确性

目标为 Ascend 310P3 / CANN 9.0.0。A1 固定 M=16、group_size=128，支持
**K=256/512/1024 × N=64/128/256，共九种形状**。实现、注册、构建及测试都在本目录，
不修改 Qwen/DFlash 模型、导出或运行主流程；真实 gate/up/down 属于下一阶段。

原版 tiny `(M,K,N)=(16,256,64)` 已由用户在服务器确认全 PASS，包括修正 scale 视图后的
native eager 比较。A1 最新服务器反馈中，K=512/1024 的六种形状通过，K=256 的三种形状
在 Host Tiling 的过严检查处被拒绝。本次按执行路径修正了该检查，**修复后的 K=256 仍待
服务器重跑，整个 A1 尚未验收全 PASS**。本机没有 NPU；CPU 测试不能替代设备或原生 OM
验收，也不提供性能成绩。

## 接口与范围

GE 类型仍为 `DFlashGroupQuantLinear`，ACLNN 调用签名不变：
`aclnnDFlashGroupQuantLinearGetWorkspaceSize(x, w_nz, s, y, ...)`，随后执行
`aclnnDFlashGroupQuantLinear(...)`。Host 从 X 和 S 读取 K/N，只接受上述固定形状集合；
不支持动态未知维度、M 的其他档位或不对齐尾块。

| 张量 | 模型逻辑 / GE origin shape | 物理 shape | dtype / format |
| --- | --- | --- | --- |
| X | `[16,K]` | `[16,K]` | FP16 ND |
| W_nz | `[N,K]` | `[K/32,N/16,16,32]` | signed INT8 FRACTAL_NZ |
| S | `[K/128,N]` | `[K/128,N]` | FP16 ND，连续 GN |
| Y | `[16,N]` | `[16,N]` | FP16 ND |

NZ 索引保持 `W_nz[k//32,n//16,n%16,k%32]=q[n,k]`，code 范围 [-128,127]，
布局版本仍为 `nz_int8_v1`。所有支持的尺寸都已对齐，无 padding。
X/S/Y stride 分别为 `[K,1]`、`[N,1]`、`[N,1]`；NZ stride 为
`[(N/16)*512,512,32,1]`。S 必须为正且有限。输入只读，输出、输入和 workspace 不重叠。

直接 ACLNN 测试的 W **ViewShape 和 StorageShape 都传物理四维 NZ shape**，stride 也使用
物理 NZ stride。CANN 9.0 单算子入口把 ViewShape 同时写入 tiling 的 origin/storage shape；
host 同时兼容 GE 的逻辑 origin ND 描述与 ACLNN 的物理 origin NZ 描述，严格校验各自组合。
不能把物理载体标成 ND。失败日志打印实际 shape、dtype、format。

## N/K 分块实现

| 全局 K | N tile | K tile | 每个 N tile 的 K 次数 | 用户 UB 字节 |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 64 | 256 | 1 | 49,408 |
| 512 | 64 | 128 | 4 | 24,704 |
| 1024 | 64 | 128 | 8 | 24,704 |

一个核顺序处理 1/2/4 个 N tile。每个 tile 独占 16×64 的输出，完整归约所有 K。
这一阶段不做多核分工、预取、双缓冲或性能调优。

- **搬入：**按全局 NZ 的 K32 plane 读取当前 64 列，plane 间距使用全局 N；
  scale 使用 `(global_group * N + n_begin)` 寻址。只搬当前 K tile 的 q/scale。
- **反量化：**signed INT8→FP16，再按各 group 的 FP16 scale 相乘，得到片上的
  ND `[64,tileK]` 权重。权重服务全部 16 行，不写回 GM。
- **K=256：**保留已验证的 `IterateAll()` 整段 K 路径，仅增加 N tile 地址偏移。
- **K=512/1024：**每个 tile 重新 `SetTensorA/SetTensorB`，第一次 `Iterate(false)`
  初始化 Cube 累加器，后续 `Iterate(true)` 累加至同一个 FP32 L0C。
  全部 K 完成后才 `GetTensorC()`，只执行一次最终 FP16 输出舍入。
- **生命周期：**DMA 后显式等待 MTE2→Scalar/Vector 事件，分别保护 scale 标量读取与 q Cast。
  每次 Iterate 后等待所有流水，才能复用 UB；K tile 之间没有
  `GetTensorC()` 或 `End()`。下一个 N tile 重新初始化累加器，最后统一 `End()`。

Host 固定 `baseM=singleCoreM=16`、`baseN=singleCoreN=64` 和 `singleCoreK=tileK`。
`SetShape(16,64,tileK)` 与 `SetOrgShape(16,N,K,tileK)` 分别声明当前计算块、全局 A/C
stride 和局部 B stride。baseK 按实际执行路径检查：K=256 的 `IterateAll()` 允许 SDK 返回
不超过 256 的正数、16 对齐 baseK（包括 256）；K=512/1024 的 partial-sum 路径继续严格
要求 baseK=128。量化 group_size=128 与 Cube 内部 baseK 是不同概念，两个 group 的 scale
在送入 Cube 前已分别应用。

[CANN v9.0.0 单核 tiling 实现](https://gitcode.com/cann/asc-devkit/blob/v9.0.0/impl/adv_api/tiling/matmul/matmul_tiling_algorithm.cpp)
的 `GetL0FactorsCand()` 会按 K 形状与片上容量选择 K 候选，最终 baseK 不能仅由
`SetFixSplit(16,64,128)` 的参数推定。旧代码对所有分支都要求 baseK=128，误拒绝整段 K
路径的合法结果，报 `tiling violates the single-output-block accumulation contract`。
现在仍拒绝不匹配的 singleCore 范围和 M/N 分块；拒绝日志会同时打印路径、期望值及实际
singleCore/base M/N/K，便于直接定位差异，不修改 SDK 的返回 tiling。

依据官方 v9.0.0 的 [MatmulImplBase](https://gitcode.com/cann/asc-devkit/blob/v9.0.0/impl/adv_api/detail/matmul/matmul_impl_base.h)
和 [Norm scheduler](https://gitcode.com/cann/asc-devkit/blob/v9.0.0/impl/adv_api/detail/matmul/scheduler/base/scheduler_norm.h)：
SetTensorA/B 重启迭代位置，`Iterate(true)` 复用已有累加器。
具体 310P 编译与数值行为以服务器验证为准；跨 K 分块后仍要求与原生输出逐位一致。

| 位置 | 分配及生命周期 |
| --- | --- |
| UB codes | `64*tileK` B；当前 tile DMA→反量化 |
| UB scale | `64*(tileK/128)*2` B；当前 tile DMA→反量化 |
| UB W16 | `64*tileK*2` B；反量化→当前 tile 最后一个 Cube 消费者 |
| UB MatMul | `floor((UB_capacity-user_UB)/32)*32` B；与用户缓冲完全分开 |
| L1 | `2*baseM*baseK*depthA1 + 2*baseN*baseK*depthB1` B，由 API 管理 |
| L0A / L0B | `2*baseM*baseK*dbL0A` / `2*baseN*baseK*dbL0B` B，由 API 管理 |
| L0C | `4*baseM*baseN*dbL0C` B；跨当前 N tile 的全部 K 保留 |
| GM workspace | Host 查询 MatMul library workspace，ACLNN 返回最终字节数 |

所有用户 UB 缓冲均按 32 B 对齐。Host 打印形状、tile 数、计划缓冲容量、转换临时区信息及
workspace；这些不是实测带宽。测试 allocator 使用 512 B 前后哨兵，保留设备指针对齐。

## 服务器构建与运行

从仓库根目录执行：

```bash
export CANN_ROOT=/usr/local/Ascend/ascend-toolkit/latest  # 指向已验证的 CANN 9.0.0
export MODEL_PYTHON=/path/to/python-with-torch-and-torch_npu
export DEVICE_ID=0
bash framework/custom_ops/draft_quant/run_server.sh
```

默认一次构建后运行全部九种形状；只跑原 tiny 回归可设置：

```bash
DFLASH_SUITE=tiny bash framework/custom_ops/draft_quant/run_server.sh
```

脚本执行 msopgen、覆盖手写源码、构建、安装独立 OPP、构建 runner，再运行 A1 suite。
本次 tiling 数据结构和 runner 都有更新，**须重新构建 OPP 与 runner**，不要混用旧安装包。
每次使用独立 `.build/a1-*`、`.runs/a1-*`（tiny 模式使用 `tiny-*`），不删除旧结果，
不替换模型 OPP，产物已忽略且不进入 Git。

**CANN 9.0 路径约束：**
[NnopbaseGetOpJsonPath](https://gitcode.com/cann/opbase/blob/v9.0.0/src/nnopbase/individual_op/executor/indv_bininfo.cpp)
使用完整路径中第一个 `.o` 推导 JSON，而非只替换文件后缀。已出现的
`.build/a1.ovznlDEL/.../DFlashGroupQuantLinear_*.o` 会被截断成 `.build/a1.json`，
使 `GetWorkspaceSize` 在 kernel 注册阶段报 `161002 / NnopbaseReadJsonConfig`，尚未执行
Host Tiling。构建、安装成功不能排除这一文件查找错误。

脚本使用连字符分隔随机后缀，并在编译前执行 `path_preflight`；父目录或符号链接目标中
的 `.o` / `.json` 同样会检查。安装后 `installed_opp_preflight` 核对本算子的实际 `.o`、
SDK 将推导的 JSON 路径、JSON 可解析性及文件 SHA256。检查失败则停止，不反复运行九种
形状。检查仅验证文件路径和 JSON，不改 SDK、生成的 metadata 或 kernel 数值逻辑。

测试入口可单独使用：

```bash
# 生成一个形状的数据，不执行 NPU。
python3 framework/custom_ops/draft_quant/test/reference.py prepare /tmp/a1-k512-n128 --k 512 --n 128
# native/check 从 manifest 读取形状，不再传 --k/--n。
"$MODEL_PYTHON" framework/custom_ops/draft_quant/test/reference.py native /tmp/a1-k512-n128 --device-id 0
```

C++ runner 用法为 `dflash_group_quant_linear_test DEVICE_ID CASE_DIR [M K N]`；
省略形状时兼容原 tiny。`run_suite.py` 自动生成九种形状的输入并调用 runner，随后在独立
进程中运行 native，避免与 runner 的设备 reset 生命周期冲突。

每种形状的输出在 `data/m16-k<K>-n<N>/`；总结果是 `data/suite.json`。某个 workload
失败后保留其阶段和日志，继续检查其他形状，任意失败都会让最终进程返回非零。

- `server.log` / `source-commit.txt`：构建、tiling、执行日志和源码提交。
- `path-preflight.json` / `opp-preflight.json`：路径前缀及已安装 object/JSON 配对检查；
  明确标记 NPU 为 `NOT_RUN`。
- 每个 workload 的 `manifest.json`：ABI、形状、tile、NZ/scale 布局和输入/golden SHA256。
- 每个 case 的 `runner.log`、`execution.json`、`actual-{0,1}.bin`：设备结果、形状身份、
  workspace、输入只读和输出/workspace 哨兵检查。
- `native-eager.json` / `native-{0,1}.bin`：原生环境、scale stride、生产 helper SHA256、
  自校验与重复性。
- `comparison.json`：custom↔CPU、native↔CPU、custom↔native 三方 bit/ULP 差异和失败位置。

## 数据与数值验收

原 tiny 六个 fixture 的输入和 CPU golden 保持不变，仍使用
`dflash-group-quant-linear-tiny-v1`。新增形状使用 `dflash-group-quant-linear-a1-v1`。
默认套件共 **81 个 case，每个 case 的 custom/native 各运行两次**：

- 原有正负零、K32/N16/group 边界、两组 dense signed、第二 group 单独激活、rounding probe。
- 覆盖全部 K128 group 边界（包括 K=255/256、511/512 等）和最后一个 K 位置。
- 最后一个 group 单独激活，防止只处理前两组或误用局部 group 索引。
- 每个 group 和 N64 tile 的 scale 有区分度，不沿用每两组重复的 tiny scale 模式。
- 大中间值相消：第一组产生约 130048，后续组抵消并留下 `(n%7+1)/64`。
  所有部分和可被 FP32 精确表示，但提前输出 FP16 会溢出；该用例明确检出错误的分段舍入。

exact case 的 CPU oracle 使用精确 dyadic 数值，要求原生与 custom 都逐位一致。
rounding probe 的 CPU 累加只作诊断，custom/native 仍必须逐位一致。无容差放宽，
正负零位型差异失败，非有限值失败，重复漂移失败。ULP 使用单调 FP16 编码，零符号距离为 1。

## 原生参考的 scale 契约

custom 读取连续 GN `[G,N]`；原生参考复用生产 `weight_quant_linear()`，从 NG `[N,G]`
存储构造 GN 转置视图，stride=`[1,G]`。二者数值相同，存储不同；原生视图不得再 contiguous。
参考策略 `production-ng-backed-scale-view-v1` 保持不变，拒绝旧错误参考结果。

依据 [CANN v9.0.0 WeightQuant 实现](https://gitcode.com/cann/ops-nn/blob/v9.0.0/matmul/weight_quant_batch_matmul_v2/op_host/op_api/aclnn_weight_quant_batch_matmul_v2.cpp)，
scale 的预处理沿用 weight 的 transpose 标志。此前把连续 GN 与 `q.t()` 一起传入导致的
GN/NG 错位已修复；本地回归保留了用户历史失败统计的完整复现，防止参考入口再次漂移。
原生 exact 自校验失败会单独报错，不能据此判断 custom kernel 错误。

## 本地验证与后续范围

```bash
python3 -m unittest discover -s framework/custom_ops/draft_quant/test -p 'test_*.py' -v
bash -n framework/custom_ops/draft_quant/run_server.sh
```

Python 测试仅依赖标准库，覆盖九种形状的布局、scale 位型、边界、相消 oracle、输出尺寸
和形状身份。另用本地 C++17 编译器（kernel 模型需支持 `_Float16`）测试共享 host 契约，以及在 CPU API 模型下执行实际
kernel 源码的 27 组索引/累加案例，检查缓冲边界、输入只读、全局 stride、调用次序和结果。
CPU 模型不模拟设备流水、L1/L0 物理行为，也不验证 CANN 编译或硬件数值。

A1 结果只验收本页的合成形状。报告中原生 OM、真实模型投影、完整 Draft 和性能仍为
`NOT_RUN`。A2 再扩真实 gate/up/down 权重与激活；后续还需 M32/M64/M80、图导出以及
[OPTIMIZATION.md](../../../docs/OPTIMIZATION.md) 规定的同输入原生 OM 和完整模型验收。
