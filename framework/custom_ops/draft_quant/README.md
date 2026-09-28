# DFlashGroupQuantLinear tiny correctness

Ascend 310P3 / CANN 9.0.0 的独立 Ascend C 原型，固定
**M=16、K=256、N=64、group_size=128**。源码、构建和 ACLNN 测试均在本目录；
未接入 Qwen/DFlash 模型、AIR 导出或运行主流程。`workloads.json` 仍是完整优化需求账本。

当前状态：服务器已反馈构建、安装、ACLNN 执行、输入只读和哨兵检查通过，六个 case 的
custom 输出与 CPU 参考均为 0 bit 差异。此前原生 eager 对照存在 GN/NG scale 视图错误，
本次已修复该对照入口，修复后的 native 比较仍待服务器复核。本地 CPU 单元测试只验证
数据布局、oracle 和比较器，不替代 NPU 数值、原生 OM 或性能验收。

## 固定接口

GE 类型为 `DFlashGroupQuantLinear`，生成的 C 接口为
`aclnnDFlashGroupQuantLinearGetWorkspaceSize(x, w_nz, s, y, ...)` 和
`aclnnDFlashGroupQuantLinear(...)`。tiny ABI 为 `dflash-group-quant-linear-tiny-v1`；
M/N/K、group size 和 `nz_int8_v1` 为编译常量，无动态 shape 或可调属性。

| 张量 | 模型逻辑 shape（GE origin） | 物理/storage shape | dtype / format |
| --- | --- | --- | --- |
| X | `[16,256]` | `[16,256]` | FP16 ND |
| W_nz | `[64,256]` | `[8,4,16,32]` | signed INT8 FRACTAL_NZ |
| S | `[2,64]` | `[2,64]` | FP16 ND，GN |
| Y | `[16,64]` | `[16,64]` | FP16 ND |

```text
W_nz[k//32,n//16,n%16,k%32] = q[n,k]
W16[n,k] = RN16(half(q[n,k]) * S[k//128,n])
Y = CubeMatMul(X, W16.T), output FP16
```

NZ 字节序与现有 `weight_prepack.pack_int8_nz` 一致；tiny 维度全部对齐，无 padding。
W 的物理 stride 为 `[2048,512,32,1]`。X、S、Y 连续，stride 分别为
`[256,1]`、`[64,1]`、`[64,1]`。GE 描述保留 W 的逻辑 origin `[64,256]` / ND，
storage 为 `[8,4,16,32]` / FRACTAL_NZ。
**直接 ACLNN 测试使用物理 NZ ViewShape `[8,4,16,32]`，storage 同形，stride 为
`[2048,512,32,1]`，format 为 FRACTAL_NZ。** `aclCreateTensor` 的首个 shape 参数是
ViewShape，不能把它等同于 GE 的 origin_shape。两条路径的权重字节完全相同；
tiny 的模型 N=64、K=256 仍由固定契约声明，manifest 中 `w_origin` 记录该模型逻辑形状。
从模块 scale `[N,G]` 到 `[G,N]` 只移动 FP16 字节。

在 [CANN opbase v9.0.0 的单算子实现](https://gitcode.com/cann/opbase/blob/v9.0.0/src/nnopbase/individual_op/executor/indv_executor_tensor.cpp)
中，`NnopbaseSaveTensor` 将 ViewShape 同时写入 tiling 的 origin/storage shape，
NZ 的 origin/storage format 也都来自输入 storage format。旧 runner 用 `[64,256]`
ViewShape 搭配四维 NZ storage，会在 tiling 中变成二维 shape，触发 `ValidInput` 拒绝；
只删除 `origin_format == ND` 的判断仍不足以修复。
host 现在严格接受 GE 逻辑描述和 ACLNN 物理描述两种组合，仍拒绝二维 storage、
错误轴序、ND 物理载体及错误 dtype。校验失败会打印三个输入和输出的实际 dtype、
origin/storage format、origin/storage shape。升级时须同时重新构建 OPP 和 ACLNN runner。

GE infer/tiling 拒绝不符合固定 shape、dtype、format 的描述。S 必须为正且有限，
测试 host 在复制到设备前校验其 FP16 位型；直接调用生成 API 的调用方也必须满足这一前提。
输入只读，输出、输入和 workspace 分别分配，不能重叠。无 bias/offset、激活 INT8 量化或隐藏状态。

## 内核与存储

单核拥有全部 `[16,64]` 输出。先把 16 KiB NZ codes 和 GN scale 搬入 UB，
按两个 K group、每组 4 个 K32 片段进行 signed Cast 和 FP16 Muls，形成片上的
ND `[64,256]` FP16 tile。tiny 将整个 K 放在 UB，随后一次 MatMul API 调用完成 K 归约；
内部 base tile 固定为 M16/N64/K128。W16 复用全部 16 行，绝不写回 GM。
这依赖 tiny 尺寸，不可直接扩大为完整模型权重。

反量化阶段串行使用全流水屏障和 vector 屏障，不做预取、流水重叠、多核或 split-K。
MatMul 使用 FP16 A/B、Cube 累加和最终 FP16 输出，不物化逐 group 的 FP16 部分和。
原生 WeightQuant 的内部舍入、累加顺序和 subnormal 行为仍以设备对照为准。

| 位置 | 分配 | 生命周期 |
| --- | --- | --- |
| UB / VECIN | q 16,384 B + S 256 B | DMA 到内核结束 |
| UB / VECOUT | W16 32,768 B | 反量化到 `mm.End()` |
| UB / MatMul 临时区 | `floor((UB_capacity-49,408)/32)*32` B | MatMul 布局转换/输出 |
| L1 | `2*baseM*baseK*depthA1 + 2*baseN*baseK*depthB1` B | MatMul API 管理 |
| L0A / L0B | `2*baseM*baseK*dbL0A` / `2*baseN*baseK*dbL0B` B | MatMul API 管理 |
| L0C | `4*baseM*baseN*dbL0C` B | MatMul API 管理 |
| GM workspace | `GetLibApiWorkSpaceSize()`，由 ACLNN 返回最终字节数 | 单次调用到 stream 同步 |

用户 UB 区和 MatMul 临时区不重叠，均 32 B 对齐；tiling 使用扣除用户缓冲后的 UB 预算，
不能照搬 FP16 smoke 中“把整块 UB 都交给 MatMul”的做法。tiling 日志打印实际选择的
base/depth 所对应的计划缓冲容量和 `transLength`；这不是实测显存或流量。
GM 使用 `aclrtMalloc`，测试缓冲增加 512 B 前后哨兵，保持传入指针对齐。

该数据路径参考官方的 [Matmul 参数支持表](https://www.hiascend.com/doc_center/source/zh/canncommercial/80RC3/apiref/ascendcopapi/atlasascendc_api_07_0601.html)
中 Atlas 推理系列的 UB/VECOUT 输入，以及 [TCubeTiling 约束](https://www.hiascend.com/doc_center/source/en/canncommercial/800/apiref/ascendcopapi/atlasascendc_api_07_0673.html)。
CANN 9.0.0 的实际编译和行为必须在目标服务器验收。

## 服务器运行

在仓库根目录执行：

```bash
export CANN_ROOT=/usr/local/Ascend/ascend-toolkit/latest  # 指向已验证的 CANN 9.0.0
export MODEL_PYTHON=/path/to/python-with-torch-and-torch_npu
export DEVICE_ID=0
bash framework/custom_ops/draft_quant/run_server.sh
```

脚本依次执行 msopgen、覆盖手写源码、构建、安装独立 OPP、构建 ACLNN runner、
执行 custom、执行同输入原生 eager WeightQuant、逐位比较。
无需模型权重。原生对照复用仓库的 `weight_quant_linear()`，使用
`antiquant_group_size=128, inner_precise=0`；缺少 torch_npu、原生执行失败、重复漂移、
非有限输出或原生 exact case 未通过 CPU 自校验时直接失败，没有 CPU fallback。

每次运行保留独立的 `.build/tiny.*/` 和 `.runs/tiny.*/`，不删除此前结果。
OPP 安装在该次 build 目录内，安装器子进程清除 `ASCEND_CUSTOM_OPP_PATH` 后安装；
不会替换现有模型 OPP。生成项目、安装包和测试产物均已忽略，不进入 Git。

输出包括：

- `server.log`：编译、安装、tiling 和执行日志；失败时显示所在阶段。
- `data/manifest.json`：固定 ABI、NZ origin/storage、输入及 CPU golden 的 SHA256。
- 每个 case 的 `actual-{0,1}.bin`、`native-{0,1}.bin`、`execution.json`：
  两次执行、输入只读检查、输出/workspace 哨兵和 workspace 字节数。
- `data/native-eager.json`：原生环境、输入 manifest 绑定、输出 SHA256、scale 存储/视图/stride、
  生产 helper 源码 SHA256，以及原生输出对 CPU 的自校验结果。
- `data/comparison.json`：逐 case 位型差异数、max abs、max ULP、首个差异位置、重复漂移。
  `cpu` 为 custom 对 CPU，`native_eager` 为 custom 对 native，新增 `native_vs_cpu`
  为 native 对 CPU；控制台同时打印这三组差异，`failed_checks` 标明失败的是哪一组。

默认包含 6 组 case：正负零、NZ/group 边界 one-hot、两组 dense signed、仅第二 group、
非二进制精确 scale 的 rounding probe。所有 code 范围含 -128/127。
前 5 组的整数部分和有严格 FP32 可表示界，CPU exact oracle 与原生输出都要求逐位相等。
rounding probe 的 CPU 累加仅用于诊断，custom 必须逐位匹配原生 WeightQuant；不放宽容差。
比较 FP16 原始 bits，正负零差异也失败；NaN/Inf 失败。ULP 使用单调 FP16 编码，
`-0` 与 `+0` 编码距离为 1，另报独立位型差异。

这里的 `PASS` 仅表示 tiny 的 CPU exact + native eager 对照通过。
报告中的 native OM、完整 Draft、性能始终为 `NOT_RUN`。
接入主流程前仍须完成 [OPTIMIZATION.md](../../../docs/OPTIMIZATION.md) 要求的
同输入原生 OM 逐位对照、真实权重/激活、多档位和完整模型验证；不得用 eager 替代 OM 验收。

## 本地检查

```bash
python3 -m unittest discover -s framework/custom_ops/draft_quant/test -p 'test_*.py' -v
bash -n framework/custom_ops/draft_quant/run_server.sh
```

数据与比较器测试仅需 Python 标准库。覆盖独立 NZ permutation、signed code、scale 位型转置、
精确整数 oracle、group 边界错误检测、ULP/非有限值、文件损坏及缺少设备证据时拒绝通过。
新增的 host 元数据契约测试使用本地 C++17 编译器，直接测试共享 shape 校验函数，覆盖
GE/ACLNN 合法描述、旧二维 NZ 描述及同字节数的错误轴序；缺少 C++ 编译器时明确跳过。
该测试不依赖 CANN，不代表 Ascend C 编译、ACLNN 数值验证或性能测试通过。

## 原生对照的 scale 视图修复

custom 输入继续使用连续 GN `[2,64]`。原生 eager 调用中，weight 是 `q.t()` 转置视图，
scale 必须与生产调用一致：连续 NG `[64,2]` 存储上的 GN 转置视图，stride=`[1,2]`。
两者表示同一组 scale 数值，但物理存储不同。

[CANN ops-nn v9.0.0 的 ACLNN 实现](https://gitcode.com/cann/ops-nn/blob/v9.0.0/matmul/weight_quant_batch_matmul_v2/op_host/op_api/aclnn_weight_quant_batch_matmul_v2.cpp)
在 `CheckContiguous()` 中，把 weight 的 transpose 标志同时传给 scale 的
`TensorContiguousProcess()`；该分支的 `CreateTransposedView()` 先重建视图，
随后 310P 的 `TransposeAndTransDataForInputsGroup()` 执行 group scale 转置。
旧测试脚本把连续 GN 与 `q.t()` 一起传入，使 GN 字节被误读成 NG。
CPU 上模拟这一错位，可以完整复现服务器报告的五组非零输入差异：
764、1022、1024、763、1023 个 bit mismatch，包括各组的 max abs、ULP 和首个错误位型。

现在先把 `s_gn.bin` 逐字节转回 NG，再调用生产 `weight_quant_linear()` 生成正确转置视图。
原生参考策略标记为 `production-ng-backed-scale-view-v1`，比较器拒绝没有该标记的旧 native
结果；原有 `actual-*.bin` 和输入无需重算，CPU 与 native 的精度门槛没有放宽。
本次不修改 custom kernel、B 的 VECOUT/ND/transpose 路径或 tiling 参数。

若复用已经成功执行的 custom 结果，可在**已加载相同 CANN/torch_npu 环境**的服务器仓库
根目录运行以下命令。先复制原 data，保留旧证据；这里只重跑原生对照与比较，无需重新编译：

```bash
old_data=framework/custom_ops/draft_quant/.runs/tiny.ddgHaC8f/data  # 修改为实际已运行目录
rerun_root=$(mktemp -d "$PWD/framework/custom_ops/draft_quant/.runs/reference.XXXXXXXX")
cp -a "$old_data" "$rerun_root/data"
"${MODEL_PYTHON:-python3}" framework/custom_ops/draft_quant/test/reference.py native "$rerun_root/data" --device-id "${DEVICE_ID:-0}"
"${MODEL_PYTHON:-python3}" framework/custom_ops/draft_quant/test/reference.py check "$rerun_root/data"
```

若原生 exact case 自校验失败，`native` 返回非零并保存 `native-eager.json` 中的诊断，
不能据此认定 custom kernel 出错。`rounding_probe` 的 native/CPU 差异仍仅作诊断；
custom/native 必须逐位一致。
