# Head FP16 overflow：保留 FP32 到最终 Vector Cast

## 真实设备证据与当前状态

用户提供 Ascend310P3 / CANN 9.0.0 evidence：
`/home/w00949577/z50058744/qwen3.5-4B-dflash/framework/custom_ops/draft_head/.runs/head-5UuIaB5v`。
原始产物保存在服务器，本地未读取，以下数值来自用户回传。

双算子编译、打包、隔离安装、设备执行已成功。C16/C64 M1..15 共 30 个真实 case PASS，
synthetic 10/11 PASS，所有 Native/Custom Top1 ID 一致，NaN/Inf 用例也 PASS。
唯一失败是 `synthetic-overflow_inf`：hidden[:,0]=2，weight[63/64,0]=65504，完整
FP32 数学结果 131008。Native FP16 为 `0x7bff`（65504），旧 Audit 为 `0x7c00`
（+Inf），共 15×2=30 个位型差异，max_ulp=1。这个 1 ULP **不能容忍或豁免**。

**本修复版本的 CANN 编译、安装、完整 41-case NPU correctness 均为 NOT_RUN。**
旧版本其余 40 项 PASS 不等于本修复已经通过。负溢出/边界的新增结果目前仅为 CPU 回归，
不能写成已经获得了 Native 的负溢出实测。生产 Draft 与 A5.1 均未修改。

## 固定 SDK 核查

官方仓库 `cann/asc-devkit` tag **v9.0.0**，commit
`c8c1b9df6bb9cd61c09751c7ef414b89e06af274`，本次在临时目录读取其源码：

1. [Matmul GetTensorC 实现](https://gitcode.com/cann/asc-devkit/blob/v9.0.0/impl/adv_api/detail/matmul/matmul_impl_base.h)
   的 LocalTensor overload 只提供输出、enAtomic、enSequentialWrite，没有饱和选项。
2. [CopyCubeOut](https://gitcode.com/cann/asc-devkit/blob/v9.0.0/impl/adv_api/detail/matmul/stage/copy_cube_out/copy_cube_out_datacopy.h)
   的 local ND 路径：先 `CopyCo12Co2WithoutQuant`，再 `TransNZ2NDForDstUB`。
3. [dav_m200 DataCopy](https://gitcode.com/cann/asc-devkit/blob/v9.0.0/impl/basic_api/dav_m200/kernel_operator_data_copy_impl.h)
   的 float→half L0C→UB overload（约 526–543 行）使用
   `copy_matrix_cc_to_ubuf(..., CRMODE_F32toF16_NONE)`；float→float overload 使用
   `CRMODE_NONE`。公开 enhanced params 没有独立 FP16 saturation 开关。
4. Matmul 的 SetQuantScalar 是量化类型路径，不是此 FP16 输入/输出场景的饱和配置。
   未找到可以直接切换当前 GetTensorC 半精度输出饱和行为的公开 Matmul 参数。
5. 不使用其它芯片的 SetSaturationFlag/SetCtrlSpr 技巧：固定 SDK 中没有可用的相应
   310P 公共路径；[SetSaturationFlag 支持表](https://asc.gitcode.com/api/SIMD-API/%E5%9F%BA%E7%A1%80API/%E7%89%B9%E6%AE%8A%E5%AF%84%E5%AD%98%E5%99%A8%E8%AE%BF%E9%97%AE/SetSaturationFlag%28ISASI%29.html)
   也明确排除 Atlas 推理系列 AI Core。

合法候选采用 [CANN 9.0 Cast 文档](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0073.html)
中 **Atlas inference product AI Core 的表 5**：float→half 支持 CAST_NONE/CAST_ODD。
该文档说明浮点缩窄的默认溢出处理为饱和；有精度损失时 CAST_NONE 表示最近偶数舍入。
进一步核对 [dav_m200 vconv 实现](https://gitcode.com/cann/asc-devkit/blob/v9.0.0/impl/basic_api/dav_m200/kernel_operator_vec_vconv_impl.h)
约 217 行：CAST_NONE 分支调用 `vconv_f322f16`；显式 CAST_RINT 分支在该平台不支持，
所以不能简单替换成 CAST_RINT，也没有使用 CAST_ODD。

Native OM 的具体内部 kernel/转换指令未从 `.om` 逆向验证；目前仅确认其输出具有上述
饱和特征，不声称它必然执行同一条 Cast，或把该现象推广到所有 Matmul/硬件。

## 最小转换路径变更

```text
FP16 A/B → 原 Matmul 完整 K FP32 累加
→ GetTensorC(FP32 VECIN ND [16,64])
→ Vector Cast(FP32 → FP16, CAST_NONE, 1024 elements)
→ 原 Top1 / Audit FP16 输出
```

Host SetCType 与 Kernel HeadC 一致改为内部 FP32；外部 Audit tensor 仍为 FP16，正式
输出仍为 INT64。只在全 K 完成后、Top1 之前执行一次 FP16 转换，没有中间 FP16 累加。
没有扫描 FP16 输出寻找 Inf/NaN，没有根据 token ID 特判，没有新逐元素 Scalar 修补循环。
不在已丢失 FP32 来源信息的 `0x7c00` 上猜测是有限值溢出还是输入特殊值。
已有 NaN/Inf 原始权重、其 Native 实测规则、signed zero 和 Better() 全部保留；如新
转换路径在设备上表现不同，完整 Audit 位型检查必须失败，而不是改 Native 预期。

新增一个私有 FP32 tile TBuf（4096 B）；从原 Matmul UB 配额扣除 4096 B，保持每核
UB 总预算不变，两个 buffer 不重叠。p=7 时：

| 预算 | 原路径 | 本修复 |
| --- | ---: | ---: |
| 用户显式 UB | 84352 B | 88448 B |
| Matmul UB（256 KiB 总量） | 177792 B | 173696 B |
| GM 用户 workspace | 1344 B | 1344 B |
| GM system workspace | 原平台值 | 同一平台接口值 |

GM `partial_max`、`partial_id`、sync offsets 和 `WorkspaceBytes(P)=192*P` 不变，
跨核 SyncAll、失败核心参与归约、最小 ID tie-break 不变。Cast 只增加必要的核内依赖。
Host launch 新增 `output_conversion=fp32-ub-cast-none-v1` / `fp32_tile_bytes=4096`，
并检查新 UB 账本，避免错误加载旧转换实现。未知 SDK tiling 返回值继续使用原严格校验。

这是 correctness 修复，不声明性能改善；增加的 FP32 搬运和 Vector Cast 成本须由 NPU
实测衡量。没有新全词表 Scalar 循环，也没有新增全词表 GM 浮点缓冲区。

## 回归与设备门槛

- CPU kernel 模型经 GetTensorC 保留 FP32，调用实际 kernel 的 Cast 位置后才转 FP16。
- 增加负溢出的完整词表/Fused/Audit 路径，以及正负 65472、65488 tie、其相邻 FP32、
  65503、65504、65519、65520、65536、131008；检查 RNE ties、±0 和 subnormal 位型。
- CPU Cast 模型是请求转换契约的 oracle，不是 CANN intrinsic 的硬件仿真。
  其 NaN/Inf 行为不证明真实硬件；Better() 的直接 NaN 优先测试仍保留。
- 回归复现用户的 30 mismatches / 1 ULP，必须 FAIL；ID 相同不能豁免。
- 对原 `synthetic-overflow_inf` 的两次 Native 输出增加全部 15 行 token63/64=`0x7bff`
  的实测规则校验；原 hidden/weight 和 expected ID=63 不修改，不新增/删除 41-case 清单。
- 完整真实与特殊值用例、重复、Guard、输入只读、row permutation、timed-tail/postcheck
  继续严格执行。与 Native 全词表 FP16 bitwise、INT64 ID 必须同时一致。

复用既有 capture / 30 个 Native OM，沿用原 MODEL_PYTHON、CANN_ROOT、DEVICE_ID：

```bash
git pull --ff-only
export HEAD_CAPTURE=/home/w00949577/qwen35-dflash-run/draft-head-UCt9vR8q/capture/manifest.json
export HEAD_NATIVE_MANIFEST=/home/w00949577/qwen35-dflash-run/head-native-gpJaP6vN/models/native.json
test -f "$HEAD_CAPTURE" && test -f "$HEAD_NATIVE_MANIFEST" && \
HEAD_NATIVE_ONLY=0 HEAD_WARMUP=5 HEAD_REPETITIONS=30 \
  bash framework/custom_ops/draft_head/run_server.sh
```

不单独重跑 Native-only；Custom 套件内 Native 执行用于同输入 golden 对照。保留所有旧
证据，脚本创建新 build/OPP/run 目录。回传 `data/suite.json`、`server.log` 和失败 case 的
`comparison.json` / 对应 `runner.log`；重点检查 overflow、nan_first、signed_zero_underflow、
rounding cases 和真实 M1..15。任何新差异都保持 FAIL。
