# Custom FP16 同 Cube recipe 对照（Gate only）

目标是测量 **W8 Off − Custom FP16 = 数据准备路径及相关访存/同步的净差异**，
不是将差值直接称为纯 Dequant time。只运行 C16 layer-0 gate/up：M16/K2560/N19456。
当前本地完成源码准备和 CPU 回归；本候选的真实 CANN 编译、NPU 数值、未插桩性能与
msprof 均 **NOT_RUN**。不运行完整 Simulator，不实现第二阶段 Dequant-only。

## 冻结输入和历史证据

默认只读复用用户已经通过的报告：

`/home/w00949577/qwen35-dflash-run/gate-same-weight-fp16-w8-20261021-014911/comparison.json`

用户回传历史 median：Native FP16 0.782460 ms，W8 Off 2.535155 ms，W8 Raw 2.332320 ms；
三者与同一 Frozen Golden bitwise equal。入口检查报告、180 个样本/模式、输出与
Golden、guard/readonly/repeat/tail、原构建/OPP、profiling CSV 和 SHA256；不重新运行
Native FP16 或 Raw。历史数字只作不同时次的参考，不与新样本混合统计。

直接读取其 `weights/weights.json` 指向的有效 FP16 `[19456,2560]` 权重。重新核对冻结
signed INT8/Group128 GN scale 的 `RN16(q * scale)` 全部位型；不换 checkpoint 或
Golden。仅在新目录生成一份 `[304,10,64,256]` 连续 FP16 tile 权重，布局为
`[N tile, K tile, N within tile, K within tile]`，逐 Tile 与原 NK FP16 比较并保存 SHA256。
CPU 计算、预打包、设备上传均不在正式计时内。

## 隔离源码与不变部分

新的算子名是 **DFlashGateFp16Tiled**，两输入 X 和 FP16 tile 权重；独立 msopgen
工程、OPP 安装根和 Runner。现有 W8 Kernel、Host、OPP、Runner、Golden、Source Lock
完全只读。生成前校验冻结 W8 源码身份，输出 `source-adaptation.json` 和逐文件 diff。
不提交生成工程或编译产物。

`tiled_sources.py` 对已验收 W8 源码作受限转换，锚点变化立即阻断：

- Host 只换输入 ABI/InferShape，严格仅支持 X `[16,2560]`、W `[304,10,64,256]`。
  platform/core 数选择、UB 预算、所有 MatmulApiTiling 配置调用、GetTiling 和容量检查
  保留原代码。按原 Off 配置生成私有 config，关闭 Gate Raw 和 2D Copy。
- Kernel 保留原 N64 cyclic 分工、K256 顺序与全 K FP32 累加结构。
  `SetTensorA/SetTensorB/Iterate(enPartialSum)/GetTensorC/End` 的整个计算和收尾区段
  逐字一致，只在全 K 结束后进行一次 FP16 输出舍入。
- 保留原 TPipe/Matmul 注册、UB InitBuffer 顺序和大小。未使用的 INT8/scale buffer
  仍保留；broadcast buffer 的一次 Duplicate 初始化及 PIPE_V 也保留。这样不会将
  更大的可用 UB 交给 SDK，改变 Matmul 的 buffer depth。
- 每 K 从 packed GM 直接 DataCopy 到原 VECOUT W16，保留原 MTE2_S/MTE2_V 的
  Fetch/Set/Wait 和搬入后的 PIPE_ALL，再保留 pre-Cube、post-Cube PIPE_ALL。
  删除 Cast/Brcb/Mul 及其内部六处 PIPE_V，不改 Matmul 内部同步。
- VECOUT W16 的 GM→UB DataCopy 使用现有 Basic DataCopy 模板。接收机保存 CANN 9.0
  dav_m200 SDK 检查结果；实际编译与数值验收才证明该通路兼容。若不支持，明确失败，
  不自动增加额外 staging buffer 或改 tiling 作为替代实现。

主机打印 `DFLASH_GATE_FP16_PLAN_*`，明确表示**基线资源计划**；baseline_dequant/scale
字段不是正在执行反量化的声明。另有 `DFLASH_GATE_FP16_PATH` 记录真实准备路径。
测试对比器生成单独的 allocation-comparison.log 以复用旧资源校验器，不改原始日志。

必须与本次 W8 Off 的所有已记录 Cube/UB/blockDim/stride/workspace-plan 字段完全一致，
否则禁止计时。冻结 Off 没有完整 TCubeTiling 二进制 dump，因此不声称独立比较过整个
不透明 tiling blob；证明范围是同源 API recipe、同参数/预算及全部可观测计划字段。

## 源码层面的搬运与同步账目

共有 304×10=3040 个 K Tile；以下是请求的 payload/调用数量，不是实测 DDR bytes/cycles。

| 每 K256/N64 Tile | W8 Off | Custom FP16 |
| --- | ---: | ---: |
| GM 权重/Scale payload | 16,640 B | 32,768 B |
| GM DataCopy 调用 | 10 | 1 |
| Cast / Brcb / Mul 调用 | 8 / 2 / 2 | 0 / 0 / 0 |
| 数据准备 PIPE_V | 6 | 0 |
| MTE2_S、MTE2_V Set/Wait 对 | 各 1 | 各 1 |
| PIPE_ALL（每 K） | 3 | 3 |
| 最终输出后 PIPE_ALL（每 N） | 1 | 1 |

总权重/scale payload：50,585,600 → 99,614,720 B。保留 codes 16,384 B、scale 256 B、
W16 32,768 B、broadcast 2,304 B 及 8,192 B SDK 预算保留；user UB budget 59,904 B，
256 KiB UB 下 Matmul 预算 202,240 B。实际 InitBuffer 总量为 253,952 B，剩余保留空间
与 Off 一致。这不是扩大 W16 或增加双缓冲的实验。

因此净差包含：反量化计算、相关 Scalar/Vector 同步、10→1 的搬运组织、两倍左右的
FP16 payload、预打包布局和相应缓存影响。FP16 原始输入 ABI 也与 W8 不同。不能仅将
时延差归为 Cast/Brcb/Mul cycles，更不能用可重叠 Pipe 活跃时间相加解释 Kernel 时间。

## 验收顺序与输出

1. 只读验证旧报告/数据，CPU tile packing 验证。
2. 新 OPP 编译、隔离安装、对象/JSON/CANN 路径规则及 ACLNN 头文件检查；构建 Runner。
3. W8 Off 与 Custom FP16 各执行两次无计时检查；FP16 也必须 bits=0、ULP=0。
   失败保存逐输出 comparison，不换 Golden、不添加容差、不进入性能阶段。
4. 逐字段确认当前 Off 与候选资源计划一致。
5. 仅两组 AB/BA，每组每轮 warmup=5、reps=30，共 60 个未插桩 samples；检查重复、
   输入只读、guards、timed-tail/postcheck，以及跨进程输出和 tiling 计划稳定。
6. 单独采集两组 msprof PipeUtilization，重新执行相同数值与计划检查。Task Duration、
   MTE2/VECTOR/SCALAR/CUBE 原始列与正式 samples 分开。

输出 `comparison.json/md`、`packed-weights.json`、`custom/build-report.json`、源码 diff、
全部原始 samples 和 profiler CSV。报告给出每轮及汇总的净差，并用当前 Off / 历史 Off
提示时次漂移。Custom FP16 − 历史 Native FP16 的差仍不是纯 Cube 或同步耗时：Native
使用不同 tiling/布局/运行时，且不是同一时次。

CPU 回归包含执行**实际派生 Kernel**的原有 CPU API/依赖模型，覆盖完整 Gate、7 个
owner、全部 10 次 K、第一/最后 Tile、FP32 中间溢出抵消、唯一输出 owner、未改输入、
缓冲预算和同步计数。这不是 Ascend Simulator，也不是 CANN/NPU 正确性证据。

## 服务器入口

```bash
CANN_ROOT=/usr/local/Ascend/cann-9.0.0 bash framework/custom_ops/draft_quant/fp16_tiled/run.sh \
  --output-dir "/home/w00949577/qwen35-dflash-run/gate-custom-fp16-tiled-$(date +%Y%m%d-%H%M%S)"
```

默认 prior-report 已绑定上述用户提供路径；可用 `--prepare-only` 只验证旧证据与打包。
请回传 `comparison.json/md`；失败时附 phase 日志以及 `custom/source-adaptation.json`。
第二阶段 Dequant-only、任何新 Simulator、其它投影、完整 Draft 均 NOT_RUN。
