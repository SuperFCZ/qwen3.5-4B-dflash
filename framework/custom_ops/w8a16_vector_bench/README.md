# 310P3 Vector 吞吐微基准

在服务器仓库根目录运行下面一条命令。默认测设备 0 的 UB 向量运算，以及 Gate/Up 和 Down 两种实际权重尺寸的 GM 读入。脚本在本次实验目录用 CANN 9.0 编译并直调一个 dav-m200 AiCore 核函数，结果直接打印在终端。无须模型、OPP 安装或 msprof。

```bash
bash framework/custom_ops/w8a16_vector_bench/run.sh
```

只看 Vector 时在末尾加 `--vector-only`。可用 `--device 1` 指定设备，或用 `--blocks 8` 对齐已测的 ATC 启动宽度。默认通过 ACL 查询 AI Core 数，查询不可用时尝试当前线程的 Cube core 资源值。终端打印所用查询和实际请求的 block 数。安装目录默认 `/usr/local/Ascend/cann-9.0.0`，可用 `CANN_ROOT` 指定。编译失败直接展开日志。运行错误带测试行、ACL 调用及返回码。每次的终端副本和编译日志保存在 `/home/w00949577/z50058744/qwen35-dflash-run/runs/vector-*/`，终端开头会打印完整路径。

## 怎么读结果

| 行 | 指标 | 用途 |
|---|---|---|
| `FP16 MAC`、`FP16 Mul` | TFLOP/s | UB 中连续的 16,384 个 FP16 元素。MAC 用 `MulAddDst`，每元素计 2 FLOP，Mul 计 1 FLOP。单 block 和整组 block 的 MAC 可直接对照。 |
| `Cast tile-ND`、`Cast NZ -> ND` | 十亿权重/秒 | 前者连续满宽转换，后者按当前生产路径分 8 个 K32 面，将 INT8 转为 FP16，写入同一 `[64,256]` UB 布局。 |
| `NZ + Brcb + Mul`、`NZ + B16 + Mul`、`tile-ND + B16 + Mul` | 十亿权重/秒 | 完整 Group128 展开与 FP16 乘法。前两行隔离广播方式，后两行隔离输入布局。B16 调用已验收的 Gate helper。 |
| `GM stream` | µs/一次权重扫描、十亿权重/秒、有效输入 GB/s | 逐次读完 Gate 或 Down 的 codes 和 GN scales。copy 行保留实际拷贝与同步，组合行增加反量化。 |
| `M16 projection rate model` | ms、等效矩阵 TFLOP/s | 按实测 Vector 和拷贝速率，估算 M16 每个投影在这两项资源上的时间预算。 |

**布局是实验变量。** NZ 的 codes 为 `[K/32,N/16,16,32]`。另一种离线打包方式为 `[N/64,K/256,64,256]`（tile-ND）。二者使用同一批确定性的逻辑 INT8 权重、相同的 FP16 `[K/128,N]` scales，产生相同的 FP16 权重。tile-ND 是用来评估连续搬运和满宽 Cast 的候选布局，离线打包耗时不计入。两个流式用例分别读取 50,585,600 和 25,292,800 个逻辑输入字节/次。

**计时口径。** 预热 3 次，取 21 个设备事件样本，每个样本包含同 stream 的 3 次 launch，按次报告 Median/P95。各行交替正反序采样。内核注册、设备内存分配、输入构造、H2D、读回和数值检查都在事件之外。UB 的 Cast 和反量化行每个 block 将两个 tile 放入 UB 轮用，热循环 2,047 次，单次启动的初始化和最后一次写回摊入结果。连续算术也使用 2,047 次循环。GM 行按 N tile 循环分工，使用单缓冲、同步及小块写回来观察每个 tile 的结果。其有效 GB/s 为请求的 codes 和 scale 字节除以设备时间。时间包括这些写回，实际 DDR 事务数和缓存命中率未测。生产算子的双缓冲、Cube 和其他图节点有各自的开销与重叠关系。

**估算上限。** 令 `R` 为对应布局完整反量化的实测 weights/s，`B` 为对应 copy 行的有效 bytes/s。每个权重供 M=16 行共用一次、block 均衡时，Vector 的等效矩阵处理率为 `32 × R` FLOP/s。内存项为 `(NK + 2NK/128) / B` 秒。表里取向量项与内存项的较大值，并对不能整除的 N tile 用最忙 block 校正向量项。纳入实测 Cube 吞吐 `P` 时还要考虑 `2MNK/P` 秒。这里的 dense MAC 是该内核实测可持续参考值，完整反量化的设备吞吐还包括 Cast、布局搬运和屏障。该预算适用于指定投影和假设，整图延迟须单独测量。[CANN 9 向量计算与数据转换 API](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_00120.html)
