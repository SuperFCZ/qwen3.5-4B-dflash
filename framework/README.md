# AIR/OM/C++ 源码索引

Qwen3.5-4B W8A8 Target + FP16 DFlash：TorchAir 导出 AIR，ATC 编译 OM，AscendCL 执行。
运行、导出、多长度对照和 msprof 统一见 [OM/C++ 使用](../docs/GDR_CHUNK_AIR_OM.md)。

| 路径 | 内容 |
|---|---|
| `python/qwen35_dflash/ascend310p/` | 图工厂、AIR 导出、ATC 编译、清单和 runner 控制面 |
| `runtime/cpp/` | AscendCL 执行、增量状态、生成调度和阶段采集 |
| `abi/dflash-chunk-v4.json` | Chunk 两遍验证，普通与 DFlash 均使用 FP32 recurrent state |
| `abi/dflash-mtp-v2.json` | MTP 验证，共用普通图与 FP32 状态接口 |
| `abi/performance-v1.json` | 计时与测量合同 |
| `scripts/lock_quant_inputs.py` | 外部模型、量化输入和 receiver 锁定 |
| `scripts/compare_cpp_closed_runtime.py` | 同设备、同输出、同计时范围的性能比较 |
| `FRAMEWORK_LOCK.json` | 框架合同 |

生成流程见 [架构](../docs/DFLASH_ARCHITECTURE.md)，张量与接口细节见
[AIR/OM 接口参考](../docs/QUANT_AIR_OM_FRAMEWORK.md)。

增量部署必须包含 `draft_context`，生成 Draft 固定 16 行。单路线最多 5 张图，双路线共用 6 个 OM 文件。
