# Qwen3.5-4B DFlash

Ascend 310P 投机解码：Target 支持 FP16 / W8A8，官方 DFlash Draft 使用 FP16。
Torch-NPU 和 AIR/OM/C++ 均支持 **GDR Chunk 两遍**与 **GDR MTP** 验证。

| 要做什么 | 文档 |
|---|---|
| 直接运行 Torch-NPU，切换 Chunk / MTP | [Torch-NPU 使用](docs/DFLASH_RUN_AND_VALIDATE.md) |
| 运行 OM、多 prompt、32～1024 token 对照、msprof | [OM/C++ 使用](docs/GDR_CHUNK_AIR_OM.md) |
| 了解 Draft → Verify → Commit | [流程与架构](docs/DFLASH_ARCHITECTURE.md) |
| 查看已有速度、接受率、deterministic 漂移问题 | [结果与已知问题](docs/DFLASH_CURRENT_USAGE_AND_RESULTS.md) |

已有 Chunk 设备报告：8 条 prompt，各生成 128 token，整体 **1.50×** 加速、
接受率 **20.69%**；其中 7 条更快，1 条变慢。允许输出差异，各模式重复稳定，
任务质量未评估。MTP 和更长输出的设备结果待测。

<details>
<summary>开发参考</summary>

- [算子清单](docs/DFLASH_OPERATORS.md)
- [AIR/OM 接口与 ABI](docs/QUANT_AIR_OM_FRAMEWORK.md)
- [DFlash 源码索引](models/dflash_v1/README.md)
- [框架源码索引](framework/README.md)

</details>
