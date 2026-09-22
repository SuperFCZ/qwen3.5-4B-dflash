# Qwen3.5-4B DFlash

Ascend310P 投机解码：OM 使用 W8A8 Target，Draft 可选 FP16、W4A16、W8A16；
支持 GDR Chunk 与 MTP 验证。所有 Draft 共用导出、编译、测试入口和 C++ runner。

| 文档 | 内容 |
|---|---|
| [整体架构](docs/ARCHITECTURE.md) | 模型、Draft 数据流、验证与状态、OM 划分 |
| [配置、使用与测试](docs/USAGE.md) | 环境配置、统一 AIR/OM 命令、测试参数、profiling |
| [测试结果](docs/RESULTS.md) | 原版基线、20 条逐题及数据集结果、阶段与单次 OM 时延 |
| [未来优化](docs/OPTIMIZATION.md) | W8 Draft ≤13.33 ms 目标、自定义算子接口、实现与收益预算 |

所有模型权重、构建、日志和测试产物放在源码外的 `AI_RUN_DIR` 中。
第三方来源与许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
