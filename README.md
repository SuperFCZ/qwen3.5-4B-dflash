# Qwen3.5-4B DFlash

Ascend310P 投机解码：OM 使用 W8A8 Target，Draft 可选 FP16、W4A16、W8A16；
支持 GDR Chunk 与 MTP 验证。W8 离线 NZ 与其他 Draft 共用导出、编译、测试入口和 C++ runner。

| 文档 | 内容 |
|---|---|
| [整体架构](docs/ARCHITECTURE.md) | 模型、Draft 数据流、验证与状态、OM 划分 |
| [配置、使用与测试](docs/USAGE.md) | 环境配置、统一 AIR/OM 命令、测试参数、profiling |
| [测试结果](docs/RESULTS.md) | 接受率、Decode 加速比、阶段时延 |
| [未来优化](docs/OPTIMIZATION.md) | 优先级、自定义算子输入输出、精度要求与收益分析 |

所有模型权重、构建、日志和测试产物放在源码外的 `AI_RUN_DIR` 中。
第三方来源与许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
