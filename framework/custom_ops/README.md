# 自定义算子开发目录

| 内容 | 文档 |
|---|---|
| 当前离线 NZ W8 Draft 结构、逐算子读写/计算收益、成本及 Decode 换算 | [优化分析](draft_quant/README.md) |
| 调用签名、输入输出顺序、shape/dtype/stride、数学功能、mask/cache 语义与精度要求 | [算子需求](draft_quant/OPERATORS.md) |
| 数值、模型集成和性能验收 | [验收要求](draft_quant/VALIDATION.md) |
| 固定形状、字节数和测量数据 | [workloads.json](draft_quant/workloads.json) |

各算子实现放在独立子目录；工具链、权重、编译产物和日志放在源码外。
既有算子依赖见 [算子清单](../../docs/DFLASH_OPERATORS.md)。
