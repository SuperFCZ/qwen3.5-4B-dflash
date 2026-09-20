# 自定义算子开发目录

这里保存与本仓库模型图对应的算子需求、接口、数值门槛和验证方法。
当前新增的是开发需求，尚无新增 kernel、编译产物或设备性能承诺。

| 范围 | 入口 | 状态 |
|---|---|---|
| Ascend310P 量化 Draft：group-128 Linear、W4 解包、完整词表 Top-1 | [架构与优化分析](draft_quant/README.md) | 待开发 / 实测 |
| 开发人员可直接采用的张量、布局、精度约束 | [算子需求](draft_quant/OPERATORS.md) | 接口提案 v1 |
| 独立与整图精度、重复执行、速度验收 | [验收要求](draft_quant/VALIDATION.md) | 未执行设备验收 |
| 固定形状与用户提供的热点数据 | [workloads.json](draft_quant/workloads.json) | 分析输入 |

每个实际算子后续放在自己的子目录，包含 host/tiling、kernel、注册、golden/compare 和测试入口；
共享工具链、权重、模型产物、日志和 profile 必须放到源码外的运行目录。
开发包使用隔离注册目录，不覆盖现有 CANN 安装。既有 GDR 等依赖见
[原算子清单](../../docs/DFLASH_OPERATORS.md)。
