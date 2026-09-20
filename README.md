# Qwen3.5-4B DFlash

Ascend 310P 投机解码：Target 支持 FP16 / W8A8；OM Draft 可选 FP16 / W4A16 / W8A16。
Torch-NPU 和 AIR/OM/C++ 均支持 **GDR Chunk 两遍**与 **GDR MTP** 验证。
同一目录可编译 7 个 OM，运行时只加载所选 Draft 与 Verify；Draft 自动切换 16/64 行上下文档位，最多 15 个候选。

## 环境配置

首次在仓库目录复制一份配置到源码之外，填写路径和参数：

```bash
cp -n config/dflash_env.sh.example /absolute/path/dflash-env.sh
vi /absolute/path/dflash-env.sh
```

以后每次打开 Bash 终端只需：

```bash
source /absolute/path/dflash-env.sh
```

配置集中保存模型、Python/CANN、OM 路径、Chunk/MTP、长度和量化选项。
必须用 `source`，直接 `bash dflash-env.sh` 无法恢复当前终端环境。
修改配置后重新 `source` 即可；升级仓库时保留自己的配置文件。

## 使用

| 要做什么 | 文档 |
|---|---|
| 直接运行 Torch-NPU，切换 Chunk / MTP | [Torch-NPU 使用](docs/DFLASH_RUN_AND_VALIDATE.md) |
| 编译三种 Draft、两条 Verify，共 7 个 OM | [导出与编译](docs/GDR_CHUNK_AIR_OM.md#导出与编译) |
| 对照量化 Draft 的 CANN MatMul 时延与误差 | [量化 MatMul](docs/GDR_CHUNK_AIR_OM.md#量化-draft-matmul) |
| 合并短 / 1K 输入与离线 JSONL/JSON 数据集，对比 Draft 精度 | [统一测试](docs/GDR_CHUNK_AIR_OM.md#统一测试) |
| 复用已有普通模型数据，只运行 DFlash | [基线复用](docs/GDR_CHUNK_AIR_OM.md#复用已有普通模型数据) |
| 了解 Draft → Verify → Commit | [流程与架构](docs/DFLASH_ARCHITECTURE.md) |
| 查看已有速度、接受率、deterministic 漂移问题 | [结果与已知问题](docs/DFLASH_CURRENT_USAGE_AND_RESULTS.md) |
| 优化量化 Draft、给自定义算子开发方提需求 | [架构、热点与算子需求](framework/custom_ops/draft_quant/README.md) |

开源数据集 Chunk 测试（输出上限 **512 token**）：5 个文件、2563 条问题，加权接受率 **26.79%**；
用户汇总报告的模型生成加速比为 **2.42×**，数学类 **2.59–2.67×**、代码类 **2.04–2.07×**。
此口径包含 Prefill + Decode 循环，不含加载、分词等完整请求开销。
这些结果对应已测 FP16 Draft 配置，W4/W8 的性能需分别测量。
允许输出差异，任务质量未评估；[详细结果与 128 token 对比](docs/DFLASH_CURRENT_USAGE_AND_RESULTS.md#开源数据集输出上限-512-token)。

<details>
<summary>开发参考</summary>

- [算子清单](docs/DFLASH_OPERATORS.md)
- [自定义算子开发目录与精度要求](framework/custom_ops/README.md)
- [AIR/OM 接口与 ABI](docs/QUANT_AIR_OM_FRAMEWORK.md)
- [DFlash 源码索引](models/dflash_v1/README.md)
- [框架源码索引](framework/README.md)

</details>
