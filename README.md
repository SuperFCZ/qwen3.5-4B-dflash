# Qwen3.5-4B DFlash

Ascend 310P 投机解码：Target 支持 FP16 / W8A8，官方 DFlash Draft 使用 FP16。
Torch-NPU 和 AIR/OM/C++ 均支持 **GDR Chunk 两遍**与 **GDR MTP** 验证。
OM 只使用一个紧凑 Draft：16 行特征投影，最多 15 个候选；长输入复用同一图分段建缓存。

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
| 运行 OM：短 + 1K 输入统一测试、Chunk/MTP、多长度、分项时延 | [OM/C++ 使用](docs/GDR_CHUNK_AIR_OM.md) |
| 读取离线 JSONL/JSON 测试集，按文件统计接受率和时延 | [离线测试集](docs/GDR_CHUNK_AIR_OM.md#离线开源测试集) |
| 了解 Draft → Verify → Commit | [流程与架构](docs/DFLASH_ARCHITECTURE.md) |
| 查看已有速度、接受率、deterministic 漂移问题 | [结果与已知问题](docs/DFLASH_CURRENT_USAGE_AND_RESULTS.md) |

开源数据集 Chunk 测试（输出上限 **512 token**）：5 个文件、2563 条问题，加权接受率 **26.77%**；
按用户日志汇总估算整体加速 **1.91×**，数学类 **2.05–2.11×**、代码类 **1.60–1.64×**。
这些数据采集早于紧凑 Draft 默认化，当前执行路径待设备重测。
允许输出差异，任务质量未评估；[详细结果与 128 token 对比](docs/DFLASH_CURRENT_USAGE_AND_RESULTS.md#开源数据集输出上限-512-token)。

<details>
<summary>开发参考</summary>

- [算子清单](docs/DFLASH_OPERATORS.md)
- [AIR/OM 接口与 ABI](docs/QUANT_AIR_OM_FRAMEWORK.md)
- [DFlash 源码索引](models/dflash_v1/README.md)
- [框架源码索引](framework/README.md)

</details>
