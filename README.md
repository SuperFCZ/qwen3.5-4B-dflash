# Qwen3.5-4B DFlash

Ascend 310P 投机解码：Target 支持 FP16 / W8A8，官方 DFlash Draft 使用 FP16。
Torch-NPU 和 AIR/OM/C++ 均支持 **GDR Chunk 两遍**与 **GDR MTP** 验证。

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
| 运行 OM、多长度 / 1K 上下文测试、msprof | [OM/C++ 使用](docs/GDR_CHUNK_AIR_OM.md) |
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
