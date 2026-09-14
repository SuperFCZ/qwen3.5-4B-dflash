# Draft W8A16 / GPTQ W4A16

本分支从 `feature/gdr-chunk-verify` 的 `8c44da7` 创建。`run_npu`、
`benchmark_npu` 和 AIR/OM 工厂均接受 `draft_quantization=fp16|w8a16|w4a16`。
默认 `fp16` 保留旧 checkpoint；量化版本直接导入指定发布者的固定权重，
不会把旧六层 Draft 做一次 RTN 后冒充 GPTQ。

| 参数 | checkpoint | 固定 revision |
| --- | --- | --- |
| `fp16` | 原 `z-lab/Qwen3.5-4B-DFlash` 六层版本 | `9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf` |
| `w8a16` | [naveenrajk W8A16](https://huggingface.co/naveenrajk/Qwen3.5-4B-DFlash-W8A16/tree/b4bf9f0f7f79bd8589d34b457e136f644c122180)，RTN | `b4bf9f0f7f79bd8589d34b457e136f644c122180` |
| `w4a16` | [nota-ai GPTQ W4A16](https://huggingface.co/nota-ai/Qwen3.5-4B-DFlash-GPTQ-W4A16/tree/c5fb290e47e30c81d06e48b0495ec06f2560dd4e) | `c5fb290e47e30c81d06e48b0495ec06f2560dd4e` |

两个量化 checkpoint 都是 **5 层、intermediate=9728、全注意力、mask=248070**，
提取 Target 第 `[1,8,15,22,29]` 层的输出，feature 宽度为 12800。
旧 FP16 Draft 则为 6 层、feature 宽度 20480、mask=248077。
配置、36 个 Linear 和 norm 都按所选 checkpoint 加载；不能混用目录、mask 或 feature 层。
两份量化配置均是 compressed-tensors `pack-quantized`、对称 group=128、无激活量化。
固定 config/model SHA-256、tensor 名称/形状/类型由 `draft_quantization.py` 校验。

## 获取所选权重

先使用 workspace 分配的 run 和声明的模型 Python。在本分支源码根目录运行：

```bash
export PYTHONPATH="$PWD:$PWD/framework/python${PYTHONPATH:+:$PYTHONPATH}"
DRAFT_VARIANT=w8a16   # 改为 w4a16 即选择发布的 GPTQ 权重
DRAFT_DIR="$AI_RUN_DIR/checkpoints/$DRAFT_VARIANT"
"$AI_MODEL_PYTHON" -B -m models.dflash_v1.prepare_draft \
  --draft-quantization "$DRAFT_VARIANT" --output "$DRAFT_DIR"
```

输出目录必须是新的外部目录。已有本地 checkpoint 可以直接用其路径作为 `DRAFT_DIR`，
但必须通过同样的固定哈希校验。下载器不执行仓库中的远程 Python 代码，不安装量化工具，
不需要重新校准 GPTQ。权重和报告均不得放进源码仓库。

## torch_npu

沿用原来的 Target、接收方 wrapper 和自定义算子环境。Draft 仅新增标准 Tensor 运算，
不需要新增自定义算子。

```bash
"$AI_MODEL_PYTHON" -B -m models.dflash_v1.run_npu \
  --target-dir "$TARGET_DIR" --draft-dir "$DRAFT_DIR" \
  --draft-quantization "$DRAFT_VARIANT" \
  --kv-cache-max-len 2048 --device npu:0 \
  --prompt '请用一句话解释为什么天空是蓝色的。' --prompt-mode chat \
  --max-new-tokens 32 --block-size 16 --execution-mode validate \
  --report "$AI_RUN_DIR/$DRAFT_VARIANT-npu.json"
```

Target W8A8 仍通过原有 `--config /path/to/qwen3.5.yaml --quant_mode enable`
单独开启。Draft 的 embedding 和 LM head 继续复用 Target 提供的 FP16 权重。
`benchmark_npu` 接受相同的 `--draft-quantization` 参数；按原 benchmark 文档运行
ordinary 和 DFlash 的 3 warmup + 10 次测量。报告包含具体量化版本、权重哈希和设备。
Target 的 two-pass chunk-GDR verify/commit 与严格贪心零 token-ID 差异门禁继续生效。

## AIR / OM

该基线的 OM 路径是固定 gear 的全前缀重算；torch_npu 的 chunk-GDR 回滚调度不等于
OM 已经有显式增量 cache ABI。沿用 [AIR/OM 指南](QUANT_AIR_OM_FRAMEWORK.md)
配置 Target W8A8、receiver、input manifest、实际 SoC 和 C++ runner。

在 factory JSON 中设置匹配的 `draft_dir` 和 `"draft_quantization": "w8a16"`
或 `"w4a16"`，并对该目录重新运行 `framework/scripts/lock_quant_inputs.py`。
也可以由命令行覆盖版本：

```bash
"$AI_MODEL_PYTHON" -B -m qwen35_dflash.ascend310p export-air \
  --factory qwen35_dflash.ascend310p.quant_factory:create_quant_recompute_graph \
  --factory-config "$AI_RUN_DIR/factory-$DRAFT_VARIANT.json" \
  --draft-quantization "$DRAFT_VARIANT" \
  --bundle-dir "$AI_RUN_DIR/om-$DRAFT_VARIANT"
```

`build-om`、`probe-pytorch`、`run-e2e`、`run-e2e-cpp` 的 factory 参数也支持此覆盖。
后续 `compile-om` 和 `infer-cpp` 使用生成的 manifest，不再另选一个量化版本。
**必须重建 runner 1.1.0 或后续兼容版本，并重新导出/编译对应版本的 OM。**

量化图前两个输入仍是 `input_ids`、`attention_mask`，随后是 72 个只读输入：
36 组压缩权重与 FP16 scale。输出仍为 Target/Draft Top1。导出器将这些输入写成
bundle 内的 `.bin`、`constant-inputs.tsv` 和带哈希的 manifest 记录。
C++ 和 pyACL 均校验实际 OM 的有序 shape/dtype/bytes，在启动时上传一次，生成循环只传 token。
`infer-cpp` 自动从 manifest 传入 table 路径和哈希；无需用户手工组织 72 个参数。
移动或交付时必须保留整个 bundle，不能只复制 `.om`。

显式图输入使压缩权重保持为运行时数据，避免常量折叠产生 FP16 权重副本。
标准算子先在设备上反量化当前 Linear，再进行 FP16 矩阵乘；没有 CPU 回退，也不保留
完整的常驻 FP16 Linear 权重。W8 常驻 Draft 权重/scale/norm 为 545856000 bytes，
W4 为 277158400 bytes，另需当前反量化矩阵、cache 和运行时工作区。
这条路线用于保留指定量化值和减少常驻权重；逐层反量化有额外开销，实际速度需真机测量。

不调用 `npu_weight_quant_batchmatmul(..., antiquant_group_size=128)`：
[该接口的 310P 推理卡说明](https://www.hiascend.com/document/detail/zh/Pytorch/600/apiref/apilist/ptaoplist_000164.html)
仅支持 per-channel，不能直接表达这些 checkpoint 的 per-group scale。

## 验证和证据范围

```bash
"$AI_MODEL_PYTHON" -B tools/validate_draft_quantization.py \
  --draft-dir "$DRAFT_DIR" --draft-quantization "$DRAFT_VARIANT" \
  --output "$AI_RUN_DIR/$DRAFT_VARIANT-reference.json"
```

该命令用独立 NumPy 字节解码对照全部 36 个真实 Linear，以及完整 Draft 的普通前向和
两轮增量缓存前向。两份固定 checkpoint 在 CPU 上均达到零差异。
另有 actual `torch.export` 数据流检查、位序/分组/版本拒绝测试和 C++ fake-ACL 常驻权重测试。
这些是主机证据，尚不能证明 TorchAir、ATC 或 310P 真机通过；最终要求两种版本各完成
实际 AIR/OM 构建、无 fallback 执行、ordinary/DFlash token-ID/EOS 零差异和配对 3+10 测量。

## 已验证的实现经验

- compressed-tensors 的 packed word 使用 offset-binary：W4 解码后减 8，W8 减 128，
  不能直接当作二补码的 int4/int8。独立字节解码验证覆盖两份真实权重。
- mask、RoPE 配置位置、nullable sliding_window 和 target feature 层属于 checkpoint
  合同。只替换 Linear 精度而保留六层配置会加载错误的 Draft。
- 所选 feature 合同绑定到 Target 实例；普通 Target 数学和其他实例不随之改变。
- OM 量化权重作为显式只读输入，C++ 每次加载检查描述符、哈希并只上传一次。
  源码检查和测试已扩展；未把 CPU/fake-ACL 通过提升为真机通过，未修改通用技能。
