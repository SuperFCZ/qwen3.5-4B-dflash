# 单个 gate/up 的 TorchAir / GE opt-in 接入设计

状态：**设计与源码调查完成；Torch binding、converter、AIR/OM、完整 Draft replay、性能和
端到端验证均为 `NOT_RUN`。本阶段不实现或启用生产默认替换。**

范围仅为 W8 Draft 的 `layers[0].gate_up_linear`。C16/C64 两档均替换同一层这一处投影；
其余 25 个量化投影、SwiGLU、residual、norm、attention、functional cache update、head
和 Top1 保持原路径。新 `DFlashDraftLmHeadTop1` 本轮不接回生产 Draft。

## 已有入口与缺口

- `op_host/d_flash_group_quant_linear.cpp` 已注册 GE `DFlashGroupQuantLinear`，生成的 OPP
  提供 ACLNN API；独立 runner 已验证 GE logical ND origin / ACLNN physical NZ view 两套描述。
  **生成 ACLNN 不等于自动得到 `torch.ops` 算子或 TorchAir converter。**
- [PackedDraftLayer / DraftGraph](../../python/qwen35_dflash/ascend310p/incremental.py) 在加载时合并
  gate/up 权重，`project_gate_up()` 调用 `gate_up_linear`，是最小替换点。图里仍执行原 FP16
  Linear 输出 → SiLU → multiply，不能借此融合或取消舍入边界。
- [CustomOpExportSpec](../../python/qwen35_dflash/ascend310p/contracts.py) 描述 torch/GE 名称与
  最低出现次数；[custom_op_export.py](../../python/qwen35_dflash/ascend310p/custom_op_export.py)
  使用明确的 `_ADAPTERS` 白名单，注册 Fake/Meta 与 converter 并审计 GE 节点。当前未知
  `dflash_experimental::group_quant_linear` 会直接拒绝，不能只添加一个 Spec 就声称接入完成。
- [runtime_input_export.py](../../python/qwen35_dflash/ascend310p/runtime_input_export.py) 与
  [weight_prepack.py](../../python/qwen35_dflash/ascend310p/weight_prepack.py) 负责输入 ABI、
  immutable tensor/FX metadata、FileConstant descriptor 和 offline NZ。现有修复专门针对
  native WeightQuant；新节点的权重描述也要显式审计，不能绕过 hash/shape/immutable 检查。
- [draft_constants.py](../../python/qwen35_dflash/ascend310p/draft_constants.py) 的 offline-NZ
  校验按 native WeightQuant 节点数核对全部 26 组权重。换掉一个节点后只有 25 个 native
  节点；只降低 `minimum_occurrences` 还不够，现有 `count*2 == expected_count` 会拒绝。

## 建议的 opt-in 结构

后续代码放入 `framework/custom_ops/draft_quant/integration/`，由独立 experimental export
命令显式选择；建议配置 `experimental_gate_up_layer=0`，默认 `None`，其它层号首版拒绝。
使用当前已验证的 W8 offline-NZ/static-C16-C64 路径。默认命令不加载实验注册模块，不注册
替换 hook，不改变全局 `AirDFlashOps.linear` 或 `GroupQuantLinear.forward`。

构建独立 native/candidate 工厂实例，避免共享 module 导致基线也被替换。在 pinned
checkpoint 审计和 PackedDraftLayer 合并完成后、创建最终 export Spec/constant 审计前，
只将 `graph.layers[0].gate_up_linear` 替换为薄包装模块；包装仅注册审计过的 NZ/GN buffers，
不把原 GroupQuantLinear 继续作为注册子模块保留，避免多计 native 模块或携带未使用常量。
若调用现有 factory 后处理 Spec，
先确认模型不是被 `DraftConstantInputs` 包装的 runtime-weight 分支；不静默解包或更换分支。

包装模块的逻辑接口：

```text
输入 [1,16,2560] FP16
  -> 保值 reshape [16,2560]
  -> dflash_experimental::group_quant_linear(x, w_nz, s_gn)
  -> [16,19456] FP16
  -> reshape [1,16,19456]
```

预处理仅在加载/导出时进行：

- q 必须来自原 layer-0 合并 gate/up，逻辑 `[19456,2560]` signed INT8；用已有 pack 规则
  得到 `[80,1216,16,32]` 的 `nz_int8_v1`，逆变换 byte-equal。
- 原 module scale 是 `[19456,20]` NG，离线保值转成连续 `[20,19456]` GN；不能直接复用
  native helper 的 NG-backed GN view。保存转换前后 FP16 位型及哈希。
- custom 仍使用 immutable weight/scale 常量，不增加 Draft 运行时输入，不在 forward
  解包、重排或反量化整张权重。不改变 gate/up 前后两半的顺序。

## Torch dispatcher 与 converter

建议独立 schema（名称示意，待实现）：

```text
dflash_experimental::group_quant_linear(Tensor x, Tensor w_nz, Tensor s_gn) -> Tensor
```

Meta/Fake 只检查固定 dtype/shape、M16、group128 和 layout，返回 FP16 `[16,19456]`；
它不得计算数值或充当 fallback。NPU eager backend 必须实际调用已安装的 ACLNN 算子，
CPU backend 不提供替代计算。也可先做明确标记为 export-only 的 schema/Fake 原型；如果
receiver TorchAir 要求 eager 执行而 binding 尚缺失，应直接失败，不冒充 NPU 路径。

NPU binding 需要使用 receiver torch_npu 当前 stream、allocator 生命周期和实际 CANN API。
尤其不能把普通 ND 的四维载体交给自动格式推断后就当作 NZ：ACLNN weight 描述必须像已
验证 runner 一样使用 physical NZ view/storage 与 `ACL_FORMAT_FRACTAL_NZ`。若扩展拿到的是
ND torch byte carrier，应验证其连续物理字节，并显式构造 ACL tensor descriptor；不能
再次 `npu_format_cast` 重排已经 NZ 的 bytes。异步输出和 workspace 必须与 PyTorch stream
生命周期绑定。该桥接仍需在 receiver 2.8/torch_npu 源码和 NPU 上单独确认。

TorchAir converter 复用当前仓库的版本探测方式，通过 `register_fx_node_ge_converter`
注册到明确 overload，调用 GE custom op，输入端口严格为 `x/w_nz/s`，输出为 `y`。
实际 GE TensorConst 描述要求：

| 项 | 要求 |
| --- | --- |
| x | FP16 ND `[16,2560]` |
| w_nz | DT_INT8；origin ND `[19456,2560]`，storage FRACTAL_NZ `[80,1216,16,32]` |
| s | FP16 ND `[20,19456]`，实际数据连续 GN |
| y | FP16 ND `[16,19456]` |

不能把 `DT_UNDEFINED/[]` FileConstant 当作已修好的权重，也不能为了通过 prepack 把权重
变为 runtime Data。应为这一 immutable custom 常量增加专用 descriptor/byte-roundtrip 审计，
复用现有可信的 tensor/FX 来源；新常量无需再次套用 native WeightQuant 的 TransData 重写。

现有 exporter 没有通用 external-adapter 注入接口。下一步最小共享改动应是一个显式、可
审计的实验 adapter 注册入口，或等价的独立 experimental exporter；默认 allowlist 不变。
不通过 monkeypatch 关闭审计，也不将 native WeightQuant converter 全局替换为 custom。

## 图、常量与数量验收

仅在含版本化 `experimental_gate_up` 元数据的 candidate Draft Spec 中：

1. 新 op 最低出现次数设为 1，native WeightQuant 最低次数由 26 改为 25；同步更新
   `Spec.custom_ops` 和 `metadata.custom_op_export_contracts`，不能留下互相矛盾的声明。
2. 导出后额外执行**精确数量**校验：1 个 custom + 25 个 native。最低次数不能代替精确检查。
3. 唯一 custom 节点绑定 layer-0 gate/up 的 checkpoint/NZ/scale 哈希和 shape；不能只凭
   模糊节点名字选第一个大矩阵，也不能同时替换五层。
4. 独立审计 custom 的常量，再与 native 的 25 组审计合并，证明共 26 个投影、权重/scale
   一组不少。`verify_constant_inputs` 对这份实验策略检查“25 audited native + 1 audited
   custom”，默认策略仍严格要求原 26 个 native；不全局放宽 count/shape/hash 检查。
5. Draft 公共输入/输出、feature layer 顺序、functional row update、C16/C64 gears、KV
   storage/current-next 规则保持原 ABI。Target、Verify、ordinary 图的 hash/内容保持原样。

ATC 只编译到新的 artifact 根目录，显式加载隔离 custom OPP，记录 so/kernel/header/源码
哈希。常量留作 immutable model buffers，不能进入 `DraftConstantInputs` 的 runtime-weight
兜底分支。新策略和证据清单须被 compiler/plan validator 显式识别后才允许运行。

## 后续验收顺序（全部 NOT_RUN）

| 阶段 | 门槛 | 当前状态 |
| --- | --- | --- |
| Torch schema / Fake / converter | fixed metadata、拒绝错误 NZ/NG，真实 NPU binding 无 fallback | NOT_RUN |
| 单节点 AIR/OM | 只含一个 DFlashGroupQuantLinear，常量格式/bytes/hash 正确 | NOT_RUN |
| 单个 layer-0 gate/up | 同输入 native OM，FP16 bits=0/ULP=0，重复/只读/guards | NOT_RUN |
| C16/C64 candidate Draft | 节点数量/常量审计；同 frozen 输入候选 ID 和全部 KV 位型一致 | NOT_RUN |
| 压力回归 | 63/64/65、零/全接受、重复请求、current/next、EOS/stop reason | NOT_RUN |
| 完整 Draft / 20 条 Decode | 同配置 baseline 配对、median/p95/显存、真实端到端效果 | NOT_RUN |

A4/A4.1 的独立算子 PASS 不能代替任何一项图集成 PASS。上述设计目前只保存在文档，
本阶段没有修改默认 factory、exporter、adapter allowlist、compiler 或生产 DraftGraph。
