# A5.1 + C：第一阶段 opt-in GE/AIR/OM 集成

本目录是独立实验入口。生产 factory、Target、Verify 和默认 Draft 不切换；A5.1/C 的 Kernel、Host Tiling、FP16 数学、Top1 比较及 workspace 源码不变。尚不能把新 AIR/OM 当作正式生产部署包。

## 基线与当前状态

- A5.1：用户提供的真实 310P3 验收 PASS，来源 `a51-fallback-20261012-063342/suite.json`，C16/C64 独立投影合计相对 Native 1.331×/1.172×。
- C：`b96dde9`，用户提供的 `head-release-20261014-020633` Release PASS，41/41；M15 Custom ≈17.1 ms、Native ≈23.25 ms。冻结 `compare_select + Vector Top1`。
- **本次联合 OPP 编译/安装、GE 注册、TorchAir 真机导出、ATC、图内数值、Draft 时延和端到端生成：全部 NOT_RUN。** CPU Fake/Protobuf/C++ 声明测试不能替代这些门槛。
- 历史独立算子成绩不代表图内收益。报告中 `PASS` 只属于明确的检查范围；只有新设备执行才能产生新的图内 PASS。

## 接口审查结果

现有 `quant_factory.create_quant_incremental_graphs` 构造 `incremental.DraftGraph`：FC 将冻结 feature layer 顺序选出的特征投影，随后 5 层各有 packed KV、Q、O、packed gate/up、down，共 26 个 WeightQuant 投影。

| 投影 | C16 M | C64 M | K | N |
|---|---:|---:|---:|---:|
| FC | 16 | 64 | 12800 | 2560 |
| 每层 KV | 32 | 80 | 2560 | 2048 |
| 每层 Q | 16 | 16 | 2560 | 4096 |
| 每层 O | 16 | 16 | 4096 | 2560 |
| 每层 gate/up | 16 | 16 | 2560 | 19456 |
| 每层 down | 16 | 16 | 9728 | 2560 |

A frontend 接 FP16 `[M,K]`、绑定的 signed INT8 `[N,K]` 和连续 GN scale `[K/128,N]`。原 `GroupQuantLinear.scales` 是 NG；必须转置并连续化，不能 reshape。保存 AIR 时才从真实 immutable tensor 创建**新的物理 NZ Const**，严格检查来源 Const/FileConstant 的 dtype、shape、字节数、SHA256，NZ inverse bit-exact，scale 和全部 26 组权重哈希对应 Capture。不能将 ND 字节直接改名为 NZ，也不能把常量变为运行时权重。

C 只代理原 `DraftGraph.ops.top1`，仍由原 `forward` 计算 `self.norm(hidden)[:,1:]`，输入恰好 `[1,15,2560]`，调用 C 后还原 `[1,15]`。原 `proposal_count` mask、anchor、原词表 token ID、所有层的 functional whole-row ScatterNdUpdate 均保留。

新增 `dflash_ac::*` 仅提供 Fake/Meta 和命名 GE converter；没有 CPU 数值实现或 NPU eager fallback。GE 名称/端口必须与冻结 Host 注册一致：

| frontend | GE | 输入 → 输出 |
|---|---|---|
| group_quant_linear | DFlashGroupQuantLinear | x,w_nz,s → y |
| head_top1 | DFlashDraftLmHeadTop1 | hidden,weight → token_id |
| head_audit | DFlashDraftLmHeadTop1Audit | hidden,weight → token_id,logits |

只有显式集成 spec 会使用保存钩子和编译审计，未带 `draft_ac` 元数据的导出规则不变。GE 输出必须保留准确节点数，不能将 compiler 的原生 WeightQuant 审计放宽成“任意数量”。

### TorchAir Converter 接口兼容

用户提供的 `/home/w00949577/qwen35-dflash-run/ac-ge-20261014-075251` 已通过 Preflight、三算子联合 OPP 编译安装和 C++ runner 构建，但首次 AIR 导出被旧的接口检查拒绝。旧报错没有区分 API 缺失与 `_custom_op_call_mode()` 返回合法的 `registered-ir-positional`，因此不能仅凭该日志断言服务器的具体签名。

A/C 现复用已有框架的两种调用约定：命名接口 `custom_op(OpType, inputs={...}, outputs=[...], attrs={})`，以及已注册 IR 的位置接口 `custom_op(OpType, *ordered_tensors)`。位置顺序严格遵循上表，A/C 不添加属性。显式暴露 `inputs/outputs/attrs` 的 variadic wrapper 仍按命名形式处理；无法 introspect 的 callable 按框架原有位置调用约定尝试一次，并明确记录签名不可见，不能视为 SDK 已验证。

每个图在导出前保存 `converter-capabilities.json`：TorchAir 版本/来源、两个 API 是否 callable、真实签名和 wrapper 签名、框架原分类、选定调用方式、端口及静态 bind 错误。API 缺失、签名不支持、注册失败和 SDK 调用失败分别报错；失败不会尝试另一种映射，不会产生多余 GE 节点。输出数量严格检查，Audit 必须同时返回 ID 和 logits。Fake dtype/shape、Converter 调用计数、AIR 节点数量和 NPU bitwise gate 均保留。

本轮本地 Converter/集成/runtime-input 回归 **249 PASS**。服务器真实 API 诊断、修复后的 AIR/ATC 和 NPU 正确性在运行前仍为 **NOT_RUN**。请随 `suite.json` 和 `.server.log` 回传失败图目录的 `converter-capabilities.json`、AIR 导出/ATC 日志及已有 `dynamo.pbtxt`（若生成）。

### FileConstant 写入与验证

A/C 的外置权重检查安排在 SDK `_save_weight2file` 完成后、图 dump/GE 保存前；导出结束再检查 Tensor/文件与绑定，避免 `_convert_data_to_const` 刚创建节点时就访问尚未写入的文件。诊断记录实际 SDK 源码、节点属性、目录布局、大小、offset 与 SHA256，所有原验证保留。详见 [FileConstant 物化说明](FILECONSTANT_MATERIALIZATION.md)。新版本真实 ATC/NPU 验收在执行前仍为 NOT_RUN。

## OPP 共存与 provenance

A/C 独立包都使用 `vendors/customize` 和同名动态库，不能简单叠放覆盖。入口从冻结源码及**已验收的生成 header**新建一个联合 msopgen 工程，含 A、C Fused、C Audit 三个算子。保留 A K256/broadcast/A-UB 及原 pipeline/core 配置；保留 C compare_select/vector/profile=none。没有 Probe 或新算法。

前置检查链：

1. A5.1 paired `runs["256"]` 路径/哈希 → K256 stage → `build_config_path`/SHA → full A4 报告。
2. full A4 `inputs.c16/c64` 自动解析 Capture 和 Native projection manifest；使用既有完整验证器核对 26 个投影、权重/scale/input、AIR/deployment/OM。显式传入的 capture 路径必须与报告一致。
3. 生成 header 必须位于已知 `CustomOp/op_host` 布局；追溯真实 build root 的 `opp`，与原 `opp-preflight.json` 一致。不存在就失败，不猜随机构建目录。
4. C Release 的冻结源码、生成 header、构建产物、已安装包、release suite、Capture/Native SHA 都要对应。
5. Native Deployment 所引用 AIR 的哈希、Draft 元数据、两 gear OM，以及 frozen replay 的 OM/输入哈希、feature 顺序、A/C checkpoint 对应。
6. Factory 优先从**被 Deployment 哈希绑定的 AIR `factory_config`**原样提取到新目录 `factory-resolved.json`。可选独立 JSON 只能等于该记录。若缺失/不是 W8 offline-NZ，明确失败，不制造配置或推测 checkpoint 路径。
7. offline-NZ 同时支持自动 `draft_weight_prepack="nz"` 和历史 `draft_weight_prepack_manifest` 两条互斥路径。历史 Manifest 必须与 Draft metadata 指向相同文件，其 SHA256 必须等于 AIR prepack 审计；自动模式从 AIR 的哈希绑定 payload 中定位生成的 Manifest。复用原有 Manifest policy/status/schema、NZ carrier 和 Const descriptor 检查，验证 26 个节点、group128/GN，以及每份实际 INT8 权重的 NZ 反解、零 padding 和 Capture 哈希。完整 Draft 再次使用同一检查。原始 factory 配置不改写，绝不在 Manifest 路径上补入 `draft_weight_prepack="nz"`。

只读 Preflight 在独立子进程中执行并记录实际导入栈；协调进程不继承其 torch_npu 模块状态，仍保留初始化顺序 guard。联合 OPP 安装并设置路径后，沿用 exec 启动新的设备解释器，使 LD_LIBRARY_PATH 在 ELF loader 启动时生效。`TORCH_DEVICE_BACKEND_AUTOLOAD=0` 只是辅助控制，不能阻止第三方显式 import。详见[进程顺序及导入证据](PROCESS_ORDER.md)。联合安装之后继续检查三组非空 `.o/.json`、CANN9 首个 `.o/.json` 路径解析规则、三个 ACLNN 头、GE prototype 动态库及哈希。**这些只证明文件完整，GE 注册是否能使用仍由后续 ATC 决定。** 活跃环境中发现旧 A/C 注册路径会拒绝；请在干净 shell 中保留接收端原有 ADN/GDR 环境，不 source 旧 A/C set_env。若 ADN/GDR 与联合包同名库产生真实加载冲突，应回传 ATC 日志进一步定位，不能把文件检查当作共存成功。

## 完整工厂接口早期检查

`--full-draft` 会在联合 OPP 构建前的隔离 Preflight 中检查锁定的 GDR dispatcher schema 和 ADN/GDR 原型，并在新设备解释器、56 个图执行前复查。旧九参数 GDR 不能代替包含 `effective_length` 的 v2 接口；原 schema、正式 prepare_export 和数值标准保持不变。只读历史/当前环境诊断及恢复步骤见 [GDR 接口调查](GDR_INTERFACE.md)。

## 服务器命令

在现有 Ascend310P3/CANN9.0、TorchAir/torch_npu/接收端 ADN/GDR 可用的模型环境中：

```bash
cd /home/w00949577/z50058744/qwen3.5-4B-dflash
git pull --ff-only
export MODEL_PYTHON=python
# 新目录，不能已有；不触碰旧 Capture、OM 或正式 OPP。
export AC_OUTPUT_DIR=/home/w00949577/qwen35-dflash-run/ac-preflight-$(date +%Y%m%d-%H%M%S)
bash framework/custom_ops/draft_ac/run_known_evidence.sh --preflight-only
```

`run_known_evidence.sh` 使用本轮用户给出的真实路径；A 的 Native manifest/OPP、factory 参数由报告链自动解析。其它机器可使用 `run_server.sh --help` 所对应的 `test/run.py --help` 参数（`run_server.sh` 要求先设置 `AC_OUTPUT_DIR`）。

前置检查成功后，运行最小图编译/数值闭环：

```bash
export AC_OUTPUT_DIR=/home/w00949577/qwen35-dflash-run/ac-ge-$(date +%Y%m%d-%H%M%S)
bash framework/custom_ops/draft_ac/run_known_evidence.sh
```

这会新建联合 OPP；先对 C16/C64 各 26 个 A 单节点图编译和执行，与对应旧 Native projection OM 逐字节比较，再对两 gear 的 M15 C Fused/Audit 图执行：Audit 全词表 FP16 logits、两路 INT64 IDs 必须严格匹配旧 Native Head/Top1 OM。C **单算子探针**沿用旧 Native 的 runtime-readonly full-head-weight ABI，清楚标记 `single-op-C`；它不代表完整 Draft 常量绑定已通过。完整 Draft C 权重仍必须 immutable。

单算子图通过后，下一次可选择完整静态 Draft：

```bash
export AC_OUTPUT_DIR=/home/w00949577/qwen35-dflash-run/ac-draft-$(date +%Y%m%d-%H%M%S)
bash framework/custom_ops/draft_ac/run_known_evidence.sh --full-draft
```

此入口重新做前置和单算子 gate，再导出 C16/C64 × `off/a/c/ac`。`off` 必须先与旧 Native OM 一致。所有模式来自原 `DraftGraph.forward`，15 输入/11 输出 ABI 逐项对原 contract 校验，比较 IDs 和十个完整 KV buffers，包括未更新行。保存相同输入下 Native/四模式 median、p95 及原始 30 个样本，`draft-timing.md` 不用独立投影求和推算收益。

为避免把未验证模型交给生产，本阶段只导出 **Draft-only diagnostic envelope**：原 incremental contract 存为 `source_incremental_contract`，不生成生产 runner-plan、不替换 Target/Verify。端到端生成、接受率、Verify/anchor 多轮状态闭环仍是下一道 gate，始终 `NOT_RUN`。

所有执行使用独立 C++ 多输入/输出 OM runner：严格查询模型 dtype/shape/逻辑字节数，物理 payload 按 32B 分配，guard 在物理末尾；输出比较只读取逻辑值。输入只读、两次重复、timed-tail、poisoned postcheck 保留；单算子图另在同一 model/workspace 做两次行置换及恢复检查。计时 `continuous-v1`，warmup=5，repetitions=30，读回/检查在计时段外。Audit 的时间是诊断输出成本，不能当作 Fused 性能。

## 缺失元数据的定位与回传

Factory 缺失时先只读定位：

```bash
python - <<'PY'
import json
from pathlib import Path
p=Path('/home/w00949577/qwen35-dflash-run/w8-nz-full-tsUMDuhH/bundle/air-manifest.json')
a=json.loads(p.read_text())
print('factory:',a.get('factory'))
print('factory_config:',json.dumps(a.get('factory_config'),indent=2))
print('draft metadata:',json.dumps([g.get('metadata') for g in a.get('graphs',[]) if g.get('name')=='draft'],indent=2))
PY
```

请回传新目录的 `request.json`、`suite.json`、`preflight.json`、旁边的 `<AC_OUTPUT_DIR>.server.log`。构建失败补 `joint/opp.json`；GE/ATC 失败补该 graph 的 `artifacts` 下导出/编译日志、`dynamo.pbtxt`、`ac-constants.json`；执行失败补对应 `runner.log`、`execution.json`、`comparison.json`。成功补 `a-probes.json`、`c-probes.json`、可选 `draft-suite.json`/`draft-timing.md`。禁止把历史 A/C PASS 填入这些新门槛。

## 本地回归记录（不是设备验收）

- 新增集成回归 37 项：Fake-only/无 CPU fallback、Torch export 保留自定义节点及 immutable buffers、GN/NZ 内容与哈希、错误 descriptor/运行时 A 权重拒绝、四模式 ABI、报告链解析、三算子 OPP 完整性、C++ runner 声明编译。
- 相关 exporter/runtime-input/WeightQuant 回归与新增测试合计 200 项通过，使用隔离测试依赖；原 Head CPU 回归 45 项、A5.1 CPU 回归 9 项通过。
- 扩展静态 gear 回归：7 PASS、1 SKIP、2 FAIL。两项失败来自 macOS 上测试硬编码 Linux `/bin/true`（不存在/编译器路径身份不同）；在未修改 `b96dde9` 的独立源码副本上同样复现。未修改原编译器以绕过这些检查。
- Python AST、Bash 语法、C++17 `-Wall -Wextra -Werror -fsyntax-only` 检查通过；不代表真实 CANN 编译。

常规 CPU 命令（使用具备仓库所需 Torch/Transformers/Protobuf/Pytest 的环境）：

```bash
PYTHONPATH=framework/python python -B -m pytest -q   tests/test_draft_ac_integration.py tests/test_custom_op_export.py   tests/test_runtime_input_export.py tests/test_weight_quant_layout.py   tests/test_weight_quant_matmul_export.py
python -B -m unittest discover -s framework/custom_ops/draft_head/test -p 'test_*.py'
python -B -m unittest discover -s framework/custom_ops/draft_quant/test -p 'test_a51*.py'
```


### 显式 Draft-only 接口隔离（当前环境诊断）

服务器环境调查表明 `ls_llm` / `cpro` 的 torch_npu 二进制相同、GDR dispatcher
均为九参数；其它已检查环境没有证明具备十参数接口。历史 GDR effective_length-v2
与 ADN GE prototype 的哈希已匹配 Native AIR，但 GE prototype 不能补上 PyTorch
前端缺失的 `effective_length` 参数。A/C 56 个单算子图的设备验收已由服务器确认，
这不代表完整 Draft 或端到端已通过。

新增 `--draft-only` 与 `--full-draft` 互斥，默认不开启。它是独立诊断入口：

- 隔离 Preflight 子进程先完整校验历史 AIR/Deployment 的 incremental_contract、
  15 输入/11 输出 ABI、原 Factory config 和身份元数据。保持原有 Evidence、NZ
  反解、Checkpoint、OPP、Capture、Replay 和 SHA256 检查。
- 只准备 Draft 使用的 `adn_rms_norm`、functional `npu_scatter_nd_update` 和
  `npu_weight_quant_batchmatmul` 的严格 schema/Meta/Converter；A/C Converter
  仍由原导出器按各模式注册。读取 GDR schema 仅用于记录 Target 接口状态，
  不注册、不伪造、不放宽 GDR。
- 使用**未修改的** `create_quant_incremental_graphs` 加载原 Target/Draft 权重并
  构造完整 contract，保留其 source lock、输入 Manifest、Embedding/Head 来源验证。
  这是 metadata/权重构造，仍需要当前接收端加载器可导入及其所引用符号存在；
  不承诺能在任意缺少 torch_npu 符号的环境运行。构造期间任何 module forward
  都明确报错；只保留原 `DraftGraph` spec，其它 Target/Verify specs 不导出、不执行。
  不调用公共 Factory 的 `prepare_export`，也不改变该 hook 的 GDR 十参数检查。
- 重建后的**完整** contract、ABI、Target/Embedding/Checkpoint 身份必须与历史
  Native 元数据一致；26 个投影的权重/scale 与 Head 权重仍逐一比较既有 Capture
  SHA256。15 候选行、anchor、proposal_count 和完整 KV 输出保持原语义。
- 在 ATC 前进一步审计唯一 Draft AIR：原 Const/NZ、26 个投影及 Head 节点规则保留，
  出现 GDR、Target attention、CacheUpdate、DynamicQuant 或 Target QuantMatmul
  节点会明确失败，不得带入 Target 路径。
- 新目录重建联合 OPP，执行既有 56 个单算子 gate，再执行 C16/C64 × off/a/c/ac
  八组 Draft OM。每组对冻结 Native OM 的 IDs 和所有十个完整 KV buffer 逐字节比较，
  保留 guards、只读、repeat、timed-tail/postcheck 和 continuous-v1 5/30 median/p95。
  不重新 Capture、导出 Native Golden，不写生产 deployment plan。

服务器命令（使用原 Ascend310P3/CANN9、TorchAir 与 ADN/GDR 环境；不要 source 旧 A/C OPP）：

```bash
git pull --ff-only
export AC_OUTPUT_DIR=/home/w00949577/qwen35-dflash-run/ac-draft-scope-$(date +%Y%m%d-%H%M%S)
bash framework/custom_ops/draft_ac/run_known_evidence.sh --draft-only \
  --native-base /home/w00949577/LLM_infer/qwen3.5-4B-dflash-gdr-chunk \
  --preflight-only
```

回传 `$AC_OUTPUT_DIR.server.log`、`preflight.log`、`draft-interface-preflight.json`、
`draft-interface-device.json`、`suite.json`、`draft-suite.json`、`draft-timing.md`；
失败时另回传对应 `draft-*-c*/` 下的 `draft-scope.json`（若已生成）、AIR/ATC 日志、
`weight-bindings.json`、`comparison.json` 与 `native/runner.log`、`candidate/runner.log`。

报告明确区分：`draft_only=PASS` 仅在本次八组 Draft 全通过后产生；当前九参数环境
`target_interface=BLOCKED`，`full_draft=NOT_RUN`、`end_to_end=NOT_RUN`。
原 `--full-draft` / 公共 Target/Draft 导出仍受 GDR 十参数 gate 阻塞。
CPU 回归包含原 Draft forward 的 Target/GDR 禁用哨兵、范围白名单、历史 contract 与
文件篡改、全部 IDs/KV bitwise 拒绝测试。**新 Draft-only GE/ATC/NPU 与性能：NOT_RUN**，
不能沿用 56 个历史单算子 PASS 或 CPU 测试冒充新验收。

### SOURCE_LOCK 恢复与提前验收（5e5b0d7 后）

服务器 `ac-draft-only-20261016-025158` 在完成 56 个单算子 gate 后，首次加载 Factory
因 `docs/ARCHITECTURE.md` 哈希不一致失败；完整 Draft / 端到端仍为 `NOT_RUN`。
全量 CPU 审计发现 **201 条声明、191 个唯一文件，11 条声明涉及 9 个不一致文件**：

| 文件 | 与锁定哈希匹配的 Git 版本 |
|---|---|
| `docs/ARCHITECTURE.md` | `9018673` |
| `framework/python/qwen35_dflash/ascend310p/compiler.py` | `119fa26` |
| `framework/python/qwen35_dflash/ascend310p/custom_op_export.py` | `ea560cb` |
| `framework/python/qwen35_dflash/ascend310p/exporter.py` | `93be6b0` |
| `framework/python/qwen35_dflash/ascend310p/runtime_input_export.py` | `93be6b0` |
| `framework/python/qwen35_dflash/ascend310p/weight_prepack.py` | `93be6b0` |
| `tests/test_weight_prepack.py` | `93be6b0` |
| `tests/test_draft_variant_bundles.py` | `93be6b0` |
| `tests/test_weight_quant_layout.py` | `d50d203` |

文档差异由 `6d86508` 追加 A/C 诊断说明造成。恢复后的文档 SHA256 为
`088552cca7461263612fda36779f3a9b67ab4fe963ac35cac5c2a5620cf8eb6e`。
其它文件包含有价值的 A/C/FileConstant 改动，不能直接丢弃：

- 九个锁定文件恢复为逐字节匹配的历史内容。完整差异保存在
  [changes-through-5e5b0d7.patch](source_lock_recovery/changes-through-5e5b0d7.patch)，
  包括文档说明、原导出扩展、测试与 macOS 测试路径适配。
- A/C 专用导出器、公共输入/FileConstant 生命周期、INT8 descriptor 修复分别移到
  `draft_ac_export_pipeline.py`、`draft_ac_runtime_input_export.py`、`draft_ac_weight_prepack.py`。
  这些是显式诊断扩展，不伪装成历史 Native 源码。未使用模块替换、运行时源码改写或
  修改 `sys.modules` 来绕过锁；原 Factory、原 verifier 和公共导出器保持锁定内容。
- A/C 服务入口显式调用独立 pipeline。原 converter 调用计数、Const descriptor、
  SHA256、NZ bitwise、FileConstant 延迟物化和精确节点数检查均保留；编译前仍严格执行
  `validate_ac_graph`，然后调用锁定的通用编译器。额外的 descriptor/protobuf 回归移到
  独立测试文件，原锁定测试也保留。
- 本地恢复后 **201/201 声明匹配，191 个文件匹配**。
  [恢复前完整审计](source_lock_recovery/audit-before.json) 与
  [恢复后完整审计](source_lock_recovery/audit-after.json) 保存每条声明及实际/期望哈希。

`SOURCE_LOCK.json` 原始字节未改，SHA256 仍为
`107ba0425348620cdc7edf5f09e845d04d435df3e123e083e63ff5f752ba3b1f`。
这证明本地与该锁匹配，**尚不能证明它与服务器冻结 AIR 中的 quant_source_lock 相同**。
Draft-only 隔离 Preflight 现在在模型加载、联合 OPP 构建及 56 个图测试前：

1. 遍历全部声明并汇总所有缺失/篡改/非法路径，保存 `source-lock-preflight.json`。
2. 将完整 `{path, sha256, verified_file_count}` 与 Deployment 哈希绑定的 Native AIR
   Draft 元数据严格逐字段比较；路径、计数、哈希或额外字段不一致均失败，不自动升级锁。
3. 再调用未改动的 `_verify_quant_source_lock()`，核对其返回身份与完整审计一致。
4. 单独记录 `draft_ac_*.py` 扩展 SHA256。设备子进程再执行同样检查，保存
   `source-lock-device.json`，Factory 加载时的原检查也不删除。

本节记录 `2ac51d1` 的当前锁恢复过程；现已确认 Native 使用另一份 205 条声明的锁，
必须使用下节历史源码隔离入口。请额外回传 `source-lock-preflight.json`
及 `source-lock-device.json`。若冻结 Native 的锁身份不同，先根据报告继续调查，不覆盖
Native AIR、Deployment、SOURCE_LOCK 或 Golden。

本轮本地 CPU 回归：463 PASS，88 项因未配置 C++ fake ACL runner 或依赖该 runner 的
host fixture 跳过；macOS 的 Linux fixture `/bin/true` 使用本地 `/usr/bin/true` 适配，
未修改锁定测试。新隔离 pipeline 的 GE/ATC/NPU、完整 Draft 与端到端：`NOT_RUN`。

### 历史 Native Base + 独立 Diagnostic Extension

服务器已验证历史目录 `/home/w00949577/LLM_infer/qwen3.5-4B-dflash-gdr-chunk`
完整匹配冻结 Native AIR：SOURCE_LOCK SHA256
`7872c0f687aca12421786ed9508b8416a987ec38d5d8bc116dee39861c119395`，
205/205 条声明、197/197 个文件通过，identity_diff 为空。Git 对应 `319f51d`。
它与当前 201 条声明的锁之间有 18 个共同文件内容不同，不能替换 Factory 或伪造身份。

`--draft-only` 现在必须显式提供 `--native-base`。该目录是唯一 Native 源码根，
不创建服务器替代 checkout，不写入历史文件，不修改历史 SOURCE_LOCK 或 Native Golden。

导入及进程边界：

- 协调进程只编排；独立 `native_worker.py preflight` 在导入模型框架之前，验证
  Deployment 哈希绑定的 Native AIR、明确指定的历史路径、原始锁 SHA256、205 条身份
  及全部声明。已加载 `models` / `qwen35_dflash` 模块会被拒绝，不清空 sys.modules。
- 历史 `models`、`qwen35_dflash` 模块由只读历史目录加载。导入器从已校验源码字节
  编译，不读取/生成历史 pyc；Python 写操作指向历史根会被拒绝。还记录全部历史
  Python 文件快照，检测包初始化文件及未列入锁的辅助 Python 文件变化。
- 仅六个新模块名允许从当前仓库加载：`draft_ac_export`、`draft_ac_constants`、
  `draft_ac_payload`、`draft_ac_export_pipeline`、`draft_ac_runtime_input_export`、
  `draft_ac_weight_prepack`。它们使用历史框架依赖，不替换任何历史模块；不把当前框架目录加入历史包搜索路径。
  历史根若已存在同名扩展会明确拒绝，避免覆盖。
- 原 Factory 还依赖接收端 wrapper。Preflight 校验冻结 AIR 的 input manifest SHA256、
  `receiver_wrapper` group 哈希、wrapper 字节数与文件哈希、原 `receiver_models_dir`，
  然后调用历史 `_append_receiver_models()` 按原流程追加该目录并导入 wrapper（不实例化模型）。
  接收端 Python 文件快照独立记录为 Receiver 来源，不能覆盖已有 Native 模块；wrapper
  若被非 Manifest 绑定路径遮蔽则拒绝。接收端漂移也会在 OPP 构建前被检查。
- 每个已加载模块记录真实 `__file__`、SHA256、Native/Extension/Receiver 分类及模块对象身份。
  Factory 和 verifier 的代码文件必须来自历史路径；随后实际调用历史
  `_verify_quant_source_lock()`，严格要求返回 Native AIR 的路径、SHA256 和 205 条记录。
  在每次 Factory 加载前再次检查，而不是伪造返回值或只验证当前 Factory。
- 扩展代码及诊断工具的文件哈希、Git revision、导出设置单独记录。Preflight 返回后，
  协调进程在构建 OPP 前再次检查快照；联合 OPP 设置后 exec 新设备进程，以相同的已接受
  快照启动。任何源码或扩展变化都会失败，不沿用旧模块/旧动态库。
- 历史预打包接口没有 `output_dir`。已有 NZ Manifest 路径使用原四个位置参数调用，
  保留全部原始预打包检查；历史模式拒绝自动预打包。A/C 的 descriptor、FileConstant
  物化、SHA256、NZ bitwise、GE 节点数审计继续在显式扩展 pipeline 中执行。

预检在模型加载、OPP 构建和 56 个单算子图之前完成。Target GDR 仍执行原十参数
schema 判定并单独记录；当前服务器九参数环境为 `target_interface=BLOCKED`，
不阻止只依赖实际 Draft 接口的诊断。Target/Verify 和端到端始终 `NOT_RUN`。

**本轮只运行轻量 Preflight：**

```bash
git pull --ff-only
export AC_OUTPUT_DIR=/home/w00949577/qwen35-dflash-run/ac-native-base-preflight-$(date +%Y%m%d-%H%M%S)
bash framework/custom_ops/draft_ac/run_known_evidence.sh \
  --draft-only \
  --native-base /home/w00949577/LLM_infer/qwen3.5-4B-dflash-gdr-chunk \
  --preflight-only
```

回传 `native-base-preflight.json`（包含模块来源、两套源码身份和扩展快照）、
`source-lock-preflight.json`、`draft-interface-preflight.json`、`preflight-result.json`、
`suite.json`、`preflight.log` 和旁侧 `.server.log`。只有本次真实 Preflight 通过后，
再提供正式设备验收命令；不得把导入/CPU PASS 写成八组 Draft NPU PASS。

后续设备路径保留 C16/C64 × off/a/c/ac 八组，复用冻结 Native OM，严格比较 Token IDs
和全部十个 KV 输出，保留 26 个投影及 Head 权重哈希、guard/readonly/repeat、
timed-tail/postcheck 与 continuous-v1 median/p95。设备 worker 另写
`native-base-device.json`，且边界最终检查失败会将 suite 标记 FAIL。

CPU 回归使用 `319f51d` 的 Git 内容作为临时测试数据，覆盖原 Factory 205 条身份、
扩展/Native 来源隔离、预加载混用、锁篡改、Native 身份差异、扩展变化、只读历史根、
两进程快照一致性，以及历史 FileConstant + NZ Manifest 接口兼容性、原接收端 wrapper 的 Manifest 绑定和防遮蔽检查。这些测试不运行
服务器 checkpoint、CANN 或 NPU。历史隔离入口的八组 GE/ATC/NPU 验收：**NOT_RUN**。

本轮本地回归结果：427 PASS、84 项因未配置 C++ fake ACL runner 跳过；shell 语法及
差异空白检查通过。macOS 的 `/bin/true` fixture 使用本地 `/usr/bin/true` 适配，
没有改动历史锁定文件。服务器历史隔离 Preflight 和八组设备验收均仍为 NOT_RUN。

### a6b5498 generator tracing：单图复测

服务器 `ac-draft-only-20261016-060918` 已完成历史源码/接口 Preflight、联合 OPP 安装
和 Runner 构建；失败发生在首次 `a-c16-fc` 的 `torchair.dynamo_export`。该轮日志没有
保存 `from user code` 或 generator 内层帧，不能从外层 traceback 直接归因 Transformers。

本地 PyTorch 2.10 CPU 受控复现（不是服务器 PyTorch 2.8/NPU 结果）：

| 同一普通 nn.Module + fullgraph 捕获 | 结果 |
|---|---|
| 不安装只读审计钩子 | 捕获成功 |
| 安装 a6b5498 的 `protect_writes()` | `Attempt to trace generator` |
| 安装修复后的钩子 | 捕获成功，仍严格拒绝历史目录写入/删除/重命名 |

失败栈指向 `native_base.py` 的 `audit()`：`any(inside(p) for p in paths)`。
父 callback 虽然被 Dynamo 判为没有 torch 操作，其 generator expression 仍会作为独立
frame 遇到 eval-frame hook。修复仅把两个 generator expression 改为普通条件/循环，
并对不涉及写入的审计事件直接返回。没有移除写保护，没有 `torch.compiler.disable`、
没有 suppress_errors、没有放宽 formal export 或数值检查。

`5e5b0d7` 原 exporter 与拆分后的 pipeline 中，包围 `torchair.dynamo_export` 的
inference/工作目录/ABI context block 逐字相同。CPU 对照也保留 host contextmanager；
无需先删除生成器式 contextmanager 或修改历史 exporter。服务器仍需以下单图门槛确认。

`Config not found for dflash-v1` 单独处理：官方 Transformers 5.13.1 的
[auto_docstring 源码](https://github.com/huggingface/transformers/blob/v5.13.1/src/transformers/utils/auto_docstring.py)
在文档配置查找失败时打印该提示，并使用 `ModelConfig` 占位值继续返回；该分支没有
直接抛出 generator 异常。新的 `environment.json` 记录服务器实际安装版本、源文件
SHA256、消息所在函数/行号/附近代码及当时的 alias 值。它不被当作已证实根因，也不
通过修改 Transformers 或历史模型配置来隐藏提示。PyTorch 对直接 generator frame
的限制见 [v2.8.0 convert_frame](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/convert_frame.py)。

独立入口 `run_single_graph.sh`：

- 从旧 `worker-input.json`、已通过的 Native boundary、`suite.json` 和 `joint/opp.json`
  核对已有 OPP/Runner，复查安装文件/注册/二进制哈希及冻结构建配置，不重建或覆盖旧包。
- 在新目录重新执行全部只读 provenance 和历史 Native Base Preflight。新代码使用新的
  Extension 哈希快照，绝不拿旧 extension 快照冒充当前版本。选择已验证的旧 OPP 环境后，
  启动另一个全新 Python/ELF 进程。
- 只选 `C16/fc`；复用正式 exporter、compiler、Runner 和 Native bitwise 比较函数。
  不执行其余 55 图、Head 或八组 Draft。完整模式仍检查 guards、只读、重复、行置换、
  timed-tail/postcheck；输入、FileConstant、NZ、GE OpType/端口审计均保留。
- 默认记录 `TORCH_LOGS="+dynamo"`、`TORCHDYNAMO_VERBOSE=1`；`--trace-bytecode` 额外启用
  `trace_source` / `trace_bytecode`。日志写入 `preflight.log` 和 `single-graph.log`。
  异常写入完整 `traceback.txt` 及结构化 `exception.json`，记录 frame 函数、文件、行号、
  源码哈希，并提取 traceback 局部变量中尚存的 generator code/frame 信息；不 repr Tensor。
- 默认完成 AIR→ATC→NPU。可附加 `--air-only` 定位导出；其状态为 `AIR_ONLY`，ATC/NPU
  保持 `NOT_RUN`，不能算单图数值 PASS。`export-stages.json` 保留实际失败阶段。

在原先 Transformers 5.13.1 隔离 venv 和 CANN9 环境下执行（不要切换到其它模型环境）：

```bash
git pull --ff-only
bash framework/custom_ops/draft_ac/run_single_graph.sh \
  --source-evidence /home/w00949577/qwen35-dflash-run/ac-draft-only-20261016-060918 \
  --output-dir /home/w00949577/qwen35-dflash-run/ac-fc-generator-$(date +%Y%m%d-%H%M%S) \
  --trace-bytecode
```

回传新目录的 `suite.json`、`single-graph.log`、`environment.json`、
`native-base-single.json`、`a-c16-fc/export-stages.json`；失败时附 `exception.json`、
`traceback.txt`（Preflight 失败则附 `preflight.log`）。成功时附
`a-c16-fc/comparison.json` 以及 `native/runner.log`、`custom/runner.log`。
单图 AIR/ATC/NPU 在服务器执行前均为 `NOT_RUN`；八组 Draft 和端到端仍为 `NOT_RUN`。
只有首图实际 bitwise PASS 后才重启 56 图及完整 Draft，不用 CPU 复现替代设备验收。

本轮相关 CPU 回归：353 PASS，包括 legacy 钩子受控复现、fullgraph 捕获、写保护不变、
单图范围、AIR-only 状态、OPP/Runner 篡改拒绝和新进程日志/环境设置。没有本地 CANN/NPU PASS。

### C 完整 Draft 性能回退归因（只读 Profiling）

用户已报告 `ac-draft-full-20261016-071408` 的 52 A + 4 C 单算子图及 8 组完整 Draft
bitwise PASS。C16 的 c−off / ac−a 分别为 9.3474 / 9.3077 ms；C64 分别为
9.3457 / 9.3134 ms。这说明 C 的整图增量相近，**尚不能定位到 MatMul、Top1、布局转换
或调度中的某一项**。独立 runtime-weight Native Head 的 23.3 ms 与 immutable-weight
整图不是同一实验，不能直接从整图减去独立 Head 时延。

新增 `profile_c.sh`，不调用 Factory，不加载 checkpoint，不导出 AIR、不编译新 OM、
不重建 OPP/Runner。它只用现有 Runner 加载冻结 OM 并执行相同输入，所有输出在新目录：

- 核验已有 52/4/8 状态、每个选中 OM/AIR/Deployment/Const 审计及哈希、15/11 ABI、
  冻结输入、Native/候选输出和旧 continuous-v1 5/30 median/p95；各模式必须使用相同输入。
  复用原隔离 OPP、Loader 环境和已绑定的 C++ Runner，结束后复核原始文件没有变化。
- msprof 包裹原 Runner，启用 AscendCL、Runtime API、task time、AI CPU、AI Core 和
  PipeUtilization；可用 `--metrics Memory` 或 `MemoryUB` 做后续独立采集，不把不同指标
  的采集时间混为同一轮 benchmark。所有 profiled timing 独立保存，不覆盖历史数据。
- 原 Runner 的 38 次执行为 2 次正确性 + 5 次 warmup + 30 次计时 + 1 次 postcheck。
  根据同一 Chrome timeline 的 Execute/Sync 配对，严格识别后才选择索引 7..36。
  38 个窗口不完整、缺硬件轨迹或多个候选 timeline 时，保留原始数据并标记未解决。
  不简单拿 op_summary 总时间除以 38，也不跨 CSV/JSON 混用不同时间原点。
- 模型加载、权重上传、poison 和验证读回不在所选 Execute/Sync 窗口内。
  Host API 时间、Host 同步等待、Device envelope、重叠任务的区间并集及空隙分别报告。
  Host Sync 与设备计算重叠，不能相加；Pipe 指标也不能直接相加。
- Native Head 根据完整词表 descriptor/shape 和图中相邻 Argmax/转换定位，不能凭
  “最大的 MatMul”猜测。C 的 fused Kernel 单独识别；未映射 MatMul/Argmax 明确列出。
  记录原始 task 行、输入/输出格式、dtype、shape 及可用 Pipe 字段。
- 流式读取原 `dynamo.pbtxt`，跳过巨大 Const payload，仅保留描述符/属性和边。
  默认尝试 `atc --mode=1 --om=已有文件 --json=新文件` 查看编译后结构；这是已有 OM
  的只读结构转换，不是模型重新编译。若版本不支持或输出无法解析，记录 UNAVAILABLE，
  保留原始日志，不能把 pre-ATC 图误称为执行图。可用 `--skip-om-dump` 显式跳过该检查。
- Profiling 后所有 11 个输出仍须匹配冻结 Native，guards、输入只读、重复、
  timed-tail/postcheck 不变。运行成功不自动意味着根因已找到。

在服务器现有 CANN/msprof 环境中，先采集 C16 的四个模式：

```bash
git pull --ff-only
bash framework/custom_ops/draft_ac/profile_c.sh \
  --evidence /home/w00949577/qwen35-dflash-run/ac-draft-full-20261016-071408 \
  --output-dir /home/w00949577/qwen35-dflash-run/c-draft-profile-$(date +%Y%m%d-%H%M%S) \
  --gears 16
```

采集两档用 `--gears 16 64`。附加 `--prepare-only` 时只做文件/图审计并生成 NOT_RUN
报告，不启动 msprof 或 ATC。新目录包含 `c-profile-report.json`、`c-profile-analysis.md`，
以及每模式 `profile.log`、`msprof-command.json`、`profiler/`、`inspect/`、`replay/`。
请回传两份总报告和原始 profiler 导出（至少 op_summary CSV 与 msprof timeline JSON）。

[仓库中的初始报告](c-profile-analysis.md) 仅记录用户提供的 median，p95 未提供；
[初始 JSON](c-profile-report.json) 的 Profiling 状态为 NOT_RUN。服务器工具会在独立目录
生成经文件校验的报告，不改写此记录或历史 evidence。

后续仅评估：若同一次设备时间线证明 Native MatMul 明显更快，则比较 C 的 MatMul
数据流优化与 Native MatMul + 独立 Top1。后者必须保留最终 FP16 logits 舍入和所有
Top1 特殊值/最小 ID 语义，并计入 M15 logits 的 7,449,600 B 写回、读取及调度成本。
没有测量前不承诺收益，也不预先认定融合一定更快。C 内部 MatMul/Top1/跨核归约无法
只凭一个 fused Kernel 时间拆分，本轮不修改 Kernel 来做插桩。

参考：[华为 OM→JSON 参数说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/devaids/atctool/atlasatcparam_16_0009.html)，
[msprof 数据格式](https://www.hiascend.com/document/detail/en/mindstudio/2610/TITools/msProf/docs/en/user_guide/profile_data_file_references.md)。
文档版本不代表服务器实际可用能力；以保存的命令、返回码和原始产物为准。

本轮本地新增 CPU 回归：16 PASS（标准库图/时间线解析、38→30 调用选择、B/E 事件、
重叠去重、缺失或重复导出拒绝归因、文件篡改门槛、只读重放及 prepare-only）。
没有本地或服务器 Profiling/NPU 性能归因 PASS。

## 独立 FP16 六层 vs W8 A-only 五层对比

`profile_fp16_w8.sh` 不更改历史 AIR/OM、SOURCE_LOCK、A/C Kernel、Target/Verify
或 W8 Evidence。FP16 与 W8 checkpoint、层数及 feature width 不同，分别用自己的
冻结输入和 Golden；比值是系统延迟对照，不能直接归因成纯量化收益。

### 先运行轻量兼容检查

```bash
bash framework/custom_ops/draft_ac/profile_fp16_w8.sh --w8-evidence /home/w00949577/qwen35-dflash-run/ac-draft-full-20261016-071408 --search-root /home/w00949577/qwen35-dflash-run --output-dir "/home/w00949577/qwen35-dflash-run/fp16-prepare-$(date +%Y%m%d-%H%M%S)" --prepare-only
```

这一步不加载 torch/NPU、不调用 msprof、不构建 Runner/OPP/OM、不采集数据，
也不重复整套 W8 Evidence 重哈希。输出 `fp16-w8-report.json` 中的 `fp16_preflight`：
每份候选 Deployment 的路径、Manifest 校验、动态/静态契约、逐文件原始 SHA256、
实测大小、原始 `bytes` 是否存在、C16/C64 Replay 和采集来源缺失项。
存在 Chunk/MTP 多份候选时不猜选；后续用 `--fp16-deployment` 显式选择。
未满足重放条件则 `status=BLOCKED`、退出码 2；所有设备项仍为 NOT_RUN。
轻量检查成功也仅代表 FP16 文件契约，不代表 W8 或实际设备 gear 已验收。

旧 Manifest 缺少 `bytes` 时，FP16 专用读取器验证 SHA256、非空普通文件、读取前后
size/mtime/inode 一致，并记录 `actual_bytes`；有 `bytes` 时仍严格相等。
拒绝绝对 artifact 路径、`..`、symlink 逃逸、缺失/篡改/空文件。不补写历史 Manifest，
也不修改共用 `checked_file()` 或 W8 检查行为。

### 动态 OM 与冻结输入

`static_gear_oms=[]` 不再直接报“不支持”。校验历史 ATC 的 `--dynamic_dims=16;64`、
`--input_format=ND`、完整 Data 端口/shape、仅 features axis 1 动态，以及原
incremental_contract。实际设备 gear 尚不能由 CPU Preflight 证明，明确标记 NOT_RUN。

若已有完整 FP16 Replay，传 `--fp16-c16-replay` / `--fp16-c64-replay` 直接复用。
如果只有 W8 Replay，不能混用。显式 `--capture-fp16` 才会创建新 FP16 输入/Golden：

- 必须指定 FP16 Deployment，以及 `--fp16-c16-request` / `--fp16-c64-request`。
  每个参数是原 FP16 `tools/profile_om.py` 的 `profile-request.json` 或
  `tools/debug_draft_om.py` 的 `request.json`，旁边必须有原 `plan.txt`。
- 核对 request 内 Deployment/Runner 绝对路径与 SHA256、plan 所有 Graph OM 的
  SHA256、AIR/Deployment 元数据及完整 ABI；不创建替代历史计划，不改变 Target。
- 使用原 request 的 token IDs、padding、EOS。原 prompt 最后一个 64-token chunk
  的有效长度必须选到请求的 C16/C64 gear，且满足原 capacity/vocab；不生成伪 hidden。
- 调用原 C++ Runner 的 `--debug-draft-replay 20 --debug-draft-workspace private`。
  该原流程会执行原 FP16 Target Prefill 来产生真实 features，然后进行 isolated /
  interleaved-prefill 两阶段重放，生成完整输入及全部 Token/KV Golden。**采集需要
  原 Runner 和原 OPP 环境确实可运行**；失败即保留日志，不用 Python/eager 回退。
- 新文件只写本次输出目录的 `fp16-capture/`，检查 fake_acl=false、设备、prompt、
  gear、所有输入/输出 SHA256、完整输出重复一致与只读门槛。采集不是正式计时，
  也不代表 Target/Verify 或端到端验收。

尚未确认原 FP16 Runner、两种 prompt 请求的真实路径，不硬编码或伪造这些路径。
如果旧 profile-request 只有 C64，需要提供原 FP16 路径的合法 C16 请求及其原计划；
工具不会截断 W8 features、复制 W8 Golden 或猜造请求元数据来让检查通过。

### 5/30 动态重放与 Profiling

原完整 C++ Runtime 支持动态 gear 和捕获，但 debug replay 夹有状态恢复/读回，
不是 continuous-v1 5/30 benchmark。为保持正式计时可比，在独立目录构建小型
`fp16_dynamic_runner`（仅 C++ 应用，不编译新 OM/OPP）。也可通过
`--fp16-dynamic-build /path/to/build.json` 复用源码/二进制 SHA256 匹配的适配器。
原历史 Runtime、A/C 静态 Runner 和原动态 FP16 OM 均不改写。

动态适配器沿用仓库 `acl_chunk.cpp` 的 gear 查询方式：读取动态控制输入索引、
查询恰好两个 gear，将全 17 个输入的 flattened shape 与 C16/C64 逐项匹配，
再调用 `aclmdlSetInputDynamicDims`。保留最大 C64 输入 allocation 及原快照字节，
只由 SDK 选择 gear；绝不当静态 OM 运行。`dynamic-gear.json` 保存实际 gear 和选择。
动态控制 buffer 由 SDK 管理，前后 Guard 同样校验；17 个公开输入保持逐字节只读。

所有四组先跑无 Profiling 的正式 5 warmup / 30 repetitions，再单独跑 msprof，
使用相同设备 ID、OPP/动态库环境和 `runner_common.h` continuous-v1 调度。
gear 选择和模型加载在计时外；保留两次正确性输出、repeat、Guard、输入只读、
timed-tail/postcheck 与各自 Golden 的全部 Token/KV bitwise 校验。
W8 A-only 继续使用已验收的原静态 Runner，不重建 56 个单算子或八组 Draft OM。

产物为 `fp16-w8-report.json`、`fp16-w8-analysis.md`，以及各模型/gear 的
`formal/`、`formal.log`、`profiled/`、`profile.log`、`profiler/`、可用时的
`om-graph.json`（ATC mode=1 只读检查）。分别报告正式 median/p95 与插桩后的
Host launch/sync、设备窗口、Linear/LM Head/Attention/Norm/KV/TransData。
不把未知 MatMul 强归类为 Linear，也不将重叠 Pipe/Kernel 时间相加当墙钟时间。

本地 CPU 测试及 fake ACL 头文件的 C++ 语法检查不是 CANN/NPU 验收。实际动态 gear、
新 FP16 capture、正式延迟与 msprof 在服务器执行前均为 **NOT_RUN**。
原 FP16 动态 OM 若存在真实 SDK 限制，则明确 BLOCKED；当前没有自动重编静态 OM 的回退。

API 语义参考：[aclmdlSetInputDynamicDims](https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/apiref/appdevgapi/aclcppdevg_03_0299.html)。
文档版本不替代 CANN 9.0.0 设备验证；接口使用和 full-shape 校验优先复用仓库现有动态 Runtime。

## C16-only：已知 FP16 KV 差异下的独立性能诊断

`profile_c16_diagnostic.sh --diagnostic-only` 是单独入口，不是正式验收的降级模式。
它要求原 FP16 C16 报告仍为 `FAIL_REPLAY_CHECKS`，且历史 Token、输入恢复、只读
检查完整；有效 KV/全输出的不一致计数原样保留。正式 `fp16_w8_profile.py` 仍拒绝
失败 Replay。共用的新 `replay_payload()` 只读取、验证字节身份，不决定验收状态。

默认只从指定 `private.json` 的相邻 `request.json` 解析哈希绑定的 FP16 Deployment，
若没有该文件则查找 Draft OM 哈希匹配的 Deployment；多解必须显式指定
`--fp16-deployment`，不猜 Chunk/MTP。继续验证 SHA-only/带 bytes Manifest、AIR/OM、
六层 ABI、原快照 `contract.txt`/`sha256.txt`、输入和历史 Native reference 文件。
后者只是失败 Replay 的原始参考输出，不能被称为已验收 Golden。

诊断构建由独立 CMake target 定义 `DFLASH_C16_DIAGNOSTIC_ONLY`，复用现有动态
Runner 的 ACL/gear/内存/计时逻辑。正式 target 不定义此宏，仍然在任何输出 repeat
差异时失败。诊断 target 只把数值差异改为观测记录，**Guard、输入只读、ABI、OM、
实际 C16/C64 flattened gear 校验错误依然中止**。可用 `--diagnostic-build build.json`
复用源码与二进制 SHA256 匹配的诊断 Runner；正式 Runner 不能冒充该构建。
不编译新 AIR/OM/OPP，不重新运行 Target Prefill，也不重复 56 图或八组 Draft。

执行三个互相分开的阶段：

1. FP16 单独执行 30 次未计时数值调用。每次 poison 输出、执行/同步、检查 Guard
   和输入只读，再读取完整 Token/KV 并计算指标。逐次写 `observations.jsonl`，
   不保存 30 份大型 KV 二进制。
2. FP16 和 W8 A-only 各自执行无 msprof 的 continuous-v1，warmup=5、repetitions=30。
   计时循环不 poison、不读取 KV、不计算误差、不写文件。只观测两次前置输出、
   最后一个计时输出和一次 postcheck，保存四份原始输出。**其余计时输出未逐次观测，
   不把独立的 30 次数值观察说成这 30 次计时输出的正确性证明。**
3. 两模型分别在 msprof 下重放相同 5/30 协议，时间只用于归因，不混入无插桩样本。

每份 FP16 输出记录全部 INT64 Token 的 bitwise 比较及实际 ID；每个 KV 记录
bit mismatch 数/比例、max_abs、RMSE、nearest-rank p99_abs、双方非有限值数量。
误差统计明确限定 finite pairs；非有限 pair 单独计数，不被当成零误差。
同时报告全 tensor、有效前缀 `[0,start+valid_rows)`、本次有效写区和 padding；
按真实 `[B,H,S,D]` row stride 检查原输入中应保持不变的 `[0,start)` 与
`[start+16,capacity)`，不把各 head 的有效行误当成一个平坦前缀。空区域指标为 null。
记录 repeat_equal，但即使 repeat_equal=true 也不代表数值正确。

总报告固定 `status=DIAGNOSTIC_ONLY`、`formal_acceptance=NOT_RUN_NOT_ELIGIBLE`、
`end_to_end=NOT_RUN`；执行完成与失败由另一个 `execution` 字段表示。W8 复用原严格
Runner，并继续与自己的冻结 reference 比较；其原始 `execution.json.status=PASS`
仅表示该 Runner 的执行检查，不能提升本诊断或 FP16 的验收状态。

输出：`c16-diagnostic-report.json`、`c16-diagnostic-analysis.md`，及 `fp16/numerics/`、
两模型 `timing/`、`profile/` 内日志、观测和 profiler 原始数据。记录独立 OM SHA256、
模型元数据、输入 SHA256、gear、valid_rows、proposal_count、median/p95、设备环境。
归因标签包括 W8_LINEAR、FP16_LINEAR、TRANSDATA_LAYOUT、LM_HEAD、ATTENTION、NORM、
KV_CACHE、OTHER_KNOWN、UNKNOWN，以及 Host launch/sync。各区域 union 时间仍可能相互
重叠，不相加作为 wall time；无法映射的 MatMul 保持 UNKNOWN。

服务器命令（仓库根目录，更新代码后执行）：

```bash
bash framework/custom_ops/draft_ac/profile_c16_diagnostic.sh --diagnostic-only --fp16-replay /home/w00949577/qwen35-dflash-run/debug-draft-l1bxgjqh/private.json --w8-evidence /home/w00949577/qwen35-dflash-run/ac-draft-full-20261016-071408 --search-root /home/w00949577/qwen35-dflash-run --output-dir "/home/w00949577/qwen35-dflash-run/fp16-w8-c16-diagnostic-$(date +%Y%m%d-%H%M%S)" --device-id 0
```

可加 `--prepare-only` 做只读身份检查。所有产物写新目录，原失败状态、Frozen Native、
历史参考输出、SOURCE_LOCK、FP16 OM 和 W8 Evidence 不变。CPU metrics/流程/假头文件
语法回归不能代替 CANN/NPU 测试；设备诊断在真实执行前保持 NOT_RUN。

### 恢复 053c25a 已完成的 C16 诊断，只补 W8 Profiling

服务器目录 `fp16-w8-c16-diagnostic-20261019-054822` 已完成双方 5/30 无插桩计时、
FP16 30 次数值观察及 FP16 msprof。异常原因已用用户提供的结构在 `053c25a` 原解析器
上复现：TorchAir 的 `dynamo.pbtxt` 用 **单引号字符串**保存另一层 GE TextFormat，
例如 `s: 'name: "y"\\ndtype: DT_INT64...attr {...}'`。旧 tokenizer 只识别双引号，
因此把字符串内的花括号及字段误当外层语法，产生 None key，随后在 `redact()` 崩溃。
这与服务器示例中的 `<class ...torch.SymInt...>` 或 tensor data 字节无关；它们是字符串内容。

新结构解析器按 TextFormat 的字符串/转义、字段、消息、列表和括号层级推进，
不会创建 None 字典 key；`values()`/`redact()` 遇到非法 key 直接报错，不用 `str(k)`
掩盖结构失步。超大 tensor 字符串有界读取并省略，不把内部字符串拼进外层图。
完整且有 dtype/shape 的短 GE TensorDesc 字符串可单独解析为描述符；不完整或普通
备注文字不能作为 Head shape 证据。图解析异常由独立 metadata helper 记为
`UNPARSED/UNKNOWN`，不能丢掉已经验证的性能数据，也不能猜测最大的 MatMul 就是 Head。

独立恢复命令：

```bash
bash framework/custom_ops/draft_ac/recover_c16_profile.sh --source-evidence /home/w00949577/qwen35-dflash-run/fp16-w8-c16-diagnostic-20261019-054822 --expected-fp16-profiler-files 109 --output-dir "/home/w00949577/qwen35-dflash-run/c16-profile-recovered-$(date +%Y%m%d-%H%M%S)"
```

恢复先检查：

- 原始 `DIAGNOSTIC_ONLY` 报告和 FAIL_REPLAY_CHECKS 身份、两模型/gear/ABI/控制参数。
- 原始 `source_watch`、联合 OPP、FP16/W8 Runner 源码与二进制 SHA256、所有输入/OM。
- 已完成阶段的 command/launch、TSV、输入/参考文件、30 次观测及四个边界输出清单。
- 独立 execution/observation 文件必须与原报告嵌入记录一致；5/30 的 median/p95 从
  原始样本重新核对（仅允许原 C++ 九位有效数字的序列化差）。
- FP16 的计时/profiling 边界原始 Token/KV 输出重新计算所有指标，与保存的观测核对；
  bit count、max_abs、p99 和 Token 精确一致。仅 RMSE 的 CPU float64 vs 原 long-double
  求和允许 `1e-12` 相对重算差，这不是 NPU tensor 容差，更不会提升 FP16 正确性。
- FP16 profiler 必须恰有指定的 109 个文件；JSON/CSV/SQLite 做格式/结构检查。
  缺失、损坏、篡改可检测记录，或 W8 profile 已存在（含部分目录），均在执行前阻断。

**历史完整性限制必须保留：**053c25a 在解析图前崩溃，没有留下采集时的 profiler /
诊断输出文件 SHA256 清单。恢复会建立新哈希清单并核对结构、嵌入观测、可保留的原始
输出，但不能倒推证明文件自采集以来从未被改动。30 次独立数值观察本来就未保存
30 份完整 KV；这部分只能核对观测记录与原报告，不能凭空补 raw outputs。二进制 raw
profiler 文件标注只做字节快照而未解码。这些限制写入 `recovery`，不伪造历史校验 PASS。

校验后，FP16 profiler **只离线解析**。双方无插桩 latency/sample 数据原样复用；
只调用一次 msprof，在新目录执行原 W8 A-only Runner。不会运行 FP16 Runner、
重新计时、Target Prefill、ATC、AIR/OM 导出、Runner/OPP 构建或八组 Draft 验收。
新 W8 profile 输出仍与它自己的冻结 Native reference 严格比较。
可加 `--validate-only`，在离线解析后退出，不执行任何设备命令。

输出仍是新目录的 `c16-diagnostic-report.json` / `c16-diagnostic-analysis.md`，包含
两模型 Kernel 区域、Host launch/sync、device occupancy、UNKNOWN task 比例及
UNKNOWN 区间 union / 全部 device 区间 union 比例。区域可重叠，比例不能相加当
wall time 分解；旧 FP16 与新 W8 profile 属于不同采集时段，不混合时间戳或声称因果。
无插桩延迟差由原始两组 median 计算，不用 profiling 数据替换。
原目录完整文件树在恢复前后核对，新文件仅写新目录，FP16 失败状态始终保留。

CPU 回归包括真实单引号嵌套描述符、字符串边界、非法 key 拒绝、完整性破坏阻断，
并以模拟子进程证明只调用一次缺失的 W8 msprof。未在服务器执行的恢复仍为 NOT_RUN。
TextFormat 语法参考：[Protocol Buffers Text Format](https://protobuf.dev/reference/protobuf/textformat-spec/)。

### 2026-10-19：只读解析现有 C16 的 38/30 窗口和 Linear 热点

服务器实际事件是 `AscendCL@aclmdlExecuteAsync` / `AscendCL@aclrtSynchronizeStream`，
并在 `args.level=acl`、`args.id` 提供 API 名称。解析器显式支持这两种拼写及原裸名称，
拒绝名称/id 冲突，不模糊匹配内部 runtime Execute。FP16 是 38 Execute / 40 Sync，
W8 是 38 / 39：按同 PID/TID、Execute 返回后且下一 Execute 之前的第一个 Sync 配对，
记录初始化/析构等额外 Sync，不能按两个列表的相同下标配对。

绝对时间戳约 1.79e15 μs，先用 Decimal 减去统一 origin，再转 float，避免先转换 epoch
丢失亚微秒 API 间隔。严格验证 38 个窗口，并选零起始 7..36 的 30 个计时调用；其余
2 次正确性、5 次 warmup、1 次 postcheck 单独保留。边界交叉的设备事件显式记录，
不能被悄悄划进某个完整窗口。38 次不完整或线程/配对歧义则保持 UNRESOLVED_WINDOWS。

以下命令**只读已有 profiler 和冻结图元数据**，不调用 msprof、ATC、Runner 或模型：

```bash
python -B framework/custom_ops/draft_ac/test/linear_attribution.py --report /home/w00949577/qwen35-dflash-run/c16-profile-recovered-20261019-070652/c16-diagnostic-report.json --output-dir "/home/w00949577/qwen35-dflash-run/c16-linear-offline-$(date +%Y%m%d-%H%M%S)"
```

输出 `linear-timeline.json` / `linear-analysis.md`：两模型 38 轮 API 配对、30 轮区域
统计、逐 Kernel 来源、Gate/Up、Down、其他 Linear、Head、OTHER_KNOWN、UNKNOWN，及
region union、device occupancy/gap、task-sum 减 union、跨区域重叠。API 和任务时间
只用同一 Chrome trace 时钟；CSV 仅提供 name/type/shape 元数据，不混用其时间戳。
分类依赖实际 logical M/K/N；CSV 缺失 shape 时，可用哈希绑定的图中**同名同类型**节点
的明确 input/output TensorDesc。仍无法匹配时保留 UNKNOWN，不用 Kernel 耗时或出现顺序猜。
FP16 Gate/Up/Down 使用其实际 9216 intermediate，与 W8 的 9728 分开；每轮 6/5 层的
FFN 数量另行核对。文件 SHA256 与原恢复清单核对，旧 report/profiler 均不修改。

当前离线工具未在本地读取服务器原始文件，真实 30 窗口/区域结果待回传。
源码瓶颈账本、同语义 Native W8 比较要求及两个最小 Gate/Up 候选见
[量化 A5.1 分析追加](../draft_quant/A5_1.md#2026-10-19c16-整图诊断后的-gateup-优化分析本轮未改-kernel)。

### C16 离线归因修订：Host 窗口与设备迭代分开验证

服务器回传的原始事件确认：`Iteration N` 的 `cat=Iteration Time`，所在线程为
`Step Trace(Model ID:1)`，包含 `Iteration ID` / `Iteration End`；实际 Kernel 在
`Stream 0`，携带 `Model Id`、`Physic Stream Id`、`Task Id`。设备 connection_id 在
多次调用中重复（例如 FP16 Head 为 560），不是 Host Execute 的 connection_id，
不能用它直接连接 Host 与 Device。FP16 的 Head、W8 的 ArgMax 确实可能跨 Host Sync
结束位置，但完整落在对应的设备 Step Trace 内。此观察不代表可以平移任意时间戳。

`linear_attribution.py` 现在使用以下可审计规则（仍完全离线）：

- 继续核对 38 个同线程 Execute/Sync，计时索引保持 7..36。
- Step Trace 必须同时满足事件名/Iteration ID、cat、线程名/Model ID。共 38 个唯一、
  连续、互不重叠的设备迭代，每个起点必须在**唯一一个** Execute 开始至 Sync 返回
  的范围内，每个 Host 调用必须恰有一个 Step。由起点实际位置建立对应，不使用
  `Iteration ID - 1` 或任务列表下标强行分配。
- `Iteration End` 与 ts+dur 按其原始时间戳的小数位核对序列化一致性；任务包含检查
  不增加时间容差，不裁剪区间，不平移时钟。范围标记不计入 Kernel occupancy。
- Kernel 必须有 Model/Task/物理 Stream 身份，且其完整区间只落在同 Model 的一个
  Step 内。一轮内重复的 Task/Stream/Batch/Subtask 身份拒绝归属。每个原始事件有
  `event_index`，最多一个 owner；边界列表可以引用同一事件，但统计只计算 owner 一次。
- 缺少或冲突的 Step/任务单独保存到 `assignment_evidence`，状态为
  `UNRESOLVED_DEVICE_ASSIGNMENT`。旧式 Host 包含统计仅为 provisional，不能升级为
  完整设备归因。没有任何按耗时、出现顺序或数量凑齐 30 轮的补救路径。

Shape 解析接受 CSV 中额外 JSON 引号层，拒绝部分解析、未知维度和非法结构。
CSV/trace 同时保留 shape、format、dtype。FP16 物理 NZ Shape **不能单独证明逻辑维度**，
尤其不能把 padded M16 当成 Head M15。仅精确同名同类型的哈希绑定冻结图 TensorDesc
可提供逻辑维度；CSV 明确为 FP16/FRACTAL_NZ 时，还核对标准
`[ceil(N/16),ceil(M/16),16,16]` 物理 extent。缺少精确节点或布局证据时仍为 UNKNOWN，
不删除 `_ascend_mbatch_batch_0` 后缀猜节点。ND descriptor 与运行时冲突也会保留 UNKNOWN。

报告逐轮验证不同 checkpoint 的清单：FP16 6 Gate/Up + 6 Down + 19 其他 Linear；
W8 5 Gate/Up + 5 Down + 16 其他 Linear。要求 30 轮节点集合稳定，每轮完整 Head
MatMul 和 ArgMax 各一个。输出 `head_components`、`linear_node_inventory`、完整 Kernel
区间、region union、设备 occupancy/gap、跨区域重叠及 UNKNOWN；区域 union 不可相加
作为 Host wall time。`WINDOWS_VERIFIED` 仅出现在 Host 字段，归因另有
`UNRESOLVED_DEVICE_ASSIGNMENT` / `INCOMPLETE_KERNEL_CLASSIFICATION` / `ATTRIBUTION_VERIFIED`。
最后一个状态只是离线证据解析完整，**不是 FP16 数值或端到端 PASS**。

复测仍使用上面的单条 `linear_attribution.py --report ... --output-dir ...` 命令，
必须创建新目录。原报告、profiler、冻结图 SHA256 前后核对；不执行 msprof、Runner、
ATC 或 NPU。完整服务器离线重分析尚未执行；真实设备性能与 FP16 FAIL_REPLAY_CHECKS
不变。回传新目录的 `linear-timeline.json` 和 `linear-analysis.md` 即可定位剩余 UNKNOWN。
