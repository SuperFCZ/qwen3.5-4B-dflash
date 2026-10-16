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
bash framework/custom_ops/draft_ac/run_known_evidence.sh --draft-only --preflight-only

# 上一步通过后，用另一个新目录执行设备验收：
export AC_OUTPUT_DIR=/home/w00949577/qwen35-dflash-run/ac-draft-only-$(date +%Y%m%d-%H%M%S)
bash framework/custom_ops/draft_ac/run_known_evidence.sh --draft-only
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

沿用上面的 `--draft-only --preflight-only` 命令即可先轻量复测；只有源码及历史身份
预检通过，才在另一个新目录执行 `--draft-only`。请额外回传 `source-lock-preflight.json`
及 `source-lock-device.json`。若冻结 Native 的锁身份不同，先根据报告继续调查，不覆盖
Native AIR、Deployment、SOURCE_LOCK 或 Golden。

本轮本地 CPU 回归：463 PASS，88 项因未配置 C++ fake ACL runner 或依赖该 runner 的
host fixture 跳过；macOS 的 Linux fixture `/bin/true` 使用本地 `/usr/bin/true` 适配，
未修改锁定测试。新隔离 pipeline 的 GE/ATC/NPU、完整 Draft 与端到端：`NOT_RUN`。
