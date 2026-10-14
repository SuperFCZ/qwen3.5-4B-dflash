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
