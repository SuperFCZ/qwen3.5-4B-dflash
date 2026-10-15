# Full Draft GDR 前端环境阻塞

服务器 `/home/w00949577/qwen35-dflash-run/ac-full-draft-20261015-030613` 的 52+4 A/C 单算子图已通过（用户提供）。随后公共 factory 的 `prepare_export` 拒绝缺少 `effective_length` 的 9 参数 PyTorch GDR schema。该失败尚未进入完整 Draft 的数值验收，不能解释成 A/C Kernel 回退。

仓库 `_prepare_quant_export` 在加载模型前检查 Target 自定义前端。`_validate_schema` 锁定 10 个参数，第六个为 `Tensor effective_length`；HIAI Target 显式传入该参数，含义是 INT16[B] 本次有效行数。它关系到 padding 和递归状态更新，不能删除或默认忽略。公共工厂包含 Target/Verify 准备，虽然本轮最终只导出 Draft，仍必须满足该契约。

应分别确认两层：

- PyTorch dispatcher：注册完整 effective-length-v2 schema 的 torch_npu 或已有定制扩展。`ASCEND_CUSTOM_OPP_PATH` 指向正确 GE 包不等于已经获得正确的 Python/C++ dispatcher 注册。
- GE/OPP：ChunkGatedDeltaRule 的有效原型和实现，不能混入多个或旧版本。GE 输入顺序是 query/key/value/g/beta、可选 initial_state、effective_length，和 PyTorch 位置顺序不同。ADN 原型与 310P kernel 也必须匹配。

历史 AIR 保存 `environment`、`operator_preflight.operators[].torch_schema`、`ge_prototypes`，图 audit 还保留 schema。它们可以证明导出时的契约及版本字符串，但旧报告不一定包含 torch_npu wheel 哈希或实际模块路径，不能凭版本号推测安装哪个私有构建。已编译 Native OM 的执行并不重新验证 PyTorch 前端，单算子 PASS 也不能替代该环境检查。

## 只读诊断

在原任务使用的模型 Python 环境和仓库目录执行：

```bash
MODEL_PYTHON=python
"$MODEL_PYTHON" -B framework/custom_ops/draft_ac/test/full_interface.py \
  --evidence /home/w00949577/qwen35-dflash-run/ac-full-draft-20261015-030613 \
  --report /tmp/ac-gdr-interface-$(date +%Y%m%d-%H%M%S).json
```

脚本从该 evidence 的 `device-process.json` 恢复实际 worker 的两条 OPP/library 环境路径，并 exec 新 Python 后才按正常顺序 import torch、torch_npu、TorchAir。仍需使用当时的 Python/conda 环境；脚本不会自动切换或安装包。

报告包含 Native Deployment 哈希绑定的历史 AIR 环境/契约、当前 Python executable、torch/torch_npu 的版本/路径/哈希、GDR schema 每个参数、dispatcher registration dump（SDK 支持时，可提供注册源码位置）、Python wrapper、已加载扩展/相关 `.so`、ADN/GDR 原型及 SHA256。没有调用 GDR、weight-quant probe、手动 Fake/Converter 注册或 schema 替换、模型加载、ATC 或 NPU 推理。仅写新的诊断 JSON；不修改环境包、OPP、checkpoint 或 golden。BLOCKED 返回非零是预期的诊断结果。

如果历史 AIR 也记录九参数或未记录 schema，尚不能证明原环境存在完整 v2 前端，应报告历史基线与当前源码契约的证据缺口，不能反向放宽锁定规则。

恢复策略由报告决定：若 Python 路径/版本改变，先恢复历史导出环境；若版本字符串一致但 schema 不同，核对定制 wheel/扩展来源及原激活步骤；若 dispatcher 正确而 GE 原型旧或重复，恢复唯一匹配 v2 的 ADN/GDR vendor。不要手写十参数空壳、用 Meta 声明覆盖真实九参数接口，或盲目 load 未核实的库。仓库中没有足够证据定位历史私有安装包时，应报告缺失的包/注册来源，保持阻塞。

## 提前门控与回归

`--full-draft` 在隔离 Evidence Preflight 子进程内写 `full-interface-preflight.json`，不匹配即停止，不重建联合 OPP、不执行 56 图。联合 OPP 启动后还会在 runner/单算子图执行之前写 `full-interface-device.json` 再检查一次，防止加载路径改变环境。单算子-only 模式保持独立。

```bash
export AC_OUTPUT_DIR=/home/w00949577/qwen35-dflash-run/ac-full-interface-$(date +%Y%m%d-%H%M%S)
MODEL_PYTHON=python bash framework/custom_ops/draft_ac/run_known_evidence.sh --full-draft --preflight-only
```

两个检查复用锁定的 schema 验证器及 ADN/GDR 原型检查，不用完整 `prepare_export` 充当轻量检查，因为该函数还会注册 Fake/Converter 并执行 WeightQuant probe。正式 full_draft 的原 `prepare_export` 保留；新门控不能替代真正导出和数值验收。

CPU 回归覆盖合法 10 参数、旧 9 参数、错序、错类型、keyword-only 和返回类型漂移、GE 原型缺失及检查失败时不交付构建阶段。没有真实服务器环境恢复证据之前，完整 Draft/ATC/NPU 验收为 **NOT_RUN**。

本轮相关 CPU 回归 208 PASS；未在服务器执行环境恢复或完整 Draft 验收。
