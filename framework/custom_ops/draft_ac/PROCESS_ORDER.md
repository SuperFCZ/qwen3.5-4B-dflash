# A/C Preflight 与联合 OPP 初始化顺序

## 已知证据与导入链

真实服务器 `cb0bce2` evidence：
`/home/w00949577/qwen35-dflash-run/ac-ge-20261014-071350`。
用户已确认只读 Preflight PASS（20 项 Evidence），但随后协调进程发现 `torch_npu` 已加载；联合 OPP、GE/ATC、NPU 均 NOT_RUN。旧日志没有首次导入栈，不能从该错误断言是哪一条第三方导入先触发。

仓库源码存在一个比 `validate_offline_nz()` 更早的触发入口：

```text
resolve(): from qwen35_dflash.ascend310p.draft_gears import ...
  → ascend310p/__init__.py
  → workflow.py: from transformers import AutoTokenizer
  → 第三方依赖的设备可用性探测
  → 显式 import torch_npu
```

本地 CPU 导入复现使用真实 Torch 2.10.0、Transformers 5.19.0、Accelerate 1.13.0，提供一个仅含 `CPU_ONLY_IMPORT_FIXTURE=True` 的 `torch_npu.py`，不含任何设备实现。即使设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`，记录到的真实 Python 导入链仍继续经过：

```text
AutoTokenizer → tokenization_auto → EncoderDecoderConfig / auto_docstring
  → modeling_auto → auto_factory → generation.utils
  → accelerate.hooks → accelerate.__init__ → accelerator
  → accelerate.utils.modeling → accelerate.state
  → is_npu_available(check_device=False)
  → accelerate.utils.imports: import torch_npu
```

观察结果是 `torch_npu` 进入 `sys.modules`。这证明关闭 Torch backend entry-point 自动加载不能阻止第三方的显式导入；该 CPU stub 复现不代表服务器具体版本的完整导入链，也不代表 NPU 执行。

新 `preflight-imports.json` 在服务器记录首次模块查询、首次实际导入的文件/行号/调用栈、导入前后模块清单、模块来源及相关 Python 源码 SHA256。区分 `find_spec()` 可用性查询与真正的 import；若在观测开始前已加载，也单独记录，不虚构首次调用栈。

## 修复后的进程边界

1. **协调进程 P**：仅导入标准库和已有轻量路径/构建辅助代码，保留初始化顺序 guard。写入 `preflight-input.json`，启动独立 Python 子进程。
2. **Preflight 子进程 Q**：执行原 `resolve()`，保留所有 SHA256、Manifest、NZ 实际反解、Capture、OPP、Native Replay/Golden、冻结源码检查。允许其 CPU 验证依赖间接加载 `torch_npu`，记录原因后退出；该进程永远不构建联合 OPP、不进入设备验收。
3. **协调进程 P**：核对子进程成功退出、请求哈希、结果/审计文件哈希、不同 PID，以及全部 evidence 文件哈希；再次确认自身没有 `torch_npu`。用原逻辑生成、构建、隔离安装联合 OPP，复验安装产物及冻结配置。
4. **设备解释器**：设置联合包优先的 `ASCEND_CUSTOM_OPP_PATH` 和 `LD_LIBRARY_PATH` 后，沿用 `os.execve` 替换解释器。exec 可以保留 P 的 PID，但不会保留原 Python 模块或旧动态库映像。正式导入前要求模块状态干净、当前 run 的联合 OPP 构建完成、报告和安装产物匹配、两条 loader 路径正确；任何失败都不能进入图测试。

不删除安全检查、不从 `sys.modules` 移除设备模块、不复用已加载的旧 A/C 动态库。`TORCH_DEVICE_BACKEND_AUTOLOAD=0` 仍保留作为辅助控制，不再被当作完整的隔离机制。

## 回归与服务器入口

CPU 回归使用真正的独立 Python 子进程模拟 `Preflight → offline_weights → torch_npu` 间接导入，并验证协调进程仍干净；另启干净解释器检验设备启动 gate。设备库/OPP 为明确的 CPU 测试替身，不能产生硬件 PASS。另覆盖 OPP 缺失、构建未完成、产物变化、loader 环境错误和提前加载设备模块的拒绝路径。

在原 CANN9/模型/ADN/GDR 环境中运行，使用新目录：

```bash
git pull --ff-only
export AC_OUTPUT_DIR=/home/w00949577/qwen35-dflash-run/ac-ge-$(date +%Y%m%d-%H%M%S)
MODEL_PYTHON=python bash framework/custom_ops/draft_ac/run_known_evidence.sh
```

仍可附加 `--preflight-only` 仅做只读检查。无需重新 Capture、导出 Native OM 或覆盖旧 OPP。

请回传 `suite.json`、`preflight.json`、`preflight.log`、`preflight-result.json`、`preflight-imports.json`、`coordinator-process.json`，以及目录旁的 `.server.log`。构建或设备阶段有产物时，补 `joint/opp.json`、`device-process.json`、失败图的 ATC/runner 日志。新版本联合 OPP、GE/ATC、单算子图和整网设备验证在执行前继续为 **NOT_RUN**。

本轮本地结果：A/C CPU 回归 106 PASS；连同 prepack/layout 回归合计 212 PASS、84 SKIP（既有测试需要未构建的 fake ACL runner）。Bash 语法检查、无 Torch/NPU/TorchAir/Transformers 的协调入口导入检查通过。以上均不构成真实 CANN/NPU 验收。
