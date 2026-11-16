# A-only Raw：历史 Python 环境复原与 import-only Export 门控

只新增本辅助目录。`draft_a_raw/`、历史 A/C helper、Native Base、SOURCE_LOCK、Kernel、
OPP、Golden、旧报告均未修改。不安装/升级包，不添加 initialization.py，不准备替代接口。

## 本次环境根因

用户回传的历史 H 命令使用：

`/home/w00949577/qwen35-dflash-run/venv-hf513-probe/bin/python`

H/preflight-imports.json 中 Transformers 来自该 venv；Torch/Torch-NPU 来自
`/root/anaconda3/envs/ls_llm/lib/python3.11/site-packages`。
H 的 TorchAir 来自 `torch_npu/dynamo/torchair/__init__.py`，历史 SHA 为
`385cc1c640b08d1dde135d9e32a28a3a6d803f2980586ffe8497e6e3139fe585`。
这是历史已使用的组合，不能仅因 Torch 位于 base Conda 而判断混包。

失败时实际运行的是 `/root/anaconda3/envs/ls_llm/bin/python3`，Transformers distribution、
模块版本都是4.57.6，模块文件也来自同一 Conda site-packages；initialization spec 为 None。
这组回传没有显示项目中的 transformers.py 抢占，优先原因是选错解释器/环境。
完整 sys.path 和新的运行时来源由本检查补充记录，不凭版本字符串排除所有 shadowing。

历史 `draft_ac/README.md:380` 明确写有 Transformers5.13.1 隔离 venv。辅助程序不根据
venv 名字猜版本，而是从 H 的已绑定 preflight-imports 中取 Transformers init 的历史
SHA，复核原文件，再静态读取其 `__version__`。原 venv 丢失或文件漂移则 BLOCKED。
不自动找一个“相近版本”、不把5.14.1自动装进旧环境。

`transformers.initialization` 是 Hugging Face 官方模块，例如
[官方v5.13.1源码](https://github.com/huggingface/transformers/blob/v5.13.1/src/transformers/initialization.py)
定义了本项目使用的 `copy_`。所查
[v4.57.6包入口](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/__init__.py)
及服务器实际安装均不提供该模块。这里引用官方源码只确认模块归属，**不作为安装指令**。

Native Base 中直接导入者（用户服务器 grep 与本地319f51d源码相符）：

| 文件 | 行 | 导入 |
|---|---:|---|
| models/modeling_qwen3_5_hiai_nd.py |22| initialization.copy_ |
| models/modeling_qwen3_5_hiai_nd_dflash_rollback.py |35| initialization.copy_ |
| models/dflash_v1/modeling_qwen3_5_dflash.py |33| initialization as init |

本次附件没有完整的原始异常内层栈，不能断言三者中哪一个首先触发异常。
源码调用次序也不能简化为“prepare_custom_op_export内部缺模块”：

```text
worker.main
  require_draft_interfaces
    native_history / require_source_lock
    native_base.check_extension_compatibility
      导入已锁定 framework + 6个诊断模块
      绑定 receiver → 导入 models.export_model_wrapper_qwen3_5
    import torch / torch_npu / torchair
    prepare_draft_interfaces
      draft_operators → 三个声明
      prepare_custom_op_export（分别准备三个接口）
```

Native/receiver模型依赖可以在 check_extension_compatibility 中、准备三个接口之前触发。
`draft_operators` 本身只声明算子；prepare_custom_op_export负责 dispatcher/schema/Meta/GE，
没有直接导入 transformers.initialization。新检查保存实际 traceback 与模块源身份。

仓库另有 embedded CLI 的5.14.1断言，属于不同入口；不修改它，也不用它覆盖 H 已验证
的 Draft-only OM 导出环境。`pip show torchair` 找不到 distribution 不能证明模块缺失：
H 使用的是 Torch-NPU 随包路径。`pip check` 的 schedule-search/absl-py 缺失单独保留，
本轮不将其推断为 initialization 错误，也不预先安装额外依赖。

## 门控和复用

`check` 在新目录写报告，B/R分别使用原已验收私有 OPP 的环境启动新 Python 进程：

- 调用原 `prior()` / `verify_build()` / Projection验证器，不修改或豁免 Source Hash。
- 要求原 Precheck、Build、104项Probes准确PASS、完整前序链接、源码/SDK/文件/输入/
  Golden/OPP/ELF/Runner身份及原Source Lock仍有效；只重核证据，不重新执行Probes。
- 核对 distribution实际模块归属、版本、package搜索路径、历史模块路径及SHA；拒绝
  项目同名文件 shadowing、错误版本、混合 package path。
- initialization必须来自所选Transformers包、提供真实copy_入口，并符合安装的SHA256
  RECORD；没有安装记录或内容被改不能靠空兼容桩通过。
- Torch、Torch-NPU、TorchAir入口身份与H一致。记录解释器调用别名、真实二进制、
  sys.prefix/base_prefix、pyvenv.cfg、sys.path和相关环境。
- 使用原Native import boundary和完整Source Lock校验；调用原只读
  check_extension_compatibility，确保其传递依赖可导入；取得三个Draft算子声明。
- **不调用** prepare_draft_interfaces、prepare_custom_op_export、model constructor、
  model forward、acl/ACLNN、ATC或设备执行；状态不冒充正式Interface PASS。
- 保存所有实际已加载模块文件身份。B/R依赖快照必须相同。

成功状态：`A_RAW_EXPORT_ENVIRONMENT_IMPORTABLE`；子报告为
`DRAFT_IMPORTS_READY_INTERFACES_NOT_PREPARED`。原接口验收仍由后续 worker 完整执行。

新目录位于原 `draft_a_raw.sources()` 清单之外，所以此次Git提交本身不使前序源身份失效。
复用依据是**旧清单逐文件完全一致且原验证器通过**，不是“忽略源变更”。若旧目录源码、
OPP、输入或任一前序报告已变，仍BLOCKED，不能修改旧报告使其适配。
本辅助工具另有独立源码清单；辅助代码或包文件在check之后变化会阻断resume。

## 第一组服务器命令：原解释器 + import-only检查

当前失败目录保持不变。拉取提交后，在原CANN shell内使用明确解释器，无需pip或activate：

```bash
cd /home/w00949577/z50058744/qwen3.5-4B-dflash
git pull --ff-only origin feature/ascend310p-custom-ops
PY=/home/w00949577/qwen35-dflash-run/venv-hf513-probe/bin/python
OLD=/home/w00949577/qwen35-dflash-run/a-only-raw-full-fixed-tl2rME
NEW=$(mktemp -d /home/w00949577/qwen35-dflash-run/a-only-raw-env-XXXXXX)
E=framework/custom_ops/draft_a_raw_env
printf 'NEW=%s\n' "$NEW"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PY" -B "$E/run_environment.py" check \
  --source-root "$OLD" --output-dir "$NEW/check"
```

不要对 `$PY` 执行readlink后改用其真实base二进制；两个路径可能同ELF，但只有venv调用
别名会选择原sys.prefix和site-packages。不要通过PYTHONPATH拼装Transformers路径。
若该解释器不存在，停止；本工具不会安装或创建环境。

回传终端的 status/error/variants 即可。失败补充 `$NEW/check/B/traceback.txt`（或R）及
对应 imports.log；不要修改该报告或重用同一个输出目录。

## 环境检查通过后，显式恢复 Export

下面是第二阶段命令，不由check自动执行：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PY" -B "$E/run_environment.py" export \
  --source-root "$OLD" --environment-report "$NEW/check/report.json" \
  --output-dir "$NEW/resume"
```

先重新执行import-only门控并逐项对照已冻结环境，再调用**原run_experiment.py export**。
该原流程会再次执行原require_draft_interfaces和全部图/资源验收，不绕过任何门槛。
新完整Export报告位于 `$NEW/resume/export/report.json`；旧报告没有复制、改写或覆盖。
成功需同时满足：

- 原Export：`A_RAW_C16_C64_OMS_COMPILED`。
- 辅助总报告：`A_RAW_EXPORT_RESUMED_WITH_IMPORT_GATE`。

Export阶段会进行原有设备Tensor准备/设备查询和ATC编译，仍不执行完整Draft。
这不同于前面的import-only检查。原Build与104 Probes不重新运行；若身份复核失败则停止。
后续Smoke/Benchmark保持NOT_RUN；本轮不自动启动它们。由于新的Export报告在新目录，
后续须显式传入原Precheck/Build/Probes与新的Export报告，不能套用要求全部阶段同根的旧shell入口。

当前本地只能验证CPU合同；真实历史venv import和完整Export仍由服务器验收。

## 进程临时模块身份（后续修复）

已确认PyTorch的 `_remote_module_non_scriptable.py` 在子进程退出时正常删除，原先把它
作为持久路径复核会误报。仅该已确认模块使用“存活时来源/内容核验 + 持久快照”，普通
模块身份不放宽。已有Admit可经固定Git源码校验与原资产严格复核做只读续接。
详见 [MODULE_LIFETIME.md](MODULE_LIFETIME.md) 的根因、限制和只运行Import-only的命令。
