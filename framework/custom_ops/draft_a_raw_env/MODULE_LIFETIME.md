# Import-only 临时模块生命周期修复

本轮不运行ATC、NPU、Smoke或Benchmark，不修改Kernel/Tiling/OPP/AIR/OM/Golden。

## 精确定位

用户实验 `a-only-raw-atc-fixed-4ol1aJ`：Admit PASS；Resume在
`check_workers → validate_child → verify_tree(child)` 阻塞。
这说明 `load_stage()` 和 `bound_admission()` 已通过，ATC尚未开始。

失效记录是B子进程报告的 `loaded_files._remote_module_non_scriptable`：

- 原路径 `/tmp/tmpubau8xpb/_remote_module_non_scriptable.py`
- 2355 bytes
- SHA256 `8205b16956fb264841ecd8644784a0d157f87df79b17c16825dc1163433ce5d8`
- 子进程报告 `DRAFT_IMPORTS_READY_INTERFACES_NOT_PREPARED`
- 父进程随后检查时目录已不存在。

[PyTorch v2.8.0 instantiator.py](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/distributed/nn/jit/instantiator.py)
19–23行创建TemporaryDirectory、注册atexit cleanup并将目录加入sys.path；75–91和131–143行
在该目录生成并导入这个模块。该模块是进程内产物，不是安装目录中的持久Python文件。
官方源码用于解释生命周期；新工具还会绑定服务器**实际加载**的生成器和模板文件身份。

`check_workers` 自己使用的是持久输出目录，没有清理该临时目录。它等待子进程结束后再
验身份，恰好晚于Torch的正常退出清理。子进程末尾 `verify_loaded_files` 可以成功，父进程
复核却失败。不是当前Source Lock失效，也不是父进程删了B目录。

`split_atc.verify_resources` 的 `a-raw-host-verify-*` 临时目录尚未参与此流程；它返回的
资源证据指向原stdout，不把临时block路径传给下游。当前报错不是该处的with退出问题。
用户提供的python.sys_path是重依赖导入前快照，所以没有后来Torch追加的临时目录。

## 严格修复

只处理名称、长度、内容哈希都已确认的 `_remote_module_non_scriptable`，不豁免整个/tmp。

子进程退出前：

1. 检查Torch版本2.8.0+cpu、实际生成器/模板模块来自当前Torch安装根。
2. 检查生成器 `_TEMP_DIR` 真的是TemporaryDirectory，目录属性、当前sys.path、模块
   `__file__`、`__spec__.origin` 和记录身份一致；不替换或修改这些属性。
3. 检查文件为已确认的2355B/固定SHA；内容变化或不可读直接BLOCKED。
4. 将实际字节封存至自己的 `module-evidence/_remote_module_non_scriptable.snapshot`，
   复核原文件在复制前后未变化，再记录snapshot、生成器/模板身份、原始来源和PID。
5. 保留Torch原cleanup，不重建旧/tmp目录、不禁用atexit、不创建兼容模块或软链接。

父进程检查持久snapshot的完整SHA，以及生成器/模板的原路径和SHA；记录明确标为
`PYTORCH_CHILD_TEMPORARY_MODULE_V1`，不冒充原临时文件仍存在。普通loaded_files仍检查
原文件，不丢弃已记录模块；这个模块从持久列表移到受严格验证的process_modules。
未知进程模块不获得此例外，未完成快照、缺失、篡改和来源冲突继续BLOCKED。

B/R只对这一已验证进程模块比较内容SHA、大小、类型和生成器身份，不比较随机临时目录、
私有snapshot位置及PID。其他解释器、包、持久模块路径、Source Lock和import boundary
比较规则保持不变。CPU模拟不能替代真实服务器import验收。

## 保留已有Admit的正式源码续接

修复改变 `draft_a_raw_env` 的源码清单，不能把旧Admit.environment_sources直接当作新代码。
`renew_admission.py` 做一次只读续接，不改旧报告，也不重跑Build或104项Projection：

- 要求旧Admit准确PASS，ATC辅助目录的源码清单仍逐文件一致。
- 旧环境检查器清单必须完整匹配固定Git基线
  `f2688ad377787ca7c388751a89b179b935a4a21d` 中的所有环境目录文件（bytes/SHA逐项核对）。
- 将已核对的旧Git源码字节归档到新的证据目录，供后续校验；归档不是运行时代码路径。
- 原Precheck/Build/104 Probes、冻结Worker/AIR/OM/OPP等全部调用原严格验证器再核对。
- 只有这些条件通过，才写新的Admission，绑定新环境源码，保留旧Admit的SHA和完整续接记录。

没有Git对象、旧源码SHA不匹配、ATC源码发生未授权变化或任一资产漂移都BLOCKED。
这不是忽略Source Hash；旧版本按不可变Git内容验证，新版本按当前文件验证。
旧Admit PASS和旧Resume BLOCKED均原样保留。

## 服务器：只续接Admit并复测Import-only

```bash
cd /home/w00949577/z50058744/qwen3.5-4B-dflash
git pull --ff-only origin feature/ascend310p-custom-ops
PY=/home/w00949577/qwen35-dflash-run/venv-hf513-probe/bin/python
E=framework/custom_ops/draft_a_raw_env
AD=/home/w00949577/qwen35-dflash-run/a-only-raw-atc-fixed-4ol1aJ/admit/report.json
NEW=$(mktemp -d /home/w00949577/qwen35-dflash-run/a-only-raw-import-life-XXXXXX)
printf 'NEW=%s\n' "$NEW"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PY" -B "$E/renew_admission.py" \
  --admission-report "$AD" --output-dir "$NEW/admit"
```

预期 `A_RAW_ATC_RESUME_ADMITTED`；仍为只读资产/源码验证，ATC/NPU NOT_RUN。
之后仅运行import-only：

```bash
BASE=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_root"])' "$NEW/admit/report.json")
TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PY" -B "$E/run_environment.py" check \
  --source-root "$BASE" --output-dir "$NEW/import-only"
```

预期 `A_RAW_EXPORT_ENVIRONMENT_IMPORTABLE`。回传终端状态及process_modules中的snapshot
身份即可；若失败，提供该新目录report/traceback。不要使用已删除的原/tmp路径。
**先验收此结果，本轮不要执行resume/export。** 后续恢复时仍会再做相同import-only门控。
