# EqxSwk 离线指令审计

目标是判断实际调度的 `double-w-serial` 和 `double-w-overlap` 是否有可逐字节验证的 dav_m200 指令证据。两个输入来自实验 `gate-w16-atc8-EqxSwk` 的 OM。原实验的 ATC 已通过，static 曾阻断。审计仅在旧实验的同级新目录写报告、工具日志和只读 ELF 副本。

## 服务器命令

从仓库根目录单独执行以下阶段。无需 Source、Build 或 ATC。

```bash
BASE=/home/w00949577/qwen35-dflash-run
PY="$BASE/venv-hf513-probe/bin/python"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PY" -B \
  framework/custom_ops/draft_gate_w16_isa_audit/audit_isa.py \
  --root "$BASE/gate-w16-atc8-EqxSwk"
```

终端最后两行给出 `w16-isa-audit-*/static-audit.json` 和 `static-audit.md` 的实际绝对路径。退出码 1 表示阻断。前驱或文件身份检查出错时看 `errors`；独立 ELF 没有一套合格解码时看 `tool_trials` 和 `coverage`。可通过 `--extra-tool-root /absolute/tool/installation` 增加实际安装位置。仅清点的用户 PATH 文件和专有 ABI 符号不获得解码资格。

## 如何复核结果

1. 复用原验证器重验 Source、Build、ATC 和旧失败报告。分别从两份模型的图节点取得 ELF 长度、偏移、内核符号和 SHA256，将封存提取文件与该 OM 字节区间逐字节比较。
2. 独立标尺是另一个算子 `DFlashDraftLmHeadTop1`。它在 Scheduler Flow A、Gate Raw OPP 和两个 W16 OPP 中的 SHA256 必须为 `47ffde8366f14aa97267a8d52b3d55fc7fec7684126c578831152ade2d5c343b`，长度 18,632 B。校验安装路径、原包内 JSON 和 ELF 头，保存输入文件身份与可执行段。其冻结源码含从 GM 拷入 UB、Cube Matmul 和全流水 barrier，分别作为带操作数的独立指令见证。一个仅支持部分指令或不同语法的工具会保留原始列表供复查。
3. 清点所选 CANN、编译器内部目录、系统工具、额外工具根及项目中的 ELF/ISA 读取代码。记录程序和关联源文件的绝对路径、别名和 SHA256，保存 LLVM/GNU 的目标、版本、选项、实测命令、返回码和原始输出。只对可信安装的文件检查接口发出调用；调试器仅使用文件模式。对 simulator 动态库仅查看导出符号。msopgen 的列表解析器调用外部 objdump。华为对 msobjdump 给出的接口用于检查或提取容器中的内核。 [CANN 9.0 msobjdump 文档](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_0103.html)
4. 同一工具与参数先检查独立 ELF，再检查候选。另对封存 SDK objdump 保留两份候选的负向实测。覆盖须能把真实 PC、编码、助记符及适用的操作数对应到 ELF 的原始字节。`.text` 部分覆盖以单次调用中去重的实际字节数计；通过独立标尺后的覆盖单列。完整解码核对所有非空可执行段、顺序、边界和编码。
5. 有完整机器证据时，原 `admit_static` 还要对两个不同 ELF 检查控制流、重定位、UB 实参、失败分支、原 fence、K 顺序及 FP16 写出。机器评审材料可使用 `--proof-index` 和 `--proof-index-sha256` 指定。原静态验证器认可的命令格式、来源和准入条件保持有效。审计保留其它命令的真实输出，出现新的有效方言时应据实审查。

`static-audit.md` 并列给出两版每项指令结论。缺少 PC 证据的项目标记 `UNPROVEN`。详细 JSON 另附原始 OM、工具/清单、安装范围及待调查接口。两份 ELF 相差 511 个执行字节仅能说明编译产物有差异。

## 已实测的独立本地标尺

ARM64 Linux CANN 9.0 的只读容器有一份既存 dav-m200 `probe.o`，SHA256 为 `f96a0c6ef3cfa824ff1095da08936ae4105317304d66ac5e36d2aa23e22a0d00`，文件长 936 B，执行段 116 B。其独立源码只有 `dest[0] = x + 17u`。版本 15.0.5 的 CANN LLVM objdump，SHA256 为 `2b15d0cbe2eebdf8d230b76550c6d713e8c4855acf9cfb9f2a6089b113881b3c`，列有 hiipu64 与 dav-m200。对该真实对象，常规、显式 CPU、架构名、完整 triple 和组合 AI Core 选项均得到 29 个占位 PC、0 个已核验编码。单独 AI Core 选项没有 PC 行。GNU AArch64 objdump 返回 1。该次保存的本地报告 SHA256 为 `962a3c32407771e697b462f24c1ca0dfab3e9efae451c80fb9d43da6480f4331`。

```bash
python3 -B -m unittest discover -s framework/custom_ops/draft_gate_w16_isa_audit/test -v
# 已有独立 ELF 的本地 Linux ARM64 CANN 容器
DAV_M200_KNOWN_DIR=/work/build/w16-driver-check-734tile4 \
  python3 -B -m unittest discover -s framework/custom_ops/draft_gate_w16_isa_audit/test -v
```

若实际找到可用的正式 SDK 解码器，按同仓库 `draft_gate_w16_decoder_retry/README.md` 的 `--prepare-flow` 命令建立离线续接，并用各自真实的机器证明运行原静态门禁。仅当续接工作流的正式报告已是 `W16_STATIC_PASS`，下一次独立设备实验才是 Layer0 Smoke：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PY" -B \
  framework/custom_ops/draft_gate_w16_atc8/run_w16.py smoke --root "$REVALIDATED_FLOW" --allow-device
```

本审计没有设备启动入口。独立的 Down 2D DMA 已完成 Smoke 和计时，三轮改善、比值 1.016006×，决定为 `DOWN_NO_GO`。
