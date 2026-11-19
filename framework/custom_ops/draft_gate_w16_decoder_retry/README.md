# W16 静态准入的离线解码复核

适用于 C16 Layer0 的 `W16_ATC_PASS` 与随后因解码失败产生的 `BLOCKED_STATIC_ADMISSION`。此入口与原 W16 源码清单分开。可以在现有 OM 上试验另一份有明确来源的 SDK decoder，复用真实的 Source、Build、ATC 证据。工具仅在新目录写报告，所有设备阶段均由原 W16 Driver 控制。

## EqxSwk 的实际阻塞

以下是服务器终端回传。原 ATC 使用同一份 CANN 9.0.0 毕昇 llvm-objdump，SHA256 为 `2b15d0cbe2eebdf8d230b76550c6d713e8c4855acf9cfb9f2a6089b113881b3c`，32304 B 的两个对象都来自各自 OM 中绑定的图节点。

| 变体 | 完整 ATC ELF SHA256 | .text 字节 | 默认 / 指定 dav-m200 |
|---|---|---:|---|
| double-w-serial | `7de3096e7697b9056366321c2585d3e77079651957a9f9f44611c4d4a7e0f323` | 25964 | 均 rc=0，6491/6491 个地址行只有占位符 |
| double-w-overlap | `195d076c225e54db57d50dda7a550df65a5c6fcf3842b4f256ecaa0b1cf90eb2` | 25964 | 均 rc=0，6491/6491 个地址行只有占位符 |

两个 .text 有 511 个字节不同。回传的 compiler-cache 中没有匹配的 .o，也没有留存 .s/.asm/.lst。当前数据能绑定代码身份，尚无法恢复操作数、定位真实 W0/W1 读写或证明 F(k) 与 D(k+1) 的次序。W16 Smoke、Benchmark 均为 NOT_RUN。

**本地 CANN 9.0.0 ARM64 检查。** 同 SHA 的 SDK 工具对真实 W16 包内 ELF 的常规解码也只输出地址。帮助中还有 `--disassemble-aicore`。对独立 hiipu ELF 单独指定该开关只输出文件头，与 `--disassemble --mcpu=dav-m200` 合用仍返回占位符。对一个由毕昇正常编译的微型 AArch64 混合对象，`--show-aicore-bins` 能找到内嵌设备对象，`--disassemble-aicore` 给出 29 个设备地址占位符；Host 指令的解码不能用于设备证明。对同一个能以 `-c` 正常编译的设备样例，`-S` 返回 `unsupported option '-S' on device side`，`-save-temps -c` 留下预处理 .ccei 和对象。这个实验的目录为 `/work/build/w16-driver-check-734tile4` 和 `/work/build/w16-offload-cap-1zwcqfzu`。它们只检验工具能力。厂商较早的 8.1.RC1 指南第 5 章也说明了该工具在算子错误定位时提供符号与地址偏移。本次 9.0.0 判定以实际工具与 ELF 输出为依据。[毕昇编译器用户指南](https://www.hiascend.com/doc_center/source/zh/canncommercial/81RC1/developmentguide/opdevg/BishengCompiler/CANN%E5%95%86%E7%94%A8%E7%89%88%208.1.RC1%20%E6%AF%95%E6%98%87%E7%BC%96%E8%AF%91%E5%99%A8%E7%94%A8%E6%88%B7%E6%8C%87%E5%8D%97%2001.pdf)

## 服务器使用

```bash
cd /home/w00949577/z50058744/qwen3.5-4B-dflash
git pull --ff-only origin feature/ascend310p-custom-ops
BASE=/home/w00949577/qwen35-dflash-run
OLD="$BASE/gate-w16-atc8-EqxSwk"
PY="$BASE/venv-hf513-probe/bin/python"
RETRY="$PWD/framework/custom_ops/draft_gate_w16_decoder_retry/retry_decoder.py"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PY" -B "$RETRY" --root "$OLD"
```

默认在 OLD 同级创建唯一的 `gate-w16-decoder-*/`。可用 `--output-parent` 选择另一个独立目录。每次先按原验证器重验三份成功前驱、源码/SDK/私有 OPP/OM 和失败 static 的身份，再从 OM 的指定 offset 核对选中 ELF 的全部字节及节表。逐项核验已封存的 argv、两次尝试的顺序、返回码、日志 SHA 和结果。对同一工具直接复核既有 canonical 日志，仅对帮助中实际存在的 AI Core 选项做补充探测。在新目录中对只读的字节相同副本调用工具。输出同时列两版结果、首个占位 PC、真实地址行数、经逐字节解码的覆盖量及两段 .text 的差异。

- `BLOCKED_STATIC_ADMISSION / SDK_OFFSETS_ONLY`：本次没有可用的设备指令文本。退出 1 是有效的阻断结论。EqxSwk 的原 Source、Build、ATC 和失败报告全部保留。此状态下重做同一 ATC 或填写证明模板无法增加设备指令证据。
- `DECODER_EVIDENCE_INCOMPLETE` 或 `REPLAY_VALIDATION_FAILED`：看报告中的原始命令、退出码或具体身份错误。不能用十六进制 word、CPU 模型、动态单路径、Host 反汇编或通用 OPP .o 补齐。
- `NEEDS_REVIEW_OF_ACTUAL_DECODER_INVOCATION`：补充选项产生了完整的字节匹配列表，但真实命令不在原 machine-proof 合同内。保留该命令及结果再审查。不会将它重写成其它 argv。
- `W16_DECODER_COVERAGE_VERIFIED_PROOF_PENDING`：两个新 OM 的每个可执行字节均已由受支持的设备解码调用核验。还需独立的 per-ELF PC、CFG、UB 操作数、事件和失败路径证明。

若确实取得原 CANN 身份边界内的另一份厂商 dav-m200 decoder，先从可信分发记录取得其预期 SHA256。使用绝对路径和固定摘要：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PY" -B "$RETRY" --root "$OLD" \
  --decoder "$SDK_DECODER" --decoder-sha256 "$EXPECTED_VENDOR_SHA256" --prepare-flow
```

仅在两份候选都满足原 decoder 的命令合同、目标和逐字节完整覆盖后，`--prepare-flow` 才在新目录的 flow/ 为三个已通过阶段生成**带原报告身份的离线复核记录**。重新调用原版 Source/Build/ATC 验证程序，源清单相同，原模型、真实 Tiling、Kernel、API、OPP、Runner、输入与 Golden 继续引用冻结身份。Source 只更新补充选择的 decoder，ATC 保留原始编译模型并绑定新的解码证据。每份记录的 replay 字段明确原始阶段报告、新工具代码和 `new_build=false/new_atc=false/new_npu=false`。本程序不生成 static 成功记录。若有完整解码，使用本次生成的逐 ELF 模板进行实际机器审查，再对打印的 REVALIDATED FLOW 手动调用原入口：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PY" -B framework/custom_ops/draft_gate_w16_atc8/run_w16.py \
  static --root "$REVALIDATED_FLOW" --proof-index "$INDEX" --proof-index-sha256 "$INDEX_SHA256"
```

若当前只有输出占位符的 SDK，可直接进入原先设计的独立 Down 二维搬运实验。开始时只运行 Source：

```bash
D=$(mktemp -d "$BASE/down-raw-dma-XXXXXX")
echo "D=$D"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PY" -B framework/custom_ops/draft_gate_w16_atc8/down_fallback.py source --root "$D"
```

## CPU 测试

```bash
python3 -B -m unittest discover -s framework/custom_ops/draft_gate_w16_decoder_retry/test -v
# 在本地 CANN 容器中也可指定历史 package 编译 fixture，实际运行同一 decoder
W16_LOCAL_BUILD_REPORT=/work/build/gate-w16-px0mgc_g/report.json \
  python3 -B -m unittest discover -s framework/custom_ops/draft_gate_w16_decoder_retry/test -v
```

测试覆盖全占位、混合/缺失 PC、Host 格式、错误退出码、编码不匹配、OM/ELF/日志篡改、工具别名/越界/哈希、两版不完整时阻断，以及在构造测试数据上的只读前驱续接。无设备测试。

本地回归结果为 macOS arm64 10 项通过、1 项实际 CANN 包测试跳过，Linux aarch64 容器 11/11 通过（含真实包内 ELF）。原 W16 套件分别为 28 项通过、1 项跳过和 29/29 通过。可解码替代工具的续接路径在 CPU 夹具上验证；当前 310P3 没有取得新的机器指令或静态 PASS。
