# 审查实际 ATC 专用 ELF

每个 Gate 的 `<atc阶段>/<变体>/<层或档>/machine/<序号>/` 保留整份 selected.elf、sections.json、selected.disassembly。适用时另存 instructions.json 和未填写的 review-input.json。阶段报告按图节点名绑定这些文件、OM、图中符号、偏移、大小、完整 SHA256、工具及命令。单 W 使用历史 Raw OM 的实际字节。B/C 的 Build 包内通用 .o 只属于构建证据。

**解码能力和语义证明分开。** 从图的 _kernelname、_kernel_bin_id、_tbeKernelSize、tvm_blockdim 和 AIcoreEngine 绑定到唯一完整 ELF；验证每段范围、符号和摘要。用 Source 指定、位于所选 CANN 的 decoder 反汇编。每个可执行段必须由原始实际编码连续覆盖，未知指令、缺字节、截断、嵌错 OM、无法关联的 PC 均阻断。CANN 9 本地容器的 llvm-objdump 能读 hiipu 的节表，指令栏却为 `<not available>`。工具退出 0 也会在 **static** 阶段给出 BLOCKED_STATIC_ADMISSION。服务器的能力需看它自己的输出。一个 SDK 模拟器执行样本不能代替整条 CFG 的静态证明。

若两份新 OM 实际选择相同完整 ELF 且父 Tiling 已核对相同，报告 NO_TESTABLE_OVERLAP。如果机器路径中下一次 Dequant 之前已经把该次 Cube 完全等空，保留相应 PC 证据，也停止试跑。实际发射/受理后仍有多少可利用的时间需要设备测量。

## 完整解码后的证据格式

编译器可内联、改寄存器、重排基本块。语义审查对应**这个 ELF 的完整二进制**，没有基于形状名自动生成“安全”签字的路径。工具先输出绑定好的 review-input.json。具备可核对的解码和实际地址时，在新的审查目录填写一份独立的逐 PC 证据。当前仓库没有服务器新 OM 的实测指令档案。下列是输入合同。

- binding 逐项相等，含生成 Kernel 源、实际 ELF、decoder、规范化指令表、23 个 SDK 文件、父 Tiling、Cube、唯一符号、具体变体。
- regions 覆盖 q0/q1、s0/s1、w0/w1、expanded、sdk、reserved、brcb_fixed。每项有半开字节区间、从真实指令推导的 derivation 和一个或多个 `{"pc":整数,"asm":"完整已解码指令"}`。核对 TPipe 从当前 UB 高水位分配和 early-return 的支配关系，检查长度、32 B 对齐、Brcb rear 的最后 256 B、保留区与所有触及范围。
- accesses 另行检查**真实内存指令的地址操作数**。每个实查点提供 blocks 和末端 pc；解释器从未知寄存器起步，从 MOV/MOVI 和受支持的算术/移位推导被消费的立即数/寄存器值。q/s 初次 DMA 目的、两个 W 的 Cast 目的和 SDK UB→L1 源、expanded、Brcb 固定临时区、CO2 scratch 写入必须分别落在已审查基址。未建模的寄存器定义导致未知。不能手填初始寄存器数值，不能用理论 UB 表代替指令。
- obligations 逐项说明对完整可达代码、重定位、direct same-core 调用、所有内存读写区间、事件所有权与失败路径的推导，附实际 PC。还要审查 K 代数/奇偶切换、V_S 最后写交接、最后一次 SDK B 读、末组、N 边界和 L0C 归约。局部片段的地址算术是可机械复核的；库全局语义仍需要结合已绑定 SDK 的独立技术审查。
- cfg 是整段可执行文本的基本块数组。每块提交 first、last、successors（字节 PC 整数列表）和 derivation。程序从 ELF 符号表解析图入口，拒绝未解析的外部文本重定位；逐指令复算支持的控制流，检查所有块的覆盖和可达性。未支持的调用或 opcode 不能标记成顺序执行。地址和阶段见证必须在可达代码内。
- paths 给出 first、odd、even、rebind、last、false_next、false_last 各条真实 CFG 路径。`blocks` 为按执行先后排列的 [首PC,末PC]（均含）列表，允许循环回边。`events` 为该 walk 中的 action、walk_index（从零开始）、pc、完整 asm。直跳须能从解码文本独立解析目标，不能通过交换两段文本伪造前后顺序。当前验证器只处理可核验的直接边，不接受未闭合的调用、间接跳转或不认识的操作数编码。
- 完整 action 序列由 w16_static.check_proof 定义。一次 D 包含 8 次 Cast、两组各 8 次 Vector 转置和一次 FP16 Mul。记录 raw 两方向的 Set/Wait、V_S Set/Wait、UB→L1、MMAD、K/N fence、最终 copyout 及失败的两次释放。首、中、末必须区分；B 和 C 的 D/F 顺序分别为 after_fence 和 before_fence。见 [类型与证据检查](w16_static.py) 及[操作数复核](w16_operands.py)。
- 只有已检查的完整反汇编、身份、地址推导、CFG 和数据流同时成立，才给出 W16_STATIC_PASS。CPU 单元测试使用的构造指令不能进入实际 capture。若真机工具输出其它 ISA 语法或不等价的路径，保留 BLOCKED 和原始文件，先审查解码合同。

证据可来自当前 feature 分支某个已发布 commit 中的 `machine_reviews/<完整ELF SHA256>.json`，也可作为独立只读数据提交给 static。外部索引的格式如下，尖括号表示须由真实文件替换的字段。

```json
{
  "schema": 1,
  "kind": "W16_EXACT_ATC_MACHINE_EVIDENCE",
  "reports": {
    "<ELF SHA256>": {"path":"/absolute/path/to/review.json","bytes":123,"sha256":"<review file SHA256>"}
  }
}
```

```bash
w16 static --proof-index "$INDEX" --proof-index-sha256 "$EXPECTED_SHA256"
# 或从指定的、已拉取的审查数据提交读取
w16 static --review-commit "$REVIEW_COMMIT"
```

外部索引及逐 ELF 报告会按预先指定的摘要绑定并在后续阶段重验。两种来源经过同一套检验，缺失的指令不能靠索引弥补。如果同一静态阶段此前 BLOCKED，新尝试写到 `retries/<阶段>/attempt-*/report.json`。成功前驱必须唯一，不能替换旧报告或将 NO_GO 后的完整 Draft 阶段放行。

采集会先尝试 SDK 默认目标，再在默认解码不完整时显式指定 dav-m200。两条尝试均保留原始命令、返回码、日志与 SHA。若全部地址仍为占位符，不需要填写 review-input.json。先解决该 ELF 的实际解码能力，或者按主 README 使用独立 Down 路线。
