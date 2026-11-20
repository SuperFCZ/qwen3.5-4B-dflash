# Gate Scale 完整 OM Export 续接

适用归档为 `gate-scale-expansion-dFFDXz` 的两档五层 GO 和其已生成的 C16 候选 OM。原报错发生在 ATC 成功后对非目标节点的整 ELF 比较。恢复工具独立于原实验的 Source Lock。新根记录复核结果，归档报告、两份私有 OPP、原始 OM 与 Golden 全部只读。原驱动的整文件比较仍可用于要求 ELF 全部字节相同的实验。

## 服务器步骤

在仓库根目录执行。每一阶段单独启动，查看状态后再执行下一条。

```bash
cd /home/w00949577/z50058744/qwen3.5-4B-dflash
git pull --ff-only origin feature/ascend310p-custom-ops
BASE=/home/w00949577/qwen35-dflash-run
PY="$BASE/venv-hf513-probe/bin/python"
S="$PWD/framework/custom_ops/draft_gate_scale_export_recovery"
R2=$(mktemp -d "$BASE/gate-scale-full-recovery-XXXXXX")
echo "R2=$R2"
scale_full() { TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PY" -B "$S/recover_full.py" "$@" --root "$R2"; }
scale_full admit-c16
```

| 手动命令 | 成功状态 | 工作范围 |
|---|---|---|
| `scale_full admit-c16` | `FULL_SCALE_ARCHIVED_C16_ADMITTED` | CPU 复核两个 GO、15 个前驱，重读 C16 ATC Worker、图、两份 OM 和真实调度。 |
| `scale_full full-draft-export` | `FULL_SCALE_EXPORT_RECOVERED` | 复用已准入的 C16，在新目录从冻结 C64 Raw AIR 编译 C64 一次，应用相同的独立验收。 |
| `scale_full full-draft-smoke --allow-device` | `FULL_SCALE_TOKEN_KV_MATCH` | 原 Runner 与 DSO socket 门控。C16/C64 两版逐位比较 15 个 Token 和 10 份完整 KV，包括未更新区，复验两种 Poison、Guard、Readonly、Repeat 和同步。 |
| `scale_full full-draft-benchmark --allow-device` | `FULL_SCALE_PAIRED_MEASURED` | 新鲜 Raw Brcb/Transpose 配对、三轮交错，每轮每版 5 次 Warmup 和 30 次 Execute+Sync，保留 Timed Tail 和 Postcheck。 |
| `scale_full final-comparison` | `FULL_SCALE_GO` 或 `FULL_SCALE_NO_GO` | 从四组 90 个正式样本复算原预注册门槛，分别报告 C16/C64。 |

每阶段报告为 `$R2/<阶段>/report.json`。失败查看同目录 `traceback.txt`。原始完整文件哈希与执行合同的比较结果分别保留，准入不产生任何模型执行记录。若需在另一终端继续，重新设定上面的变量，把 `R2` 设为已有恢复根。

## C16 实物与校验范围

- 原失败报告为 329,533 B，SHA256 `f26db9ca99d0910d74f7fafe78514f7fe1c4aa5810c03866d4fb6b237fbbbc08`。C16 候选 OM 为 3,090,429,133 B，SHA256 `a34a7e2139a9e79082e9de8c859d221c70a19a777dc48e43680c1fd13fee8b57`。代码另锁定 Worker 和两档 Compare 的原 SHA，调用原实验的前驱及数值/计时证据复核器。
- 分区表、边界与总长先验证，再对类型 1（权重）、2（设备任务）、8（模型输入输出信息）比对完整字节数与 SHA256。C16 权重为 3,088,792,576 B，SHA256 `03fd2f3a04baf9e209cbf515072d18cb5d2571e217d0f57eff4539f0cc5c291d`。GE 官方枚举给出了分区类型。[GE v9.0.0 模型分区定义](https://gitcode.com/cann/ge/blob/261bdc399000f5bbe1211b8a09dd42a5d1ba92f6/inc/framework/common/types.h)
- C16 两边各有 583 个节点。51 对生成的 Const 通过稳定消费节点、输入槽位、生产者输出端口和出现次数建立唯一双射，再比较完整 Const 张量数据、外置权重引用和描述符。无消费者、歧义、同名端口调换、权重或真实 shape/dtype/layout 漂移都会阻断。GE 属性按键比较，已观测的重复来源描述符保留每个类型、值和出现次数。记录单独保留原名称和配对名称。
- `fusion_failed` 是编译器给回退的融合组写入的 scope ID。逐节点保存新旧 ID、核对一一重编号和完整组成员，仍比较最终融合图的边、每个节点、实际选中的设备代码和任务分区。[GE 写入 scope ID](https://gitcode.com/cann/ge/blob/261bdc399000f5bbe1211b8a09dd42a5d1ba92f6/compiler/engines/nn_engine/optimizer/adapter/tbe_adapter/tbe_op_store_adapter.cc) · [FE 按 pass 与 scope 统计](https://gitcode.com/cann/ge/blob/261bdc399000f5bbe1211b8a09dd42a5d1ba92f6/compiler/engines/nn_engine/fusion/common/fusion_statistic/buffer_fusion_info_collecter.cc)
- 五个目标 Gate 仍须实际选中不同可执行段；独立核对编译路径、真实 8-block、六种回调、Workspace、Cube Tiling 和 ELF 可写资源。其余 21 个投影和所有原生代码分别从各自 OM 的真实偏移取证。每个图上的 `_kernel_bin_id`、长度、入口均绑定自己的原始 ELF。完整相同记作 `EXACT_ELF`。
- `LOCAL_STT_FILE_TAG_ONLY` 仅允许当前入口对应的 `<编译键>_<PID>_kernel.cpp` 中十进制 PID 改变。验证非装载的字符串表、独占引用它的本地绝对 STT_FILE、全部指令、程序头、重定位、非调试元数据和符号。不同位数需能仅由该字符串长度导出偏移和对齐变化。CANN 9.0 的 `compile_op.py` 与 `ascendc_common_utility.py` 在准入时按 SHA256 验证。
- `ET_REL_DFX_ARGUMENT_DESCRIPTORS_ONLY` 只处理观测到的原生 fused Cast/Mul 和 StridedSlice 家族。函数级 `.ascend.meta.<entry>` 的 Type 4 是 DFX，内部 Type 5 使用大端字数。校验参数条目顺序、固定前缀、形状编码、具体描述与单个 `-1` 诊断描述的转换，并复算 Runtime 的 `userArgsNum`。其余 TLV、每个可执行字节、其他节、符号及其顺序均须相同；仅允许本地元数据对象长度与随节大小发生的文件偏移变化。实际运行形状和 ABI 来自独立核验的 GE 描述符与任务分区，DFX 的 `-1` 只写入诊断差异报告。异常 dump 的参数细节可能不同。该判断限于无插桩的完整 OM 执行；原始 ELF 和诊断元数据都保留。[Runtime 枚举，区分函数级与二进制级 Type 4](https://gitcode.com/cann/runtime/blob/3542a202286c1aa93b56b9aede6f25c5c11a253c/pkg_inc/runtime/runtime/elf_base.h) · [Runtime 对 DFX 参数计数及元数据加载实现](https://gitcode.com/cann/runtime/blob/3542a202286c1aa93b56b9aede6f25c5c11a253c/src/runtime/feature/src/kernel/elf.cc)
- 整 ELF 两侧 SHA、可执行段、被允许的每项差异及所有未知差异的阻断理由均可核查。新 export 在 C16/C64 各自重做上述比较。真实 Token/KV 及延迟分别由后续手动阶段决定。生产默认值不在本实验的发布范围。

CPU 回归可用 `python3 -B -m unittest discover -s framework/custom_ops/draft_gate_scale_export_recovery/test -v`。只读摘要仍可用 `python3 -B framework/custom_ops/draft_gate_scale_export_recovery/inspect_export.py`。
