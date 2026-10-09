# Native Top1 输出 Guard 定位与物理对齐修复

2026-10-09 用户报告：30 个 native OM 已导出，native logits 可执行；native-top1
仅 M=4/8/12 PASS，其余 M=1..15 在 `device buffer guard overwritten` 失败。
`acfc07d` 增加具名 Guard 与分配尺寸诊断，没有改变分配大小。
随后用户回传了 Ascend310P3 / CANN 9.0 的 NPU 定位结果：

| 检查 | M1 | M4 |
| --- | --- | --- |
| logical / model-reported / physical | 8 / 8 / 8 B | 32 / 32 / 32 B |
| 首次执行前 Guard | 全部完整 | 完整 |
| 首次执行后 | 仅 y suffix 损坏 24 B，data offset 8..31 | 全部完整 |
| x / w / workspace / modelWeights | 完整 | 完整 |
| 数值比较 | Guard 阶段失败 | PASS |

这是用户提供的实际设备证据，不是 CPU 模拟结果；原始诊断目录路径未提供，后续补录，
不编造证据路径。它确认了本次 Native Top1 小输出的 32B 写回边界问题。
**本修复的 CANN 编译、Native-only NPU 全套回归、正式 Custom Head 验收仍为 `NOT_RUN`。**
未修改 Custom Head Kernel、native OM 或数学。

## 已确认的源码事实

`test/runner.cpp` 在 `acfc07d` 中先为 Top1 分配 `M*8` 字节，再读取
`aclmdlGetOutputSizeByIndex(desc,0)`；若其返回值更大，重新按返回值分配。
输出 shape/dtype 必须仍为 INT64 `[M]`，返回大小小于逻辑长度会失败。
Guard 在 payload 前后各 512 B，尾部从 `data + payload_bytes` 开始。
因此，“没有使用模型报告的输出大小”并非当前源码的问题。

官方 [aclmdlGetOutputSizeByIndex 文档](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/API/ascendgraphapi/aclcppdevg_03_1452.html)
说明返回模型输出的字节数；这份 API 说明本身不能证明本机 CANN 9.0/310P 的 native
Top1 一定按 32 B 写回，也不能代替本次故障的实际 Guard 证据。

模型报告值等于 `M*8`，写回至 `ceil(M*8/32)*32` 时，只有 M 为 4 的倍数不会触碰
原尾哨兵。M1 的设备诊断已直接观察到 8..31 B 被覆盖。CPU 回归保留原始分配的失败
模式，并验证修复后逻辑长度和物理边界分离；不能用它替代修复后的 NPU 全形状复测。

## 新的证据

每个执行目录在执行前生成 `output-allocation.json`，即使执行失败也保留：

- `m`、`top1_logical_bytes=M*8`、当前模式的 `logical_output_bytes`。
- `model_reported_output_bytes`：实际 API 返回值；Custom 模式为 null。
- `physical_payload_bytes`：runner 实际为 y 保留的 payload 容量，也是 dataset buffer 大小。
- `dataset_output_buffer_bytes`：传给 `aclCreateDataBuffer` 的容量；native 为物理 payload，
  Custom ACLNN 不使用此 dataset，记录 null。
- `allocation_request_bytes`：传给 `aclrtMalloc` 的总大小，含两侧 512 B Guard。
  不声称知道 allocator 内部额外分配的页空间。
- `suffix_data_offset`：尾部 Guard 相对于 y.data 的位置。

`runner.log` 的 `native_output` / `output_allocation` 同时打印以上关键大小。
`guard_check phase=...` 区分 setup、首次执行前、每个 correctness 验证点及 row permutation。
每次都检查 x、w、y、logits、workspace、modelWeights 的全部前后 Guard，再统一报错，
不会因第一个 buffer 损坏而遗漏后面的 buffer。未分配的区域明确写为 `UNALLOCATED`。

每侧记录 `INTACT/CORRUPT`、预期字节 `0xa5`、损坏字节数；损坏时还记录：

- `first_changed_guard_offset` / `last_changed_guard_offset`：相对该侧 Guard，0-based。
- `first_changed_data_offset` / `last_changed_data_offset`：相对 payload 起点，前哨兵为负数。
- `first_actual`：首个损坏位置的实际字节值。

这些是**观察到的不同字节**，不是 DMA 指令写入长度；如果写入值恰好为 `0xa5`，
哨兵不能单独证明该位置未被写入。连续计时中不增加 readback/logging，诊断在既有
计时前后的 correctness 检查以及第一次未计时执行前完成。

## 复用已有 capture / OM 复测

沿用成功导出时的 MODEL_PYTHON、CANN_ROOT、DEVICE_ID 环境。下面仅重新编译 native runner，
创建新的验收目录，原 C16/C64 capture、30 个 OM 和旧证据目录均不修改。
`old_head_run` 填本次失败的 `.runs/head-...` 目录，不是 A4 目录。

```bash
git pull --ff-only
old_head_run=framework/custom_ops/draft_head/.runs/head-替换为原失败目录
export HEAD_CAPTURE="$(cat "$old_head_run/capture-path.txt")"
export HEAD_NATIVE_MANIFEST="$(cat "$old_head_run/native-path.txt")"
test -f "$HEAD_CAPTURE" && test -f "$HEAD_NATIVE_MANIFEST" && \
HEAD_NATIVE_ONLY=1 HEAD_WARMUP=5 HEAD_REPETITIONS=30 \
  bash framework/custom_ops/draft_head/run_server.sh
```

不要删除或重新导出 capture/native。仍遍历全部 M 和完整用例，任何物理 Guard、输入只读、
repeat/permutation 或数值差异都会使该用例 FAIL；修复的实际效果以这次 NPU 回归为准。
优先查看新目录中的：

```bash
new_head_run=framework/custom_ops/draft_head/.runs/head-替换为新目录
cat "$new_head_run/data/real-c16-m01/native-top1/output-allocation.json"
cat "$new_head_run/data/real-c16-m01/native-top1/runner.log"
cat "$new_head_run/data/real-c16-m04/native-top1/output-allocation.json"
cat "$new_head_run/data/real-c16-m04/native-top1/runner.log"
```

## 最小修复与验收边界

`test/output_buffer.h` 定义 runner 使用的大小计算和逻辑读取：

```text
logical_bytes = M * 8
physical_payload_bytes = align_up(max(logical_bytes, model_reported_bytes), 32)
dataset_output_buffer_bytes = physical_payload_bytes
allocation_request_bytes = physical_payload_bytes + 2 * 512
```

物理补齐只应用于 `native-top1`；native-logits 继续按模型报告大小分配，Custom 的分配
与 kernel 不改。保持模型 INT64 `[M]` 校验及 `reported >= logical` 的原有 ABI 检查；
新增大小溢出检查。Guard 的位置为 `[data-512,data)` 和 `[data+physical,data+physical+512)`。
padding 允许被写入，物理 payload 后的任何 Guard 字节变化仍失败。

按照 [aclCreateDataBuffer 的参数定义](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha003/apiref/appdevgapi/aclcppdevg_03_1431.html)，
size 描述内存字节容量；因此 dataset 传物理 payload，不传带 Guard 的总分配大小。
逻辑 shape/dtype 继续从原 OM 检查，没有构造新的 tensor shape 或修改模型描述。
CANN 9.0 实际接收此容量的运行结果仍需本轮 NPU 回归确认；不存在失败后自动降级、
跳过形状或放宽检查的路径。

回读直接以 `logical_bytes` 为 D2H 长度，输出文件、repeat/permutation 和 bitwise 比较
只得到 M 个 INT64。padding 不回读、不参与比较，也不视为 token。具名 Guard 诊断完整保留。
预期 M1 的新报告为 logical=8、reported=8、physical=32、dataset=32、allocation=1056 B，
尾哨兵从 offset 32 开始；M4 则仍为 logical=reported=physical=dataset=32。

只有 Native-only **41 个完整用例、M1..15 全部 PASS** 后，才能进入正式 Custom Head
NPU 验收。沿用相同 `HEAD_CAPTURE` / `HEAD_NATIVE_MANIFEST`，再执行：

```bash
# 前置条件：本修复的 Native-only suite.json.status == PASS。
HEAD_NATIVE_ONLY=0 HEAD_WARMUP=5 HEAD_REPETITIONS=30 \
  bash framework/custom_ops/draft_head/run_server.sh
```

现在不执行或宣称该阶段通过，完整 Draft/Decode 也仍为 `NOT_RUN`。

## 本地回归

```bash
python -B -m unittest discover -s framework/custom_ops/draft_head/test -p 'test_*.py'
python -B -m unittest discover -s framework/custom_ops/draft_quant/test -p 'test_*.py'
```

`guard_model.cpp` 通过 CPU ACL 内存桩调用真实 shared Guard 代码，逐个破坏前后各 512
个位置，检查名称、偏移、期望/实际值、聚合多个损坏缓冲区；模拟 M1..15 的精确和
32B 写回，确认原始越界仍被拒绝。新增 M1..15 的修复后边界：模型报告等于/大于逻辑
大小、不同 padding 内容、精确逻辑 D2H 长度与文件内容、逻辑 ID 改变仍不相等、物理
payload 后多写一字节仍失败、非法回读长度与分配溢出被拒绝。
其结果不替代 CANN 编译、NPU 或 full-model 验收。
