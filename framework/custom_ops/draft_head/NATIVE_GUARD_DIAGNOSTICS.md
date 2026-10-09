# Native Top1 输出 Guard 定位

2026-10-09 用户报告：30 个 native OM 已导出，native logits 可执行；native-top1
仅 M=4/8/12 PASS，其余 M=1..15 在 `device buffer guard overwritten` 失败。
**实际损坏缓冲区、设备写回边界以及修复后的 NPU 回归尚待确认。** 本次为定位补丁，
不调整任何缓冲区容量，不修改 Custom Head 算法或数值门槛。

## 已确认的源码事实

`test/runner.cpp` 原实现先为 Top1 分配 `M*8` 字节，再读取
`aclmdlGetOutputSizeByIndex(desc,0)`；若其返回值更大，重新按返回值分配。
输出 shape/dtype 必须仍为 INT64 `[M]`，返回大小小于逻辑长度会失败。
Guard 在 payload 前后各 512 B，尾部从 `data + payload_bytes` 开始。
因此，“没有使用模型报告的输出大小”并非当前源码的问题。

官方 [aclmdlGetOutputSizeByIndex 文档](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/API/ascendgraphapi/aclcppdevg_03_1452.html)
说明返回模型输出的字节数；这份 API 说明本身不能证明本机 CANN 9.0/310P 的 native
Top1 一定按 32 B 写回，也不能代替本次故障的实际 Guard 证据。

若模型报告值等于 `M*8`，而实际末次写回覆盖至 `ceil(M*8/32)*32`，则只有 M 为 4
的倍数时不会触碰原尾哨兵。这与现象相符，但目前只是候选解释。
CPU 回归模拟该写回可以复现此规律，**不是 NPU 证实**。

## 新的证据

每个执行目录在执行前生成 `output-allocation.json`，即使执行失败也保留：

- `m`、`top1_logical_bytes=M*8`、当前模式的 `logical_output_bytes`。
- `model_reported_output_bytes`：实际 API 返回值；Custom 模式为 null。
- `physical_payload_bytes`：runner 实际为 y 保留的 payload 容量，也是 dataset buffer 大小。
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
test -f "$HEAD_CAPTURE" && test -f "$HEAD_NATIVE_MANIFEST"
HEAD_NATIVE_ONLY=1 HEAD_WARMUP=5 HEAD_REPETITIONS=30 \
  bash framework/custom_ops/draft_head/run_server.sh
```

不要删除或重新导出 capture/native。仍遍历全部 M 和完整用例，原有 Guard 失败仍会
使该用例 FAIL；这个定位版本**不承诺消除原故障**。
优先查看新目录中的：

```bash
new_head_run=framework/custom_ops/draft_head/.runs/head-替换为新目录
cat "$new_head_run/data/real-c16-m01/native-top1/output-allocation.json"
cat "$new_head_run/data/real-c16-m01/native-top1/runner.log"
cat "$new_head_run/data/real-c16-m04/native-top1/output-allocation.json"
cat "$new_head_run/data/real-c16-m04/native-top1/runner.log"
```

## 确认后才实施的最小修复

先确认首次执行前所有 Guard 完整，执行后只有 y 的 suffix 损坏，且 M1/M2/M3 的
损坏位置与各自逻辑尾部到 32 B 边界相符；同时核对模型实际报告大小。
如果 workspace/modelWeights、prefix 或更远位置损坏，按真实区域继续定位，不套用输出对齐解释。

只有上述 NPU 证据支持输出物理写回对齐后，才修改 native-top1 的物理容量计算：
逻辑输出保持 `M*8`，物理 payload 覆盖已确认的写回范围且不小于模型报告值；Guard
放在物理 payload 的两端，dataset 声明容量与之相同。输出文件、repeat/permutation
和数值比较仍只有逻辑 M 个 INT64，不比较 padding，不以 padding 当成有效 token。
随后必须重跑 M1..15 native-only 全套；任何 Guard 或数值差异继续失败。

## 本地回归

```bash
python -B -m unittest discover -s framework/custom_ops/draft_head/test -p 'test_*.py'
python -B -m unittest discover -s framework/custom_ops/draft_quant/test -p 'test_*.py'
```

`guard_model.cpp` 通过 CPU ACL 内存桩调用真实 shared Guard 代码，逐个破坏前后各 512
个位置，检查名称、偏移、期望/实际值、聚合多个损坏缓冲区；模拟 M1..15 的精确和
32B 写回，确认原始越界仍被拒绝。其结果不替代 CANN 编译、NPU 或 full-model 验收。
