# Gate/Up Raw Prefetch：独立首轮实验

当前新候选 CANN 编译、NPU 正确性/性能、完整 Draft Token/KV、Decode 均 **NOT_RUN**。
只比较 C16 layer-0-gate_up；不自动推广五层，不改完整 Draft 配置、Native Golden 或旧 Evidence。

## 最小改动与可复用性

新增 `DFLASH_GATE_PREFETCH_MODE=off|raw`，默认 off，精确限定 M16/K2560/N19456、
N64/K256、batched/Broadcast。原 `DFLASH_PIPELINE_MODE` 的 Down/合成形状策略保持
不变；新开关独立于它。两种构建均使用原始 Loop 搬运，**2D DataCopy 必须关闭**。
配置器、Shell 和 Kernel static_assert 同时拒绝混合候选。

Kernel 只把这个精确 Gate predicate 加入原 prefetch 分支的选择；原 Down 分支内部
代码不改。它可以作为首轮候选复用，因为已有的依赖链覆盖当前 raw bank、下一 raw bank、
单份 W16 和 Matmul 消费者，而不是假设 Iterate 返回就代表 Cube 已完成。

每个 N tile 的操作保持：

1. 首次只搬 K0 到 bank0，分别 signal 预留的 MTE2_S / MTE2_V。
2. 每轮先 wait 当前两个事件，再向另一 bank 搬下一 K tile（仅 hasNext 时）。
   只有一轮 lookahead；两事件用 TPipe AllocEventID 保留，防止 Matmul 内部 FetchEventID
   与尚未消费的信号冲突。
3. 当前 bank 的 INT8 Cast、两个 group128 的 Broadcast/Mul 和 FP16 边界保持不变。
   Vector 写 W16 后用 V_S set/wait 完成交接，再进入原 Matmul。
4. **每个 Iterate 之后的 PIPE_ALL 原样保留**，保护下一轮复用 raw/W16/SDK 工作区。
   这里也会等尚未完成的 lookahead；没有增加更激进的跨轮 Cube 异步方案。
5. 最后 K tile 不再预取；仅一次 GetTensorC 最终 FP16 输出。N tile 结束的 PIPE_ALL
   保留，下一个 N tile 重新从 bank0/K0 开始，绝不预取到下一个 N tile。
6. Iterate false 时，有 lookahead 则先 wait 两个事件，然后 PIPE_ALL、ReleaseEventID、
   End；首/中/末 K、首/次 N tile 的异常路径均由 CPU 模型验证。

Gate 的 304×10=3,040 次 tile 迭代、每 tile 8+2 次 Loop raw DataCopy、GM 逻辑读取量、
FP32 完整 K 累加、单份 W16、N tile owner 和全部输出精度不变。Prefetch 可能与
Matmul 的 A 搬运竞争 MTE2，因此不能预先保证收益。

## 每核 UB 账本

| 部分 | off 基线 B | raw 候选 B |
| --- | ---: | ---: |
| INT8 raw | 16384 | 32768 |
| FP16 GN raw scale | 256 | 512 |
| 单份 FP16 W | 32768 | 32768 |
| Broadcast 临时区 | 2304 | 2304 |
| SDK Brcb 固定保留区（未交给 TPipe） | 8192 | 8192 |
| Host user UB（含保留区） | 59904 | 76544 |
| 交给 SDK Matmul 的 UB | 202240 | 185600 |
| 总计 | 262144 | 262144 |

额外 raw bank 为 16640 B。Host 用原计算公式减少 Matmul UB，不改 SetFixSplit 或容量
校验；SDK 返回的 Cube/base/step/depth/DB 参数及 L1/L0A/L0B/L0C 资源必须在 A/B 配对
中一致。只允许 raw_banks、selected_pipeline、上述两项 UB 预算和 Gate 开关证据不同。
如果 SDK 因 UB 缩小选出不同的其他计划，比较器明确失败，不能把两个变量混为收益。

Host 日志新增 `DFLASH_GROUP_QUANT_GATE_PREFETCH`；build-config 绑定 Host/Kernel
同一配置头 SHA256，SDK 检查记录 dav_m200 和可定位的 TPipe 头文件。资源/事件合同
属于源码和 Host 计划证据，不是已测硬件 overlap。

## 服务器最小入口

```bash
bash framework/custom_ops/draft_quant/run_gate_prefetch.sh --a51-evidence /home/w00949577/qwen35-dflash-run/a51-fallback-20261012-063342/suite.json --output-dir "/home/w00949577/qwen35-dflash-run/gate-raw-prefetch-$(date +%Y%m%d-%H%M%S)"
```

复用已验证的 A5.1 C16 Capture、Native W8 OM 和冻结输出；从旧报告解析实际路径并
校验哈希，不重新 capture/export。新 OPP 与 Runner 各自放在新 build/run 目录。
自动执行两个独立进程：off 原 A5.1 Loop/K256/Broadcast 与 raw 候选，原 Down 配置
和 core cap 均继承接受的 A5.1。支持 `--order candidate-first`，便于另建目录反序复测。

每个构建阶段为：SDK 检查 → CANN 编译/隔离安装 → **两次无计时 Custom 对冻结
Native bitwise** → 仅通过后进行 Native/Custom warmup=5 / repetitions=30。
保留 guards、输入只读、原始 NZ/Scale 哈希、重复输出、timed-tail/postcheck；
计时阶段所有输出还须匹配相同冻结 Native。任何正确性失败停止该构建，不计时。

输出新目录 `suite.json`、`comparison.json`、`off.log`、`raw.log` 及单次运行指针。
比较报告包含 Native-before/after、baseline、candidate 的原始 samples / median / p95、
bitwise、完整 build 和 launch；单次报告的 `prefetch_resources` 列出 UB/事件依赖。
PASS 仅为数值门槛，adoption 仍 NOT_DECIDED。没有明显、可复现的 median/p95 收益，
就不推广五层，更不把独立算子差值推算成 Draft/Decode 收益。

## CPU 验证

- 实际 Kernel 源码在 CPU API 模型中执行，覆盖全部原形状、多核、Gate 双 bank、
  首末 K、跨 N、完整 K 输出及只读/guard；额外验证原 Down pipeline 开/关均不受干扰。
- 检查原始搬运/反量化次数、单份 W、UB 分配、每轮 PIPE_ALL 数量及事件分配/释放。
- 对首/中/末 K 与第二个 N tile 注入 Iterate false，验证无泄漏信号或未完成消费者，
  已完成输出与未写 poison 的边界正确。
- 故意删除 consumer fence、遗漏 MTE2_V wait、借用非预留 ID、覆盖当前 bank，
  真实 Kernel 的变异版本必须被 CPU 模型拒绝。
- 配置/模式混用、UB 预算、SDK 计划变化、错误 launch log、分阶段及配对入口均有回归。

CPU 模型不能证明 NPU 并发、cache、SDK 内部生命周期或性能；最终采用只依据真实
Ascend310P3 正确性和相同输入/协议的可复现性能结果。
