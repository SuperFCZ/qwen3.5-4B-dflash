# 已测对照与候选裁决

历史配对仅对应 **C16 Layer0 单 Gate/Up，resident Execute+Sync**。原实验目录 `/home/w00949577/qwen35-dflash-run/gate-w16-experimental-fixed-YHsAKu`。本轮依据服务器只读回传；下方命令可以重新计算原样本的配对统计。

| 原配对 | 控制 Median / P95 (ms) | double-w-overlap Median / P95 (ms) | 三轮控制/候选 Median 比 | 总倍率 | 结果 |
|---|---:|---:|---|---:|---|
| schedule，控制 double-w-serial | 2.113535 / 2.13334 | 2.064200 / 2.68437 | 1.017407、1.024333、1.024187 | 1.023900× | NO_GO |
| net，控制 single-w-raw | 2.004735 / 2.04253 | 2.064585 / 2.10123 | 0.970687、0.970689、0.971103 | 0.971011× | NO_GO |

每格每版 90 个新鲜样本。schedule 的中位延迟缩短 49.335 µs，但 P95 增加 551.03 µs。net 三轮都比 single-W 慢，总体多 59.850 µs，P95 多 58.70 µs。两组的 overlap 属于不同独立进程集合，不合并其 P95，也不能从两个离散比较分配出“栅栏耗时”或“双 W 的纯资源开销”。真实决定为 `W16_EXPERIMENTAL_NO_GO`。现有数据没有测窄 Fence。

| 封存文件 | SHA256 |
|---|---|
| `compare/report.json` | `7d3c15d762b9ddcb3efc67f111b5987ee430d8087b73d7c6ba28f01bc10de74a` |
| `schedule-benchmark/report.json` | `d8a14c40324e334b43ff71be9cfc7035e30b21e8072cb31ed095999cc2608609` |
| `net-benchmark/report.json` | `2814877e85df90f365cb97edbe6270b638cc7b585b655287cb578290ebce46d0` |
| double-w-serial 的生成 C++ / 选中 ATC ELF | `144ab97b82cf91d7a3ce8cf3d20018f8394fc180e2bb41a06f727656fc17a13b` / `7de3096e7697b9056366321c2585d3e77079651957a9f9f44611c4d4a7e0f323` |
| double-w-overlap 的生成 C++ / 选中 ATC ELF | `4fd99adb3fb57b8946ca1ecebec4d1bf3100946632e05bd6c97e66f6a62972d5` / `195d076c225e54db57d50dda7a550df65a5c6fcf3842b4f256ecaa0b1cf90eb2` |

两份旧 ELF 各有 25,964 B 可执行 .text；独立解码复核均为 0 B 有效覆盖。那次运行按 `EXPERIMENTAL_ONLY / MACHINE_PROOF_UNRESOLVED / PRODUCTION_NOT_APPROVED` 放行。新研究要求机器级准入，既有实验放行的范围没有扩展。

## 在相同的双 W 布局上比较

| 方案 | 保证范围与仍需的独立边 | 事件与可能的等待 | 首/中/末 K、N 和失败 |
|---|---|---|---|
| A，`PipeBarrier<PIPE_MTE3>` | 约束 MTE3 自身的前后操作。旧 W 的最后读在 MTE3，W 复写在 V。单流水 barrier 给不出跨流水确认，L1_A 读→写也没有因此闭合。 | 无新增 ID。仍可等待同一 MTE3 上无关搬运。 | 所有方案均保留首轮预热、F9、FN 及失败排空，仅替换成功 k0…8。中间轮有未保护端点，A 不进入设备试验。 |
| B 的直接尝试，MTE3→V | 保护 W 的实际下一次 Vector 写。没有显式保护上一代 MTE1 读 A1 到下一代 MTE2 写 A1。 | 一个相应独占 ID 的设计。还须保护原 raw 标志的代数和 SDK CopyOut 的同方向 Fetch。 | k0 仅成功后 Set/Wait，中间仍有 A1 缺边。首尾或已经发出的 ID 不能在 Wait 前回收。 |
| B 的设计候选，MTE1→V 再 V→MTE2 | 通过 B1 前向队列保护 W，直接保护下一代 A1 写和 raw 标志重用。Cube 的独立 M 栅栏、SDK L0 槽位协议以及完整尾部仍保留。 | 每 block 两个新增方向各一份 reservation，每成功非尾 K 两次 Set、两次 Wait。load_done 等旧 L1 读，raw_ack 仍可能推迟下一次预取直至 D 完成。 | k0/稳态按同一规则。k9 与 FN 保留全栅栏，失败只 drain 已发代。首代硬件标志和全部真实 PC 仍待证。 |
| C，原 overlap 的 `PIPE_ALL` | 同核所有流水以前发出的读写在复用前退休。保留本身不能证明未来调用不会再读旧地址；这一条件来自 direct CFG_NORM 的 SDK 调用链。 | 无新增应用 ID。可等待 M、V 和多种搬运。实际暴露的等待时间未知。 | 历史源码三处全栅栏的语义均保留，失败不发布本 N。历史功能有实测，原严格机器证据保持未解决。 |

CPU 模型用两个连续 N，每 N 十个 K，将发射与退休分开，不引入未证明的编译器补同步。A 缺 16 个同 bank 的 W 复用边和 18 个 A1 复用边；直接 MTE3→V 保住 W，剩 18 个 A1 边；B 的设计候选覆盖模型所列存储边。单独去掉 SDK PIPE_M 后，一个 N 的九个相邻 L0C 累加边均失去证明。上述是各约束的敏感性检查。共同的 raw 首代 Wait 消费问题在 [事件设计](narrow-fence-design.md) 单列，不能以“模型存储边无缺失”升级为设备 PASS。

## 离线复核

在仓库根目录执行下列命令只读取文件并打印结果。SDK 检查也可单独运行，省略 `--experiment` 即可。

```bash
R=/home/w00949577/qwen35-dflash-run/gate-w16-experimental-fixed-YHsAKu
S=framework/custom_ops/draft_gate_w16_narrow_fence
python3 -B "$S/review_evidence.py" --sdk-root /usr/local/Ascend/cann-9.0.0 --experiment "$R"
```

工具核对 43 份 SDK 文件、compare 的原始 SHA 和前驱身份、冻结输入、两个双 W 已选 ELF 及实际 Tiling，从原 Runner 的 execution.json 核验每轮 Execute+Sync 样本并按旧协议复算。其 `device_launch_authorized` 恒为 false，因为这是只读审计入口。
