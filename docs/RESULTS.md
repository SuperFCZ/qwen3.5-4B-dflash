# 测试结果

Ascend310P，Chunk Verify，thinking 开。以下均为用户提供的设备测量，
状态为 `PASS_WITH_DIFFERENCES`：生成完成，ordinary token 一致性未通过，任务质量未评估。

## 指标口径

- **Decode 加速比** = Σ ordinary decode-loop 时间 / Σ DFlash decode-loop 时间。
  DFlash 循环包括 Draft、Verify/commit 和调度，不含 Prefill、循环前 context 构建及启动。
- **Generation tok/s** = 实际输出 token / (Prefill + Decode)。此列包含 Prefill，与 Decode 加速比口径不同。
- **接受率** = Σ accepted / Σ proposed；warmup 不参与统计。
- 两模式分别生成自己的输出，EOS 可能改变实际工作量。图级 ms/call 为同步 OM 调用时间；
  Draft 图统计可能包含 Prefill 调用，不能与阶段表重复相加。

## 自定义输入：128 token

三种 checkpoint 的比较使用 20 条输入；FP16 为六层，W4/W8 为五层。

| Draft / 权重方式 | 预热 / 重复 | 接受率 | token/轮 | Generation tok/s | Decode 加速比 | Draft ms/call | Verify ms/call |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP16 | 0 / 1 | 20.83% | 3.92 | 39.35 | 1.92× | 20.22 | 51.14 |
| W4A16 runtime | 0 / 1 | 18.96% | 3.65 | 19.13 | 0.87× | 94.68 | 51.38 |
| W8A16 runtime | 0 / 1 | 21.03% | 3.93 | 25.41 | 1.19× | 63.52 | 51.25 |
| **W8A16 离线 NZ** | **1 / 3** | **21.03%** | **3.93** | **29.16** | **1.39×** | **47.89** | **51.17** |

前三行来源 `gdr-lengths-waw8h36d`，加速比按已记录 Decode 阶段重算；
末行来源 `w8-nz-full-tsUMDuhH/gdr-lengths-mjhazgip`。
跨运行的预热、重复次数不同，速度变化作为观测对比，不作配对实验结论。

W8 接受率接近 FP16，但当前 Draft 调用仍比 FP16 慢；量化未带来相对 FP16 的速度优势。
离线 NZ 的 Draft 平均调用由 63.52 降至 47.89 ms，约下降 24.6%。

### 当前 W8 离线 NZ 分组

20/20 条输入完成，每条每模式 1 次预热、3 次测量；接受 `5718 / 27192`。

| 分组 | 测量 / 选中 | 接受率 | token/轮 | ordinary gen tok/s | DFlash gen tok/s | Decode 加速比 | Draft ms/call | Verify ms/call |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 全部 | 20 / 20 | 21.03% | 3.93 | 24.67 | 29.16 | 1.39× | 47.89 | 51.17 |
| 短输入 | 8 / 8 | 21.20% | 3.98 | 28.41 | 39.73 | 1.41× | 47.54 | 51.13 |
| 长输入 | 12 / 12 | 20.91% | 3.90 | 22.68 | 24.77 | 1.38× | 48.04 | 51.20 |

短输入为 28–59 token，长输入为 978–1024 token。逐题 Decode 加速比为 0.65–2.49×；
`zh_plan` 与 `long_zh_extract` 接受率较低，分别为 0.65×、0.70×。

阶段耗时（平均 ms/次生成）：

| 分组 | ordinary Prefill | ordinary Decode | DFlash Prefill | DFlash Decode |
|---|---:|---:|---:|---:|
| 全部 | 752.38 | 4436.34 | 1195.57 | 3193.72 |
| 短输入 | 74.20 | 4431.58 | 74.89 | 3146.91 |
| 长输入 | 1204.50 | 4439.52 | 1942.69 | 3224.93 |

完整报告：`w8-nz-full-tsUMDuhH/gdr-lengths-mjhazgip/summary.json`；
逐题数据位于同次运行的 `w8a16/cases.csv`。

## FP16 Draft 数据集：512 token

来源 `gdr-lengths-x2l5aydx`；0 次预热、1 次测量，2563/2563 条问题完成。

| 数据集 | 测量 / 选中 | accepted / proposed | 接受率 | token/轮 | DFlash gen tok/s | Decode 加速比 |
|---|---:|---:|---:|---:|---:|---:|
| gsm8k.jsonl | 1319 / 1319 | 540146 / 1864893 | 28.96% | 5.28 | 72.92 | 2.62× |
| humaneval.jsonl | 164 / 164 | 64701 / 284489 | 22.74% | 4.37 | 59.68 | 2.11× |
| math500.jsonl | 500 / 500 | 209575 / 683780 | 30.65% | 5.53 | 76.16 | 2.71× |
| mbpp.jsonl | 500 / 500 | 194904 / 904058 | 21.56% | 4.19 | 58.48 | 2.05× |
| mtbench.jsonl | 80 / 80 | 30994 / 145503 | 21.30% | 4.15 | 57.46 | 2.04× |
| 全部 | 2563 / 2563 | 1040320 / 3882723 | 26.79% | 4.96 | 68.61 | 2.44× |

总平均 Decode：ordinary 17787.30 ms，DFlash 7278.87 ms；
Draft / Verify 平均调用为 19.80 / 51.35 ms。
加速比由原报告的 Decode 阶段时间重算，未使用含 Prefill 的旧比值。
