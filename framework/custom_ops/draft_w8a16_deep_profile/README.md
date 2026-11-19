# W8A16 完整 Draft 深度归因（独立实验）

A 是对新的 Gate8 **完整 OM** 做受控 Profiling；B、C 是独立的资源/生命周期与 Head 研究。正式性能从 `gate-raw-atc8-Np20yu/full-benchmark/report.json` 的当轮样本读取。旧 `a-only-draft-profile-OC8pCW` 只提供历史证据链、工具身份和研究背景，不替代新 Off 采集。没有未插桩 benchmark、Kernel/OPP/OM 构建或生产切换入口。

## 本轮先做只读 admit

```bash
cd /home/w00949577/z50058744/qwen3.5-4B-dflash
git pull --ff-only origin feature/ascend310p-custom-ops
PY=/home/w00949577/qwen35-dflash-run/venv-hf513-probe/bin/python
S=$PWD/framework/custom_ops/draft_w8a16_deep_profile
R=$(mktemp -d /home/w00949577/qwen35-dflash-run/w8a16-deep-profile-XXXXXX)
echo "R=$R"
deep_stage() { TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PY" -B "$S/run_deep.py" "$@" --root "$R"; }
deep_stage admit
```

`DEEP_ADMIT_PASS` 才表示服务器重新验证成功。默认基线可以用 `--gate8-root`、`--historical-off` 指向实际已验收目录，不能改变验证合同。admit 复用原 Gate8 前驱链和重放验证器，核对 full-export/smoke/benchmark、四个 OM/实际嵌入代码、隔离 OPP/Runner、原输入/Golden、15 IDs、10 KV、运行环境、原 DSO manifest 加 22 项身份绑定和实际 Workspace。输出 admit/asset-index.json、head-om-evidence.json、head-graph-review.md 和 profile-report.json。会重新哈希大文件，**不重新执行任何阶段**。

## 后续手动真机采集（不是本地测试）

| 单条命令 | 作用 |
|---|---|
| `deep_stage profile-raw-c16 --allow-profile` | 只采 Raw C16 |
| `deep_stage profile-raw-c64 --allow-profile` | 只采 Raw C64 |
| `deep_stage analyze-raw` | 只读以上两份报告，先看 Raw 排名 |
| `deep_stage profile-off-c16 --allow-profile` | 同环境与数据，重新采本次 Gate8 对照 Off C16 |
| `deep_stage profile-off-c64 --allow-profile` | 同协议 Gate8 Off C64 |
| `deep_stage profile-paired-analysis` | 离线对照四个捕获，输出 profile-paired-analysis/profile-hotspots.md 与 optimization-next-step.md |

每次只执行所选命令，成功状态是 `DEEP_<大写阶段名，连字符换为下划线>_PASS`。四个采集目录独立且不可覆盖，不强迫做两份 Off 才能分析 Raw。在未来授权设备采集时，建议先验证 Raw C16；为了控制顺序漂移，也可先采 Raw C16→Off C16，再 Off C64→Raw C64。一次采集/变体仅是描述性 profile，不是新的正式速度比测量。

使用原 positional msprof + exec shim + SO_PASSCRED 的应用进程门禁。复用当前 full-export OM、full-smoke/benchmark 用过的 Runner，ATC mode 1 只检查图。实际执行程序、PID、原 FD、argv、scope、私有 OPP/loader 环境和 pre/post maps 必须符合原检查。2 次 poisoned correctness、5 warmup、30 次 Execute+Sync、timed tail 与单独 postcheck 共 38 次计算。profile 进程使用既有 Runner 的 benchmark **调用结构**实现这 38 次；其 timing 绝不写回正式 Benchmark。新增依赖/丢 FD、窗口或身份不符均 BLOCKED，无替代执行路径。保留 CLI、日志、全部原始 CSV/Timeline、maps、二进制输出和 SHA。若失败，回传阶段 status/error、traceback、msprof.log、application/runner.log（若有），别修改旧报告。

## 分析产物和局限

- 选择原 PID 对应的唯一 38 对窗口，只使用 7..36 的设备任务作统计。保留原严格 Step 结果及错误原因；无严格归属时，用独立 Decimal Host 窗口完整包含与按窗口唯一 task 身份的规则。跨窗/重复/缺身份不计，不把结果升级为严格 PASS。配对时某档一侧不严格，则两边用相同 provisional 规则；存在严格的一侧另存 -strict 子目录。
- 每个变体/gear 有 case-analysis.json、timed-tasks.csv（stream/task ID）、projection-map.json、head.json、attention.json、notify.json、counters.json。26 个投影按不可变 NZ 权重/Scale 成对哈希、图节点及逻辑形状绑定；所有期待的名字都列出。缺失覆盖保留 UNRESOLVED。Head 的 ArgMax、Attention 中有证据的 softmax/QK/PV 路径与普通未定位 MatMul 分开。wait/record 提供原始身份、同 ID 候选和邻近/重叠；没有生产者证据绝不按距离强行配对。
- task-based PipeUtilization 使用实际的 ascendcl、runtime-api、task-time、aicpu、ai-core、aic-mode、aic-metrics 选项。Counter 只来自同一 op_summary 的已唯一匹配行。按 stream/task/name/type 与明确同单位的原始时间匹配；多重命中或同一 CSV 行被不同事件认领则剔除，绝不按行序映射，不混合 task_time。Cube/MAC、MTE1/2/3、Vector/Scalar 和 Stall 字段按实际列名/原单位，分 type/core 报告覆盖数。没采到或无法绑定为 UNAVAILABLE，不把 busy ratio 当 stall，不凑总和。原始基线是未插桩 wall latency；Task Sum 与通知等待/各 pipe 可重叠。
- 分析成功不是所有任务都已归因。先看每档 attribution、每个投影 counts_per_execute、counter 覆盖和 unmatched，再解读 delta。报告动态选择的最大已识别计算族是下一次验证候选，不是确定的关键路径节省。终端可 `cat "$R/profile-paired-analysis/optimization-next-step.md"`；热点较长时逐档 `sed -n '1,120p' "$R/profile-paired-analysis/profile-hotspots.md"`。

## B/C 交付及本地复核

见 w16-double-buffer-design.md、w16-event-lifecycle.md、w16-risk-register.md、w16-ub-budget.json、head-optimization-review.md、optimization-next-step.md。可在已加载 CANN 环境的 Linux ARM64 运行 `bash "$S/w16-query.sh"` 重查两个预算，仅 CPU SDK named-platform 查询；输出新临时目录。它不授予设备运行许可。

CPU：`python3 -B -m unittest discover -s framework/custom_ops/draft_w8a16_deep_profile/test -v`。本地已实际编译、链接、运行 Host Tiling probe；没有运行 msprof、NPU、OM 推理或 Benchmark。真实 Raw/Off captures 和 paired analysis 都待服务器。
