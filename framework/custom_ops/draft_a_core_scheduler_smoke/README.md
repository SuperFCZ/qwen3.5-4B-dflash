# 310P3 单 FC 调度 Smoke

只验收 C16 FC `[16,12800,2560]`。输入 X FP16 ND、W INT8 NZ `[400,160,16,32]`、Group128 Scale FP16 `[100,2560]` 和 Native OM Golden 均来自已验收的历史 `fc` case。设备 0。A/B 分别使用 Host Compare 中原封不动的私有 OPP、API 和设备 Kernel。Build 只构建两个主机 Runner，不安装 OPP，不调用 ATC。

| 阶段 | 成功状态 | 设备执行 |
|---|---|---|
| Source | `FC_SMOKE_SOURCE_BOUND` | 0。校验历史五份报告的前驱、源码、OPP、API、Kernel、输入/Golden、DSO 和真实回调。`sources` 为路径/字节/SHA256 清单。 |
| Build | `FC_SMOKE_RUNNERS_BUILT` | 0。对每一版单独链接私有 API，ldd 路径与 SHA 必须匹配对应历史 Runner。 |
| Smoke | `FC_AB_FOUR_EXECUTIONS_MATCH` | A0、A1、B0、B1；每次独立进程、设备会话、输入上传、Workspace、executor；每次一对 GetWorkspaceSize/Execute，无计时。 |
| Compare | `FC_SCHEDULER_SINGLE_SHAPE_NPU_MATCH` | 0。重新读取证据并比较 40960 个元素、两轮 Repeat、两组 A/B、资源和身份。 |

每个 Runner 只看本版私有 OPP。GetWorkspaceSize 触发真实 callback 后，Runner 通过私有 socket 等待父进程许可。实时 platform/scheduling 各一行，必须完整等于已验收 Runtime 回调（包括 40 Tile 分配、Tiling bytes、资源以及 Host DSO）；A/B 的实际 `blockDim` 分别为 7/8，generic/AIC/AIV 为 7/8/8，`CalcTschBlockDim` 为 8。这里的整数只用于拒绝漂移，不写回调度参数。Workspace 必须为 2097152 B。加载身份在许可前与同步后分别检查，沿用历史 DSO Manifest；如果 Execute 新引入未绑定库则阻断，不按路径临时放行。

预填充 `0x7f7f` / `0xffff` 两种 FP16 NaN 并在执行前读回。每次同步后保存 `actual.bin`，使用历史 `draft_quant/test/reference.py::compare` 的 `finite && bitwise_equal` 门槛；保留每个 mismatch 的坐标、bits、有限值 ULP 及全部非有限位置。对输入三份 payload 做逐字节只读检查；五个 GM 分配（含 Workspace）每侧 512 B Guard 的每处变化记录在 `*-guards.jsonl`。没有新加也不声称检查任何片上 Guard。设备错误写入 `events.jsonl`/`runner.log`；若同步失败，不把未完成输出当作有效读回。没有 CPU fallback、性能采样或额外投影。

## 服务器（每段单独执行，仅上段成功才继续）

```bash
cd /home/w00949577/z50058744/qwen3.5-4B-dflash
git pull --ff-only origin feature/ascend310p-custom-ops
PY=/home/w00949577/qwen35-dflash-run/venv-hf513-probe/bin/python
S=$PWD/framework/custom_ops/draft_a_core_scheduler_smoke
H=/home/w00949577/qwen35-dflash-run/a-core-scheduler-host-MzJtLl
R=$(mktemp -d /home/w00949577/qwen35-dflash-run/a-core-scheduler-fc-smoke-XXXXXX)
echo "R=$R"
fc() { TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PY" -B "$S/run_smoke.py" "$@"; }
fc source --host-root "$H" --output-dir "$R/source"
```

```bash
fc build --source-report "$R/source/report.json" --output-dir "$R/build"
```

以下是唯一会执行计算 Kernel 的命令：

```bash
fc smoke --source-report "$R/source/report.json" --build-report "$R/build/report.json" --output-dir "$R/smoke"
```

```bash
fc compare --source-report "$R/source/report.json" --build-report "$R/build/report.json" --smoke-report "$R/smoke/report.json" --output-dir "$R/compare"
```

任何错误退出即停止；保留全目录，不覆盖旧报告。回传阶段 `report.json` 的 `status/error/integrity_error`、`traceback.txt` 及失败 run 的 `events.jsonl`、`runner.log`；若是数值或保护错误，再附 `native.json` 的汇总或非空 Guard 变化。单 FC PASS 才值得单独设计其余投影验收；不据此推断物理 Core 驻留、完整 Draft 正确性或加速。

CPU 合同回归：`python3 -B -m unittest discover -s framework/custom_ops/draft_a_core_scheduler_smoke/test -v`。仅真实服务器生成的四次同步执行及 Compare 报告是 NPU 证据。
