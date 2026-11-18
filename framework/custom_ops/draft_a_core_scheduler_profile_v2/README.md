# A-only Draft 离线归因诊断

运行 `python3 -B analyze.py --root /home/w00949577/qwen35-dflash-run/a-only-draft-profile-OC8pCW`。只读本次已验收的采集和 analyze 证据，所有新文件位于实验根目录的 **analysis-v2/**。已存在即拒绝覆盖。没有 NPU、ATC、msprof、模型执行或重新测时入口；无需进入 CANN 环境。终端回传 `analysis-v2/summary.md` 即可。成功状态 `OFFLINE_V2_RECORDED` 仅表示离线审计完成。

输入以原 analyze/report 身份链（文件名为 profile-report.json）绑定 C16/C64 的捕获报告、原 gear 结果、windows.json、实际选中的 timeline、单份 op_summary、同一导出目录中的 step_trace CSV 和 OM graph JSON。检查路径、字节数、SHA256，重新运行原严格解析器并比较 assignment_evidence。没有根据文件名挑选另一份 timeline，也不会拿另一个导出目录的时间戳补证。

每个 gear 的输出：

- **step-audit.json**：确认/未确认 Marker、完整 ID 顺序、内部缺失 ID 范围、重复次数、未获得严格 anchor 的 Host 调用；逐 Step 的原始 Start/End/Duration、Model/PID/TID/track、Decimal 检查、同 ID 的 CSV 行号与原字段、端点差值和不能旁证的原因。分别列出第一个未经端点旁证的精度异常与严格解析器的第一条映射错误。精度与端点旁证是两个判断：没有独立 Start 列时，即使 End 一致也只能是 UNPROVEN_ENDPOINTS。不会将 FP Start 改名为 Iteration Start，不用 End − Duration 推导 Start，不修改阈值。没有外部起始 ID 依据时不猜两端缺号。
- **strict.json**：原严格状态及重放结果。`DEVICE_STEPS_VERIFIED` 时单列已验收的严格分类；否则此栏为 null。
- **all-device-tasks.csv / crossing-tasks.csv / unassigned-tasks.csv**：逐原始 event_index 核算。用 runner lane 中原 38 对 Execute/Sync 的精确时间戳独立构造 Host 区间，以 Decimal 比较两个端点。只把完整落在唯一窗口、且有可区分 Task ID 的任务计入。跨窗、窗外、边界歧义、缺 ID 均不分配。同一调用、PID/model、物理 stream、task/batch/subtask 下有重复就排除整个歧义组；不同调用合法复用 Task ID。在所有报表中此方法始终为 **PROVISIONAL_HOST_ENCLOSURE**，即使同一采集另有严格 PASS。
- **host-windows.json**：全部 38 次的 Host 起止、已归属任务数、区间并集、每类任务统计。排名仅使用 calls 7–36；不计两次前检、5 次 warmup 和 postcheck。
- **provisional.json / unmapped-timed.csv**：分类任务数、累计 µs、30 次平均 µs/Execute、单 task 最大/平均 µs、按名称聚合的 Top 20 及单次事件 Top 20；未分类任务逐项保留映射依据。按唯一 op_summary 名称/类型、可辨认的 stream/task、OM 节点与原分类规则关联；普通 MatMul 不自动算 Linear。图缺失或有歧义时保留不确定性。覆盖率分别为 38 个窗口归属数 / 全部设备任务、30 个计入窗口的已分类数 / 已归属数、已分类耗时和 / 已归属耗时和；无分母时为 null。

累计 Task Duration 可重叠，不是整图 Wall Time。某类任务的耗时和排名只能提名单变量验证候选，不能证明优先优化它一定降低整图延迟。服务器运行前，真实异常 Step ID、覆盖率及排序均未知。

测试：`python3 -B -m unittest discover -s framework/custom_ops/draft_a_core_scheduler_profile_v2/test -v`。测试全为 CPU 合成数据。
