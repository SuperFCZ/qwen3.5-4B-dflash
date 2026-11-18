#!/usr/bin/env python3
"""Print a bounded, read-only summary of an accepted A-only Draft msprof run."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median


def short(value, limit=180):
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def bound_json(path, identity):
    data = path.read_bytes()
    if identity != dict(path=str(path.resolve()), bytes=len(data), sha256=hashlib.sha256(data).hexdigest()):
        raise ValueError(f"报告身份与 analyze 不一致：{path}")
    return json.loads(data)


def accepted(report, status):
    if report["status"] != status or report.get("error") or report.get("integrity_error"):
        raise ValueError(f"需要 {status}，实际为 {report['status']}")


def summary(root):
    root = root.resolve()
    analyze = root / "analyze"
    main = json.loads((analyze / "profile-report.json").read_bytes())
    accepted(main, "A_OM_PROFILE_ANALYZED")
    lines = [f"实验目录: {root}", f"Analyze: {main['status']}"]
    for gear in ("16", "64"):
        phase = f"profile-c{gear}"
        capture = bound_json(root / phase / "profile-report.json", main["predecessors"][phase])
        accepted(capture, f"A_C{gear}_RUNNER_CAPTURE_VERIFIED")
        report = bound_json(analyze / f"c{gear}" / "profile-report.json", main["gears"][gear]["report"])
        calls = report["measured_calls"]
        if [call["call"] for call in calls] != list(range(7, 37)):
            raise ValueError(f"C{gear}: 需要第 7–36 次调用的完整窗口")
        base = report["formal_baseline"]
        us_per_ms_times_calls = 1000 * len(calls)
        lines.extend([
            f"\n========== A / C{gear} ==========",
            f"采集: {capture['status']} | {capture['execute']}",
            f"归属: {report['attribution']} | 图解析: {short(report['graph_status'])}",
            f"无插桩基线: median={base['median_ms']:.4f} ms  P95={base['p95_ms']:.4f} ms",
            f"插桩窗口 median: Host={median(x['host_window_us'] for x in calls)/1000:.4f} ms  "
            f"已归属任务区间并集={median(x['task_union_us'] for x in calls)/1000:.4f} ms",
            f"计入任务={report['measured_task_count']}  可排名计算任务={report['rankable_kernel_task_count']}  "
            f"provisional={report['provisional_measured_tasks']}",
            f"边界交叉观察={report['boundary_crossing_observations']}  重复事件={len(report['duplicate_events'])}  "
            f"窗口外任务={report['outside_windows']}",
        ])
        for key in ("step_unassigned_reasons", "rejected_step_reasons", "classification_uncertain"):
            lines.append(f"{key}: {short(json.dumps(report[key], ensure_ascii=False), 500)}")
        lines.append("类别 | 任务数(30次合计) | 任务和(ms/调用) | 单任务最大(µs)")
        types = report["operator_types"]
        for row in types[:20]:
            lines.append(f"{short(row['category'])} | {row['task_count']} | "
                         f"{row['sum_us']/us_per_ms_times_calls:.4f} | {row['max_us']:.2f}")
        if len(types) > 20:
            lines.append(f"只显示前 20 类，共 {len(types)} 类")
        ranks = report["top20_kernels"]
        lines.append("计算 Kernel Top 20" if ranks else "无可排名的计算 Kernel；改列已归属设备任务")
        lines.append("排名 | 任务数 | 任务和(ms/调用) | 名称")
        for index, row in enumerate((ranks or report["top20_device_tasks"])[:20], 1):
            lines.append(f"{index:2} | {row['task_count']:5} | {row['sum_us']/us_per_ms_times_calls:.4f} | {short(row['kernel'])}")
        if not types:
            lines.append("没有可汇总的已归属任务；不能解释为设备耗时为零。")
    lines.append("\n任务可能重叠，任务耗时和不是整图墙钟延迟。")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="成功运行 prof_stage analyze 的实验根目录 ($R)")
    args = parser.parse_args()
    try:
        print(summary(args.root))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"无法读取已验收的 Profile 摘要：{short(error)}\n")


if __name__ == "__main__":
    main()
