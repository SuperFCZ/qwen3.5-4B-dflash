"""Disjoint component accounting from thirty actual integrated-model calls."""

from collections import Counter, defaultdict
import csv
from decimal import Decimal, localcontext
from pathlib import Path
from statistics import median

from selection import c, require
from module_profile import csvfile

LABELS = {"fc": "FC", "kv": "KV projection", "q": "Q", "o": "O", "gate_up": "Gate/Up", "down": "Down"}
W8_TOTAL = "All 26 W8 projections"


def component(module):
    if module.startswith("Fused/"):
        parts = {component(m) for m in module[6:].split(",")}
        return parts.pop() if len(parts) == 1 else "Cross-module fusion"
    if module == "fc.activation_layout":
        return "FC activation layout"
    suffix = module.rsplit(".", 1)[-1]
    if suffix in LABELS:
        return "Projection glue"
    if suffix in ("input_norm", "q_norm", "k_norm", "post_norm", "fc_norm", "final_norm"):
        return "Norm"
    if module == "head.matmul":
        return "Head MatMul"
    if module.startswith("head."):
        return "Head Top1 / output"
    if module.startswith("layout."):
        return "Layout / copy / cast"
    return {
        "kv_update": "KV cache update",
        "rope": "RoPE",
        "swiglu": "SwiGLU",
        "attention": "Attention",
        "glue": "Layer glue",
        "embedding": "Embedding",
        "setup": "Setup",
    }.get(suffix, module)


def aggregate(tasks, inventory, *, with_fc_layout=False):
    roles = {}
    for role, row in inventory.items():
        for name in (row["node"], row.get("source_node", row["node"])):
            require(roles.setdefault(name, role) == role, "ambiguous projection alias: " + name)
    require(len(inventory) == len(set(roles.values())) == 26, "missing integrated projection identities")
    expected = defaultdict(set)
    for role in roles.values():
        expected[LABELS[role.split(".")[-1]]].add(role)
    expected[W8_TOTAL] = set(roles.values())
    samples = defaultdict(lambda: defaultdict(list))
    seen = set()
    for t in tasks:
        call = int(t["call"])
        require(7 <= call < 37 and t["idx"] not in seen, "invalid or duplicate formal task")
        seen.add(t["idx"])
        role = roles.get(t["op"]) if not t["module"].startswith("NOTIFY_") else None
        value = Decimal(t["end"]) - Decimal(t["start"])
        require(value.is_finite() and value >= 0, "invalid device interval")
        name = LABELS[role.split(".")[-1]] if role else component(t["module"])
        samples[name][call].append((role, value))
        if role:
            samples[W8_TOTAL][call].append((role, value))
    require(
        sum(len(v) for name, calls in samples.items() if name != W8_TOTAL for v in calls.values())
        == len(tasks),
        "component accounting differs from task inventory",
    )
    records = []
    by_call = []
    for name in sorted(
        set(samples)
        | set(expected)
        | {"Head MatMul", "Head Top1 / output"}
        | ({"FC activation layout"} if with_fc_layout else set())
    ):
        times = []
        observed = 0
        count = 0
        for call in range(7, 37):
            group = samples[name][call]
            count += len(group)
            observed += bool(group)
            complete = bool(group)
            if name in expected:
                complete = Counter(r for r, _ in group) == Counter(expected[name])
            value = float(sum((d for _, d in group), Decimal(0)) / 1000) if complete else None
            if value is not None:
                times.append(value)
            by_call.append(
                dict(component=name, call=call, tasks=len(group), complete=complete, task_sum_ms=value)
            )
        records.append(
            dict(
                component=name,
                calls_observed=observed,
                complete_calls=len(times),
                tasks=count,
                median_ms=median(times) if len(times) == 30 else None,
                p95_ms=sorted(times)[28] if len(times) == 30 else None,
            )
        )
    return records, by_call


def summarize(out, modules_path, gear, inventory, formal):
    summary = c.read(c.verify(modules_path))["result"]
    c.verify_tree(summary["files"])
    directory = Path(modules_path["path"]).parent
    with (directory / "timed-tasks.csv").open(newline="") as file, localcontext() as ctx:
        ctx.prec = 60
        rows = list(csv.DictReader(file))
        require(len(rows) == summary["timed_tasks"], "different timed task set")
        parts, calls = aggregate(rows, inventory, with_fc_layout=gear == 64)
    csvfile(out / "components.csv", parts, list(parts[0]))
    csvfile(out / "component-calls.csv", calls, list(calls[0]))
    b, o = (formal["variants"][mode] for mode in ("baseline", "optimized"))
    md = [
        f"# C{gear} 全部 W8 投影的完整 Draft",
        "",
        f"Gate/Down 对照与全 W8 的正式 Execute+Sync 中位数分别为 {b['median_ms']:.6f} 和 {o['median_ms']:.6f} ms。"
        f"P95 分别为 {b['p95_ms']:.6f} 和 {o['p95_ms']:.6f} ms。每边 90 个无插桩样本。",
        "",
        f"设备归属为 `{summary['scope']}`。按正式 30 个窗口分别汇总各类任务的区间时长，再求 Median 和 P95。"
        "流水和 Notify 等待可以重叠。各行之和不能用于还原墙钟时间。全 W8 行为单独的小计。",
        "",
        "| 模块 | Median（ms） | P95（ms） | 完整调用 |",
        "|---|---:|---:|---:|",
    ]
    print(
        f"C{gear} per-Execute device task sums (ms), {summary['scope']}; "
        f"unmapped {summary['unmapped']}/{summary['timed_tasks']}",
        flush=True,
    )
    print("  Component                      Median       P95     complete", flush=True)
    for row in parts:
        a = row["median_ms"]
        b95 = row["p95_ms"]
        n = row["complete_calls"]
        label = row["component"]
        a, b95 = (f"{a:.4f}", f"{b95:.4f}") if a is not None else ("—", "—")
        print(f"  {label:28s} {a:>9s} {b95:>9s}      {n}/30", flush=True)
        md.append(f"| {label} | {a} | {b95} | {n}/30 |")
    print("  Layer                          Median       P95     observed", flush=True)
    md.extend(["", "| 层 | Median（ms） | P95（ms） | 已观测调用 |", "|---|---:|---:|---:|"])
    for layer in summary["layers"]:
        name = layer["module"]
        calls = layer["calls_observed"]
        p50 = layer["p50_task_sum_ms"]
        p95 = layer["p95_task_sum_ms"]
        lo, hi = (f"{p50:.4f}", f"{p95:.4f}") if p50 is not None else ("—", "—")
        print(f"  {name:28s} {lo:>9s} {hi:>9s}      {calls}/30", flush=True)
        md.append(f"| {name} | {lo} | {hi} | {calls}/30 |")
    md.extend(
        [
            "",
            "各层与层内模块的明细分别在下方 CSV 中。",
            "",
            f"- [五层汇总]({directory/'layers.csv'})",
            f"- [逐模块]({directory/'module-latency.csv'})",
            f"- [逐调用]({directory/'module-calls.csv'})",
            f"- [原始任务归属]({directory/'task-ownership.csv'})",
        ]
    )
    (out / "summary.md").write_text("\n".join(md) + "\n")
    return dict(
        components=parts,
        layers=summary["layers"],
        attribution=summary["scope"],
        strict=summary["strict"],
        unmapped=summary["unmapped"],
        timed_tasks=summary["timed_tasks"],
        module_report=modules_path,
        files={
            p.name: c.identity(p)
            for p in (
                out / "components.csv",
                out / "component-calls.csv",
                out / "summary.md",
                directory / "module-latency.csv",
                directory / "layers.csv",
                directory / "module-calls.csv",
                directory / "timed-tasks.csv",
                directory / "task-ownership.csv",
            )
        },
    )
