"""Device task attribution for the optimized checkpoint graph over exactly 30 steady-state calls."""

import csv
import math
from collections import Counter, defaultdict
from decimal import localcontext
from pathlib import Path
from statistics import median
import sys

from joint import OPS, c, require

sys.path.insert(0, str(OPS / "draft_w8a16_deep_profile"))
from hotspots import choose_owners, complete, host_windows, partition, deep, trace
from attribution import recover
from c_profile_graph import read_om_json


def scopes(graph, nodes):
    source = graph["modules"].copy()
    for name, owner in list(source.items()):
        if owner == "head":
            kind = graph["node_types"][name]
            source[name] = (
                "head.matmul" if kind == "MatMul" else "head.top1" if kind == "ArgMaxV2" else "head.other"
            )
    result = {}
    for n in nodes:
        origins = []
        for a in n.get("attributes", []):
            if a.get("key") in ("_datadump_original_op_names", "_original_op_names"):
                origins.extend((a.get("value") or {}).get("list", {}).get("s", []))
        known = {source[k] for k in [n["name"], *origins] if k in source}
        unknown = [k for k in origins if k not in source]
        if len(known) == 1 and not unknown:
            result[n["name"]] = next(iter(known))
        elif known:
            result[n["name"]] = "Fused/" + ",".join(sorted(known | ({"Unmapped"} if unknown else set())))
    return source, result


def module(e, mapping):
    if e["signal"]:
        return "NOTIFY_" + e["signal"].upper()
    if e["op"] in mapping:
        return mapping[e["op"]]
    kind = trace.norm(e["kind"])
    if "transdata" in kind:
        return "layout.TransData"
    if "memcpy" in kind or kind == "cast":
        return "layout.CopyCast"
    return "Unmapped"


def union(intervals):
    total = 0
    end = None
    for lo, hi in sorted(intervals):
        total += hi - lo if end is None else max(0, hi - max(lo, end))
        end = hi if end is None else max(end, hi)
    return total


def summarize(entries, expected=()):
    groups = defaultdict(lambda: defaultdict(list))
    for e in entries:
        groups[e["module"]][e["call"]].append(e)
    rows = []
    calls = []
    for name in sorted(set(groups) | set(expected)):
        sums = []
        unions = []
        for call in range(7, 37):
            es = groups[name][call]
            duration = sum((e["hi"] - e["lo"] for e in es), 0)
            busy = union([(e["lo"], e["hi"]) for e in es])
            sums.append(float(duration) / 1000)
            unions.append(float(busy) / 1000)
            calls.append(dict(module=name, call=call, tasks=len(es), sum_ms=sums[-1], union_ms=unions[-1]))
        counts = [len(groups[name][call]) for call in range(7, 37)]
        rows.append(
            dict(
                module=name,
                calls_observed=sum(x > 0 for x in counts),
                tasks=sum(counts),
                p50_task_sum_ms=median(sums) if any(counts) else None,
                p95_task_sum_ms=sorted(sums)[math.ceil(0.95 * len(sums)) - 1] if any(counts) else None,
                mean_task_union_ms=sum(unions) / 30 if any(counts) else None,
            )
        )
    return rows, calls


def csvfile(path, rows, fields):
    with path.open("x", newline="") as f:
        w = csv.DictWriter(f, fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def analyze(folder, capture, model, graph):
    raw = Path(capture["profiler"])
    timeline, tables, diagnostics = recover(raw, capture["runner_pid"])
    c.write(folder / "windows.json", timeline)
    source, compiled = scopes(graph, read_om_json(c.verify(model["dump"])))
    _, meta = trace.csv_tables(raw)
    raw_json = deep.v2.read(Path(timeline["file"]))
    events = raw_json.get("traceEvents", []) if isinstance(raw_json, dict) else raw_json
    kinds = defaultdict(set)
    for n in read_om_json(c.verify(model["dump"])):
        kinds[n["name"]].add(n["type"])
    with localcontext() as ctx:
        ctx.prec = 60
        done, tracks, hardware = complete(events)
        windows = host_windows(done, hardware, timeline, capture["runner_pid"])
        _, _, tasks = partition(done, tracks, hardware)
        rows, scope = choose_owners(tasks, windows, timeline)
        require(
            len(tasks) == len(rows) == timeline["assignment_evidence"]["real_event_count"],
            "task inventory drift",
        )
        for r, e in zip(rows, tasks):
            name, kind, category, basis = deep.v2.label(e, meta, kinds, set())
            r.update(
                call="" if r["call"] is None else r["call"],
                op_name=name,
                op_type=kind,
                category=category,
                basis=basis,
            )
        hw, _ = deep.task_data(Path(timeline["file"]), rows, tables[0])
        timed = [e for e in hw if e["call"] in range(7, 37)]
        require(timed, "no uniquely attributed steady-state tasks")
        for e in timed:
            e["module"] = module(e, compiled)
        expected = set(source.values()) - {"setup", "head", "head.other"}
        stats, calls = summarize(timed, expected)
        layer_entries = [
            dict(e, module=e["module"].split(".")[0]) for e in timed if e["module"].startswith("layer")
        ]
        layers, _ = summarize(layer_entries, [f"layer{i}" for i in range(5)])
    fields = list(stats[0])
    csvfile(folder / "module-latency.csv", stats, fields)
    csvfile(folder / "layers.csv", layers, fields)
    csvfile(folder / "module-calls.csv", calls, ["module", "call", "tasks", "sum_ms", "union_ms"])
    csvfile(
        folder / "timed-tasks.csv",
        timed,
        ["idx", "call", "module", "op", "kind", "pid", "stream", "task_id", "start", "end", "us"],
    )
    csvfile(
        folder / "task-ownership.csv", rows, ["event_index", "call", "state", "op_name", "op_type", "basis"]
    )
    return dict(
        scope=scope,
        strict=timeline["device_assignment"],
        reasons=timeline["assignment_evidence"]["remaining_failure_reasons"],
        runner_pid=capture["runner_pid"],
        formal_calls=30,
        timed_tasks=len(timed),
        unmapped=sum(e["module"] == "Unmapped" for e in timed),
        ownership=dict(Counter(r["state"] for r in rows)),
        modules=stats,
        layers=layers,
        timeline=c.identity(Path(timeline["file"])),
        op_summary=c.identity(raw / tables[0]["file"]),
        parse_diagnostics=diagnostics,
        quantity="Device task duration sums per profiled Execute; interval unions account for overlap within each module",
    )
