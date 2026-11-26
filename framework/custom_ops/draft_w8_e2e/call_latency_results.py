"""Read same-boundary samples. Device events and host timings have separate clocks."""

import csv
import math
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
import re
from statistics import mean, median

from winners import c, require


def dist(x):
    x = sorted(x)
    require(x and all(math.isfinite(t) for t in x), "empty or nonfinite sample")
    return dict(n=len(x), mean=mean(x), p50=median(x), p95=x[math.ceil(0.95 * len(x)) - 1])


def samples(path, mode, n):
    with Path(path).open(newline="") as f:
        rows = list(csv.DictReader(f))
    gears = {}
    for gear in (16, 64):
        group = [r for r in rows if int(r["gear"]) == gear]
        expected = []
        for rep in range(-3, n):
            seq = (
                ["native"]
                if mode == "clean"
                else ["after_target", *(("hot_bare", "hot_full") if rep % 2 else ("hot_full", "hot_bare"))]
            )
            expected.extend((rep, k) for k in seq)
        require([(int(r["rep"]), r["sequence"]) for r in group] == expected, f"incomplete C{gear} rounds")
        got = {}
        for rep, r in [(int(v["rep"]), v) for v in group]:
            numeric = {k: float(v) for k, v in r.items() if k.endswith("_ms")}
            require(
                all(math.isfinite(t) for t in numeric.values()) and numeric["wall_ms"] > 0,
                "invalid host duration",
            )
            if mode == "events":
                require(
                    numeric["event_model_ms"] > 0
                    and (
                        r["sequence"] == "hot_bare"
                        or min(numeric["event_input_ms"], numeric["event_output_ms"]) >= 0
                    ),
                    "missing device interval",
                )
            if rep >= 0:
                got.setdefault(r["sequence"], []).append(numeric)
        gears[str(gear)] = got
    require(
        sum(len(g) for g in gears.values()) > 0 and len(rows) == 2 * (n + 3) * (1 if mode == "clean" else 3),
        "unexpected sample inventory",
    )
    return gears


def compare(archives, clean, repeated, isolated, events, memory):
    evidence = {}
    for gear, r in repeated.items():
        seq = {k: dist(t["wall_ms"] for t in v) for k, v in r.items()}
        full, hot, bare = (r[k] for k in ("after_target", "hot_full", "hot_bare"))

        def paired(a, b, field="wall_ms"):
            return dist(x[field] - y[field] for x, y in zip(a, b, strict=True))

        cold_dispatch = dist(t["native_call_ms"] - t["wall_ms"] for t in full)
        final_stock = memory["stock-end"][gear]["hot_bare"]
        reference = (median(t["wall_ms"] for t in bare) + median(t["wall_ms"] for t in final_stock)) / 2
        controls = {
            name: {side: dist(v["wall_ms"] for v in data[gear][side]) for side in r}
            for name, data in memory.items()
        }
        delta = dict(
            predecessor=paired(full, hot),
            hot_io=paired(hot, bare),
            resident_minus_isolated_ms=seq["hot_bare"]["mean"] - isolated[gear]["mean"],
            outer_dispatch=cold_dispatch,
        )
        independent = dist(v["wall_ms"] for v in clean[gear]["native"])
        print(
            f"C{gear} clean real {independent['p50']:.3f} | after Target {seq['after_target']['p50']:.3f} "
            f"| repeat + IO {seq['hot_full']['p50']:.3f} | repeat execute+sync {seq['hot_bare']['p50']:.3f} "
            f"| isolated same inputs {isolated[gear]['p50']:.3f} ms (P50)",
            flush=True,
        )
        print(
            f"C{gear} mean deltas: Target->Draft predecessor {delta['predecessor']['mean']:+.3f}, "
            f"per-call IO on repeat {delta['hot_io']['mean']:+.3f}, "
            f"resident bare - isolated {delta['resident_minus_isolated_ms']:+.3f}, "
            f"outer dispatch {cold_dispatch['mean']:+.3f} ms",
            flush=True,
        )
        api = {k: dist(v[k] for v in full) for k in full[0] if k.startswith("host_")}
        print(
            f"C{gear} after Target host API occupancy P50 ms, upload/submit/download/sync "
            + "/".join(
                f"{api[k]['p50']:.4f}"
                for k in ("host_upload_ms", "host_submit_ms", "host_download_ms", "host_sync_ms")
            ),
            flush=True,
        )
        split = {}
        if events:
            for side, values in events[gear].items():
                split[side] = {
                    k: dist(v[k] for v in values) for k in values[0] if all(v[k] >= 0 for v in values)
                }
            for side in ("after_target", "hot_full", "hot_bare"):
                v = split[side]
                print(
                    f"C{gear} {side} device event ms H2D/model/D2H "
                    + "/".join(
                        f"{v[k]['p50']:.4f}" if k in v else "-"
                        for k in ("event_input_ms", "event_model_ms", "event_output_ms")
                    )
                    + f", marked wall {v['wall_ms']['p50']:.3f}",
                    flush=True,
                )
        print(
            f"C{gear} allocation controls (resident bare P50 ms, stock bracket "
            f"{seq['hot_bare']['p50']:.3f}/{controls['stock-end']['hot_bare']['p50']:.3f}): "
            + ", ".join(
                f"{k} {v['hot_bare']['p50']:.3f} ({v['hot_bare']['p50'] - reference:+.3f})"
                for k, v in controls.items()
                if k != "stock-end"
            ),
            flush=True,
        )
        evidence[gear] = dict(
            clean=independent,
            paired_wall=seq,
            contrasts=delta,
            marked=split,
            allocations=controls,
            isolated=isolated[gear],
            archive=archives[gear],
            host_apis=api,
        )
    return evidence


def task_contrasts(root, specs, sequences, *, single=None):
    """Identify the Model ID using its actual FC, then enclose each operator between successive FC launches."""
    from c_profile_trace import norm

    files = list(root.rglob("op_summary*.csv"))
    if len(files) != 1:
        return dict(status="UNRESOLVED", reason=f"{len(files)} op_summary tables")
    names = {name for info in specs.values() for name in info["nodes"]}
    kept = defaultdict(list)
    with files[0].open(encoding="utf-8-sig", newline="") as f:
        for raw in csv.DictReader(f):
            fields = {norm(k): v for k, v in raw.items() if k}
            name = fields.get("opname", "")
            if name not in names:
                suffixes = [n for n in names if name.endswith("/" + n)]
                if len(suffixes) != 1:
                    continue
                name = suffixes[0]
            model = fields.get("modelid", "").strip()
            if model in ("", "N/A", "-1"):
                model = fields.get("modelname", "").strip()
            try:
                start = Decimal(fields["taskstarttimeus"].strip())
                duration = float(fields["taskdurationus"])
                if start.is_finite() and math.isfinite(duration) and duration >= 0:
                    shape = re.findall(r"\d+", fields.get("outputshapes", "").strip())
                    pipe = {}
                    for k in ("mactimeus", "mte2timeus"):
                        try:
                            value = float(fields[k])
                            if math.isfinite(value) and value >= 0:
                                pipe[k] = value
                        except (KeyError, ValueError, TypeError):
                            pass
                    kept[model].append((name, fields.get("optype"), start, duration / 1000, shape, pipe))
            except (ArithmeticError, ValueError, KeyError):
                continue
    # When the CSV has no usable Model ID, the real trace still has two different
    # FC types. Validate the complete serialized C64/C16 schedule before using
    # global FC windows. Require the full vocabulary shape for an unscoped Head.
    all_rows = [r for rows in kept.values() for r in rows]
    anchors = sorted((r[2], g) for r in all_rows for g, i in specs.items() if r[:2] == i["fc"])
    expected_order = []
    if not single:
        for rep in dict.fromkeys(i for i, _ in sequences["64"]):
            for g in ("64", "16"):
                expected_order.extend(g for i, _ in sequences[g] if i == rep)
    global_ok = (
        bool(expected_order)
        and [g for _, g in anchors] == expected_order
        and len({t for t, _ in anchors}) == len(anchors)
    )
    reports = {}
    for gear, info in specs.items():
        fc, typ = info["fc"]
        seq = sequences[gear]
        possible = [
            (mid, rows) for mid, rows in kept.items() if sum(r[:2] == (fc, typ) for r in rows) == len(seq)
        ]
        scoped = len(possible) == 1 and (
            single
            or (
                possible[0][0] not in ("", "N/A", "-1")
                and not any(
                    r[:2] == other["fc"] for k, other in specs.items() if k != gear for r in possible[0][1]
                )
            )
        )
        if not scoped and not global_ok:
            reports[gear] = dict(status="UNRESOLVED", reason="no unique model with complete FC invocations")
            continue
        mid, rows = possible[0] if scoped else (None, all_rows)
        starts = sorted(r[2] for r in rows if r[:2] == (fc, typ))
        # Some older exports have no Model ID. The single-OM case still needs unique FC windows.
        if len(set(starts)) != len(seq):
            reports[gear] = dict(status="UNRESOLVED", reason="nonunique FC start")
            continue
        times = defaultdict(lambda: defaultdict(list))
        spans = defaultdict(list)
        head_counters = defaultdict(lambda: defaultdict(list))
        missing = set()
        for i, (rep, condition) in enumerate(seq):
            if rep < 0:
                continue
            end = starts[i + 1] if i + 1 < len(starts) else Decimal("Infinity")
            if not scoped:
                end = next((t for t, _ in anchors if t > starts[i]), Decimal("Infinity"))
            hits = defaultdict(list)
            head = []
            for name, kind, start, duration, shape, pipe in rows:
                if starts[i] <= start < end and name in info["nodes"]:
                    if info.get("types") and info["types"][name] != kind:
                        continue
                    if not scoped and info["nodes"][name] == "Head" and shape != ["15", "248320"]:
                        continue
                    hits[name].append(duration)
                    if info["nodes"][name] == "Head":
                        head.append((float((start - starts[i]) / 1000) + duration, pipe))
            if len(head) == 1:
                spans[condition].append(head[0][0])
                for k, val in head[0][1].items():
                    head_counters[k][condition].append(val)
            by_component = defaultdict(list)
            for name, kind in info["nodes"].items():
                by_component[kind].append(hits[name])
            if info.get("types"):
                by_component["W8"] = [hits[k] for k, kind in info["nodes"].items() if kind != "Head"]
            for kind, chunks in by_component.items():
                if all(len(v) == 1 for v in chunks):
                    times[kind][condition].append(sum(v[0] for v in chunks))
                else:
                    missing.add(kind)
        n = sum(rep >= 0 for rep, _ in seq)
        result = {
            k: {mode: dist(items) for mode, items in modes.items()}
            for k, modes in times.items()
            if k not in missing and sum(map(len, modes.values())) == n
        }
        reports[gear] = dict(
            status="COMPLETE" if set(info["nodes"].values()) | {"Head", "W8"} <= set(result) else "PARTIAL",
            model_id=mid,
            task_sums_ms=result,
            missing=sorted(missing),
            fc_to_head_span_ms={k: dist(v) for k, v in spans.items()}
            if sum(map(len, spans.values())) == n
            else {},
            head_pipeline_us={
                k: {s: dist(v) for s, v in modes.items()}
                for k, modes in head_counters.items()
                if sum(map(len, modes.values())) == n
            },
        )
    return dict(csv=c.identity(files[0]), gears=reports)
