"""Paired host timings, complete FP16 checks, and device-task attribution for the Gate/Up pair."""

from collections import defaultdict
import csv
from decimal import Decimal
import math
from pathlib import Path
import re
from statistics import mean, median
import subprocess

import numpy as np

from pair_graph import K, N, MM, DQ, OP, c, require
from pair_reference import ints, matrix_view, read_case
from run_prefill_head import capture
from full_replay import prepare_case
from c_profile_graph import read_om_json
from diagnose_calls import export_csv
from snapshot_profiles import check_application


WEIGHT_BYTES = 2 * K * N
PEAK = 204.8


def stats(values):
    a = sorted(values)
    require(a and all(math.isfinite(x) and x > 0 for x in a), "invalid duration")
    return dict(n=len(a), p50=median(a), p95=a[math.ceil(0.95 * len(a)) - 1], mean=mean(a))


def validate_two_rounds(d, ref):
    rows = ref["rows"]
    result = {}
    for b in ("gate", "up"):
        src = ref["weights"][b]
        layout = ref["math"]["branches"][b]["layout"]
        w = matrix_view(np.fromfile(c.verify(src["weight"]), np.int8), layout)
        sw = np.fromfile(c.verify(src["scale"]), "<f4")[0]
        errs = {}
        for name, case in ref["cases"].items():
            q, scale, y = read_case(case, rows)
            e = 0
            for start in range(0, N, 512):
                a = ints(q, w[:, start : start + 512])
                with np.errstate(over="ignore", invalid="ignore"):
                    interim = (a.astype("<f4") * sw).astype("<f2")
                    pred = (interim * scale.astype("<f2")).astype("<f2").view("<u2")
                e += int(np.count_nonzero(pred != y[b][:, start : start + 512]))
            errs[name] = e
        result[b] = dict(layout=layout, weight_scale_bits=f"{int(sw.view('<u4')):08x}", errors=errs)
    print(
        f"M{rows} two-round CPU oracle, full outputs", {k: v["errors"] for k, v in result.items()}, flush=True
    )
    c.write(d / "two-rounds.json", result)
    return result


def compiled_inventory(model):
    names = {}
    for v in read_om_json(c.verify(model["dump"])):
        if v["type"] not in (DQ, MM, OP):
            continue
        a = {x["key"]: x.get("value") or {} for x in v["attributes"]}
        entry = dict(
            type=v["type"],
            engine=a.get("_ge_attr_op_kernel_lib_name", {}).get("s"),
            vendor=a.get("custom_op_file_path", {}).get("s"),
            blocks=a.get("tvm_blockdim", {}).get("i"),
            kernel=a.get("_kernel_bin_id", {}).get("s"),
        )
        require(
            v["name"] not in names and entry["engine"] == "AIcoreEngine",
            "wrong engine or duplicate node: " + v["name"],
        )
        if v["type"] == DQ:
            require(
                entry["vendor"] and Path(entry["vendor"]).name == "customize_dynamic",
                "different DynamicQuant provider",
            )
        names[v["name"]] = entry
    return names


def smoke(d, mode, model, ref, runner, env):
    require(set(model["abi"]["outputs"][i]["name"] for i in range(2)) == {"gate", "up"}, "unexpected outputs")
    mismatches = {}
    receipts = {}
    for case, x in ref["x"].items():
        root = d / case
        outs, updates = capture(root, runner, model, {"x": x}, env)
        require(not updates, "input mutation")
        receipts[case] = c.identity(root / "case/execution.json")
        for k in ("gate", "up"):
            old = ref["cases"][case][k]
            if outs[k]["sha256"] == old["sha256"]:
                continue
            a = np.fromfile(c.verify(old), "<u2")
            b = np.fromfile(c.verify(outs[k]), "<u2")
            require(a.shape == b.shape, "different output lengths")
            ids = np.flatnonzero(a != b)
            j = int(ids[0])
            mismatches[case + "/" + k] = dict(count=len(ids), first=j, expected=int(a[j]), actual=int(b[j]))
    ok = not mismatches
    print(
        f"M{ref['rows']} {mode} two FP16 outputs, three inputs, two poisons: "
        + ("bitwise PASS" if ok else str(mismatches)),
        flush=True,
    )
    return dict(ok=ok, mismatches=mismatches, executions=receipts)


def execute(d, model, ref, runner, workspace, env, cli=None):
    case = d / "case"
    prepare_case(
        case,
        model["abi"],
        {"files": {"x": ref["x"]["real"]}},
        {b: ref["cases"]["real"][b] for b in ("gate", "up")},
    )
    args = list(
        map(str, [runner, 0, c.verify(model["om"]), case, case / "inputs", workspace, "huge-first", "normal"])
    )
    if cli:
        args = [cli["alias"], *cli["flags"], "--output=" + str(d / "profiler"), *args]
    c.write(d / "command.json", args)
    log = d / "runner.log"
    with log.open("xb") as f:
        rc = subprocess.run(args, stdout=f, stderr=subprocess.STDOUT, cwd=d, env=env).returncode
    receipt = case / "execution.json"
    check_application(rc, receipt.is_file(), log.read_text(errors="replace"))
    e = c.read(receipt)
    t = e["timing"]
    require(
        e["status"] == "PASS"
        and not e["cpu_fallback"]
        and all(e[k] for k in ("guards", "readonly", "repeat"))
        and e["allocation"]["golden_outputs"] == 2
        and e["allocation"]["workspace_bytes"] == workspace
        and t["protocol"] == "continuous-v1"
        and t["warmup"] == 5
        and t["repetitions"] == 30
        and t["timed_tail_checked"]
        and len(t["execute_sync"]["samples_ms"]) == 30,
        "incomplete checked timing",
    )
    return dict(
        raw_us=[1000 * x for x in t["execute_sync"]["samples_ms"]], execution=c.identity(receipt), root=str(d)
    )


def paired(d, models, ref, runner, cap, envs):
    modes = sorted(set(models) - {"native"})
    runs = {k: [] for k in models}
    # Two flanking controls in each round allow for drift during the candidate sweep.
    for i, order in enumerate((modes, list(reversed(modes)), modes[1:] + modes[:1])):
        for j, mode in enumerate(["native", *order, "native"]):
            run = execute(d / f"r{i + 1}-{j}-{mode}", models[mode], ref, runner, cap, envs[mode])
            runs[mode].append(dict(round=i, **run))
    result = {}
    for name in models:
        series = stats([x for run in runs[name] for x in run["raw_us"]])
        ratios = []
        for i in range(3):

            def midpoint(k):
                return median(x for run in runs[k] if run["round"] == i for x in run["raw_us"])

            ratios.append(midpoint("native") / midpoint(name))
        result[name] = dict(stats=series, ratios=ratios, runs=runs[name])
    base = result["native"]["stats"]
    viable = ["native"]
    for name, v in result.items():
        z = v["stats"]
        if name != "native" and min(v["ratios"]) > 1 and z["p95"] <= base["p95"]:
            viable.append(name)
        print(
            f"M{ref['rows']} {name} Execute+Sync Median/P95 {z['p50']:.2f}/{z['p95']:.2f} us, "
            f"{base['p50'] / z['p50']:.4f}x, rounds {[round(x, 4) for x in v['ratios']]}",
            flush=True,
        )
    selected = min(viable, key=lambda name: result[name]["stats"]["p50"])
    return dict(modes=result, selected=selected)


def device(path, inventory, rows):
    # One OM/process. Each named task occurs once in every checked execution.
    # Validate window membership across streams before aggregating the timed 30 calls.
    series = defaultdict(list)
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for raw in csv.DictReader(stream):
            r = {re.sub("[^a-z0-9]", "", k.lower()): v for k, v in raw.items() if k}
            name = (r.get("opname") or "").rsplit("/", 1)[-1]
            typ = r.get("optype", "")
            if typ not in (MM, OP, DQ, "Transpose", "TransposeD", "TransData") and name not in inventory:
                continue
            start = Decimal(r["taskstarttimeus"].strip())
            duration = Decimal(r["taskdurationus"].strip())
            require(start.is_finite() and duration.is_finite() and duration > 0, "bad profiler row")
            metrics = {}
            for k in (
                "aicoretimeus",
                "mactimeus",
                "mte1timeus",
                "mte2timeus",
                "mte3timeus",
                "vectimeus",
                "scalartimeus",
            ):
                try:
                    value = float(r[k])
                    if math.isfinite(value) and value >= 0:
                        metrics[k] = value
                except (ValueError, KeyError):
                    pass
            series[(r.get("modelid", ""), name, typ)].append((start, start + duration, metrics))
    modelids = {
        mid for (mid, n, t) in series if n in inventory and inventory[n]["type"] == t and t in (MM, OP)
    }
    require(len(modelids) == 1, "unresolved profiled model")
    (mid,) = modelids
    sets = {n: t for m, n, t in series if m == mid and t in (MM, OP, DQ)}
    require(sets == {n: v["type"] for n, v in inventory.items()}, "profiler/compiled node inventory mismatch")
    selected = [(t, sorted(v, key=lambda e: e[0])) for (m, n, t), v in series.items() if m == mid]
    require(
        all(len(v) == 38 for _, v in selected),
        "expected 2 correctness + 5 warmup + 30 measured + 1 postcheck calls",
    )
    spans = []
    cubes = []
    quant = []
    prep = []
    pipes = defaultdict(list)
    prev = None
    for i in range(38):
        visits = [(t, *v[i]) for t, v in selected]
        lo = min(v[1] for v in visits)
        hi = max(v[2] for v in visits)
        require(prev is None or prev <= lo, "overlapping adjacent model executions")
        prev = hi
        if not 7 <= i < 37:
            continue
        mm = [v for v in visits if v[0] in (MM, OP)]
        cubes.append(sum(float(b - a) for _, a, b, _ in mm))
        spans.append(float(hi - lo))
        quant.append(sum(float(b - a) for t, a, b, _ in visits if t == DQ))
        prep.append(sum(float(b - a) for t, a, b, _ in visits if t not in (MM, OP, DQ)))
        for k in set.intersection(*(set(v[3]) for v in mm)):
            pipes[k].append(sum(v[3][k] for v in mm))
    core = stats(cubes)
    span = stats(spans)
    return dict(
        core_us=core,
        span_us=span,
        dq_us=median(quant),
        layout_us=median(prep),
        logical_kernel_GB_s=WEIGHT_BYTES / (core["p50"] * 1000),
        logical_span_GB_s=WEIGHT_BYTES / (span["p50"] * 1000),
        matched=30,
        pipe_sums_us={k: median(v) for k, v in pipes.items() if len(v) == 30},
        csv=c.identity(path),
    )


def profile(d, mode, model, inv, ref, runner, cap, env, cli):
    run = execute(d, model, ref, runner, cap, env, cli)
    export_csv(d, cli, env)
    (path,) = (d / "profiler").rglob("op_summary*.csv")
    data = device(path, inv, ref["rows"])
    data["execution"] = run["execution"]
    print(
        f"M{ref['rows']} {mode} device projection tasks {data['core_us']['p50']:.2f} us, "
        f"DQ/prep→outputs span {data['span_us']['p50']:.2f} us, {data['logical_kernel_GB_s']:.2f} logical GB/s, "
        f"{100 * data['logical_kernel_GB_s'] / PEAK:.1f}% weight-only roof; matched 30/30 {data['pipe_sums_us']}",
        flush=True,
    )
    return data
