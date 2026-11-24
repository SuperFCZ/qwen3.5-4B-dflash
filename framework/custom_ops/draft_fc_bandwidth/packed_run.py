"""Retest C64 against the sealed, numerically correct A1 OM. Charge activation layout to the caller."""

import csv
from decimal import Decimal
import math
import os
from pathlib import Path
import shutil
from statistics import median

import fc_panel as f
import packed_panel as pp
from fc_runtime import execute, profile
from experiment import Stages
from run_q import run_path
from run_gate import pair_statistics
from fresh_run import environment
from fresh_dso import seal
from air import protobuf_classes

M = 64


def admit(out, old):
    st = Stages(out, old)
    src = st.completed("source")["result"]
    exe = st.completed("runner")["result"]
    pkg = st.completed("build-" + f.LOCAL)["result"]
    st.completed("atc-c64-" + f.CONTROL)
    model = st.completed("atc-c64-" + f.LOCAL)
    refs = {
        label: st.completed("reference-c64-" + label)["result"]["files"] for label in ("features", "stress")
    }
    refs["boundaries"] = st.completed("boundaries-c64")["result"]
    sm = st.completed("smoke-c64")
    b = st.completed("benchmark-c64")
    original = f.c.read(old / "result.json")
    f.require(
        original["status"] in ("MEASURED", "MEASURED_AND_PROFILED")
        and original["measurements"]["64"]["report"] == st.refs["benchmark-c64"]
        and model["contract"]["pkg"] == pkg
        and model["contract"]["source"] == st.refs["source"]
        and sm["contract"]["runner"] == exe
        and sm["contract"]["cases"] == refs
        and sm["contract"]["models"] == [st.refs["atc-c64-" + k] for k in (f.CONTROL, f.LOCAL)]
        and b["contract"]["smoke"] == st.refs["smoke-c64"],
        "different measured C64 FC controls",
    )
    rows = b["result"]["runs"]
    for row in rows:
        e = f.c.read(f.c.verify(row["execution"]))
        f.require(
            e["status"] == "PASS"
            and e["golden_match"]
            and e["sync_completed"]
            and row["validation"]["timing"]["samples_ms"] == e["timing"]["execute_sync"]["samples_ms"],
            "changed prior C64 execution",
        )
    stats = pair_statistics(
        rows, f.LOCAL, f.CONTROL, flops=2 * M * f.K * f.N, logical_bytes=f.logical_bytes(M)
    )
    f.require(
        b["result"]["statistics"][f.CONTROL + "__" + f.LOCAL] == stats
        and stats["decision"] == "GO"
        and b["result"]["selected"] == f.LOCAL,
        "C64 requires a measured/admitted A1 control",
    )
    f.c.verify(exe)
    f.c.verify_opp(pkg["opp"])
    f.c.verify(model["result"]["om"])
    f.c.verify_tree(src["weights"])
    for x in refs.values():
        f.c.verify_tree(x)
        f.require(all(x[k] == v for k, v in src["weights"].items()), "crossed C64 weight/scale references")
    return dict(source=src, cases=refs, runner=exe, package=pkg, control=model["result"], receipts=st.refs)


def charged_span(folder, model, row):
    # Device interval from first verified layout task to the end of its dependent FC (includes any gap).
    with (folder / "timed-tasks.csv").open(newline="") as fobj:
        tasks = list(csv.DictReader(fobj))
    per = []
    prep = set(model.get("prep_nodes", []))
    metadata = {"Reshape", "Squeeze", "ExpandDims", "Flatten"}
    essential = {n for n in prep if model.get("compute_names", {}).get(n) not in metadata}
    fc_name = model["selected"]["node"]
    for call in range(7, 37):
        events = [r for r in tasks if int(r["call"]) == call and not r["module"].startswith("NOTIFY_")]
        core = [r for r in events if r["op"] == fc_name and r["module"] == "fc"]
        transforms = [r for r in events if r["op"] in prep]
        if (
            len(core) != 1
            or (prep and not transforms)
            or not essential.issubset({r["op"] for r in transforms})
        ):
            return dict(status="UNRESOLVED", reason="missing or ambiguous packing/FC task", call=call)
        start, end = (Decimal(core[0][k]) for k in ("start", "end"))
        if any(Decimal(t["end"]) > start for t in transforms):
            return dict(status="UNRESOLVED", reason="input producer/consumer time conflict", call=call)
        first = min([start] + [Decimal(t["start"]) for t in transforms])
        work = sum((Decimal(t["end"]) - Decimal(t["start"]) for t in transforms), Decimal(0))
        per.append(
            dict(call=call, layout_tasks=len(transforms), layout_us=float(work), span_us=float(end - first))
        )
    spans = [x["span_us"] for x in per]
    p50 = median(spans)
    return dict(
        status=row["attribution"],
        calls=per,
        median_us=p50,
        p95_us=sorted(spans)[math.ceil(0.95 * len(spans)) - 1],
        pack_task_median_us=median(x["layout_us"] for x in per),
        logical_gbps=f.logical_bytes(M) / p50 / 1000,
        floor_ratio=p50 / (f.logical_bytes(M) / 204800),
    )


def run(out, args, state):
    runs = Path(os.environ["DFLASH_RUN_ROOT"]) / "runs"
    old = run_path(runs, args.packed_a, "fc-bandwidth-")
    prior = run_path(runs, args.resume, "fc-bandwidth-") if args.resume else None
    stages = Stages(out, prior)

    def stage(name, key, action):
        state["phase"] = name
        return stages.run(name, key, action)

    archive = admit(out, old)
    control = f.LOCAL
    runner = archive["runner"]
    cases = archive["cases"]
    print("Reusing measured C64 FC A1 control", archive["control"]["om"]["sha256"], flush=True)
    cann = Path(os.environ["CANN_ROOT"]).resolve(strict=True)
    tools = {n: f.c.identity(shutil.which(n)) for n in ("atc", "msopgen", "cmake")}
    shared = environment(cann, tools)
    atc = [
        str(cann / "bin/atc"),
        "--mode=0",
        "--framework=1",
        "--model=INPUT",
        "--output=OUTPUT",
        "--soc_version=Ascend310P3",
        "--precision_mode=must_keep_origin_dtype",
        "--deterministic=0",
    ]
    types = protobuf_classes((out / "cann-ge.desc").read_bytes())
    models = {control: archive["control"]}
    pkgs = {control: archive["package"]}
    envs = {control: environment(cann, tools, pkgs[control]["opp"]["install"])}
    dsos = {}
    root_key = dict(archive=archive["receipts"]["benchmark-c64"], runner=runner, inputs=cases)
    state.update(
        parent_run=str(old),
        control=archive["control"]["om"],
        roofline_us=f.logical_bytes(M) / 204800,
        comparisons={},
        profiles={},
    )

    def call(d, m, data, timed, i):
        f.c.verify_opp(pkgs[m]["opp"])
        return execute(d, M, runner, models[m], envs[m], dsos[m], data, timed, i)

    def smoke(d, m):
        rows = [dict(call(d / k, m, x, False, -1), case=k) for k, x in cases.items()]
        print("C64 bitwise + guards PASS", m, "(163,840 outputs, three cases, two poisons each)", flush=True)
        return rows

    def compare(d, a, b):
        rows = []
        for i in range(3):
            order = (b, a) if i == 1 else (a, b)
            print(f"C64 round {i+1}: " + " → ".join(order), flush=True)
            for m in order:
                rows.append(call(d / str(i) / m, m, cases["features"], True, i))
        return dict(
            rows=rows,
            statistics=pair_statistics(rows, b, a, flops=2 * M * f.K * f.N, logical_bytes=f.logical_bytes(M)),
        )

    def measured_pair(a, b):
        key = a + "__" + b
        stat = stage(
            "benchmark-" + key,
            dict(
                root_key,
                smoke=[stages.refs["smoke-" + m] for m in (a, b)],
                stats=f.c.sha(f.OPS / "draft_gate_bandwidth/run_gate.py"),
            ),
            lambda d: compare(d, a, b),
        )["statistics"]
        state["comparisons"][key] = stat
        x, y = (stat["variants"][m] for m in (a, b))
        print(
            f"C64 FC including input preparation, Median/P95 ms ({a} → {b}): {x['median_ms']:.6f}/{x['p95_ms']:.6f} → "
            f"{y['median_ms']:.6f}/{y['p95_ms']:.6f}, {stat['speedup']:.4f}x {stat['decision']}",
            flush=True,
        )
        return stat["decision"] == "GO"

    def prof(m, cli):
        def measure(d):
            r = profile(d, M, runner, models[m], envs[m], dsos[m], cases["features"], cli)
            return dict(r, charged=charged_span(d, models[m], r))

        r = stage(
            "profile-" + m,
            dict(
                root_key,
                model=models[m]["om"],
                cli=cli["file"],
                flags=cli["flags"],
                code=f.c.sha(Path(__file__)),
            ),
            measure,
        )
        state["profiles"][m] = r
        z = r["charged"]
        print(
            f"C64 {m} device FC alone {1000*r['kernel']['p50_task_sum_ms']:.2f} us; "
            + (
                f"pack task sum {z['pack_task_median_us']:.2f} us, pack→FC device span Median/P95 {z['median_us']:.2f}/{z['p95_us']:.2f} us, "
                f"{z['logical_gbps']:.2f} logical GB/s, {z['floor_ratio']:.3f}x reference, {z['status']}"
                if z["status"] != "UNRESOLVED"
                else f"pack→FC device span unresolved ({z['reason']})"
            ),
            flush=True,
        )
        print(
            "  FC matched counters",
            r["pipeline"]["joined"],
            {
                k: round(v, 2)
                for k, v in r["pipeline"]["medians"].items()
                if "time" in k.lower() and "us" in k.lower()
            },
            flush=True,
        )

    widths = (64, 320) if args.width == "both" else (int(args.width),)
    passing = []
    cli = None
    for width in widths:
        mode = pp.MODES[width]
        pkg = stage(
            "build-" + mode,
            dict(
                tools=tools,
                device=f.q.sha(pp.device_source().encode()),
                host=f.q.sha(pp.host_source().encode()),
                scheduler=f.q.sha(pp.scheduler_source().encode()),
                header=f.q.sha(pp.header(width)),
                builder=f.c.sha(f.HERE / "packed_panel.py"),
            ),
            lambda d: pp.build(d, cann, shared, width),
        )
        f.c.verify_opp(pkg["opp"])
        pkgs[mode] = pkg
        envs[mode] = environment(cann, tools, pkg["opp"]["install"])
        model = stage(
            "atc-" + mode,
            dict(
                build=stages.refs["build-" + mode],
                control=archive["receipts"]["atc-c64-" + control],
                source=archive["receipts"]["source"],
                atc=tools["atc"],
                width=width,
            ),
            lambda d: pp.compile_model(d, types, pkg, envs[mode], atc, width, archive),
        )
        f.c.verify(model["om"])
        models[mode] = model
        print(
            f"C64 {mode}: ATC 8 blocks, N{width}/K{pp.QUERY[width][0]}, charged layout",
            model["prep_nodes"],
            flush=True,
        )
        if args.compile_only:
            continue
        for m in (control, mode):
            if m in dsos:
                continue
            dsos[m] = seal(out / ("dso-" + m), pkgs[m], runner, envs[m])
            stage("smoke-" + m, dict(root_key, model=models[m]["om"]), lambda d: smoke(d, m))
        if measured_pair(control, mode):
            passing.append(mode)
        if not args.no_profile:
            if cli is None:
                from panel_profiler import bind_msprof

                cli = bind_msprof(out, cann, envs[control])
                prof(control, cli)
            prof(mode, cli)
    if args.compile_only:
        return "COMPILED_NOT_EXECUTED"
    picked = passing[0] if passing else control
    if len(passing) == 2 and measured_pair(*passing):
        picked = passing[1]
    state["selected_c64"] = picked
    print("Selected C64 (preparation included):", picked, flush=True)
    return "MEASURED" if args.no_profile else "MEASURED_AND_PROFILED"
