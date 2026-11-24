"""Read the measured full control, its frozen outputs and the winning single-op receipts."""

from pathlib import Path

from selection import FULL, FULL_OM, LATENCY, SINGLE, c, require, selections, sites
from experiment import Stages
from profile_archive import read_gear
from bench import embedded, read_om_json, selected_image
from run_gate import pair_statistics


def chosen(final, family, gear, mode, records, receipt):
    if family in ("q", "o"):
        pick = final["selected"]
        accepted = final["statistics"]
        require(final["benchmark_report"] == records["pair_ref"], "different Q/O benchmark")
    elif family == "kv":
        view = final["gears"][str(gear)]
        pick = view["selected"]
        key = "raw-nz-brcb__" + mode
        require(view["pairs"][key]["report"] == records["pair_ref"], "different KV benchmark")
        accepted = records["stats"]
        require(view["pairs"][key]["decision"] == accepted["decision"], "KV decision drift")
    elif gear == 16:
        view = final["measurements"]["16"]
        pick = view["selected"]
        require(view["report"] == records["pair_ref"], "different FC benchmark")
        accepted = view["stats"]["raw-nz-brcb__" + mode]
    else:
        pick = final["selected_c64"]
        accepted = final["comparisons"][receipt.pair.removeprefix("benchmark-")]
    require(
        pick == mode and accepted == records["stats"] and accepted["decision"] == "GO",
        "missing measured winner: " + mode,
    )


def load_single(out, runs, family, gear, spec):
    pin = SINGLE[gear][family]
    root = runs / pin.root
    final_file = c.identity(root / "result.json")
    require(final_file["sha256"] == pin.result, "different sealed result: " + pin.root)
    final = c.read(root / "result.json")
    require(final["status"] == "MEASURED_AND_PROFILED", "unfinished single projection")
    stages = Stages(out, root)
    pkg = stages.completed(pin.build)["result"]
    rec = stages.completed(pin.atc)
    model = rec["result"]
    ref = stages.completed(pin.pair)
    stats = ref["result"]["statistics"]
    if "decision" not in stats:
        stats = stats["raw-nz-brcb__" + spec.mode]
    chosen(final, family, gear, spec.mode, dict(stats=stats, pair_ref=stages.refs[pin.pair]), pin)
    rows = ref["result"].get("runs", ref["result"].get("rows"))
    (control,) = set(stats["variants"]) - {spec.mode}
    for row in rows:
        e = c.read(c.verify(row["execution"]))
        require(
            e["status"] == "PASS"
            and e["golden_match"]
            and e["sync_completed"]
            and not e["cpu_fallback"]
            and row["validation"]["timing"]["samples_ms"] == e["timing"]["execute_sync"]["samples_ms"],
            "single-op execution/samples changed",
        )
    measured = pair_statistics(
        rows,
        spec.mode,
        control,
        flops=2 * spec.m * spec.k * spec.n,
        logical_bytes=spec.k * spec.n + 2 * (spec.k // 128) * spec.n + 2 * spec.m * (spec.k + spec.n),
    )
    require(
        measured == stats
        and model["op"] == spec.op
        and model["mode"] == spec.mode
        and (rec["contract"].get("pkg") == pkg or rec["contract"].get("build") == stages.refs[pin.build]),
        "mismatched ATC, package or paired decision: " + spec.mode,
    )
    c.verify_opp(pkg["opp"])
    c.verify(pkg["source"])
    c.verify(pkg["host_source"])
    om = c.verify(model["om"])
    require(model["om"]["sha256"] == pin.om, "different single-op OM")
    (node,) = (n for n in read_om_json(c.verify(model["dump"])) if n["type"] == spec.op)
    image = embedded(om, [selected_image(node, 8, spec.op)])[node["name"]]
    require(
        image == model["selected"] and [a["sha256"] for a in image["executable_sections"]] == [pin.text],
        "different single-op machine code",
    )
    if "constants" in model:
        digests = [x["sha256"] for x in model["constants"]]
    elif "packing" in model:
        digests = [x["sha256"] for x in model["packing"]["constants"]]
    else:
        digests = [model["weight_sha256"], model["scale_sha256"]]
    return dict(
        device=pkg["source"],
        host=pkg["host_source"],
        query=model["query"],
        image=image,
        layer0_constants=digests,
        mode=spec.mode,
        evidence=dict(
            final=final_file,
            om=model["om"],
            build=stages.refs[pin.build],
            atc=stages.refs[pin.atc],
            benchmark=stages.refs[pin.pair],
        ),
    )


def admit(out, runs, cann, gears):
    old = runs / FULL
    st = Stages(out, old)
    latency = c.identity(old / "latency.json")
    require(latency["sha256"] == LATENCY, "different UFqlys latency receipt")
    sealed = c.read(latency["path"])
    index = c.read(c.verify(st.completed("checkpoint-graphs")["result"]))
    private = st.completed("private-opp")
    require(private["contract"]["cann"] == str(cann), "different CANN installation")
    pkg = private["result"]
    c.verify_opp(pkg["opp"])
    c.verify_tree(pkg["sources"])
    runner = st.completed("runner")["result"]
    c.verify(runner)
    external = {name: index["target"][name]["fp16"] for name in ("embedding", "head")}
    c.verify_tree(external)
    selected_cache = {}
    cases = {}
    for gear in gears:
        key = str(gear)
        graph = index["gears"][key]
        case, historical = read_gear(st, graph, gear, runner)
        require(
            historical == sealed[key] and case["model"]["om"]["sha256"] == FULL_OM[gear],
            "different measured full Draft control",
        )
        c.verify(graph["optimized"])
        c.verify_tree(case["reference"])
        specs = selections(gear)
        require(set(graph["projections"]) == set(sites(specs)), "full projection inventory differs")
        assets = {}
        for kind in ("gate_up", "down"):
            spec = specs[kind]
            binding = pkg["sources"][spec.op]
            tag = "x".join(map(str, spec.shape))
            assets[kind] = dict(binding, query=case["model"]["plans"][tag], mode=spec.mode)
        for kind, pin in SINGLE[gear].items():
            if pin not in selected_cache:
                selected_cache[pin] = load_single(out, runs, kind, gear, specs[kind])
            assets[kind] = selected_cache[pin]
        cases[key] = dict(case, assets=assets)
    return dict(
        cases=cases,
        baseline_package=pkg,
        runner=runner,
        external=external,
        control_measurement=latency,
        control_receipts=st.refs,
    )
