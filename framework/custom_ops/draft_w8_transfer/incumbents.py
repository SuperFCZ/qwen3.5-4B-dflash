"""Recover the exact prior winners and their real activations/Goldens. Only Down needs a reference cut."""

from pathlib import Path
from recipes import CONTROL, SINGLE, gate_next, c, require, packed, fc, kv, o, q
from selection import FULL, OLD_OP
from archive import load_single
from experiment import Stages
from gate_input import check_elf, cut_reference, single_graph, DEPLOYMENT, FULL_OM
from run_gate import pair_statistics
import o_runtime, q_runtime, fc_runtime

GATE_RUN = "gate-next-n-669ScA"
GATE_RESULT = "e26a86767c00e0b582df59fa5b13efd04fa6f6926c813087690d00060b8c6d33"
GATE_OM = "5aff952b8b6d1e408299128ff0df9d32d27ccd717cd9d40f51d8519e035f9015"
GATE_ELF = "efde2ab507a0d6fd6bd1c2220cd867f524551d6f539a6b8bfa7451c0deaf2f92"
GATE_Y = (
    "f27ec0a7517763bd9a4da394406daf5d44966f6cefd718718bf960c1abce45c1",
    "30ab09c49f2d6ee06fff5f9e052e337fc84470e26180373e373c0434025d6561",
)
DOWN_TEXT = "e54d6951348154107e470606c8fde04c0cc7d1988ab19a1b5aeeeb510308fff6"
DOWN_PLAN = "a3c29fff3f62f25b78f38b4688f9a528c5b367a6bc34d0571c654da5d08745bc"
DOWN_SCOPE = "supplied real Layer0 Down reference checked in process"


def binding(r, sm, model, pkg, scope, measured, stress, receipts, constant_pins):
    key = sm["contract"]
    runner = key["runner"]
    c.verify(runner)
    cases = key["cases"] if "cases" in key else key["inputs"]
    require(
        model["abi"] == r.abi and model["mode"] == r.name and model["selected"]["block_dim"] == 8,
        "wrong single-op winner or ABI",
    )
    rows = sm["result"]
    selected = rows if isinstance(rows, dict) else {v["case"]: v for v in rows if v["mode"] == r.name}
    require(
        set(selected) == set(cases) == {measured, stress, "boundaries"}, "incomplete archived smoke cases"
    )
    for name, row in selected.items():
        e = c.read(c.verify(row["execution"]))
        d = Path(row["folder"])
        v = cases[name]
        require(
            row["mode"] == r.name
            and row["launch_authorized"]
            and not row["profiled"]
            and e["scope"] == scope
            and e["status"] == "PASS"
            and not e["cpu_fallback"]
            and e["golden_match"]
            and all(e[x] for x in ("guards", "readonly", "repeat", "sync_completed", "poison_verified"))
            and all(c.sha(d / f"y-actual-{i}.bin") == v["golden.bin"]["sha256"] for i in (0, 1))
            and all(v[n] == cases[measured][n] for n in ("w_nz.bin", "s_gn.bin")),
            "archived output/weight identity changed: " + name,
        )
    cases = dict(measured=cases[measured], stress=cases[stress], boundaries=cases["boundaries"])
    c.verify_tree(cases)
    c.verify(model["air"])
    c.verify_opp(pkg["opp"])
    m, k, n = r.shape
    require(
        all(
            row["x.bin"]["bytes"] == 2 * m * k and row["golden.bin"]["bytes"] == 2 * m * n
            for row in cases.values()
        ),
        "archived operands have wrong logical extent",
    )
    w, s = packed.packed_weights(
        *(c.verify(cases["measured"][f]).read_bytes() for f in ("w_nz.bin", "s_gn.bin")),
        n,
        k,
        r.width,
        r.panel,
    )
    require([q.sha(w), q.sha(s)] == constant_pins, "different measured checkpoint weight or scale")
    return dict(model=model, package=pkg, runner=runner, cases=cases, scope=scope, receipts=receipts)


def gate_archive(out, runs):
    root = runs / GATE_RUN
    idx = Stages(out, root)
    final = c.read(root / "result.json")
    r = CONTROL["gate"]
    require(
        c.sha(root / "result.json") == GATE_RESULT
        and final["selected"] == r.name
        and final["status"] == "MEASURED_AND_PROFILED",
        "not the measured Gate Next-N winner",
    )
    pkg = idx.completed("build-" + r.name)["result"]
    rec = idx.completed("atc-" + r.name)
    m = rec["result"]
    smoke = idx.completed("smoke-" + r.name)
    require(
        rec["contract"]["pkg"] == pkg
        and smoke["contract"]["atc"] == idx.refs["atc-" + r.name]
        and m["om"]["sha256"] == GATE_OM
        and m["selected"]["sha256"] == GATE_ELF,
        "crossed Gate build or OM",
    )
    check_elf(m, m["selected"])
    a = gate_next.CONTROL
    for left, right in ((a, r.name), (a, gate_next.EARLY), (r.name, gate_next.EARLY)):
        key = left + "__" + right
        name = "benchmark-" + key
        record = idx.completed(name)
        require(final["pairs"][key]["report"] == idx.refs[name], "unbound Gate measurement")
        if r.name in (left, right):
            require(idx.refs["smoke-" + r.name] in record["contract"]["smokes"], "Gate smoke differs")
        for row in record["result"]["runs"]:
            e = c.read(c.verify(row["execution"]))
            require(
                e["golden_match"]
                and e["sync_completed"]
                and e["timing"]["execute_sync"]["samples_ms"] == row["validation"]["timing"]["samples_ms"],
                "Gate sample drift",
            )
        stats = pair_statistics(record["result"]["runs"], right, left)
        require(
            stats == final["pairs"][key]["statistics"] == record["result"]["statistics"]
            and stats["decision"] == ("NO_GO" if left == r.name else "GO"),
            "different Gate decision",
        )
    src = idx.completed("source")["result"]
    require(src["full_om"]["sha256"] == FULL_OM and m["air"] == src["single_air"], "crossed Gate source")
    result = binding(
        r,
        smoke,
        m,
        pkg,
        gate_next.SCOPE,
        "measured",
        "stress",
        idx.refs,
        [x["sha256"] for x in src["packing"]["constants"]],
    )
    require(
        tuple(result["cases"][label]["golden.bin"]["sha256"] for label in ("measured", "stress")) == GATE_Y,
        "Gate outputs for Down changed",
    )
    return result


def archived(out, runs, key):
    if key == "gate":
        return gate_archive(out, runs)
    r = CONTROL[key]
    pin = SINGLE[r.gear][r.role]
    st = Stages(out, runs / pin.root)
    evidence = load_single(out, runs, r.role, r.gear, r.spec)
    pkg = st.completed(pin.build)["result"]
    model = st.completed(pin.atc)["result"]
    bench = st.completed(pin.pair)
    smoke_name = {"q": "smoke", "o": "smoke", "fc16": "smoke-c16", "fc64": "smoke-" + r.name}.get(
        key, f"smoke-c{r.gear}-{r.name}"
    )
    sm = st.completed(smoke_name)
    sm_ref = bench["contract"]["smoke"]
    require(
        st.refs[smoke_name] in (sm_ref if isinstance(sm_ref, list) else [sm_ref]), "different benchmark/smoke"
    )
    key_sm = sm["contract"]
    require(
        key_sm.get("model") == model["om"]
        or key_sm.get("atc") == st.refs[pin.atc]
        or st.refs[pin.atc] in key_sm.get("models", []),
        "smoke did not measure the selected OM",
    )
    labels = (
        ("measured-tokens", "dense-token-stress")
        if key == "q"
        else ("features" if r.role == "fc" else "measured", "stress")
    )
    scope = {"fc": fc_runtime.SCOPE, "q": q_runtime.REFERENCE, "o": o_runtime.REFERENCE, "kv": kv.SCOPE}[
        r.role
    ]
    return dict(
        binding(r, sm, model, pkg, scope, *labels, st.refs, evidence["layer0_constants"]),
        evidence=evidence["evidence"],
    )


def down_source(out, runs, types):
    r = CONTROL["down"]
    m, k, n = r.shape
    gate_input = gate_archive(out, runs)
    st = Stages(out, runs / DEPLOYMENT)
    accepted = st.completed("admit")["result"]
    case = accepted["cases"]["16"]
    graph = case["graph"]
    row = graph["projections"]["layer0.down"]
    full = st.completed("audit-c16")["result"]
    pkg = st.completed("opp-c16")["result"]
    require(full["om"]["sha256"] == FULL_OM, "different all-W8 full graph")
    down = full["inventory"]["layer0.down"]
    query = full["plans"]["down"]
    check_elf(full, down["selected"])
    c.verify_opp(pkg["opp"])
    require(
        down["mode"] == r.name
        and down["selected"]["executable_sections"] == [dict(bytes=7000, sha256=DOWN_TEXT)]
        and query["tiling_sha256"] == down["tiling_sha256"] == DOWN_PLAN,
        "different deployed Down",
    )
    rawst = Stages(out, runs / FULL)
    raw = rawst.completed("atc-c16-baseline")["result"]
    require(
        raw["air"] == graph["baseline"]
        and rawst.refs["atc-c16-baseline"] == accepted["control_receipts"]["atc-c16-baseline"],
        "wrong checkpoint raw graph",
    )
    raw_image = raw["kernels"][row["node"]]
    check_elf(raw, raw_image)

    def reference():
        msg = types[0].FromString(c.verify(raw["air"]).read_bytes())
        cut = graph["projections"]["layer0.gate_up"]["node"]
        return cut_reference(
            out,
            types,
            msg.graph[0],
            graph,
            row,
            {cut: ("gate_y", [m, 2 * k])},
            r.shape,
            "layer0.down",
            "layer0.swiglu",
            "Mul",
            "layer0_swiglu_raw_down",
        )

    src = reference()
    air, packing = single_graph(
        out,
        types,
        types[0].FromString(c.verify(full["air"]).read_bytes()),
        row,
        src["weights"],
        shape=r.shape,
        op_type=r.spec.op,
        panel=r.panel,
        name="down",
    )
    require(
        [v["sha256"] for v in packing["constants"]] == [v["sha256"] for v in down["constants"]],
        "wrong Layer0 Down constants",
    )
    tag = "x".join(map(str, r.shape))
    return dict(
        src,
        air=air,
        packing=packing,
        query=query,
        selected=down["selected"],
        pkg=accepted["baseline_package"],
        control_package=pkg,
        replays={
            label: {"gate_y": gate_input["cases"][label]["golden.bin"]} for label in ("measured", "stress")
        },
        source_plans={tag: raw["plans"][tag]},
        source_implementations={"layer0.down": raw_image},
        capture_origin="DEPLOYED_CHECKPOINT_GATE_OUTPUT_SPLIT_SWIGLU_DOWN",
        receipts=st.refs,
    )
