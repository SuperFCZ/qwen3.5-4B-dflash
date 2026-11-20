"""Read-only evidence binding shared by the two projection gears and full Draft."""

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "draft_gate_scale_broadcast"))
import run_scale as pilot
from scale_contracts import (
    CANDIDATE,
    CONTROL,
    FIRST,
    GATE,
    GATE8_ROOT,
    MODES as MODES,
    SHAPE,
    SHAPE_KEY as SHAPE_KEY,
    c,
    config_graph,
    historical,
    one,
    require,
    verify_compiled_graph,
)
from scale_reuse import PREFIX, open_prefix
from gate_common import GEARS, LAYERS, case_contract
from atc_evidence import payloads

BASE = GATE8_ROOT.parent
OLD = BASE / "gate-scale-broadcast-ABnQDE"
PILOT = BASE / "gate-scale-reviewfix-bXR45o"
PILOT_END = {
    "review": (64910, "b26e5691778966d0e73ea479bd7ac2f601ac76366ae274cf7dc5e812e6110290"),
    "smoke": (160768, "fbe5574cd30aa93f456f785013415995967612c727b1cbd8b79a95b8307335c0"),
    "benchmark": (242961, "7d73e34f41af5175cae973730c86dc881aaa98c52a157f44cf09156fd3557a23"),
    "compare": (48294, "3b495f2fc68270b97ab7121166b0b39f7227167f39c287de6c3924facc62e297"),
}
FULL_EXPORT_SHA = "83b95910b5a3f31a59a1706f903c1c02d073eeaae7b5cd81dd14810aa1142127"
FULL_OMS = {
    "16": "eeb1510326908a7890fd9d62ce7b2ec8801241abadf749e041cb16ce3d9fd17e",
    "64": "a120bc6e6b47ae8de9d0a771229fec363477f93bf6b0ba1e706987dee223075e",
}
PILOT_ELF = "d822a9f91266ebea904bb5022f61d35ec543b20fa503941190ffeb814da14ac6"
NODES = {layer: f"DFlashGroupQuantLinear_{4 + 5 * i}" for i, layer in enumerate(LAYERS)}


def source_files():
    own = [c.identity(p) for p in sorted(HERE.rglob("*")) if p.is_file() and "__pycache__" not in p.parts]
    return own + pilot.sources()


def pinned(item, digest, size=None):
    require(
        item["sha256"] == digest and (size is None or item["bytes"] == size),
        "archived digest or size changed: " + item["path"],
    )
    return c.read(c.verify(item))


def pilot_chain(ctx, *, replay):
    base = dict(
        schema=1,
        root=str(PILOT),
        archive=str(GATE8_ROOT),
        sources=pilot.sources(),
        scope="C16_LAYER0_GATE_SCALE_EXPERIMENT",
    )
    archived, overrides, receipt = open_prefix(OLD, GATE8_ROOT, base)
    base["reuse"] = receipt
    refs, reports = {}, {}
    for phase in pilot.STAGES:
        prefix = phase in PREFIX
        path = (OLD if prefix else PILOT) / phase / "report.json"
        item = c.identity(path)
        size, digest = (None, PREFIX[phase]) if prefix else PILOT_END[phase]
        r = pinned(item, digest, size)
        header = archived if prefix else base
        require(
            item["path"] == str(path)
            and all(r[k] == v for k, v in header.items())
            and r["predecessors"] == refs
            and r["folder"] == str(path.parent)
            and r["phase"] == phase
            and r["status"] == pilot.STATUS[phase]
            and r["error"] is None
            and r["integrity_error"] is None,
            "cross-wired or failed Pilot stage: " + phase,
        )
        if replay:
            historical.verify_tree(r, dict(overrides) if prefix else {})
            require(
                pilot.verify(phase, path.parent, r, reports, ctx) == r["status"], "Pilot replay: " + phase
            )
        reports[phase], refs[phase] = r, item
    metrics = pilot.statistics(reports["benchmark"]["runs"])
    require(
        metrics == reports["compare"]["comparison"] == reports["benchmark"]["statistics"]
        and metrics["decision"] == "GO"
        and reports["atc"]["candidate"]["atc_kernels"][GATE]["sha256"] == PILOT_ELF,
        "wrong Pilot result",
    )
    return (
        dict(
            reports=refs,
            original_failed_review=receipt["failed_review"],
            baseline=ctx[0]["reports"],
            comparison=metrics,
            candidate_elf=PILOT_ELF,
        ),
        reports,
    )


def context(replay=False):
    ctx = pilot.read_baseline(GATE8_ROOT)
    binding, src, raw, _ = ctx
    lock, reports = pilot_chain(ctx, replay=replay)
    bench = c.read(c.verify(binding["reports"]["full-benchmark"]))
    item = c.identity(GATE8_ROOT / "full-export/report.json")
    whole = pinned(item, FULL_EXPORT_SHA)
    historical.stage_status("full-export", whole)
    require(
        bench["predecessors"]["full-export"] == item
        and whole["sources"] == historical.sources()
        and all(
            whole["predecessors"][p] == binding["reports"][p]
            for p in ("source", "build", "export", "c16-smoke")
        ),
        "Raw full export provenance drift",
    )
    controls = c.read(c.verify(binding["reports"]["export"]))["models"][CONTROL]
    full = whole["models"][CONTROL]
    require(
        set(controls) == set(LAYERS)
        and set(full) == set(GEARS)
        and all(full[g]["om"]["sha256"] == FULL_OMS[g] and full[g]["gate_nodes"] == NODES for g in GEARS),
        "wrong frozen single/full model inventory",
    )
    lock["raw_full_export"] = item
    return dict(
        src=src,
        builds={CONTROL: raw, CANDIDATE: reports["build"]["candidate"]},
        pilot=reports,
        lock=lock,
        controls=controls,
        full=full,
    )


def fresh_layers(gear):
    require(gear in GEARS, "unknown gear")
    return [layer for layer in LAYERS if (gear, layer) != ("16", FIRST)]


def full_abi(spec):
    expected = [("draft_top1", "int64", [1, 15], 120)]
    for i in range(5):
        expected.extend((f"d{i}_{kind}", "float16", [1, 8, 2112, 128], 4325376) for kind in ("key", "value"))
    require(
        [(x["name"], x["dtype"], x["shape"]) for x in spec["abi"]["outputs"]]
        == [(n, t, s) for n, t, s, _ in expected]
        and {k: i["bytes"] for k, i in spec["golden"].items()} == {n: b for n, _, _, b in expected},
        "incomplete Draft token/KV ABI or Golden extent",
    )


def raw_air(ctx, gear):
    """Use the actual copied AIR attached to the successful Raw full deployment."""
    model, spec = ctx["full"][gear], ctx["src"]["full"][gear]
    dep = c.verify(model["deployment"])
    manifest = c.linked(dep.parent, c.read(dep)["air_manifest"])
    frozen = spec["frozen"]
    require(c.sha(manifest) == model["air"]["sha256"] == frozen["air"]["sha256"], "Raw AIR manifest drift")
    files = payloads(manifest, c.read(manifest))
    require(
        {k: (v["sha256"], v["bytes"]) for k, v in files.items()}
        == {k: (v["sha256"], v["bytes"]) for k, v in frozen["payloads"].items()},
        "Raw AIR payload drift",
    )
    worker = c.read(c.verify(model["worker_report"]))
    require(
        worker["om"] == model["om"]
        and worker["deployment"] == model["deployment"]
        and worker["status"] == "GATE8_FULL_AIR_COMPILED_NOT_EXECUTED"
        and worker["execute_calls"] == 0
        and worker["atc_command"] == model["atc_command"],
        "unbound Raw ATC command",
    )
    return dict(
        frozen,
        air=c.identity(manifest),
        payloads=files,
        deployment=model["deployment"],
        om=model["om"],
        command=worker["atc_command"],
    )


def bind_gear(ctx, gear):
    from om_constants import inline_constants, weight_partition

    src, model = ctx["src"], ctx["full"][gear]
    case_contract(src["cases"])
    cases = src["cases"][gear]
    flow_cases = c.read(c.verify(src["history"]["source"]))["binding"]["cases"]
    spec = src["full"][gear]["io"]
    full_abi(spec)
    historical.verify_tree(spec, {})
    verify_compiled_graph(model, src, True)
    graph = c.read(c.verify(model["deployment"]))["graphs"][0]
    all_ops = [n for n in graph["runtime_input_abi"]["draft_ac"]["nodes"] if n["ge_type"] == GATE]
    nodes = config_graph(graph, cases, True)
    require(
        len(all_ops) == 26
        and {n["name"] for n in all_ops if (n["k"], n["n"]) == (2560, 19456)} == set(NODES.values())
        and {k: n["name"] for k, n in nodes.items()} == NODES,
        "five constant-bound targets in 26 projections required",
    )
    v3 = c.read(c.verify(src["v3_projections"][gear]))
    rows = {}
    for layer, case in cases.items():
        require(case == flow_cases[gear + "/" + layer], "case differs from its own frozen gear: " + layer)
        historical.verify_tree(case, {})
        node = nodes[layer]
        match = one((x for x in v3 if x["name"] == node["name"] and x["projection"] == layer), "v3 mapping")
        require(
            match["mkn"] == SHAPE
            and match["role"] == "gate_up"
            and match["contract_matches"] == [layer]
            and match["immutable_node"] == [node]
            and src["full_nodes"][gear][layer] == node,
            "projection graph/input binding changed",
        )
        single = ctx["controls"][layer]
        verify_compiled_graph(single, src, False)
        single_graph = c.read(c.verify(single["deployment"]))["graphs"][0]
        projection = config_graph(single_graph, {layer: case})[layer]
        for field in (
            "weight_sha256",
            "nz_sha256",
            "scale_gn_sha256",
            "scale_shape",
            "storage_shape",
            "k",
            "n",
        ):
            require(projection[field] == node[field], "single OM constants differ for " + gear + "/" + layer)
        rows[layer] = dict(
            case=case,
            full_node=node,
            single_node=projection,
            raw_single_om=single["om"],
            raw_full_kernel=model["atc_kernels"][node["name"]],
            raw_single_air=single["air"],
        )
    return dict(
        gear=gear,
        raw_om=model["om"],
        raw_deployment=model["deployment"],
        raw_dump=model["dump"],
        raw_weight_partition=weight_partition(model["om"]["path"]),
        raw_const_nodes=inline_constants(model["dump"]["path"]),
        raw_air=raw_air(ctx, gear),
        projection_source=src["v3_projections"][gear],
        nodes=NODES,
        cases=rows,
        full_io=spec,
        projection_count=len(all_ops),
    )
