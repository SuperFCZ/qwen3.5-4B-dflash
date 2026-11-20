"""Bind isolated compiled projections and both complete Draft graphs to actual OM bytes."""

import json
from pathlib import Path
import shutil

from evidence import (
    CANDIDATE,
    CONTROL,
    GATE,
    NODES,
    SHAPE_KEY,
    c,
    config_graph,
    raw_air,
    require,
    verify_compiled_graph,
)
from scale_contracts import flow, historical, read_om_json, same_plan
from scale_review import allocated_sections
from gate_plan import embedded, node_binding, non_gate
from w16_export import compile_air, native_images
from w16_graph import protected_graph
from atc_evidence import payloads, resumed_command
from atc_dependency_identity import plan as atc_entry
from split_atc import one_capture
from om_constants import compare_constants
from routing import verify_dispatch


def selected_plans(folder, worker, build, src, control, *, write):
    baseline = control["plans"]
    streams = {k: flow.lines(c.verify(worker[k])) for k in ("platform", "scheduling")}
    for k, rows in streams.items():
        require(
            {"x".join(str(row[a]) for a in ("m", "k", "n")) for row in rows} == set(baseline),
            "ATC callback shape inventory: " + k,
        )
    results = {}
    for tag, ref in baseline.items():
        shape = list(map(int, tag.split("x")))
        files = []
        for k, rows in streams.items():
            group = [r for r in rows if [r[a] for a in ("m", "k", "n")] == shape]
            # The archived Pilot bound the original one-shape callback files directly.
            if len(baseline) == 1:
                p = c.verify(worker[k])
            else:
                p = folder / (tag + "-" + k + ".jsonl")
                if write:
                    with p.open("x") as file:
                        for r in group:
                            file.write(json.dumps(r) + "\n")
                require(flow.lines(p) == group, "callback differs from real ATC journal")
            files.append(p)
        q = flow.read_query(*files, build["opp"], "ATC", "A", shape)
        require(q["host_dso"]["sha256"] == ref["query"]["host_dso"]["sha256"], "different tiling DSO")
        if tag == SHAPE_KEY:
            proof = same_plan(q, src, ref["contract"])
        else:
            non_gate(q, ref["query"])
            proof = ref["contract"]
        results[tag] = dict(query=q, contract=proof)
    return results


def verify_worker(worker, model, src, selected, frozen):
    full = model["scope"] == "full"
    tag = "FULL" if full else "PROJECTION"
    require(
        worker["status"] == "GATE8_" + tag + "_AIR_COMPILED_NOT_EXECUTED"
        and worker["execute_calls"] == 0
        and not worker.get("error")
        and not worker.get("integrity_error")
        and (worker["om"], worker["deployment"]) == (model["om"], model["deployment"]),
        "wrong ATC worker",
    )
    row = one_capture(worker["captures"], worker["atc_command"])
    compiled = c.verify(model["worker_report"]).parent
    folder = compiled.parent
    env = dict(
        selected["env"]["values"],
        AI_RUN_DIR=str(folder),
        ASCEND_WORK_PATH=str(compiled),
        ASCEND_CACHE_PATH=str(folder / "compiler-cache"),
        DFLASH_PLATFORM_DIAG_PATH=str(compiled / "platform.jsonl"),
        DFLASH_PLATFORM_DIAG_ROUTE="ATC",
        DFLASH_SCHEDULER_EVIDENCE_PATH=str(compiled / "scheduling.jsonl"),
    )
    require(
        env == row["environment"] == worker["child_environment"]
        and env["ASCEND_CUSTOM_OPP_PATH"] == str(Path(selected["opp"]["install"]) / "vendors/customize")
        and not env.get("BUILD_KERNEL_SRC"),
        "unadmitted ATC environment",
    )
    dep = c.verify(model["deployment"])
    manifest = c.linked(dep.parent, c.read(dep)["air_manifest"])
    require(
        c.sha(manifest) == frozen["air"]["sha256"]
        and row["atc"] == frozen["atc_binary"]
        and row["argv"] == resumed_command(frozen, manifest, dep.parent),
        "different AIR or ATC options",
    )
    cloned = payloads(manifest, c.read(manifest))
    require(
        {k: (v["bytes"], v["sha256"]) for k, v in cloned.items()}
        == {k: (v["bytes"], v["sha256"]) for k, v in frozen["payloads"].items()},
        "cloned AIR payload differs",
    )
    wrapper = atc_entry(row["atc"], row["argv"][0], src["cann"], env)
    effective = dict(env, **wrapper["wrapper_exports"])
    require(worker["elf_environment"] == effective, "ATC wrapper environment drift")
    bisheng = effective.get("BISHENG_REAL_PATH") or shutil.which("bisheng", path=effective["PATH"])
    require(bisheng and c.identity(bisheng) == selected["compiler"]["tools"]["bisheng"], "wrong compiler")
    return dict(capture=row, atc_chain=wrapper, actual_air=c.identity(manifest), payloads=cloned)


def image_contract(left, right, targets, full):
    require(
        set(left) == set(right)
        and len(left) == (26 if full else 1)
        and set(targets) <= set(left)
        and len(targets) == (5 if full else 1),
        "invalid compiled projection inventory",
    )
    unchanged = {}
    for name, raw in left.items():
        new = right[name]
        require(
            raw["block_dim"] == new["block_dim"] == 8 and raw["engine"] == new["engine"] == "AIcoreEngine",
            "different compiled scheduler: " + name,
        )
        if name in targets:
            require(
                new["sha256"] != raw["sha256"] and new["executable_sections"] != raw["executable_sections"],
                "target executable unchanged: " + name,
            )
        else:
            require(
                {k: v for k, v in raw.items() if k != "offset"}
                == {k: v for k, v in new.items() if k != "offset"},
                "non-target embedded implementation changed: " + name,
            )
            unchanged[name] = raw["sha256"]
    return unchanged


def compile_model(folder, ctx, build, key, *, full):
    src = ctx["src"]
    control = ctx["full"][key] if full else ctx["controls"][key]
    frozen = raw_air(ctx, key) if full else src["singles"][key]
    scope = "full" if full else "projection"
    worker, dump = compile_air(folder, src, build, frozen, scope)
    graph = c.read(c.verify(worker["deployment"]))["graphs"][0]
    targets = config_graph(graph, src["cases"][key] if full else {key: src["cases"]["16"][key]}, full)
    nodes = [node_binding(n) for n in read_om_json(dump) if n["type"] == GATE]
    return dict(
        mode=CANDIDATE,
        key=key,
        scope=scope,
        air=control["air"],
        om=worker["om"],
        deployment=worker["deployment"],
        dump=c.identity(dump),
        worker_report=c.identity(folder / "compiled/report.json"),
        abi=graph["metadata"]["tensor_abi"],
        opp_root=build["opp"]["install"],
        private_kernel=build["kernel"],
        gate_nodes={k: n["name"] for k, n in targets.items()},
        atc_kernels=embedded(c.verify(worker["om"]), nodes),
        plans=selected_plans(folder, worker, build, src, control, write=True),
    )


def check_model(ctx, build, key, model, *, full):
    src = ctx["src"]
    control = ctx["full"][key] if full else ctx["controls"][key]
    frozen = raw_air(ctx, key) if full else src["singles"][key]
    require(
        control["mode"] == CONTROL
        and model["mode"] == CANDIDATE
        and model["key"] == control["key"] == key
        and model["scope"] == control["scope"] == ("full" if full else "projection")
        and model["air"] == control["air"]
        and model["abi"] == control["abi"]
        and model["gate_nodes"] == control["gate_nodes"]
        and model["opp_root"] == build["opp"]["install"] != control["opp_root"]
        and model["private_kernel"] == build["kernel"],
        "candidate/control/model binding differs",
    )
    verify_compiled_graph(control, src, full)
    verify_compiled_graph(model, src, full)
    targets = set(model["gate_nodes"].values())
    require(not full or model["gate_nodes"] == NODES, "full target identities")
    unchanged = image_contract(control["atc_kernels"], model["atc_kernels"], targets, full)
    folder = c.verify(model["dump"]).parent
    worker = c.read(c.verify(model["worker_report"]))
    historical.verify_tree(worker, {})
    require(
        model["plans"] == selected_plans(folder, worker, build, src, control, write=False),
        "actual ATC plans differ",
    )
    count = protected_graph(
        c.verify(control["dump"]),
        c.verify(model["dump"]),
        targets,
        raw_opp_root=control["opp_root"],
        new_opp_root=model["opp_root"],
    )
    resources = {}
    for n in sorted(targets):
        before, after = (allocated_sections(m, n) for m in (control, model))
        require(before["writable"] == after["writable"], "target writable storage delta: " + n)
        resources[n] = dict(raw=before, transpose=after)
    proof = dict(
        compile=verify_worker(worker, model, src, build, frozen),
        targets=sorted(targets),
        protected_nodes=count,
        unchanged_projections=unchanged,
        sections=resources,
        constants=(
            compare_constants(control, model)
            if full
            else dict(
                checkpoint=config_graph(
                    c.read(c.verify(model["deployment"]))["graphs"][0], {key: src["cases"]["16"][key]}
                )[key]
            )
        ),
    )
    if full:
        proof["dispatch"] = verify_dispatch(
            c.identity(folder / "scale-dispatch.jsonl"), build, model["atc_kernels"], targets
        )
        native = []
        for m in (control, model):
            native.append(
                {n: {k: v for k, v in item.items() if k != "offset"} for n, item in native_images(m).items()}
            )
        require(native[0] == native[1], "native compiled image including Head changed")
        proof["native_images"] = native[1]
    return proof
