"""Compile a frozen AIR in one isolated ATC worker and verify its real callbacks."""

import json
from pathlib import Path
import re
import struct
import sys

from w16_common import CUBE_SHA, GATE, GATE8, SHAPE, SHAPE_KEY, c, command, flow, historical, one, require, tree
from w16_graph import protected_graph
from gate_common import config_graph, sources as worker_sources
from gate_plan import CUBE, embedded, node_binding, non_gate, schema, sha
from export_gate import verify_compiled_graph
from c_profile_graph import read_om_json
from flow_query import assignments, read_query


def resource_contract(query, raw):
    p, s = query["platform"], query["schedule"]
    rp, rs = raw["platform"], raw["schedule"]
    require(
        {k: v for k, v in p.items() if k != "host_dso"} == {k: v for k, v in rp.items() if k != "host_dso"},
        "PlatformInfo or workspace differs from Raw",
    )
    require(
        {k: v for k, v in s.items() if k not in ("resources", "tiling_hex")}
        == {k: v for k, v in rs.items() if k not in ("resources", "tiling_hex")},
        "changed scheduler policy",
    )
    require(
        p["GetCoreNum"] == p["availableCores"] == p["blockDim"] == s["selected_block_dim"] == 8
        and p["coreLimit"] == s["core_limit"] == 0
        and p["arch"] == 2002
        and s["variant"] == "A"
        and s["route"] == p["route"] == "ATC"
        and s["n_tiles"] == 304
        and s["baseline_assignments"] == assignments(304, 8),
        "not ATC auto/8 coupled",
    )
    expect = dict(
        raw_banks=2,
        codes_bytes=32768,
        scale_bytes=512,
        user_ub=76544,
        scratch=185600,
        weight_bytes=32768,
        tile_k=256,
        l1_used=40960,
        l0a_used=16384,
        l0b_used=65536,
        l0c_used=4096,
        workspace=p["workspace"],
    )
    require(rs["resources"] == expect and p["workspace"] == 2097152, "single-W Raw resources unbound")
    require(
        s["resources"] == dict(expect, weight_bytes=65536, user_ub=109312, scratch=152832)
        and p["capacity"]["UB"] == 262144
        and p["capacity"]["L0B"] == 65536,
        "dual-W resource delta",
    )
    before = bytes.fromhex(rs["tiling_hex"])
    after = bytes.fromhex(s["tiling_hex"])
    require(
        len(before) == len(after) == 224
        and before[:24] == struct.pack("<IIIIQ", *SHAPE, 256, 185600)
        and after == before[:16] + struct.pack("<Q", 152832) + before[24:]
        and sha(before) == raw["tiling_sha256"]
        and sha(after) == query["tiling_sha256"]
        and sha(after[24:]) == CUBE_SHA,
        "unapproved parent or Cube serialization",
    )
    fields = {f["name"]: struct.unpack_from("<I", after, 24 + f["offset"])[0] for f in schema()}
    require(all(fields[k] == v for k, v in CUBE.items()), "SDK Cube getter values")
    return dict(
        block_dim=8,
        assignments=s["baseline_assignments"],
        resources=s["resources"],
        workspace=p["workspace"],
        cube_fields=fields,
        cube_sha256=CUBE_SHA,
        parent_sha256=sha(after),
    )


def plans(folder, worker, opp, baseline, *, replay=None):
    streams = {key: flow.lines(c.verify(worker[key])) for key in ("platform", "scheduling")}
    results = {}
    for key in streams:
        tags = {"x".join(str(row[x]) for x in ("m", "k", "n")) for row in streams[key]}
        require(tags == set(baseline), "unexpected callback shape inventory in " + key)
    for tag, ref in baseline.items():
        paths = []
        shape = list(map(int, tag.split("x")))
        for key, rows in streams.items():
            rows = [row for row in rows if [row[k] for k in ("m", "k", "n")] == shape]
            path = folder / (tag + "-" + key + ".jsonl")
            if replay is None:
                with path.open("x") as out:
                    for row in rows:
                        out.write(json.dumps(row) + "\n")
            require(flow.lines(path) == rows, "grouped callback differs from original")
            paths.append(path)
        query = read_query(*paths, opp, "ATC", "A", shape)
        if tag == SHAPE_KEY:
            contract = resource_contract(query, ref["query"])
        else:
            non_gate(query, ref["query"])
            contract = ref["contract"]
        results[tag] = dict(query=query, contract=contract)
    if replay is not None:
        require(replay == results, "ATC callback replay differs")
    return results


def native_images(model):
    """Bind native code-bearing nodes, including the vocabulary MatMul, to real OM bytes."""
    refs = []
    head = False
    for node in read_om_json(c.verify(model["dump"])):
        if node["type"] == GATE:
            continue
        rows = node["attributes"]
        if not isinstance(rows, list):
            continue

        def values(key):
            return [a["value"] for a in rows if a.get("key") == key]

        ids = values("_kernel_bin_id")
        sizes = values("_tbeKernelSize")
        if not ids or not any(v.get("s") for v in ids):
            # Non-code nodes and AICPU dispatch still undergo protected_graph equality.
            require(
                not sizes or all(int(v["i"]) == 0 for v in sizes),
                "native compiled payload has no bindable binary ID: " + node["name"],
            )
            continue
        bid = one(ids, "native binary id")["s"]
        size = int(one(values("_tbeKernelSize"), "native ELF length")["i"])
        symbol = one(values("_kernelname"), "native symbol")["s"]
        require(
            re.search(r"[a-f0-9]{64}$", bid) and symbol and 64 < size <= 16 * 1024 * 1024,
            "unresolved native binary identity",
        )
        refs.append(dict(node=node["name"], symbol=symbol, bytes=size, sha256=bid[-64:]))
        if node["type"] == "MatMul" and node["name"] == "MatMul":
            head = True
    require(head, "native Head MatMul has no resolvable embedded code")
    return embedded(c.verify(model["om"]), refs)


def compile_air(folder, src, selected, frozen, scope):
    folder.mkdir(parents=True)
    prior = c.read(c.verify(src["host_source"]))["binding"]["prior"]
    env = flow.build_env(prior, selected)
    cache = folder / "compiler-cache"
    cache.mkdir()
    out = folder / "compiled"
    env.update(AI_RUN_DIR=str(folder), ASCEND_WORK_PATH=str(out), ASCEND_CACHE_PATH=str(cache))
    env_id = dict(selected["env"], values=dict(selected["env"]["values"], ASCEND_CACHE_PATH=str(cache)))
    request = folder / "request.json"
    c.write(
        request,
        dict(
            output=str(out),
            scope=scope,
            flow_sources=worker_sources(),
            frozen=frozen,
            evidence={"H": {"state": {"args": prior["current"]["evidence"]["H"]["state"]["args"]}}},
            opp=selected["opp"],
            environment=env,
            runtime_environment=env_id,
        ),
    )
    try:
        command(
            [sys.executable, "-B", GATE8 / "gate_atc_worker.py", request],
            folder / "atc-worker.log",
            env,
            cwd=folder,
        )
    except RuntimeError as exc:
        if (out / "report.json").is_file():
            failed = c.read(out / "report.json")
            raise RuntimeError(
                f"ATC worker: {failed.get('error')}, integrity={failed.get('integrity_error')} ({out / 'report.json'})"
            ) from exc
        raise
    child = c.read(out / "report.json")
    require(
        child["status"] == "GATE8_" + scope.upper() + "_AIR_COMPILED_NOT_EXECUTED"
        and not child.get("error")
        and not child.get("integrity_error")
        and child["execute_calls"] == 0,
        "ATC worker failed",
    )
    historical.verify_tree(child, {})
    require(tree(Path(selected["opp"]["install"])) == selected["package_files"], "ATC private OPP drift")
    model = c.verify(child["om"])
    dump = folder / "om-graph.json"
    command(
        [c.verify(frozen["atc_binary"]), "--mode=1", "--om=" + str(model), "--json=" + str(dump)],
        folder / "graph-inspect.log",
        env,
        cwd=folder,
    )
    deployment = c.read(c.verify(child["deployment"]))
    original = c.read(c.verify(frozen["air"]))["graphs"]
    require(
        deployment["status"] == "PASS"
        and len(deployment["graphs"]) == len(original) == 1
        and c.identity(c.linked(c.verify(child["deployment"]).parent, deployment["air_manifest"]))["sha256"]
        == frozen["air"]["sha256"]
        and c.linked(c.verify(child["deployment"]).parent, deployment["graphs"][0]["om"]) == model
        and all(
            deployment["graphs"][0][k] == original[0][k]
            for k in ("air", "metadata", "runtime_input_abi", "custom_op_audit", "input_names", "output_names")
        ),
        "ATC substituted immutable AIR/ABI/constants",
    )
    return child, dump


def compile_model(folder, src, selected, mode, key, baseline, full):
    scope = "full" if full else "projection"
    frozen = src["full"][key]["frozen"] if full else src["singles"][key]
    child, dump = compile_air(folder, src, selected, frozen, scope)
    model = c.verify(child["om"])
    graph = c.read(c.verify(child["deployment"]))["graphs"][0]
    cases = src["cases"][key] if full else {key: src["cases"]["16"][key]}
    gates = config_graph(graph, cases, full)
    nodes = [node_binding(n) for n in read_om_json(dump) if n["type"] == GATE]
    result = dict(
        mode=mode,
        key=key,
        scope=scope,
        air=frozen["air"],
        om=child["om"],
        deployment=child["deployment"],
        worker_report=c.identity(folder / "compiled/report.json"),
        dump=c.identity(dump),
        opp_root=selected["opp"]["install"],
        private_kernel=selected["kernel"],
        atc_kernels=embedded(model, nodes),
        plans=plans(folder, child, selected["opp"], baseline["plans"]),
        gate_nodes={k: v["name"] for k, v in gates.items()},
        abi=graph["metadata"]["tensor_abi"],
    )
    native = check_model(result, src, selected, baseline, full)
    if full:
        result["protected_native_images"] = native
    return result


def check_model(model, src, selected, baseline, full):
    verify_compiled_graph(model, src, full)
    worker = c.read(c.verify(model["worker_report"]))
    require(
        worker["om"] == model["om"]
        and worker["deployment"] == model["deployment"]
        and worker["execute_calls"] == 0
        and worker["status"] == "GATE8_" + model["scope"].upper() + "_AIR_COMPILED_NOT_EXECUTED"
        and not worker.get("integrity_error")
        and not worker.get("error"),
        "worker/model identity mismatch",
    )
    require(
        model["opp_root"] == selected["opp"]["install"]
        and model["private_kernel"] == selected["kernel"]
        and model["air"] == baseline["air"]
        and model["abi"] == baseline["abi"]
        and model["gate_nodes"] == baseline["gate_nodes"],
        "cross-wired graph/ABI/OPP",
    )
    plans(Path(model["dump"]["path"]).parent, worker, selected["opp"], baseline["plans"], replay=model["plans"])
    gates = set(model["gate_nodes"].values())
    a, b = baseline["atc_kernels"], model["atc_kernels"]
    require(
        len(gates) == (5 if full else 1) and a.keys() == b.keys() and len(a) == (26 if full else 1),
        "projection inventory",
    )
    for name in a:
        if name not in gates:
            require(
                a[name]["sha256"] == b[name]["sha256"]
                and a[name]["executable_sections"] == b[name]["executable_sections"],
                "non-Gate code changed: " + name,
            )
        else:
            require(
                a[name]["sha256"] != b[name]["sha256"],
                "selected Gate is still the frozen single-W ELF: " + name,
            )
    protected_graph(
        c.verify(baseline["dump"]),
        c.verify(model["dump"]),
        gates,
        raw_opp_root=baseline["opp_root"],
        new_opp_root=selected["opp"]["install"],
    )
    if full:
        old_images, new_images = native_images(baseline), native_images(model)
        require(
            old_images.keys() == new_images.keys()
            and all(
                {k: v for k, v in rec.items() if k != "offset"}
                == {k: v for k, v in new_images[name].items() if k != "offset"}
                for name, rec in old_images.items()
            ),
            "native Head/other embedded code drift",
        )
        return dict(baseline=old_images, candidate=new_images)
