"""Compile and check the two real checkpoint graphs with the same CANN and private OPP."""

from collections import defaultdict
import json
import sys
from pathlib import Path

from joint import c, gate, down, OLD_OP, require
from bench import flow, selected_image, embedded, read_om_json, PARENT
from package import command


def grouped(path):
    groups = defaultdict(list)
    for line in path.read_text().splitlines():
        x = json.loads(line)
        groups[tuple(x[k] for k in ("m", "k", "n"))].append(line)
    return groups


def projection_sites(nodes, roles):
    expected = {r["node"] for r in roles.values()}
    custom = [n for n in nodes if n["type"] in (OLD_OP, gate.OP, down.OP)]
    resolved = {}
    for node in custom:
        names = {node["name"]}
        for a in node["attributes"]:
            if a["key"] in ("_datadump_original_op_names", "_original_op_names"):
                names.update((a.get("value") or {}).get("list", {}).get("s", []))
        keys = names.intersection(expected)
        require(len(keys) == 1, "unmapped or fused projection: " + str(sorted(names)))
        key = keys.pop()
        require(key not in resolved, "multiply selected projection: " + key)
        resolved[key] = node
    require(set(resolved) == expected and len(custom) == 26, "ATC projection inventory changed")
    return resolved


def compile_air(out, air, cann, env):
    cache = out / "cache"
    cache.mkdir()
    plat = out / "platform.jsonl"
    schedule = out / "schedule.jsonl"
    for p in (plat, schedule):
        p.touch(exist_ok=False)
    env = dict(
        env,
        AI_RUN_DIR=str(out),
        ASCEND_WORK_PATH=str(out),
        ASCEND_CACHE_PATH=str(cache),
        DFLASH_PLATFORM_DIAG_PATH=str(plat),
        DFLASH_PLATFORM_DIAG_ROUTE="ATC",
        DFLASH_SCHEDULER_EVIDENCE_PATH=str(schedule),
    )
    atc = cann / "bin/atc"
    prefix = out / "draft"
    argv = [
        str(atc),
        "--mode=0",
        "--framework=1",
        "--model=" + str(air),
        "--output=" + str(prefix),
        "--soc_version=Ascend310P3",
        "--precision_mode=must_keep_origin_dtype",
        "--deterministic=0",
    ]
    c.write(out / "atc-command.json", argv)
    # Exercise the same CPython listener and isolated environment used by CANN's TBE process.
    command(
        [sys.executable, "-I", "-B", Path(__file__).with_name("ipc_check.py")],
        out / "ipc-check.log",
        env,
        out,
    )
    print((out / "ipc-check.log").read_text().strip(), flush=True)
    command(argv, out / "atc.log", env, out)
    om = out / "draft.om"
    dump = out / "om-graph.json"
    command([atc, "--mode=1", "--om=" + str(om), "--json=" + str(dump)], out / "inspect.log", env, out)
    pp, ss = grouped(plat), grouped(schedule)
    return om, dump, pp, ss


def compiled(out, gear, mode, graph, pkg, cann, env, control):
    air = c.verify(graph[mode])
    shapes = {tuple(r[d] for d in ("m", "k", "n")) for r in graph["projections"].values()}
    require(len(shapes) == 6, "expected six distinct projection shapes")
    om, dump, pp, ss = compile_air(out, air, cann, env)
    require(set(pp) == set(ss) == shapes, "ATC callback shape inventory changed")
    plans = {}
    for shape in sorted(shapes):
        key = "x".join(map(str, shape))
        folder = out / key
        folder.mkdir()
        paths = [folder / "platform.jsonl", folder / "schedule.jsonl"]
        for path, rows in zip(paths, (pp[shape], ss[shape])):
            path.write_text("\n".join(rows) + "\n")
        query = flow.read_query(*paths, pkg["opp"], "ATC", "A", shape)
        require(query["platform"]["blockDim"] == 8, "unexpected actual 310P3 ATC blockDim")
        base = control["plans"][key] if control else None
        if shape == (down.M, down.K, down.N):
            down.plan(query, down.SINGLE if base else down.CONTROL, {"query": base} if base else None)
        elif shape == (gate.M, gate.K, gate.N):
            if base:
                gate.validate_plan(query, {"plans": {key: {"query": base}}}, gate.K, w16_banks=2)
            else:
                require(query["tiling_sha256"] == PARENT, "Gate control tiling changed")
        elif base:
            require(
                all(query[k] == base[k] for k in ("platform", "schedule", "host_dso", "tiling_sha256")),
                "non-target tiling differs: " + key,
            )
        plans[key] = query
    nodes = read_om_json(dump)
    rows = graph["projections"]
    custom = projection_sites(nodes, rows)
    images = []
    for role, r in rows.items():
        name = r["node"]
        node = custom[name]
        wanted = OLD_OP
        if mode == "optimized":
            if role.endswith(".gate_up"):
                wanted = gate.OP
            if role.endswith(".down"):
                wanted = down.OP
        vendor = [a["value"] for a in node["attributes"] if a.get("key") == "custom_op_file_path"]
        require(
            vendor == [{"s": str(Path(pkg["opp"]["install"]) / "vendors/customize")}],
            "unexpected ATC OPP for " + role,
        )
        images.append(
            selected_image(
                node, plans["x".join(str(r[d]) for d in ("m", "k", "n"))]["platform"]["blockDim"], wanted
            )
        )
    emitted = embedded(om, images)
    bindings = {source: emitted[node["name"]] for source, node in custom.items()}
    if control:
        for role, r in rows.items():
            if not role.endswith((".gate_up", ".down")):
                name = r["node"]
                require(
                    bindings[name]["executable_sections"] == control["kernels"][name]["executable_sections"],
                    "protected ATC executable changed: " + role,
                )
    print(
        f"C{gear} {mode}: 26 projections verified in OM, {sum(x['type']==gate.OP for x in custom.values())} panel Gate/Up, {sum(x['type']==down.OP for x in custom.values())} panel Down",
        flush=True,
    )
    return dict(om=c.identity(om), dump=c.identity(dump), air=graph[mode], kernels=bindings, plans=plans)
