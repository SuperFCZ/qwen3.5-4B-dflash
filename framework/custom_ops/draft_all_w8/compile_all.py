"""Compile the integrated AIR and bind every projection to its actual ATC instructions and inputs."""

import hashlib
import mmap
from pathlib import Path

from selection import c, require, selections, sites
from compile_model import compile_air, grouped
from bench import embedded, selected_image, read_om_json, flow


def origins(node):
    names = {node["name"]}
    for item in node["attributes"]:
        if item["key"] in ("_datadump_original_op_names", "_original_op_names"):
            names.update((item.get("value") or {}).get("list", {}).get("s", []))
    return names


def locate(mem, item, cursor=0):
    """Verify a whole payload. The short probe is only an index into the OM."""
    needle = bytes.fromhex(item["probe"])
    size = item["bytes"]
    for begin, end in ((cursor, len(mem)), (0, min(len(mem), cursor + len(needle) - 1))):
        pos = mem.find(needle, begin, end)
        while pos >= 0:
            offset = pos - item["probe_offset"]
            if (
                0 <= offset <= len(mem) - size
                and hashlib.sha256(memoryview(mem)[offset : offset + size]).hexdigest() == item["sha256"]
            ):
                return offset
            pos = mem.find(needle, pos + 1, end)
    raise ValueError("compiled OM missing verified constant payload " + item["sha256"])


def actual_sites(nodes, graph):
    names = {row["node"]: role for role, row in graph["inventory"].items()}
    by_name = {n["name"]: n for n in nodes}
    require(len(nodes) == len(by_name), "duplicate compiled node")
    result = {}
    for n in nodes:
        if not n["type"].startswith("DFlash") and "WeightQuant" not in n["type"]:
            continue
        keys = origins(n) & names.keys()
        require(len(keys) == 1, "unbound compiled W8 node: " + n["name"])
        role = names[keys.pop()]
        require(
            role not in result and graph["inventory"][role]["op"] == n["type"],
            "wrong or duplicate W8 implementation at " + role + ": " + n["type"],
        )
        result[role] = n
    require(
        set(result) == set(graph["inventory"]) and len(result) == 26,
        "incomplete compiled W8 inventory: " + str(sorted(set(graph["inventory"]) - set(result))),
    )
    return result, by_name


def compile_candidate(out, graph, cann, env):
    om, dump, _, _ = compile_air(out, c.verify(graph["air"]), cann, env)
    return dict(
        air=graph["air"],
        om=c.identity(om),
        dump=c.identity(dump),
        platform=c.identity(out / "platform.jsonl"),
        scheduler=c.identity(out / "schedule.jsonl"),
        command=c.identity(out / "atc-command.json"),
    )


def audit(out, gear, graph, pkg, case, compiled):
    require(compiled["air"] == graph["air"], "compiled a different AIR")
    om, dump = c.verify(compiled["om"]), c.verify(compiled["dump"])
    pp, ss = grouped(c.verify(compiled["platform"])), grouped(c.verify(compiled["scheduler"]))
    specs = selections(gear)
    shapes = {s.shape for s in specs.values()}
    require(len(shapes) == 6 and set(pp) == set(ss) == shapes, "incomplete actual ATC tiling callbacks")
    proof = {}
    vendor = str(Path(pkg["opp"]["install"]) / "vendors/customize")
    for kind, spec in specs.items():
        folder = out / kind
        folder.mkdir()
        files = [folder / "platform.jsonl", folder / "scheduling.jsonl"]
        for path, groups in zip(files, (pp, ss)):
            path.write_text("\n".join(groups[spec.shape]) + "\n")
        actual = flow.read_query(*files, pkg["opp"], "ATC", "A", spec.shape, tile_n=spec.columns)
        orig = case["assets"][kind]["query"]

        def platform(q):
            return {k: v for k, v in q["platform"].items() if k != "host_dso"}

        require(
            platform(actual) == platform(orig)
            and actual["platform"]["blockDim"] == 8
            and actual["schedule"] == orig["schedule"]
            and actual["tiling_sha256"] == orig["tiling_sha256"],
            f"C{gear} {kind} SDK plan, resources or launch changed; see {folder}",
        )
        proof[kind] = actual
    nodes = read_om_json(dump)
    bound, table = actual_sites(nodes, graph)
    images = []
    for role, spec in sites(specs).items():
        node = bound[role]
        require(
            [a["value"] for a in node["attributes"] if a["key"] == "custom_op_file_path"] == [{"s": vendor}],
            "different private vendor: " + role,
        )
        images.append(selected_image(node, 8, spec.op))
    emitted = embedded(om, images)
    result = {}
    cursor = 0
    mismatches = {}
    with om.open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as data:
        for role, spec in sites(specs).items():
            row = graph["inventory"][role]
            node = bound[role]
            img = emitted[node["name"]]
            expected = (
                case["model"]["kernels"][row["node"]]
                if role.endswith((".gate_up", ".down"))
                else case["assets"][role.split(".")[-1]]["image"]
            )
            if img["executable_sections"] != expected["executable_sections"]:
                mismatches[role] = dict(
                    measured=expected["executable_sections"], integrated=img["executable_sections"]
                )
            inputs = [e for e in node["inputs"] if e and not e.startswith("^") and not e.endswith(":-1")]
            require(len(inputs) == 3 and len(node["input_desc"]) == 3, "different W8 entry arity: " + role)
            x = node["input_desc"][0]
            require(
                x["layout"] == "ND"
                and tuple(map(int, x["shape"]["dim"]))
                == ((spec.k // 16, spec.m, 16) if spec.pack_x else (spec.m, spec.k)),
                "different integrated activation ABI: " + role,
            )
            offsets = []
            for edge, desc, src, dims, item in zip(
                inputs[1:],
                node["input_desc"][1:],
                row["source_constants"],
                row["packed_shapes"],
                row["constants"],
            ):
                name, port = edge.rsplit(":", 1)
                require(
                    port == "0"
                    and table[name]["type"] in ("Const", "Constant")
                    and src in origins(table[name])
                    and desc["layout"] == "ND"
                    and tuple(map(int, desc["shape"]["dim"])) == tuple(dims),
                    f"{role} compiled constant edge {edge} differs from {src}",
                )
                offset = locate(data, item, cursor)
                cursor = offset + item["bytes"]
                offsets.append(dict(node=name, offset=offset, bytes=item["bytes"], sha256=item["sha256"]))
            result[role] = dict(
                node=node["name"],
                source_node=row["node"],
                op=spec.op,
                mode=spec.mode,
                selected=img,
                constants=offsets,
                input=inputs[0],
                tiling_sha256=proof[role.split(".")[-1]]["tiling_sha256"],
            )
    c.write(out / "inventory.json", result)
    if mismatches:
        c.write(out / "different-machine-text.json", mismatches)
        raise ValueError(
            "compiled instructions differ at "
            + str(list(mismatches))
            + ": "
            + str(out / "different-machine-text.json")
        )
    print(
        f"C{gear} candidate OM verified: 26/26 measured W8 implementations, 52 packed constants, 8 blocks",
        flush=True,
    )
    return dict(om=c.identity(om), dump=c.identity(dump), air=graph["air"], inventory=result, plans=proof)
