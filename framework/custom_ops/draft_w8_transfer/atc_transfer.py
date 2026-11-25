"""Verify the actual compiled single projection, its charged inputs, sampled code and data identity."""

import mmap
from pathlib import Path
import struct
from recipes import c, packed, q, require, OLD_OP
from bench import compile_graph, embedded, selected_image
from tile_gate_graph import model_proto


def plan(query, original, r, base):
    p, s = query["platform"], query["schedule"]
    m, k, n = r.shape
    require(
        {key: v for key, v in p.items() if key != "host_dso"}
        == {key: v for key, v in original["platform"].items() if key != "host_dso"}
        and p["availableCores"] == p["blockDim"] == 8
        and p["coreLimit"] == 0
        and p["workspace"] == 2097152
        and p["capacity"]["UB"] == 262144
        and s["baseline_assignments"] == [list(range(i, n // r.width, 8)) for i in range(8)],
        "platform/ownership changed",
    )
    raw = bytes.fromhex(s["tiling_hex"])
    scratch = p["capacity"]["UB"] - r.ub
    require(
        len(raw) == 224
        and q.sha(raw) == query["tiling_sha256"]
        and struct.unpack_from("<4IQ", raw) == (m, k, n, r.panel, scratch),
        "incorrect parent tiling or UB",
    )
    cube = {f["name"]: struct.unpack_from("<I", raw, 24 + f["offset"])[0] for f in q.schema()}
    fixed = dict(
        baseM=m,
        baseN=r.width,
        baseK=r.base_k,
        singleCoreM=m,
        singleCoreN=r.width,
        singleCoreK=r.panel,
        dbL0A=2,
        dbL0B=2,
        dbL0C=1,
        isBias=0,
    )
    require(
        all(cube[key] == v for key, v in fixed.items())
        and 0 < cube["transLength"] <= scratch
        and min(cube[x] for x in ("depthA1", "depthB1", "stepKa", "stepKb")) > 0,
        "unsupported SDK Cube plan",
    )
    resources = dict(
        tile_k=r.panel,
        raw_banks=1,
        user_ub=r.ub,
        scratch=scratch,
        codes_bytes=16 * r.chunk * r.packet,
        scale_bytes=2 * (r.panel // 128) * r.width,
        weight_bytes=2 * r.banks * 16 * r.chunk,
        l1_used=r.l1 + 2 * r.base_k * (m * cube["depthA1"] + r.width * cube["depthB1"]),
        l0a_used=4 * m * r.base_k,
        l0b_used=4 * r.width * r.base_k,
        l0c_used=4 * m * r.width,
        workspace=p["workspace"],
    )
    require(
        resources == s["resources"]
        and all(
            resources[a] <= p["capacity"][b]
            for a, b in (("l1_used", "L1"), ("l0a_used", "L0A"), ("l0b_used", "L0B"), ("l0c_used", "L0C"))
        ),
        "SDK resources differ from application allocations",
    )
    if r.cube_geometry == base.cube_geometry:
        old = bytes.fromhex(original["schedule"]["tiling_hex"])
        require(
            raw[24:] == old[24:] and (r.ub != base.ub or raw == old),
            "same-geometry transfer changed Cube tiling",
        )
    return dict(
        cube=cube,
        resources=resources,
        block_dim=p["blockDim"],
        cube_sha256=q.sha(raw[24:]),
        iterates_per_core=n // (8 * r.width) * (k // r.panel),
        writes_per_core=n // (8 * r.width),
    )


def make_air(out, types, r, base, control, weights):
    m, k, n = r.shape
    nz, gn = (c.verify(weights[f]).read_bytes() for f in ("w_nz.bin", "s_gn.bin"))
    codes, scales = packed.packed_weights(nz, gn, n, k, r.width, r.panel)
    dims = ((n // r.width, k // r.panel, r.panel * r.width), (n // r.width, k // 128, r.width))
    air = out / "projection.air"
    if r.layout == base.layout:
        source = c.verify(control["air"])
        model = types[0].FromString(source.read_bytes())
        air.symlink_to(source)
    else:
        model = model_proto(types, nz, gn, projection_shape=r.shape, name=f"single_{r.role}_c{r.gear}")
        packed.add_pack_graph(
            model,
            r.width,
            r.panel,
            codes,
            scales,
            m=m,
            k=k,
            n=n,
            op_type=r.spec.op,
            prefix="transfer_a",
            pack_activation=r.activation == "packed",
        )
        graph = model.graph[0]
        target = "proj_" + ("fc" if r.role == "fc" else "layer0_" + r.role)
        for node in graph.op:
            if node.name == OLD_OP:
                node.name = target
            for i, e in enumerate(node.input):
                if e == OLD_OP + ":0":
                    node.input[i] = target + ":0"
        graph.output[:] = [target + ":0"]
        air.write_bytes(model.SerializeToString(deterministic=True))
    (graph,) = model.graph
    table = {node.name: node for node in graph.op}
    (projected,) = (node for node in graph.op if node.type == r.spec.op)
    for edge, sz, data in zip(projected.input[1:], dims, (codes, scales)):
        name, port = edge.rsplit(":", 1)
        t = table[name]
        require(
            port == "0"
            and t.type == "Const"
            and t.attr["value"].t.data == data
            and tuple(t.output_desc[0].shape.dim) == sz
            and t.output_desc[0].layout == "ND",
            "selected AIR does not use checkpoint weights/scale for " + r.name,
        )
    return air, dims, (codes, scales)


def compiled(out, types, r, base, control, weights, pkg, env, atc, *, recreate_control=False):
    air, dims, data = make_air(out, types, r, base, control, weights)
    args, om, dump, query, nodes = compile_graph(
        out, pkg, env, atc, shape=r.shape, model_name="projection", tile_n=r.width
    )
    proof = plan(query, control["query"], r, base)
    m, k, n = r.shape
    (op,) = (v for v in nodes if v["type"] == r.spec.op)
    if r.activation == "packed":
        prep = packed.activation_nodes(nodes, op, m, k)
    else:
        require(
            all(x is op or x["type"] in ("Const", "FileConstant", "Data", "NetOutput") for x in nodes)
            and tuple(map(int, op["input_desc"][0]["shape"]["dim"])) == (m, k),
            "unexpected activation preparation",
        )
        prep = set()
    require(
        len(op["input_desc"]) == len(op["inputs"]) == 3
        and op["input_desc"][0]["layout"] == "ND"
        and [a["value"] for a in op["attributes"] if a["key"] == "custom_op_file_path"]
        == [{"s": str(Path(pkg["opp"]["install"]) / "vendors/customize")}],
        "wrong private ATC implementation",
    )
    selected = embedded(om, [selected_image(op, proof["block_dim"], r.spec.op)])[op["name"]]
    require(
        (selected["executable_sections"] == control["selected"]["executable_sections"]) == recreate_control,
        "unexpected executable identity relative to incumbent for " + r.name,
    )
    constants = []
    with om.open("rb") as fd, mmap.mmap(fd.fileno(), 0, access=mmap.ACCESS_READ) as mem:
        for desc, shape, b in zip(op["input_desc"][1:], dims, data):
            offset = mem.find(b)
            require(
                desc["layout"] == "ND" and tuple(map(int, desc["shape"]["dim"])) == shape and offset >= 0,
                "missing compiled constant/descriptor for " + r.name,
            )
            constants.append(dict(bytes=len(b), sha256=q.sha(b), offset=offset))
    return dict(
        mode=r.name,
        op=r.spec.op,
        abi=r.abi,
        air=c.identity(air),
        om=c.identity(om),
        dump=c.identity(dump),
        argv=args,
        selected=selected,
        query=query,
        contract=proof,
        constants=constants,
        prep_nodes=sorted(prep),
        compute_names={node["name"]: node["type"] for node in nodes if node["name"] in prep or node is op},
    )
