"""Recover the current Layer0 Gate operands without running a full-model benchmark."""

from pathlib import Path

from gate_next import CONTROL, ROLE, SHAPE, SHAPE_KEY, c, gate, require, OLD_OP
from selection import FULL, O
from experiment import Stages
from full_replay import compare_rounds
from gate_plan import recheck_embedded
from q_source import edge

DEPLOYMENT = "full-all-w8-TglfhB"
FULL_OM = "16e822f10a339b984dc8a2d4c6bba2bf34b30dca5f3f750959e7533b6e73e2f1"


def check_elf(model, image):
    fields = {k: v for k, v in image.items() if k not in ("offset", "executable_sections")}
    recheck_embedded(c.verify(model["om"]), [fields], {image["node"]: image})


def o_operands(out, runs, case, full, receipts):
    # Both inputs were already captured and the O winner was separately checked
    # against the captured O outputs before it was admitted to the integrated run.
    root = runs / O.root
    final = c.identity(root / "result.json")
    require(final["sha256"] == O.result, "different O measurement")
    st = Stages(out, root)
    src = st.completed("source")["result"]
    model = st.completed(O.atc)["result"]
    ref = st.completed("reference-om")["result"]
    sm = st.completed("smoke")
    require(
        src["archive"]["inputs-c16"] == receipts["inputs-c16"]
        and src["sites"]["layer0.o"] == case["graph"]["projections"]["layer0.o"]
        and model["om"]["sha256"] == O.om
        and model["selected"]["executable_sections"]
        == full["inventory"]["layer0.o"]["selected"]["executable_sections"]
        and c.read(final["path"])["selected"] == model["mode"],
        "different Layer0 O replay/implementation",
    )
    out_files = {}
    for label in ("measured", "stress"):
        row = st.completed("capture-" + label)
        replay = src["replays"][label]
        require(
            row["contract"]["source"] == st.refs["source"]
            and row["contract"]["om"] == ref["om"]
            and row["contract"]["inputs"] == replay
            and row["contract"]["runner"] == sm["contract"]["runner"]
            and all(case["io"]["files"][name] == f for name, f in replay.items() if name != "embedding_rows"),
            "unbound O replay: " + label,
        )
        cap = row["result"]
        require(
            cap["outputs"]["y"] == cap["files"]["golden.bin"]
            and sm["contract"]["cases"][label] == cap["files"],
            "unbound O output",
        )
        (winner,) = (x for x in sm["result"] if x["case"] == label and x["mode"] == model["mode"])
        execution = c.read(c.verify(winner["execution"]))
        y = c.verify(cap["outputs"]["y"])
        require(
            execution["status"] == "PASS"
            and execution["golden_match"]
            and not execution["cpu_fallback"]
            and execution["sync_completed"]
            and all(
                c.sha(Path(winner["folder"]) / f"y-actual-{i}.bin") == cap["outputs"]["y"]["sha256"]
                for i in (0, 1)
            )
            and y.stat().st_size == 16 * 2560 * 2,
            "O winner outputs differ from raw captured reference",
        )
        if label == "measured":
            require(
                all(
                    cap["outputs"][name]["sha256"] == case["reference"][name]["sha256"]
                    for name in ("d0_key", "d0_value")
                ),
                "O capture differs from full Layer0 replay",
            )
        c.verify(replay["embedding_rows"])
        out_files[label] = dict(embedding_rows=replay["embedding_rows"], o_output=cap["outputs"]["y"])
    return out_files, st.refs


def reference_graph(out, types, model, graph, *, shape=SHAPE):
    """Keep the actual embedding scaling, residual add, post-norm, reshape and original raw Gate."""
    m, k, n = shape
    (g,) = model.graph
    table = {t.name: t for t in g.op}
    site = graph["projections"][ROLE]
    projection = table[site["node"]]
    o = graph["projections"]["layer0.o"]["node"]
    (embed,) = (
        op for op in g.op if op.type == "GatherV2" and table[edge(op.input[0])].name == "target_embedding"
    )
    cuts = {embed.name: ("embedding_rows", [1, m, k]), o: ("o_output", [m, k])}
    require(
        len(table) == len(g.op)
        and projection.type == table[o].type == OLD_OP
        and [edge(v) for v in projection.input[1:]] == [site["weight"], site["scale"]]
        and tuple(site[key] for key in ("m", "k", "n")) == shape,
        "wrong raw Layer0 Gate",
    )
    return cut_reference(out,types,g,graph,site,cuts,shape,ROLE,"layer0.post_norm","Add","layer0_post_norm_raw_gate")


def cut_reference(out,types,g,graph,site,cuts,shape,role,required_module,required_type,graph_name):
    """Replace frozen upstream tensors by public inputs and expose the raw projection's x/y on device."""
    m,k,n=shape
    table={t.name:t for t in g.op}
    projection=table[site["node"]]
    require(len(table)==len(g.op) and projection.type==OLD_OP and
            tuple(site[key] for key in ("m","k","n"))==shape and
            [edge(v) for v in projection.input[1:]]==[site["weight"],site["scale"]],
            "reference projection, shape or constants differ")
    x = edge(projection.input[0])
    wanted = set()

    def visit(name):
        if name not in wanted:
            wanted.add(name)
            if name not in cuts:
                for e in table[name].input:
                    visit(edge(e))

    visit(projection.name)
    require(cuts.keys() <= wanted and any(graph["modules"].get(name) == required_module for name in wanted)
            and required_type in {table[name].type for name in wanted}
            and {name for name in wanted-set(cuts) if table[name].type.startswith("DFlash")}=={projection.name},
            "unexpected "+role+" prefix dataflow")
    ref = types[0]()
    ref.name = graph_name
    target = ref.graph.add(name=ref.name)
    abi = {"inputs": [], "outputs": []}
    for old in g.op:
        if old.name not in wanted:
            continue
        op = target.op.add()
        if old.name in cuts:
            name, sz = cuts[old.name]
            d = old.output_desc[0]
            require(
                name not in table and d.dtype == 2 and list(d.shape.dim) == sz,
                "wrong reference input " + name,
            )
            op.name = name
            op.type = "Data"
            op.attr["index"].i = len(abi["inputs"])
            op.input_desc.add().CopyFrom(d)
            op.output_desc.add().CopyFrom(d)
            target.input.append(name + ":0")
            abi["inputs"].append(dict(name=name, dtype="float16", shape=sz))
        else:
            require(old.type not in ("Data", "FileConstant"), "unexpected upstream input in Gate reference")
            op.CopyFrom(old)
            for i, v in enumerate(op.input):
                if edge(v) in cuts:
                    op.input[i] = cuts[edge(v)][0] + ":0"
        op.id = len(target.op) - 1
    end = target.op.add(name="NetOutput", type="NetOutput", id=len(target.op))
    for public, name, sz in (("x", x, [m, k]), ("y", projection.name, [m, n])):
        d = table[name].output_desc[0]
        require(d.dtype == 2 and list(d.shape.dim) == sz, "wrong Gate reference output")
        end.input.append(name + ":0")
        desc = end.input_desc.add()
        desc.CopyFrom(d)
        desc.name = public
        target.output.append(name + ":0")
        abi["outputs"].append(dict(name=public, dtype="float16", shape=sz))
    path = out / "reference.air"
    path.write_bytes(ref.SerializeToString(deterministic=True))
    weights = {}
    for file, key, pin, dt, fmt, sz in (
        ("w_nz.bin", "weight", "nz_sha256", 3, "FRACTAL_NZ", (k // 32, n // 16, 16, 32)),
        ("s_gn.bin", "scale", "gn_sha256", 2, "ND", (k // 128, n)),
    ):
        p = table[site[key]]
        d = p.output_desc[0]
        t = p.attr["value"].t
        require(
            p.type == "Const"
            and t.desc == d
            and d.dtype == dt
            and d.layout == fmt
            and tuple(d.shape.dim) == sz,
            "wrong raw projection constant: " + key,
        )
        dst = out / file
        dst.write_bytes(t.data)
        weights[file] = c.identity(dst)
        require(weights[file]["sha256"] == site[pin], "checkpoint projection bytes changed: " + key)
    return dict(
        reference_air=c.identity(path),
        reference_abi=abi,
        weights=weights,
        x_node=x,
        sites={role: site},
        cut_nodes={name: spec[0] for name, spec in cuts.items()},
    )


def single_graph(out, types, model, site, weights, *, shape=SHAPE, op_type=gate.OP, panel=None, name="gate"):
    """Keep the deployed Gate, constant names, ports and descriptors; expose only its ND activation."""
    m, k, n = shape
    panel = k if panel is None else panel
    (original,) = model.graph
    table = {op.name: op for op in original.op}
    op = table[site["node"]]
    require(
        op.type == op_type
        and [edge(v) for v in op.input[1:]] == [site["weight"], site["scale"]]
        and list(op.input_desc[0].shape.dim) == [m, k]
        and list(op.output_desc[0].shape.dim) == [m, n],
        "expected full-K panel in integrated AIR",
    )
    codes, scales, _ = gate.pack_cube_panels(
        *(c.verify(weights[name]).read_bytes() for name in ("w_nz.bin", "s_gn.bin")), panel, k=k, n=n
    )
    msg = types[0]()
    msg.name = "layer0_single_" + name
    g = msg.graph.add(name=msg.name)
    x = g.op.add(name="x", type="Data", id=0)
    x.attr["index"].i = 0
    x.output_desc.add().CopyFrom(op.input_desc[0])
    x.input_desc.add().CopyFrom(x.output_desc[0])
    g.input.append("x:0")
    constants = []
    dims = []
    for i, (key, expected, sz) in enumerate(
        (("weight", codes, [n // 64, k // panel, 64 * panel]), ("scale", scales, [n // 64, k // 128, 64])), 1
    ):
        node = table[site[key]]
        d = node.output_desc[0]
        t = node.attr["value"].t
        require(
            node.type == "Const"
            and d == t.desc
            and list(d.shape.dim) == sz
            and d.layout == "ND"
            and t.data == expected,
            "integrated packing differs from original NZ/GN: " + key,
        )
        cp = g.op.add()
        cp.CopyFrom(node)
        cp.id = i
        f = out / (key + "-packed.bin")
        f.write_bytes(expected)
        constants.append(c.identity(f))
        dims.append(sz)
    proj = g.op.add()
    proj.CopyFrom(op)
    proj.id = 3
    proj.input[0] = "x:0"
    end = g.op.add(name="NetOutput", type="NetOutput", id=4)
    end.input.append(proj.name + ":0")
    d = end.input_desc.add()
    d.CopyFrom(proj.output_desc[0])
    d.name = "y"
    g.output.append(proj.name + ":0")
    p = out / (name + ".air")
    p.write_bytes(msg.SerializeToString(deterministic=True))
    return c.identity(p), dict(physical_shapes=dims, constants=constants)


def source(out, runs, types):
    full_root = runs / DEPLOYMENT
    st = Stages(out, full_root)
    admit = st.completed("admit")["result"]
    case = admit["cases"]["16"]
    pkg = st.completed("opp-c16")["result"]
    atc = st.completed("atc-c16")
    audit = st.completed("audit-c16")
    full = audit["result"]
    smoke = st.completed("smoke-c16")
    bench = st.completed("benchmark-c16")
    final = c.read(full_root / "result.json")
    require(
        final["status"] == "MEASURED_AND_PROFILED"
        and final["gears"]["16"]["candidate"] == full["om"]
        and full["om"] == atc["result"]["om"]
        and full["om"]["sha256"] == FULL_OM
        and final["gears"]["16"]["inventory"] == st.refs["audit-c16"]
        and atc["contract"]["build"] == st.refs["opp-c16"]
        and audit["contract"]["compiled"] == st.refs["atc-c16"]
        and smoke["contract"]["models"][1] == full["om"]
        and bench["contract"]["smoke"] == st.refs["smoke-c16"]
        and final["gears"]["16"]["benchmark"] == st.refs["benchmark-c16"]
        and compare_rounds(bench["result"]["rounds"]) == bench["result"]["statistics"],
        "different measured full Gate",
    )
    deployed = full["inventory"][ROLE]
    query = full["plans"]["gate_up"]
    require(
        deployed["mode"] == CONTROL
        and deployed["op"] == gate.OP
        and deployed["tiling_sha256"] == query["tiling_sha256"]
        and deployed["selected"]["block_dim"] == 8
        and pkg["inputs"]["gate_up"]["device"]["source"] == case["assets"]["gate_up"]["device"],
        "wrong deployed Gate implementation",
    )
    check_elf(full, deployed["selected"])
    c.verify_opp(pkg["opp"])
    require(
        c.sha(gate.HERE / "op_kernel/full_l1.cpp") == c.sha(c.verify(case["assets"]["gate_up"]["device"]))
        and c.sha(gate.HERE / "op_host" / (gate.STEM + ".cpp"))
        == c.sha(c.verify(case["assets"]["gate_up"]["host"])),
        "different measured Gate host/device source",
    )
    rawst = Stages(out, runs / FULL)
    raw = rawst.completed("atc-c16-baseline")["result"]
    graph = case["graph"]
    site = graph["projections"][ROLE]
    require(
        rawst.refs["atc-c16-baseline"] == admit["control_receipts"]["atc-c16-baseline"]
        and raw["air"] == graph["baseline"],
        "different reference graph",
    )
    raw_query = raw["plans"][SHAPE_KEY]
    raw_image = raw["kernels"][site["node"]]
    check_elf(raw, raw_image)
    replays, prior = o_operands(out, runs, case, full, admit["control_receipts"])
    ref = reference_graph(out, types, types[0].FromString(c.verify(raw["air"]).read_bytes()), graph)
    air, packing = single_graph(
        out, types, types[0].FromString(c.verify(full["air"]).read_bytes()), site, ref["weights"]
    )
    require(
        [v["sha256"] for v in packing["constants"]] == [v["sha256"] for v in deployed["constants"]],
        "packed single Gate weights/scales differ from TglfhB",
    )
    return dict(
        ref,
        replays=replays,
        o_receipts=prior,
        full_receipts=st.refs,
        full_om=full["om"],
        capture_origin="LAYER0_ARCHIVED_EMBEDDING_AND_O_DEVICE_POST_NORM_RAW_GATE",
        pkg=admit["baseline_package"],
        control_package=pkg,
        single_air=air,
        packing=packing,
        deployed=deployed["selected"],
        deployed_query=query,
        source_plans={SHAPE_KEY: raw_query},
        source_implementations={ROLE: raw_image},
    )
