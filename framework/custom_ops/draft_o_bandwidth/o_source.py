"""Recover real Layer0 Attention→O operands and the raw device reference from the accepted C16 graph."""

from pathlib import Path

import numpy as np

from o_panel import c, require, SHAPE, OLD_OP, TAG, ROLE
from q_source import edge, parents, index_value
from experiment import Stages
from gate_plan import recheck_embedded

NZ_SHA = "48a22ef01c4a87c5e92e55d487c71c71f4cc64e6e4ecd6e8f1cef5bdb75f567f"
GN_SHA = "dbbb820adb67e3bfcb9a1b3823ee9d0de22e0dd32290f5ce06fd130794b224d2"
PARENT = "4920dd17e7d36bd998e3f4824300058d4048f2cb6cd3b0ff1d8dc81f0088d84f"
TEXT = "07f2874c19ea2f826fd464412a5fb975e9fc6eb5655c9ee4604d83bc3d28adfb"
ROLES = ("fc", "layer0.kv", "layer0.q", ROLE)
DTYPES = {2: "float16", 5: "int16", 8: "int64"}


def embedding_rows(out, table, op, blob, anchor, m, n):
    name, indices, axis = map(edge, op.input)
    lookup = table[name]
    require(
        lookup.type == "FileConstant"
        and Path(lookup.attr["file_path"].s.decode()).resolve() == c.verify(blob)
        and lookup.attr["dtype"].dt == 1
        and op.attr["batch_dims"].i == 0
        and index_value(table, axis, anchor).item() == 0,
        "different frozen embedding table or Gather axis",
    )
    vocab, width = lookup.output_desc[0].shape.dim
    require(
        lookup.output_desc[0].dtype == 2 and width == n and blob["bytes"] == 2 * vocab * width,
        "wrong Target embedding extent",
    )
    ids = index_value(table, indices, anchor)
    require(
        ids.dtype == np.dtype("<i8")
        and ids.shape == (1, m)
        and ids[0, 0] == anchor.item()
        and (ids[0, 1:] == ids[0, 1]).all()
        and (ids >= 0).all()
        and (ids < vocab).all(),
        "invalid original Layer0 token sequence",
    )
    stress = np.random.default_rng(3102560).choice(vocab, (1, m), replace=False)
    weights = np.memmap(blob["path"], "<f2", mode="r", shape=(vocab, width))
    result = {}
    for label, tokens in (("measured", ids), ("stress", stress)):
        v = weights[tokens]
        require(np.isfinite(v).all(), "nonfinite embedding")
        if label == "stress":
            require(np.unique(v.reshape(m, width), axis=0).shape[0] == m, "stress embeddings not distinct")
        dest = out / (label + "-embeddings.bin")
        v.tofile(dest)
        result[label] = dict(tokens=tokens.tolist(), file=c.identity(dest))
    return result


def extract(out, types, model, gear, embedding, io, *, shape=SHAPE):
    m, k, n_out = shape
    (original,) = model.graph
    table = {op.name: op for op in original.op}
    require(len(table) == len(original.op), "duplicate original graph node")
    sites = {role: gear["projections"][role] for role in ROLES}
    site = sites[ROLE]
    o = table[site["node"]]
    require(
        tuple(site[n] for n in ("m", "k", "n")) == shape
        and o.type == OLD_OP
        and [edge(v) for v in o.input[1:]] == [site["weight"], site["scale"]],
        "different O projection or operands",
    )
    x = edge(o.input[0])
    output_abi = gear["abi"]["outputs"]
    require(len(output_abi) == len(original.output), "public output arity differs")
    kv = {
        a["name"]: edge(e) for a, e in zip(output_abi, original.output) if a["name"] in ("d0_key", "d0_value")
    }
    require(set(kv) == {"d0_key", "d0_value"}, "missing complete Layer0 KV")
    ancestors = parents(table, x)
    (lookup,) = (
        table[n]
        for n in ancestors
        if table[n].type == "GatherV2" and table[edge(table[n].input[0])].type == "FileConstant"
    )
    require(
        any(gear["modules"].get(n) == "layer0.attention" for n in ancestors),
        "O input not produced by Layer0 attention",
    )
    anchor = np.fromfile(c.verify(io["files"]["anchor"]), "<i8")
    require(anchor.shape == (1,) and anchor.item() == io["scalars"]["anchor"], "invalid archived anchor")
    embeddings = embedding_rows(out, table, lookup, embedding, anchor, m, n_out)
    wanted = parents(table, o.name, lookup.name)
    for name in kv.values():
        wanted.update(parents(table, name, lookup.name))
    require(
        {n for n in wanted if table[n].type.startswith("DFlash")} == {s["node"] for s in sites.values()}
        and not any(table[n].type == "FileConstant" for n in wanted),
        "unexpected prefix projections or external weights",
    )
    original_inputs = {a["name"]: a for a in gear["abi"]["inputs"]}
    replay = {label: {} for label in embeddings}
    ref = types[0]()
    ref.name = "c16_layer0_attention_o"
    g = ref.graph.add(name=ref.name)
    abi = dict(inputs=[], outputs=[])
    for old in original.op:
        if old.name not in wanted:
            continue
        op = g.op.add()
        if old.name == lookup.name or old.type == "Data":
            name = "embedding_rows" if old.name == lookup.name else old.name
            require(name == old.name or name not in table, "embedding replacement name collision")
            desc = old.output_desc[0]
            spec = dict(name=name, dtype=DTYPES[desc.dtype], shape=list(desc.shape.dim))
            if old.name == lookup.name:
                require(
                    spec["dtype"] == "float16" and spec["shape"] == [1, m, n_out],
                    "unexpected embedding shape",
                )
            else:
                require(original_inputs[name] == spec, "different prefix input ABI: " + name)
                require(
                    c.verify(io["files"][name]).stat().st_size
                    == int(np.prod(spec["shape"])) * np.dtype(spec["dtype"]).itemsize,
                    "input extent: " + name,
                )
            op.name = name
            op.type = "Data"
            op.attr["index"].i = len(abi["inputs"])
            op.input_desc.add().CopyFrom(desc)
            op.output_desc.add().CopyFrom(desc)
            g.input.append(name + ":0")
            abi["inputs"].append(spec)
            for label, data in replay.items():
                data[name] = embeddings[label]["file"] if old.name == lookup.name else io["files"][name]
        else:
            op.CopyFrom(old)
            for i, e in enumerate(op.input):
                if edge(e) == lookup.name:
                    op.input[i] = "embedding_rows:0"
        op.id = len(g.op) - 1
    require(
        {"features", "start_position", "valid_rows", "d0_key", "d0_value", "embedding_rows"}
        <= {a["name"] for a in abi["inputs"]},
        "missing real context or KV inputs",
    )
    end = g.op.add(name="NetOutput", type="NetOutput", id=len(g.op))
    public = {a["name"]: a for a in output_abi}
    for name, node, shape in (
        ("x", x, [m, k]),
        ("y", o.name, [m, n_out]),
        *((name, kv[name], public[name]["shape"]) for name in ("d0_key", "d0_value")),
    ):
        desc = table[node].output_desc[0]
        require(desc.dtype == 2 and list(desc.shape.dim) == shape, "different O prefix output: " + name)
        end.input.append(node + ":0")
        d = end.input_desc.add()
        d.CopyFrom(desc)
        d.name = name
        g.output.append(node + ":0")
        abi["outputs"].append(dict(name=name, dtype="float16", shape=shape))
    air = out / "reference.air"
    air.write_bytes(ref.SerializeToString(deterministic=True))
    files = {}
    for name, key, pin, fmt, dt, shape in (
        ("w_nz.bin", "weight", "nz_sha256", "FRACTAL_NZ", 3, (k // 32, n_out // 16, 16, 32)),
        ("s_gn.bin", "scale", "gn_sha256", "ND", 2, (k // 128, n_out)),
    ):
        node = table[site[key]]
        desc = node.output_desc[0]
        require(
            node.type == "Const"
            and node.attr["value"].t.desc == desc
            and desc.dtype == dt
            and desc.layout == fmt
            and tuple(desc.shape.dim) == shape,
            "O constant descriptor: " + key,
        )
        path = out / name
        path.write_bytes(node.attr["value"].t.data)
        files[name] = c.identity(path)
        require(files[name]["sha256"] == site[pin], "O checkpoint constant drift: " + key)
    return dict(
        reference_air=c.identity(air),
        reference_abi=abi,
        replays=replay,
        tokens={k: v["tokens"] for k, v in embeddings.items()},
        weights=files,
        o_node=o.name,
        x_node=x,
        sites=sites,
        prefix_nodes=len(g.op),
    )


def read_source(out, root, types):
    stages = Stages(out, root)
    bundle = c.read(c.verify(stages.completed("checkpoint-graphs")["result"]))
    pkg = stages.completed("private-opp")["result"]
    c.verify_opp(pkg["opp"])
    c.verify_tree(pkg["sources"])
    runner = stages.completed("runner")["result"]
    stages.completed("atc-c16-baseline")
    opt = stages.completed("atc-c16-optimized")
    io = stages.completed("inputs-c16")
    ref = stages.completed("reference-c16")
    smoke = stages.completed("smoke-c16")
    bench = stages.completed("benchmark-c16")
    gear = bundle["gears"]["16"]
    m = opt["result"]
    require(
        m["air"] == gear["optimized"] == opt["contract"]["air"]
        and opt["contract"]["pkg"] == stages.refs["private-opp"]
        and io["contract"]["abi"] == gear["abi"]
        and ref["contract"]["io"] == stages.refs["inputs-c16"]
        and ref["contract"]["runner"] == runner
        and ref["contract"]["models"] == [stages.refs["atc-c16-baseline"], stages.refs["atc-c16-optimized"]]
        and smoke["contract"] == dict(ref["contract"], reference=ref["result"]["reference"])
        and bench["contract"] == dict(smoke["contract"], smoke=stages.refs["smoke-c16"])
        and bench["result"]["statistics"]["variants"]["optimized"]["count"] == 90,
        "different accepted C16 measurement",
    )
    site = gear["projections"][ROLE]
    image = m["kernels"][site["node"]]
    query = m["plans"][TAG]
    require(
        (site["nz_sha256"], site["gn_sha256"], query["tiling_sha256"]) == (NZ_SHA, GN_SHA, PARENT)
        and image["block_dim"] == 8
        and image["executable_sections"] == [dict(bytes=14516, sha256=TEXT)],
        "different measured O constant, tiling or machine text",
    )
    emitted = {
        m["kernels"][gear["projections"][r]["node"]]["node"]: m["kernels"][gear["projections"][r]["node"]]
        for r in ROLES
    }
    recheck_embedded(
        c.verify(m["om"]),
        [
            {k: v for k, v in x.items() if k not in ("offset", "executable_sections")}
            for x in emitted.values()
        ],
        emitted,
    )
    data = extract(
        out,
        types,
        types[0].FromString(c.verify(m["air"]).read_bytes()),
        gear,
        bundle["target"]["embedding"]["fp16"],
        io["result"],
    )
    expected = {name: ref["result"]["reference"][name] for name in ("d0_key", "d0_value")}
    c.verify_tree(expected)
    return dict(
        data,
        pkg=pkg,
        query=query,
        compiled_o=image,
        expected_kv=expected,
        archive=stages.refs,
        source_implementations={r: m["kernels"][gear["projections"][r]["node"]] for r in ROLES},
        source_plans={
            "x".join(str(row[k]) for k in ("m", "k", "n")): m["plans"][
                "x".join(str(row[k]) for k in ("m", "k", "n"))
            ]
            for row in data["sites"].values()
        },
    )
