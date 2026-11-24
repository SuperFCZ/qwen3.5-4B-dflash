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


def extract(out, types, model, gear, embedding, io, *, shape=SHAPE, role=ROLE, roles=ROLES):
    m, k, n_out = shape
    (original,) = model.graph
    table = {op.name: op for op in original.op}
    require(len(table) == len(original.op), "duplicate original graph node")
    sites = {r: gear["projections"][r] for r in roles}
    site = sites[role]
    projection = table[site["node"]]
    require(
        tuple(site[n] for n in ("m", "k", "n")) == shape
        and projection.type == OLD_OP
        and [edge(v) for v in projection.input[1:]] == [site["weight"], site["scale"]],
        f"different {role} projection or operands",
    )
    x = edge(projection.input[0])
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
    if role == "layer0.kv":
        # The first rows are FC+hidden_norm(context); the last 16 use normalized token embeddings.
        flat = table[x]
        cat = table[edge(flat.input[0])]
        require(
            flat.type == "Reshape"
            and cat.type == "ConcatD"
            and cat.attr["concat_dim"].i == 1
            and len(cat.input) == 2,
            "KV input must concatenate context then proposal rows",
        )
        ctx, noise = (table[edge(v)] for v in cat.input)
        require(
            list(ctx.output_desc[0].shape.dim) == [1, m - 16, k]
            and list(noise.output_desc[0].shape.dim) == [1, 16, k]
            and sites["fc"]["node"] in parents(table, ctx.name)
            and lookup.name in parents(table, noise.name),
            "KV input lineage or row order changed",
        )
        token_rows, hidden_width = 16, k
    else:
        require(
            role == ROLE and any(gear["modules"].get(n) == "layer0.attention" for n in ancestors),
            "O input not produced by Layer0 attention",
        )
        token_rows, hidden_width = m, n_out
    anchor = np.fromfile(c.verify(io["files"]["anchor"]), "<i8")
    require(anchor.shape == (1,) and anchor.item() == io["scalars"]["anchor"], "invalid archived anchor")
    embeddings = embedding_rows(out, table, lookup, embedding, anchor, token_rows, hidden_width)
    wanted = parents(table, projection.name, lookup.name)
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
    ref.name = "c16_layer0_attention_o" if role == ROLE else f"c{m-16}_layer0_kv"
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
                    spec["dtype"] == "float16" and spec["shape"] == [1, token_rows, hidden_width],
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
        ("y", projection.name, [m, n_out]),
        *((name, kv[name], public[name]["shape"]) for name in ("d0_key", "d0_value")),
    ):
        desc = table[node].output_desc[0]
        require(desc.dtype == 2 and list(desc.shape.dim) == shape, f"different {role} prefix output: {name}")
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
            f"{role} constant descriptor: {key}",
        )
        path = out / name
        path.write_bytes(node.attr["value"].t.data)
        files[name] = c.identity(path)
        require(files[name]["sha256"] == site[pin], f"{role} checkpoint constant drift: {key}")
    return dict(
        reference_air=c.identity(air),
        reference_abi=abi,
        replays=replay,
        tokens={k: v["tokens"] for k, v in embeddings.items()},
        weights=files,
        **{role.split(".")[-1] + "_node": projection.name},
        x_node=x,
        sites=sites,
        prefix_nodes=len(g.op),
    )


def read_source(out, root, types):
    return read_projection_source(
        out, root, types, 16, ROLE, ROLES, SHAPE, (NZ_SHA, GN_SHA, PARENT, 14516, TEXT)
    )


def read_projection_source(out, root, types, gear_id, role, roles, shape, pins):
    """Bind the prefix to a successfully smoked and measured full Draft archive."""
    g = str(gear_id)
    tag = "x".join(map(str, shape))
    stages = Stages(out, root)
    bundle = c.read(c.verify(stages.completed("checkpoint-graphs")["result"]))
    pkg = stages.completed("private-opp")["result"]
    c.verify_opp(pkg["opp"])
    c.verify_tree(pkg["sources"])
    runner = stages.completed("runner")["result"]
    stages.completed(f"atc-c{g}-baseline")
    opt = stages.completed(f"atc-c{g}-optimized")
    io = stages.completed(f"inputs-c{g}")
    ref = stages.completed(f"reference-c{g}")
    smoke = stages.completed(f"smoke-c{g}")
    bench = stages.completed(f"benchmark-c{g}")
    gear = bundle["gears"][g]
    m = opt["result"]
    require(
        m["air"] == gear["optimized"] == opt["contract"]["air"]
        and opt["contract"]["pkg"] == stages.refs["private-opp"]
        and io["contract"]["abi"] == gear["abi"]
        and ref["contract"]["io"] == stages.refs[f"inputs-c{g}"]
        and ref["contract"]["runner"] == runner
        and ref["contract"]["models"]
        == [stages.refs[f"atc-c{g}-baseline"], stages.refs[f"atc-c{g}-optimized"]]
        and smoke["contract"] == dict(ref["contract"], reference=ref["result"]["reference"])
        and bench["contract"] == dict(smoke["contract"], smoke=stages.refs[f"smoke-c{g}"])
        and bench["result"]["statistics"]["variants"]["optimized"]["count"] == 90,
        f"different accepted C{g} measurement",
    )
    site = gear["projections"][role]
    image = m["kernels"][site["node"]]
    query = m["plans"][tag]
    require(
        (site["nz_sha256"], site["gn_sha256"], query["tiling_sha256"]) == pins[:3]
        and image["block_dim"] == 8
        and image["executable_sections"] == [dict(bytes=pins[3], sha256=pins[4])],
        f"different measured {role} constant, tiling or machine text",
    )
    emitted = {
        m["kernels"][gear["projections"][r]["node"]]["node"]: m["kernels"][gear["projections"][r]["node"]]
        for r in roles
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
        shape=shape,
        role=role,
        roles=roles,
    )
    expected = {name: ref["result"]["reference"][name] for name in ("d0_key", "d0_value")}
    c.verify_tree(expected)
    return dict(
        data,
        pkg=pkg,
        query=query,
        **{"compiled_" + role.split(".")[-1]: image},
        expected_kv=expected,
        archive=stages.refs,
        source_implementations={r: m["kernels"][gear["projections"][r]["node"]] for r in roles},
        source_plans={
            "x".join(str(row[k]) for k in ("m", "k", "n")): m["plans"][
                "x".join(str(row[k]) for k in ("m", "k", "n"))
            ]
            for row in data["sites"].values()
        },
    )
