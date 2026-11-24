"""Extract the Layer0 Q constants and a small device reference graph from the measured C16 AIR."""

from pathlib import Path

import numpy as np

from q_panel import c, require, SHAPE, ROLE, OLD_OP, M, K, N, TAG
from experiment import Stages
from gate_plan import recheck_embedded

NZ_SHA = "da1c1106875865efab44729da428b2194468902e285d4788f0501e5764e0ee73"
GN_SHA = "5a4edd4ab75af6e5b06b9469ccd600c2ce59ed58b0600be59be38156bd1a6bdc"
PARENT = "91cf1d40645a972189f0f0ddb9deb8c361181b71e06e29bcf8431a9ea052d824"
TEXT = "e8f0beabf801c607956c0f01458669941b8782cff4d46e0ba4e8f8308f3c34d3"


def edge(value):
    name, port = value.rsplit(":", 1)
    require(port == "0" and name, "unexpected Q prefix edge: " + value)
    return name


def parents(table, root, stop=None):
    found = set()

    def visit(name):
        if name not in found:
            found.add(name)
            if name != stop:
                for e in table[name].input:
                    visit(edge(e))

    visit(root)
    return found


def index_value(table, name, anchor):
    """The Gather indices consist only of the captured INT64 anchor and the checkpoint mask IDs."""
    op = table[name]
    if op.type == "Data":
        require(name == "anchor", "unexpected runtime input to token selection")
        return anchor
    if op.type == "Const":
        dt = {7: "<i4", 8: "<i8"}[op.output_desc[0].dtype]
        return np.frombuffer(op.attr["value"].t.data, dt).reshape(tuple(op.output_desc[0].shape.dim))
    args = [index_value(table, edge(e), anchor) for e in op.input]
    if op.type == "Reshape":
        return args[0].reshape(tuple(args[1].tolist()))
    if op.type == "ConcatD":
        return np.concatenate(args, axis=op.attr["concat_dim"].i)
    raise ValueError("unexpected token selection op: " + op.type)


def extract(out, types, model, index, embedding, anchor):
    (graph,) = model.graph
    table = {op.name: op for op in graph.op}
    require(len(table) == len(graph.op), "duplicate original node")
    record = index["projections"][ROLE]
    require(tuple(record[k] for k in ("m", "k", "n")) == SHAPE, "wrong Q geometry")
    q = table[record["node"]]
    require(
        q.type == OLD_OP and list(map(edge, q.input[1:])) == [record["weight"], record["scale"]],
        "different Q implementation or weights",
    )
    x = edge(q.input[0])
    (gather,) = (table[n] for n in parents(table, x) if table[n].type == "GatherV2")
    require(
        tuple(gather.output_desc[0].shape.dim) == (1, M, K)
        and gather.output_desc[0].dtype == 2
        and gather.attr["batch_dims"].i == 0,
        "unexpected token embedding ABI",
    )
    file, indices, axis = map(edge, gather.input)
    lookup = table[file]
    vocab, width = tuple(lookup.output_desc[0].shape.dim)
    require(
        lookup.type == "FileConstant"
        and c.verify(embedding) == Path(lookup.attr["file_path"].s.decode()).resolve(strict=True)
        and lookup.output_desc[0].dtype == 2
        and width == K
        and lookup.attr["dtype"].dt == 1
        and embedding["bytes"] == vocab * K * 2
        and int(index_value(table, axis, anchor).item()) == 0,
        "unexpected FP16 embedding binding",
    )
    ids = index_value(table, indices, anchor)
    require(
        ids.dtype == np.dtype("<i8")
        and ids.shape == (1, M)
        and ids[0, 0] == anchor.item()
        and (ids[0, 1:] == ids[0, 1]).all()
        and (ids >= 0).all()
        and (ids < vocab).all(),
        "wrong Layer0 token sequence",
    )
    weights = np.memmap(embedding["path"], dtype="<f2", mode="r", shape=(vocab, K))
    selected = weights[ids]
    require(np.isfinite(selected).all(), "nonfinite token embeddings")
    selected.tofile(out / "embedding.bin")
    stress_ids = np.random.default_rng(3104096).choice(vocab, size=(1, M), replace=False)
    stress = weights[stress_ids]
    require(
        np.isfinite(stress).all() and np.unique(stress.reshape(M, K), axis=0).shape[0] == M,
        "dense token stress requires sixteen distinct finite embeddings",
    )
    stress.tofile(out / "stress-embedding.bin")
    wanted = parents(table, q.name, stop=gather.name)
    supported = {"Const", "Cast", "Reshape", "Square", "ReduceMean", "Rsqrt", "Mul", "Add", OLD_OP}
    require(
        all(table[n].type in supported for n in wanted - {gather.name})
        and sum(table[n].type == OLD_OP for n in wanted) == 1,
        "unexpected upstream computation before Layer0 Q",
    )
    ref = types[0]()
    ref.name = "q_layer0_reference"
    g = ref.graph.add(name=ref.name)
    for old in graph.op:
        if old.name not in wanted:
            continue
        node = g.op.add()
        if old.name == gather.name:
            node.name = old.name
            node.type = "Data"
            node.attr["index"].i = 0
            node.output_desc.add().CopyFrom(old.output_desc[0])
            node.input_desc.add().CopyFrom(old.output_desc[0])
            g.input.append(node.name + ":0")
        else:
            node.CopyFrom(old)
        node.id = len(g.op) - 1
    end = g.op.add(name="NetOutput", type="NetOutput", id=len(g.op))
    for n, name, shape in ((x, "x", (M, K)), (q.name, "y", (M, N))):
        d = table[n].output_desc[0]
        require(d.dtype == 2 and tuple(d.shape.dim) == shape, "wrong Q reference outputs")
        end.input.append(n + ":0")
        e = end.input_desc.add()
        e.CopyFrom(d)
        e.name = name
        g.output.append(n + ":0")
    air = out / "reference.air"
    air.write_bytes(ref.SerializeToString(deterministic=True))
    files = {}
    for label, key, pin, fmt, dt, shape in [
        ("w_nz.bin", "weight", "nz_sha256", "FRACTAL_NZ", 3, (K // 32, N // 16, 16, 32)),
        ("s_gn.bin", "scale", "gn_sha256", "ND", 2, (K // 128, N)),
    ]:
        node = table[record[key]]
        require(
            node.type == "Const"
            and node.output_desc[0].layout == fmt
            and node.output_desc[0].dtype == dt
            and tuple(node.output_desc[0].shape.dim) == shape
            and node.attr["value"].t.desc == node.output_desc[0],
            "wrong Q constant descriptor: " + key,
        )
        p = out / label
        p.write_bytes(node.attr["value"].t.data)
        files[label] = c.identity(p)
        require(files[label]["sha256"] == record[pin], "different checkpoint bytes: " + label)
    return dict(
        weights=files,
        reference_air=c.identity(air),
        embedding=c.identity(out / "embedding.bin"),
        stress_embedding=c.identity(out / "stress-embedding.bin"),
        stress_token_ids=stress_ids.tolist(),
        selected_token_ids=ids.tolist(),
        q_node=q.name,
        x_node=x,
        prefix_nodes=len(g.op),
    )


def read_source(out, root, types):
    stages = Stages(out, root)
    graphs = c.read(c.verify(stages.completed("checkpoint-graphs")["result"]))
    pkg = stages.completed("private-opp")["result"]
    c.verify_opp(pkg["opp"])
    c.verify_tree(pkg["sources"])
    require(
        graphs["target"]["embedding"]["fp16"]["bytes"] == 248320 * K * 2, "unexpected full vocabulary table"
    )
    runner = stages.completed("runner")["result"]
    stages.completed("atc-c16-baseline")
    opt = stages.completed("atc-c16-optimized")
    io = stages.completed("inputs-c16")
    ref = stages.completed("reference-c16")
    smoke = stages.completed("smoke-c16")
    bench = stages.completed("benchmark-c16")
    g = graphs["gears"]["16"]
    query = opt["result"]["plans"][TAG]
    compiled = opt["result"]["kernels"][g["projections"][ROLE]["node"]]
    key = dict(ref["contract"], reference=ref["result"]["reference"])
    require(
        opt["contract"]["pkg"] == stages.refs["private-opp"]
        and opt["result"]["air"] == opt["contract"]["air"] == g["optimized"]
        and io["contract"]["abi"] == g["abi"]
        and ref["contract"]["io"] == stages.refs["inputs-c16"]
        and ref["contract"]["runner"] == runner
        and ref["contract"]["models"] == [stages.refs["atc-c16-baseline"], stages.refs["atc-c16-optimized"]]
        and smoke["contract"] == key
        and bench["contract"] == dict(key, smoke=stages.refs["smoke-c16"])
        and bench["result"]["statistics"]["variants"]["optimized"]["count"] == 90,
        "wrong measured full-Draft predecessor",
    )
    require(
        query["tiling_sha256"] == PARENT
        and compiled["block_dim"] == 8
        and compiled["executable_sections"] == [dict(bytes=14516, sha256=TEXT)],
        "different UFqlys Layer0 Q control",
    )
    om = c.verify(opt["result"]["om"])
    recheck_embedded(
        om,
        [{k: v for k, v in compiled.items() if k not in ("executable_sections", "offset")}],
        {compiled["node"]: compiled},
    )
    anchor = np.fromfile(c.verify(io["result"]["files"]["anchor"]), dtype="<i8")
    require(anchor.shape == (1,) and anchor[0] == io["result"]["scalars"]["anchor"], "wrong measured anchor")
    require(
        (g["projections"][ROLE]["nz_sha256"], g["projections"][ROLE]["gn_sha256"]) == (NZ_SHA, GN_SHA),
        "different Q checkpoint",
    )
    data = extract(
        out,
        types,
        types[0].FromString(c.verify(g["optimized"]).read_bytes()),
        g,
        graphs["target"]["embedding"]["fp16"],
        anchor,
    )
    return dict(
        data,
        full_run=str(root),
        source_stages=stages.refs,
        control_package=pkg,
        query=query,
        compiled_q=compiled,
    )
