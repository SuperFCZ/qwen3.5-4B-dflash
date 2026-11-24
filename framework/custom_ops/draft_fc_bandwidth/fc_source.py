"""Read the measured W8 FC, its public feature replay and the GE input-selection subgraph."""

import hashlib

import numpy as np

from fc_panel import c, require, K, N, OLD_OP
from q_source import edge, parents
from experiment import Stages
from gate_plan import recheck_embedded

WEIGHTS = (
    "6f31693910b3561a7f33a1b47dcad061db08edab6a2599ed07d4fc80b91a769c",
    "cab921d55488d4bf6877da02ec0c242ef04e85aa747c3acc5ba76bf0dad761dd",
)
PINS = {
    16: (
        "dedb05f1aef78855ea6ecda305ba5fefc0597332e4650b993fce2640c40fdf21",
        "9da6c8972644cc98329460b4f0bcf319dbc0510a43dd9c88e28e131b8c1a60c2",
        "a572d2cf4de0db464c827a3d45c9b55da63f604a06b31b1ec6dffa8e40c214c4",
    ),
    64: (
        "bc8da80b10416f148c7319e2b9f002f45d28ff18700055bfa478863dd8936589",
        "7daf84fef97c537ac3fef7dd9fcb8fc7ab83d08257361d466df27fd2576a54e1",
        "96aaa528fb99dec254c771c69b92c082a110384729bba4a8115ee9afb2057ea1",
    ),
}
TYPES = {2: np.dtype("<f2"), 5: np.dtype("<i2"), 7: np.dtype("<i4"), 8: np.dtype("<i8"), 11: np.dtype("?")}


def replay_input(graph, target, files, m, k):
    table = {n.name: n for n in graph.op}
    require(len(table) == len(graph.op), "duplicate FC input node")
    needed = parents(table, target)
    values = {}
    require(
        {n for n in needed if table[n].type == "Data"} == {"features", "valid_rows"},
        "FC input lineage differs",
    )
    for op in graph.op:
        if op.name not in needed:
            continue
        dt = TYPES[op.output_desc[0].dtype]
        shape = tuple(op.output_desc[0].shape.dim)
        args = [values[edge(e)] for e in op.input]
        if op.type == "Data":
            v = np.fromfile(c.verify(files[op.name]), dt).reshape(shape)
        elif op.type == "Const":
            t = op.attr["value"].t
            require(t.desc == op.output_desc[0], "FC constant descriptor drift")
            v = np.frombuffer(t.data, dt).reshape(shape)
        elif op.type == "Cast":
            require(args[0].dtype.kind in "iu" and dt.kind in "iu", "unexpected floating FC input conversion")
            v = args[0].astype(dt)
        elif op.type == "Less":
            v = np.less(*args)
        elif op.type == "Reshape":
            v = args[0].reshape(tuple(args[1].tolist()))
        elif op.type == "BroadcastTo":
            v = np.broadcast_to(args[0], tuple(args[1].tolist()))
        elif op.type == "SelectV2":
            v = np.where(*args)
        else:
            raise ValueError("unexpected operation before FC: " + op.type)
        require(v.dtype == dt and v.shape == shape, "wrong FC input tensor: " + op.name)
        values[op.name] = v
    raw = values["features"]
    valid = int(values["valid_rows"].item())
    x = values[target]
    require(
        raw.shape == (1, m, k) and raw.dtype == np.dtype("<f2") and 0 < valid <= m and x.shape == (m, k),
        "invalid FC feature ABI",
    )
    expected = raw.reshape(m, k).copy()
    expected[valid:] = 0
    require(
        np.array_equal(expected.view("<u2"), x.view("<u2")) and np.isfinite(x).all(),
        "masked FC feature bits differ",
    )
    return np.ascontiguousarray(x), valid


def extract(out, model, record, replay, m, weights):
    (graph,) = model.graph
    table = {n.name: n for n in graph.op}
    op = table[record["node"]]
    require(
        op.type == OLD_OP
        and tuple(record[d] for d in ("m", "k", "n")) == (m, K, N)
        and [edge(v) for v in op.input[1:]] == [record["weight"], record["scale"]]
        and (record["nz_sha256"], record["gn_sha256"]) == WEIGHTS,
        "wrong checkpoint FC",
    )
    for name, key, dtype, fmt, shape, pin in (
        ("w_nz.bin", "weight", 3, "FRACTAL_NZ", (K // 32, N // 16, 16, 32), WEIGHTS[0]),
        ("s_gn.bin", "scale", 2, "ND", (K // 128, N), WEIGHTS[1]),
    ):
        node = table[record[key]]
        d = node.output_desc[0]
        v = node.attr["value"].t
        require(
            node.type == "Const"
            and v.desc == d
            and d.dtype == dtype
            and d.layout == fmt
            and tuple(d.shape.dim) == shape,
            "invalid FC weight/scale descriptor",
        )
        data = v.data
        require(hashlib.sha256(data).hexdigest() == pin, "FC weight/scale bytes differ")
        if name not in weights:
            p = out / name
            p.write_bytes(data)
            weights[name] = c.identity(p)
        require(weights[name]["sha256"] == pin, "cross-gear FC weight drift")
    x, valid = replay_input(graph, edge(op.input[0]), replay["files"], m, K)
    require(valid == replay["scalars"]["valid_rows"], "wrong FC valid_rows")
    path = out / f"features-c{m}.bin"
    x.tofile(path)
    # Keep actual FP16 patterns and exercise a different arrangement of rows, K groups and signs.
    stress = np.roll(x[::-1], 129, axis=1).copy()
    stress.view("<u2")[:, ::2] ^= 0x8000
    stress_path = out / f"stress-c{m}.bin"
    stress.tofile(stress_path)
    return dict(
        x=c.identity(path),
        stress=c.identity(stress_path),
        valid_rows=valid,
        x_node=edge(op.input[0]),
        dtype="float16",
    )


def read(out, root, types, gears):
    stages = Stages(out, root)
    graphs = c.read(c.verify(stages.completed("checkpoint-graphs")["result"]))
    pkg = stages.completed("private-opp")["result"]
    c.verify_opp(pkg["opp"])
    c.verify_tree(pkg["sources"])
    runner = stages.completed("runner")["result"]
    weights = {}
    cases = {}
    for m in gears:
        g = str(m)
        idx = graphs["gears"][g]
        r = idx["projections"]["fc"]
        stages.completed("atc-c" + g + "-baseline")
        opt = stages.completed("atc-c" + g + "-optimized")
        io = stages.completed("inputs-c" + g)
        ref = stages.completed("reference-c" + g)
        sm = stages.completed("smoke-c" + g)
        bench = stages.completed("benchmark-c" + g)
        require(
            opt["contract"]["pkg"] == stages.refs["private-opp"]
            and opt["result"]["air"] == opt["contract"]["air"] == idx["optimized"]
            and io["contract"]["abi"] == idx["abi"]
            and ref["contract"]["runner"] == runner
            and ref["contract"]["io"] == stages.refs["inputs-c" + g]
            and ref["contract"]["models"]
            == [stages.refs["atc-c" + g + "-" + mode] for mode in ("baseline", "optimized")]
            and sm["contract"] == dict(ref["contract"], reference=ref["result"]["reference"])
            and bench["contract"] == dict(sm["contract"], smoke=stages.refs["smoke-c" + g])
            and bench["result"]["statistics"]["variants"]["optimized"]["count"] == 90,
            "wrong measured full Draft C" + g,
        )
        query = opt["result"]["plans"][f"{m}x{K}x{N}"]
        kernel = opt["result"]["kernels"][r["node"]]
        parent, text, features = PINS[m]
        require(
            query["tiling_sha256"] == parent
            and kernel["block_dim"] == 8
            and kernel["executable_sections"] == [dict(bytes=14516, sha256=text)]
            and io["result"]["files"]["features"]["sha256"] == features
            and r["node"] == "proj_fc",
            "different archived FC",
        )
        recheck_embedded(
            c.verify(opt["result"]["om"]),
            [{k: v for k, v in kernel.items() if k not in ("offset", "executable_sections")}],
            {kernel["node"]: kernel},
        )
        cases[g] = dict(
            extract(
                out, types[0].FromString(c.verify(idx["optimized"]).read_bytes()), r, io["result"], m, weights
            ),
            kernel=kernel,
            query=query,
        )
    return dict(node="proj_fc", control_package=pkg, weights=weights, gears=cases, archive=stages.refs)
