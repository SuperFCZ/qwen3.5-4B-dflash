"""Build Target Gate/Up test graphs from the deployed public GE dataflow and tensor contracts."""

# ruff: noqa: E402 -- the board runs these sibling experiment modules directly.

from collections import Counter
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "draft_w8_e2e"))
from winners import c, require
from air import protobuf_classes
from packed_panel import nd_desc
from receiver_air import node_name

K, N = 2560, 9216
MM, DQ = "QuantBatchMatmulV4444", "DynamicQuant"
OP = "TargetGateUpPanel"
DT = {"float16": 2, "float32": 1, "int8": 3}


def spec(name, dtype, shape):
    return dict(name=name, dtype=dtype, shape=list(shape))


def dtype_shape(n):
    return n.attr["dtype"].dt, tuple(n.attr["shape"].list.i)


def file_path(n):
    path = Path(n.attr["file_path"].s.decode())
    require(n.type == "FileConstant" and path.is_absolute(), "expected an absolute receiver FileConstant")
    return path


def signature(n):
    return {k: v.SerializeToString(deterministic=True) for k, v in n.attr.items() if not k.startswith("_")}


def pairs(graph):
    nodes = {n.name: n for n in graph.op}
    found = {}
    for n in graph.op:
        if n.type != MM:
            continue
        w = nodes[node_name(n.input[1])]
        mt = re.fullmatch(r"body_layers_(\d+)_mlp_(gate|up)_proj_W_q", file_path(w).name)
        if not mt:
            continue
        i, branch = int(mt[1]), mt[2]
        scale = nodes[node_name(n.input[2])]
        dq, row = nodes[node_name(n.input[0])], nodes[node_name(n.input[5])]
        require(
            len(n.input) == 6
            and not any(n.input[3:5])
            and dtype_shape(w) == (2, (K, N))
            and file_path(w).stat().st_size == K * N
            and dtype_shape(scale) == (0, (1,))
            and file_path(scale).stat().st_size == 4
            and dq.type == DQ
            and len(dq.input) == 1
            and n.input[0] == dq.name + ":0"
            and row.type == "Reshape"
            and row.input[0] == dq.name + ":1",
            "unsupported Target quantized projection: " + n.name,
        )
        slots = found.setdefault(i, {})
        require(branch not in slots, "duplicate MLP projection")
        slots[branch] = dict(
            mm=n.name, weight=w.name, scale=scale.name, dq=dq.name, rows=row.name, x=dq.input[0]
        )
    require(set(found) == set(range(32)), "missing receiver MLP layers")
    for i, p in found.items():
        require(
            set(p) == {"gate", "up"}
            and p["gate"]["x"] == p["up"]["x"]
            and signature(nodes[p["gate"]["dq"]]) == signature(nodes[p["up"]["dq"]])
            and signature(nodes[p["gate"]["mm"]]) == signature(nodes[p["up"]["mm"]]),
            f"Gate/Up input or quantization differs at layer {i}",
        )
    return found


def tensor_node(cls, name, kind, ports, outputs):
    n = cls(name=name, type=kind)
    for edge, port, dt, shape in ports:
        n.input.append(edge)
        nd_desc(n.input_desc.add(), port, dt, shape)
    for port, dt, shape in outputs:
        nd_desc(n.output_desc.add(), port, dt, shape)
    return n


def subgraph(source, abi, output_edges, extra, rewires, new_inputs, name):
    """Slice by port, preserve control edges, and renumber only the retained public Data inputs."""
    (graph,) = source.graph
    data = sorted((v for v in graph.op if v.type in ("Data", "RefData")), key=lambda v: v.attr["index"].i)
    public = {n.name: s for n, s in zip(data, abi["inputs"], strict=True)}
    public.update(new_inputs)
    table = {n.name: n for n in [*graph.op, *extra] if n.type != "NetOutput"}

    def edge(e):
        return rewires.get(e, e)

    ordered, done, pending = [], set(), set()
    # Iterative postorder also works for full-depth Target graphs.
    for root, _ in output_edges:
        stack = [(node_name(edge(root)), False)]
        while stack:
            k, finish = stack.pop()
            if k in done:
                continue
            if finish:
                ordered.append(k)
                done.add(k)
                pending.remove(k)
                continue
            require(k not in pending and k in table, "invalid edge at " + k)
            pending.add(k)
            stack.append((k, True))
            stack.extend((node_name(edge(v)), False) for v in reversed(table[k].input) if v)
    msg = type(source)(name=name, version=source.version, custom_version=source.custom_version)
    g = msg.graph.add(name=name)
    used = [k for k in public if k in done]
    for k in used:
        g.input.append(k + ":0")
    for i, k in enumerate(ordered):
        n = g.op.add()
        n.CopyFrom(table[k])
        n.id = i
        n.input[:] = [edge(v) for v in n.input]
        if k in public:
            n.attr["index"].i = used.index(k)
    net = g.op.add(name="NetOutput", type="NetOutput", id=len(ordered))
    keys = []
    for i, (e, s) in enumerate(output_edges):
        net.input.append(edge(e))
        nd_desc(net.input_desc.add(), "", DT[s["dtype"]], s["shape"])
        keys.append((f"input{i}".encode(), i))
    for k, i in sorted(keys):
        net.attr["_input_name_key"].list.s.append(k)
        net.attr["_input_name_value"].list.i.append(i)
    return msg, dict(inputs=[public[k] for k in used], outputs=[s for _, s in output_edges])


def single_graph(source, abi, rows, mode, extra_builder=None):
    (graph,) = source.graph
    branch = pairs(graph)[0]
    gate, up = branch["gate"], branch["up"]
    cls = type(graph.op[0])
    xshape, yshape = [1, rows, K], [1, rows, N]
    x = spec("x", "float16", xshape)
    new_inputs, extras, rewires = {}, [], {}
    if mode != "prefix":
        data = tensor_node(cls, "pair_x", "Data", [], [("y", 2, xshape)])
        data.input.append("")
        nd_desc(data.input_desc.add(), "x", 2, xshape)
        extras.append(data)
        new_inputs[data.name] = x
        rewires[gate["x"]] = data.name + ":0"
    outputs = [(p["mm"] + ":0", spec(key, "float16", yshape)) for key, p in branch.items()]
    outputs.sort(key=lambda v: v[1]["name"])
    if mode in ("prefix", "oracle"):
        probe = tensor_node(
            cls, "pair_q_probe", "Cast", [(gate["dq"] + ":0", "x", 3, xshape)], [("y", 1, xshape)]
        )
        probe.attr["dst_type"].i = 0  # GE DT_FLOAT
        extras.append(probe)
        outputs.extend(
            [
                (probe.name + ":0", spec("q_probe", "float32", xshape)),
                (gate["rows"] + ":0", spec("row_scale", "float32", [rows])),
            ]
        )
        if mode == "prefix":
            outputs.append((gate["x"], x))
    elif mode == "shared":
        for port in (":0", ":1"):
            rewires[up["dq"] + port] = gate["dq"] + port
        rewires[up["rows"] + ":0"] = gate["rows"] + ":0"
    elif mode == "custom":
        require(extra_builder is not None, "missing packed weight binding")
        nodes, out = extra_builder(cls, branch, rows)
        extras.extend(nodes)
        outputs = [(out + ":" + str(i), spec(key, "float16", yshape)) for i, key in enumerate(("gate", "up"))]
    else:
        require(mode == "native", "unknown Gate/Up graph")
    return subgraph(source, abi, outputs, extras, rewires, new_inputs, "target_pair_" + mode)


def load_model(desc, air):
    cls = protobuf_classes(c.verify(desc).read_bytes())[0]
    return cls.FromString(c.verify(air).read_bytes())


def save_graph(folder, model, abi):
    path = folder / "model.air"
    path.write_bytes(model.SerializeToString(deterministic=True))
    return dict(air=c.identity(path), abi=abi, counts=dict(Counter(n.type for n in model.graph[0].op)))
