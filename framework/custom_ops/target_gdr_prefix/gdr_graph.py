"""Identify the committed FP32 state by output port and slice real GDR operands from Verify."""

from collections import Counter
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "target_gate_up"))
from pair_graph import require, nd_desc, spec, subgraph, tensor_node, node_name  # noqa: E402
from receiver_air import const_int  # noqa: E402

GDR = "ChunkGatedDeltaRule"
COMMIT = "TargetGdrPrefixState"
INPUTS = ("query", "key", "value", "g", "beta", "initial", "length")
TYPES = (2, 2, 2, 1, 2, 1, 5)
DTYPES = ("float16", "float16", "float16", "float32", "float16", "float32", "int16")
STATE = [1, 32, 128, 128]


def specs(physical=16):
    shapes = [[1, physical, 32, 128]] * 3 + [[1, physical, 32]] * 2 + [STATE, [1]]
    return [spec(name, dt, shape) for name, dt, shape in zip(INPUTS, DTYPES, shapes, strict=True)]


def sites(model, abi):
    (g,) = model.graph
    table = {n.name: n for n in g.op}
    (net,) = (n for n in g.op if n.type == "NetOutput")
    ports = {v["name"]: e for v, e in zip(abi["outputs"], net.input)}
    ds = sorted((n for n in g.op if n.type in ("Data", "RefData")), key=lambda n: n.attr["index"].i)
    ins = {v["name"]: n.name + ":0" for v, n in zip(abi["inputs"], ds, strict=True)}
    users = Counter(e for n in g.op for e in n.input if e)
    result = {}
    for v in abi["outputs"]:
        name = v["name"]
        if not re.fullmatch(r"t\d+_recurrent", name):
            continue
        end, discard = ports[name], ports["verify_discard_" + name]
        first, commit = table[node_name(discard)], table[node_name(end)]
        require(
            first.name != commit.name
            and first.type == commit.type == GDR
            and len(first.input) == len(commit.input) == 7
            and list(first.input[:6]) == list(commit.input[:6])
            and first.input[6] == ins["valid_rows"]
            and first.input[5] == ins[name]
            and end == commit.name + ":1"
            and discard == first.name + ":1"
            and v["dtype"] == "float32"
            and v["shape"] == STATE
            and not users[commit.name + ":0"]
            and not any(e == commit.name + ":0" for e in g.output),
            "unexpected native state boundary: " + name,
        )
        for n in (first, commit):
            require(
                n.attr["chunk_size"].i == 64
                and n.attr["use_qk_l2norm_in_kernel"].b
                and n.attr["output_final_state"].b
                and n.output_desc[1].dtype == 1,
                "different GDR arithmetic policy",
            )
        length = table[node_name(commit.input[6])]
        require(length.type == "Cast" and length.output_desc[0].dtype == 5, "commit length is not INT16")
        inc = table[node_name(length.input[0])]
        require(
            inc.type == "Add"
            and inc.input[0] == ports["accepted_count"]
            and const_int(table[node_name(inc.input[1])]).reshape(-1).tolist() == [1],
            "wrong commit prefix",
        )
        result[name] = dict(first=first.name, commit=commit.name, operands=list(commit.input))
    require(
        len({v["commit"] for v in result.values()}) == len(result) > 0, "duplicate or empty commit inventory"
    )
    return result


def prefix_graph(model, abi, site):
    edges = zip(site["operands"][:5], specs()[:5], strict=True)
    return subgraph(model, abi, list(edges), [], {}, {}, "target_gdr_real_operands")


def isolated(model, site, chunk=64, physical=16, custom=False, level=2):
    """Keep all inputs dynamic. Only the state output is live, as in the deployed second pass."""
    (g,) = model.graph
    src = next(n for n in g.op if n.name == site["commit"])
    out = type(model)(name="target_gdr_state", version=model.version, custom_version=model.custom_version)
    graph = out.graph.add(name=out.name)
    public = specs(physical)
    if custom:
        public = public[1:]  # A state-only kernel has no Query or attention-output port.
    types = dict(zip(INPUTS, TYPES, strict=True))
    cls = type(src)
    for i, s in enumerate(public):
        dt = types[s["name"]]
        n = tensor_node(cls, "in_" + s["name"], "Data", [], [("y", dt, s["shape"])])
        n.id = i
        n.attr["index"].i = i
        n.input.append("")
        nd_desc(n.input_desc.add(), "x", dt, s["shape"])
        graph.op.append(n)
        graph.input.append(n.name + ":0")
    args = [("in_" + s["name"] + ":0", s["name"], types[s["name"]], s["shape"]) for s in public]
    if custom:
        op = tensor_node(cls, "prefix_state", COMMIT, args, [("state", 1, STATE)])
        op.attr["prune_level"].i = level
        port = 0
    else:
        op = cls()
        op.CopyFrom(src)
        op.name = "prefix_state"
        op.input[:] = [a[0] for a in args]
        for d, (_, _, dt, shape) in zip(op.input_desc, args, strict=True):
            portname = d.name
            d.Clear()
            nd_desc(d, portname, dt, shape)
        for d, dt, sh in zip(op.output_desc, (2, 1), ([1, physical, 32, 128], STATE), strict=True):
            portname = d.name
            d.Clear()
            nd_desc(d, portname, dt, sh)
        op.attr["chunk_size"].i = chunk
        port = 1
    op.id = len(graph.op)
    graph.op.append(op)
    net = graph.op.add(name="NetOutput", type="NetOutput", id=len(graph.op))
    net.input.append(op.name + ":" + str(port))
    nd_desc(net.input_desc.add(), "", 1, STATE)
    net.attr["_input_name_key"].list.s.append(b"input0")
    net.attr["_input_name_value"].list.i.append(0)
    return out, dict(inputs=public, outputs=[spec("state", "float32", STATE)])


def replace_commit(model, abi, level=2):
    """Replace only the second-pass state producer. Keep every first-pass output and public port."""
    bound = sites(model, abi)
    (graph,) = model.graph
    table = {n.name: n for n in graph.op}
    aliases = {}
    for state, entry in bound.items():
        old = table[entry["commit"]]
        name = "target_gdr_prefix_" + state
        require(name not in table, "GDR name collision")
        port_specs = specs()[1:]
        args = [
            (edge, v["name"], dt, v["shape"])
            for edge, v, dt in zip(old.input[1:], port_specs, TYPES[1:], strict=True)
        ]
        op = tensor_node(type(old), name, COMMIT, args, [("state", 1, STATE)])
        op.attr["prune_level"].i = level
        op.id = old.id
        for port in (":1", ":-1"):
            aliases[old.name + port] = name + (":0" if port == ":1" else port)
        old.CopyFrom(op)
        entry["new"] = name
    for n in graph.op:
        for i, e in enumerate(n.input):
            if e in aliases:
                n.input[i] = aliases[e]
    for i, e in enumerate(graph.output):
        if e in aliases:
            graph.output[i] = aliases[e]
    require(
        Counter(n.type for n in graph.op)[GDR] == len(bound)
        and Counter(n.type for n in graph.op)[COMMIT] == len(bound),
        "wrong first/commit rewrite counts",
    )
    return bound
