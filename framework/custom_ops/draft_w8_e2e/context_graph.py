"""Extract the cache-producing part of the measured C64 Draft AIR."""

import math


STATES = [f"d{i}_{kind}" for i in range(5) for kind in ("key", "value")]
ROLES = ["fc", *(f"layer{i}.kv" for i in range(5))]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def context_model(source, abi, inventory):
    (g,) = source.graph
    nodes = {n.name: n for n in g.op}
    names = ["features", "start_position", "valid_rows", *STATES]
    require(
        [v["name"] for v in abi["inputs"]] == [*names[:3], "anchor", "proposal_count", *STATES]
        and abi["inputs"][0]["shape"][1] == 64,
        "expected the full C64 input ABI",
    )
    require(
        [v["name"] for v in abi["outputs"]] == ["draft_top1", *STATES]
        and abi["outputs"][1:] == abi["inputs"][5:],
        "unexpected full Draft outputs",
    )
    output = nodes["NetOutput"]
    require(
        list(g.output) == list(output.input)
        and [d.name for d in output.input_desc] == ["draft_top1", *STATES],
        "cache output mapping differs",
    )

    def parent(node, port):
        name, slot = node.input[port].rsplit(":", 1)
        require(slot == "0", f"unexpected edge into {node.name}")
        return nodes[name]

    # All five layers project the same committed FC activations in rows 0..63.
    # Keep the measured M80 KV kernels, giving their discarded 16 rows exact zeros.
    joined = None
    pad = None
    changed_inputs = {}
    for role in ROLES[1:]:
        kv = nodes[inventory[role]["node"]]
        flat = parent(kv, 0)
        cat = parent(flat, 0)
        require(
            flat.type == "Reshape"
            and cat.type == "ConcatD"
            and cat.attr["concat_dim"].i == 1
            and len(cat.input) == 2,
            "unexpected KV row packing",
        )
        left, right = (list(d.shape.dim) for d in cat.input_desc)
        require(
            left[:2] == [1, 64]
            and len(left) == 3
            and right == [1, 16, left[2]]
            and all(d.dtype == 2 for d in cat.input_desc)
            and list(flat.output_desc[0].shape.dim) == [80, left[2]],
            "different KV input shape/dtype",
        )
        if joined is None:
            joined = cat
            pad = type(cat)(name="context_unused_kv_rows", type="Const")
            desc = pad.output_desc.add()
            desc.CopyFrom(cat.input_desc[1])
            desc.name = "y"
            pad.attr["value"].t.desc.CopyFrom(desc)
            pad.attr["value"].t.data = bytes(2 * math.prod(right))
            pad.attr["_out_shape_locked"].b = True
            require(pad.name not in nodes, "padding node name already in use")
            nodes[pad.name] = pad
            changed_inputs[joined.name] = [joined.input[0], pad.name + ":0"]
        require(cat.input[0] == joined.input[0], "context features differ between layers")
        changed_inputs[flat.name] = [joined.name + ":0", *flat.input[1:]]

    # Trace from the ten KV outputs after dropping the Top1 result and all noise-row dependencies.
    keep = set()
    pending = [s.rsplit(":", 1)[0] for s in output.input[1:]]
    while pending:
        name = pending.pop()
        if name not in keep:
            keep.add(name)
            pending.extend(e.rsplit(":", 1)[0] for e in changed_inputs.get(name, nodes[name].input))
    require({name for name in keep if nodes[name].type == "Data"} == set(names), "unexpected public input")
    require(
        {role for role, row in inventory.items() if row["node"] in keep} == set(ROLES)
        and not any(nodes[n].type in ("FileConstant", "MatMul", "BatchMatMulV2", "ArgMaxV2") for n in keep),
        "candidate/Head computation still reaches context KV",
    )
    model = type(source)(name="draft_context64")
    graph = model.graph.add(name=model.name)
    graph.input.extend(n + ":0" for n in names)
    graph.output.extend(output.input[1:])
    for node in g.op:
        if node.name == joined.name:
            graph.op.add().CopyFrom(pad)
        if node.name in keep:
            copied = graph.op.add()
            copied.CopyFrom(node)
            if node.name in changed_inputs:
                copied.input[:] = changed_inputs[node.name]
            if node.type == "Data":
                copied.attr["index"].i = names.index(node.name)
    end = graph.op.add(name="NetOutput", type="NetOutput")
    end.input.extend(graph.output)
    end.input_desc.extend(output.input_desc[1:])
    for i, n in enumerate(graph.op):
        n.id = i
    reduced_abi = dict(inputs=[v for v in abi["inputs"] if v["name"] in names], outputs=abi["outputs"][1:])
    return model, reduced_abi
