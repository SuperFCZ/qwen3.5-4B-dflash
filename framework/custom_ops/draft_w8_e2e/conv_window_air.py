"""Replace the published prefix-bank selection in the deployed Target AIR with a four-column gather."""

from collections import defaultdict
import re

import numpy as np

from winners import c, require
from air import protobuf_classes
from packed_panel import nd_desc
from receiver_air import const_int, node_name


def rewrite(model, abi):
    (g,) = model.graph
    original = list(g.op)
    table = {op.name: op for op in original}
    (net,) = (op for op in original if op.type == "NetOutput")
    data = sorted(
        (op for op in original if op.type in ("Data", "RefData")), key=lambda op: op.attr["index"].i
    )
    require(len(original) == len(table) and len(data) == len(abi["inputs"]), "ambiguous Target AIR ports")
    inputs = {v["name"]: op.name + ":0" for v, op in zip(abi["inputs"], data, strict=True)}
    ports = {v["name"]: i for i, v in enumerate(abi["outputs"])}
    (rows,) = (v["shape"][1] for v in abi["inputs"] if v["name"] == "input_ids")
    verify = "accepted_count" in ports
    require(rows == (16 if verify else 64), "unexpected Target gear")
    require(not g.output or list(g.output) == list(net.input[: len(ports)]), "unexpected graph outputs")
    added, branches, old_window, old_transposes = [], {}, set(), set()

    def op(edge, kind):
        n = table[node_name(edge)]
        require(edge == n.name + ":0" and n.type == kind, f"expected {kind}, found {n.type} at {edge}")
        return n

    def literal(edge, values):
        a = const_int(op(edge, "Const"))
        require(
            a is not None and a.reshape(-1).tolist() == values,
            f"unexpected literal {edge}, expected {values}",
        )

    def uncast(edge):
        while table[node_name(edge)].type == "Cast":
            n = op(edge, "Cast")
            require(
                len(n.input) == 1 and n.output_desc[0].dtype in (5, 7, 8), "unexpected prefix-length cast"
            )
            edge = n.input[0]
        return edge

    def new(name, kind):
        require(name not in table, f"duplicate Target node {name}")
        n = type(net)(name=name, type=kind, id=next_id + len(added))
        table[name] = n
        added.append(n)
        return n

    def constant(name, value, dtype=8):
        a = np.asarray(value, dtype={7: "<i4", 8: "<i8"}[dtype])
        n = new(name, "Const")
        d = n.output_desc.add()
        nd_desc(d, "y", dtype, a.shape)
        n.attr["value"].t.desc.CopyFrom(d)
        n.attr["value"].t.data = a.tobytes()
        return n.name + ":0"

    def operand(n, edge, port, dtype, shape):
        n.input.append(edge)
        nd_desc(n.input_desc.add(), port, dtype, shape)

    next_id = max(n.id for n in original) + 1
    # Prefill reads [valid_rows, ..., valid_rows+3]; Verify reads [accepted+1, ..., accepted+4].
    # Reuse the old INT64 committed length, including the original INT16 GDR boundary.
    columns = None
    for name, index in ports.items():
        if not re.fullmatch(r"t\d+_conv", name):
            continue
        spec = abi["outputs"][index]
        batch, channels, width = spec["shape"]
        require(batch == 1 and width == 4 and spec["dtype"] == "float16", "unexpected window ABI")
        squeeze = op(net.input[index], "Squeeze")
        require(list(squeeze.attr["axis"].list.i) == [1] and len(squeeze.input) == 1, "wrong prefix squeeze")
        pick = op(squeeze.input[0], "GatherV2")
        require(
            len(pick.input) == 3 and ("batch_dims" not in pick.attr or pick.attr["batch_dims"].i == 0),
            "wrong window gather",
        )
        literal(pick.input[2], [1])
        sub = op(pick.input[1], "Sub")
        literal(sub.input[1], [1])
        length = sub.input[0]
        require(table[node_name(length)].output_desc[0].dtype == 8, "window index is not INT64")
        base = uncast(length)
        if verify:
            inc = op(base, "Add")
            require(
                inc.input[0] == net.input[ports["accepted_count"]], "Verify must select the accepted prefix"
            )
            literal(inc.input[1], [1])
        else:
            require(base == inputs["valid_rows"], "Prefill must select the valid prefix")

        identity = op(pick.input[0], "Identity")
        shaped = op(identity.input[0], "Reshape")
        literal(shaped.input[1], [1, rows, channels, width])
        transpose = op(shaped.input[0], "Transpose")
        literal(transpose.input[1], [1, 0, 2])
        flat = op(transpose.input[0], "Reshape")
        literal(flat.input[1], [channels, rows, width])
        pack = op(flat.input[0], "Pack")
        require(
            pack.attr["axis"].i == -1 and pack.attr["N"].i == len(pack.input) == width, "wrong window Pack"
        )
        slices = [op(edge, "StridedSliceV2") for edge in pack.input]
        history = op(slices[0].input[0], "ConcatV2")
        require(
            len(history.input) == 3 and history.input[0] == inputs[name], f"cross-layer history at {name}"
        )
        literal(history.input[2], [-1])
        for offset, s in enumerate(slices, start=1):
            require(
                len(s.input) == 5
                and s.input[0] == history.name + ":0"
                and not any(
                    s.attr[k].i
                    for k in ("begin_mask", "end_mask", "ellipsis_mask", "new_axis_mask", "shrink_axis_mask")
                    if k in s.attr
                ),
                "slices do not select four consecutive history columns",
            )
            for edge, val in zip(s.input[1:], (offset, offset + rows, 2, 1), strict=True):
                literal(edge, [val])
        if columns is None:
            offsets = constant("target_conv_offsets", [0, 1, 2, 3])
            axis_dtype = op(pick.input[2], "Const").attr["value"].t.desc.dtype
            axis = constant("target_conv_axis", 2, axis_dtype)
            add = new("target_conv_indices", "Add")
            operand(add, length, "x1", 8, [1])
            operand(add, offsets, "x2", 8, [width])
            nd_desc(add.output_desc.add(), "y", 8, [width])
            columns = add.name + ":0"
        gather = new("target_conv_window_" + name, "GatherV2")
        operand(gather, history.name + ":0", "x", 2, [1, channels, rows + width])
        operand(gather, columns, "indices", 8, [width])
        operand(gather, axis, "axis", axis_dtype, [])
        nd_desc(gather.output_desc.add(), "y", 2, spec["shape"])
        net.input[index] = gather.name + ":0"
        if g.output:
            g.output[index] = net.input[index]
        old_window.update(n.name for n in (squeeze, pick, identity, shaped, transpose, flat, pack, *slices))
        old_transposes.add(transpose.name)
        branches[name] = dict(history=history.name, previous=transpose.name, gather=gather.name)

    require(bool(branches) and len(old_transposes) == len(branches), "missing or shared window branches")
    # Keep shared computations, graph controls and every unrelated node. Remove any orphaned
    # provisional selection from the same bank as well as its published selection.
    live = set()
    todo = [net.name, *(node_name(e) for e in [*g.input, *g.output] if e)]
    while todo:
        name = todo.pop()
        if name not in live:
            live.add(name)
            todo.extend(node_name(e) for e in table[name].input if e)
    drop = old_window - live
    require(old_transposes <= drop, "another live output needs a full window transpose")
    users = defaultdict(set)
    for n in original:
        for e in n.input:
            if e:
                users[node_name(e)].add(n.name)
    todo = list(drop)
    while todo:
        for name in users[todo.pop()] - drop:
            require(
                name not in live
                and table[name].type in ("Identity", "GatherV2", "Squeeze", "Cast", "Reshape"),
                "unexpected consumer of a discarded window: " + name,
            )
            drop.add(name)
            todo.append(name)
    out = []
    for n in original:
        if n.name == net.name:
            out.extend(added)
        if n.name not in drop:
            out.append(n)
    kept = {n.name for n in out}
    require(all(node_name(e) in kept for n in out for e in n.input if e), "dangling GE edge")
    del g.op[:]
    g.op.extend(out)
    return dict(rows=rows, branches=branches, removed=sorted(drop))


def export(d, desc, source):
    T = protobuf_classes(c.verify(desc).read_bytes())[0]
    model = T.FromString(c.verify(source["air"]).read_bytes())
    proof = rewrite(model, source["abi"])
    require(len(proof["branches"]) == 24, "expected 24 receiver GDN layers")
    path = d / "target.air"
    path.write_bytes(model.SerializeToString(deterministic=True))
    c.write(d / "windows.json", proof)
    print(
        f"Target M{proof['rows']} replaced {len(proof['branches'])} prefix banks with four-column gathers",
        flush=True,
    )
    return dict(
        air=c.identity(path),
        abi=source["abi"],
        payloads=source["payloads"],
        proof=c.identity(d / "windows.json"),
    )
