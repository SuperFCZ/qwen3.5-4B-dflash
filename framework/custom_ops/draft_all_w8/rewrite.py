"""Transform only the sixteen remaining raw projections in the accepted Gate/Down AIR."""

import hashlib
from pathlib import Path

from selection import OLD_OP, c, packed, require, selections, sites


def digest(data):
    return hashlib.sha256(data).hexdigest()


def node_hash(node):
    # A topological insertion renumbers later nodes. Edges in this AIR use name:port.
    old = node.id
    node.id = 0
    try:
        return digest(node.SerializeToString(deterministic=True))
    finally:
        node.id = old


def payload(data):
    require(len(data) >= 64, "empty or short projection constant")
    offsets = [(len(data) - 64) * i // 4 for i in range(5)]
    i = max(offsets, key=lambda j: len(set(data[j : j + 64])))
    return dict(bytes=len(data), sha256=digest(data), probe_offset=i, probe=data[i : i + 64].hex())


def dims(desc):
    return tuple(desc.shape.dim)


def transform(model, graph, specs, assets, external):
    (g,) = model.graph
    proposed = sites(specs)
    rows = graph["projections"]
    roles = set(proposed)
    table = {op.name: op for op in g.op}
    require(
        len(g.op) == len(table)
        and all(op.id == i for i, op in enumerate(g.op))
        and set(rows) == roles
        and len(roles) == 26,
        "invalid five-layer source inventory or node IDs",
    )
    groups = ("node", "weight", "scale")
    bound = {r[k] for r in rows.values() for k in groups}
    require(len(bound) == 3 * len(rows) and bound <= table.keys(), "shared or missing W8 node/constant")
    require(
        {op.name for op in g.op if op.type == "Const" and op.output_desc[0].dtype == 3}
        == {r["weight"] for r in rows.values()},
        "unaccounted INT8 weight",
    )
    require(
        {op.name for op in g.op if op.type.startswith("DFlash") or "WeightQuant" in op.type}
        == {r["node"] for r in rows.values()},
        "unexpected quantized operator",
    )
    for kind, item in external.items():
        op = table["target_" + kind]
        require(
            op.type == "FileConstant"
            and Path(op.attr["file_path"].s.decode()).resolve() == c.verify(item)
            and op.output_desc[0].dtype == 2,
            "different external FP16 " + kind,
        )
    initial = {name: node_hash(node) for name, node in table.items()}
    public = (list(g.input), list(g.output))
    original_order = list(table)
    preserved = {r["role"]: r for r in graph["replacements"]}
    require(
        set(preserved) == {r for r in roles if r.endswith((".gate_up", ".down"))},
        "expected exactly ten already integrated Gate/Down sites",
    )
    changed = set()
    insertion = None
    inventory = {}
    preparation = []
    for role, spec in proposed.items():
        row = rows[role]
        op, w, s = (table[row[k]] for k in groups)
        kind = role.split(".")[-1]
        m, k, n = spec.shape
        require(
            tuple(row[a] for a in ("m", "k", "n")) == spec.shape
            and len(op.input) == 3
            and list(op.input[1:]) == [w.name + ":0", s.name + ":0"]
            and op.input_desc[0].dtype == op.output_desc[0].dtype == 2
            and dims(op.input_desc[0]) == (m, k)
            and dims(op.output_desc[0]) == (m, n),
            "projection ABI/edge mismatch: " + role,
        )
        require(
            all(v.type == "Const" and v.output_desc[0] == v.attr["value"].t.desc for v in (w, s))
            and w.output_desc[0].dtype == 3
            and s.output_desc[0].dtype == 2,
            "invalid tensors: " + role,
        )
        wd = (n // spec.columns, k // spec.panel, spec.panel * spec.columns)
        sd = (n // spec.columns, k // 128, spec.columns)
        if role in preserved:
            prev = preserved[role]
            require(
                op.type == prev["type"] == spec.op
                and spec.panel == prev["panel_k"]
                and dims(w.output_desc[0]) == wd
                and dims(s.output_desc[0]) == sd
                and (w.output_desc[0].layout, s.output_desc[0].layout) == ("ND", "ND"),
                "different measured Gate/Down site: " + role,
            )
            codes, scales = w.attr["value"].t.data, s.attr["value"].t.data
            require(
                [digest(codes), digest(scales)] == [prev["packed_sha256"], prev["scale_sha256"]],
                "accepted Gate/Down weights changed: " + role,
            )
        else:
            nz, gn = w.attr["value"].t.data, s.attr["value"].t.data
            require(
                op.type == OLD_OP
                and (w.output_desc[0].layout, s.output_desc[0].layout) == ("FRACTAL_NZ", "ND")
                and dims(w.output_desc[0]) == (k // 32, n // 16, 16, 32)
                and dims(s.output_desc[0]) == (k // 128, n)
                and [digest(nz), digest(gn)] == [row["nz_sha256"], row["gn_sha256"]],
                "different original NZ/GN pair: " + role,
            )
            codes, scales = packed.packed_weights(nz, gn, n, k, spec.columns, spec.panel)
            op.type = spec.op
            for port, const, num, shape, data in ((1, w, 3, wd, codes), (2, s, 2, sd, scales)):
                desc = const.output_desc[0]
                desc.Clear()
                packed.nd_desc(desc, "w_tiles" if port == 1 else "s", num, shape)
                const.attr["value"].t.desc.CopyFrom(desc)
                const.attr["value"].t.data = data
                const.attr["_out_shape_locked"].b = True
                op.input_desc[port].CopyFrom(desc)
            changed.update((op.name, w.name, s.name))
            if spec.pack_x:
                require(role == "fc" and insertion is None, "only C64 FC may introduce activation packing")
                source, port = op.input[0].rsplit(":", 1)
                require(
                    port == "0" and source in table and dims(table[source].output_desc[0]) == (m, k),
                    "FC public activation source differs",
                )
                nodes = type(g)()
                ready = packed.append_activation_pack(nodes, table[source], m, k, "all_w8_fc_a")
                op.input[0] = ready.name + ":0"
                op.input_desc[0].CopyFrom(ready.output_desc[0])
                op.input_desc[0].name = "x"
                preparation = [v.name for v in nodes.op]
                require(
                    len(set(preparation)) == 4 and not set(preparation) & table.keys(),
                    "FC pack name collision",
                )
                insertion = (original_order.index(op.name), nodes)
        blobs = [payload(codes), payload(scales)]
        if role not in preserved and (role == "fc" or role.startswith("layer0.")):
            require(
                [b["sha256"] for b in blobs] == assets[kind]["layer0_constants"],
                "packed weight/scale differs from measured single operator: " + role,
            )
        inventory[role] = dict(
            node=op.name,
            op=spec.op,
            mode=spec.mode,
            shape=list(spec.shape),
            source_constants=[w.name, s.name],
            packed_shapes=[wd, sd],
            constants=blobs,
        )
    if insertion:
        position, nodes = insertion
        for i, node in enumerate(nodes.op):
            g.op.insert(position + i, node)
    for i, node in enumerate(g.op):
        node.id = i
    final = {op.name: node_hash(op) for op in g.op}
    require(
        len(changed) == 48
        and {n for n in initial if final[n] != initial[n]} == changed
        and [n for n in final if n not in preparation] == original_order
        and public == (list(g.input), list(g.output))
        and set(final) - set(initial) == set(preparation),
        "unrelated AIR node, edge, constant or public IO changed",
    )
    owners = dict(graph["modules"], **{name: "fc.activation_layout" for name in preparation})
    return dict(
        abi=graph["abi"],
        modules=owners,
        node_types={op.name: op.type for op in g.op},
        inventory=inventory,
        layout_nodes=preparation,
        changed_nodes=sorted(changed),
        protected_nodes=len(initial) - len(changed),
    )


def export(out, types, gear, case, external):
    graph = case["graph"]
    before = c.verify(graph["optimized"])
    model = types[0].FromString(before.read_bytes())
    info = transform(model, graph, selections(gear), case["assets"], external)
    path = out / "all-w8.air"
    require(model.ByteSize() < 2**31 - (64 << 20), "AIR exceeds protobuf safe size")
    path.write_bytes(model.SerializeToString(deterministic=True))
    return dict(info, air=c.identity(path), input_air=graph["optimized"])
