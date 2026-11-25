"""Bridge the documented Target feature union to the checkpoint's exact five slots in the timed OM."""

import copy
from pathlib import Path
import numpy as np

from winners import c, require, UNION, SELECT, selected
from rewrite import transform
from packed_panel import nd_desc


def slots(union, chosen):
    require(
        len(union) == len(set(union)) and len(chosen) == len(set(chosen)) and set(chosen) <= set(union),
        "missing/ambiguous Target feature layer",
    )
    return tuple(union.index(n) for n in chosen)


def insert_selector(graph, abi, gear, hidden, union, chosen):
    indices = slots(union, chosen)
    old = [1, gear, hidden * len(chosen)]
    wide = [1, gear, hidden * len(union)]
    require(
        abi["inputs"][0] == dict(name="features", dtype="float16", shape=old),
        "unexpected selected feature ABI",
    )
    (public,) = (n for n in graph.op if n.name == "features" and n.type == "Data")
    require(
        public.attr["index"].i == 0 and list(public.output_desc[0].shape.dim) == old,
        "different feature input",
    )
    fresh = type(graph)()
    cache = {}
    original = "features:0"

    def const(values):
        key = tuple(values)
        if key not in cache:
            v = fresh.op.add(name="e2e_indices_" + str(len(cache)), type="Const")
            d = v.output_desc.add()
            nd_desc(d, "y", 7, [len(key)])
            v.attr["value"].t.desc.CopyFrom(d)
            v.attr["value"].t.data = np.array(key, dtype="<i4").tobytes()
            cache[key] = v
        return cache[key]

    def add_inputs(node, entries):
        for port, src in entries:
            node.input.append(src.name + ":0")
            d = node.input_desc.add()
            d.CopyFrom(src.output_desc[0])
            d.name = port

    for d in (public.input_desc[0], public.output_desc[0]):
        nd_desc(d, d.name, 2, wide)
    cuts = []
    for idx in indices:
        off = const((0, 0, idx * hidden))
        size = const((1, gear, hidden))
        v = fresh.op.add(name=f"e2e_layer_{union[idx]}", type="Slice")
        add_inputs(v, (("x", public), ("offsets", off), ("size", size)))
        nd_desc(v.output_desc.add(), "y", 2, [1, gear, hidden])
        cuts.append(v)
    join = fresh.op.add(name="e2e_selected_features", type="ConcatD")
    add_inputs(join, [(f"x{i}", node) for i, node in enumerate(cuts)])
    join.attr["N"].i = len(cuts)
    join.attr["concat_dim"].i = 2
    nd_desc(join.output_desc.add(), "y", 2, old)
    uses = 0
    for n in graph.op:
        for i, edge in enumerate(n.input):
            if edge == original:
                require(list(n.input_desc[i].shape.dim) == old, "unexpected original feature consumer")
                n.input[i] = join.name + ":0"
                uses += 1
    names = [op.name for op in fresh.op]
    require(
        uses > 0 and not set(names) & {n.name for n in graph.op}, "missing feature consumer or name collision"
    )
    where = next(i for i, v in enumerate(graph.op) if v.name == public.name) + 1
    for i, op in enumerate(fresh.op):
        graph.op.insert(where + i, op)
    for i, op in enumerate(graph.op):
        op.id = i
    a = copy.deepcopy(abi)
    a["inputs"][0]["shape"] = wide
    return a, names


def export(out, types, gear, case, external):
    source = case["graph"]
    before = c.verify(source["optimized"])
    model = types[0].FromString(before.read_bytes())
    info = transform(model, source, selected(gear), case["assets"], external)
    for role in ("gate_up", "down"):
        require(
            [r["sha256"] for r in info["inventory"]["layer0." + role]["constants"]]
            == case["assets"][role]["layer0_constants"],
            "different measured Layer0 constants: " + role,
        )
    info["abi"], new = insert_selector(model.graph[0], info["abi"], gear, 2560, UNION, SELECT)
    info["modules"].update({n: "target_feature_selection" for n in new})
    p = out / "draft.air"
    p.write_bytes(model.SerializeToString(deterministic=True))
    return dict(
        info,
        air=c.identity(p),
        input_air=source["optimized"],
        feature_layers=list(SELECT),
        target_feature_layers=list(UNION),
        feature_slots=list(slots(UNION, SELECT)),
    )


def expand_replay(out, case, gear):
    io = case["io"]
    old = c.verify(io["files"]["features"])
    x = np.fromfile(old, "<u2").reshape(1, gear, len(SELECT), 2560)
    rng = np.random.default_rng(310200 + gear)
    wide = rng.normal(size=(1, gear, len(UNION), 2560)).astype("<f2").view("<u2")
    for i, idx in enumerate(slots(UNION, SELECT)):
        wide[:, :, idx] = x[:, :, i]
    require(np.array_equal(wide[:, :, list(slots(UNION, SELECT)), :], x), "expanded features changed payload")
    p = out / "features.bin"
    wide.tofile(p)
    return dict(
        io, files=dict(io["files"], features=c.identity(p)), source="FROZEN_REPLAY_IN_TARGET_LAYER_SLOTS"
    )
