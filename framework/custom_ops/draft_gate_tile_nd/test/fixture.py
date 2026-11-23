"""Synthetic offline GE fixture for tests only. The hardware runner always reads the frozen Layer0 AIR."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from layout import K, M, N, NZ, OLD_OP


def model_proto(types, nz, gn):
    msg = types[0]()
    msg.name = "gate_fixture"
    graph = msg.graph.add()
    graph.name = "gate_fixture"

    def descriptor(d, name, dtype, shape, layout="ND"):
        d.name = name
        d.dtype = dtype
        d.layout = layout
        d.shape.dim.extend(shape)
        d.attr["format_for_int"].i = 29 if layout == "FRACTAL_NZ" else 2
        d.attr["origin_format_for_int"].i = 2
        d.attr["origin_shape"].list.val_type = 2
        d.attr["origin_shape"].list.i.extend([N, K] if layout == "FRACTAL_NZ" else shape)
        d.attr["origin_format_is_set"].b = True
        d.attr["origin_shape_initialized"].b = True

    x = graph.op.add(name="x", type="Data")
    x.attr["index"].i = 0
    descriptor(x.output_desc.add(), "x", 2, [M, K])
    x.input_desc.add().CopyFrom(x.output_desc[0])
    w = graph.op.add(name="weight", type="Const")
    descriptor(w.output_desc.add(), "w_nz", 3, NZ, "FRACTAL_NZ")
    w.attr["value"].t.desc.CopyFrom(w.output_desc[0])
    w.attr["value"].t.data = nz
    s = graph.op.add(name="scale", type="Const")
    descriptor(s.output_desc.add(), "s", 2, [K // 128, N])
    s.attr["value"].t.desc.CopyFrom(s.output_desc[0])
    s.attr["value"].t.data = gn
    gate = graph.op.add(name=OLD_OP, type=OLD_OP)
    for v in (x, w, s):
        gate.input.append(v.name + ":0")
        gate.input_desc.add().CopyFrom(v.output_desc[0])
    descriptor(gate.output_desc.add(), "y", 2, [M, N])
    out = graph.op.add(name="NetOutput", type="NetOutput")
    out.input.append(gate.name + ":0")
    out.input_desc.add().CopyFrom(gate.output_desc[0])
    graph.input.append("x:0")
    graph.output.append(gate.name + ":0")
    for i, node in enumerate(graph.op):
        node.id = i
    return msg
