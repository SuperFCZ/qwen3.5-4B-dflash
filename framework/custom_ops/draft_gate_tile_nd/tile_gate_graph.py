"""Build a static GE single-Gate AIR with immutable NZ weight and GN scale."""

from layout import K, M, N, OLD_OP, require


def model_proto(types, nz, gn, *, projection_shape=(M, K, N), name="c16_gate_up"):
    m, k, n = projection_shape
    require(m > 0 and k > 0 and k % 128 == 0 and n > 0 and n % 16 == 0, "unaligned W8A16 projection")
    nz_shape = (k // 32, n // 16, 16, 32)
    require(len(nz) == n * k and len(gn) == 2 * (k // 128) * n, "wrong immutable weight or GN scale extent")
    msg = types[0]()
    msg.name = name
    graph = msg.graph.add()
    graph.name = name

    def descriptor(d, name, dtype, shape, layout="ND"):
        d.name = name
        d.dtype = dtype
        d.layout = layout
        d.shape.dim.extend(shape)
        d.attr["format_for_int"].i = 29 if layout == "FRACTAL_NZ" else 2
        d.attr["origin_format_for_int"].i = 2
        d.attr["origin_shape"].list.val_type = 2
        d.attr["origin_shape"].list.i.extend([n, k] if layout == "FRACTAL_NZ" else shape)
        d.attr["origin_format_is_set"].b = True
        d.attr["origin_shape_initialized"].b = True

    x = graph.op.add(name="x", type="Data")
    x.attr["index"].i = 0
    descriptor(x.output_desc.add(), "x", 2, [m, k])
    x.input_desc.add().CopyFrom(x.output_desc[0])
    w = graph.op.add(name="weight", type="Const")
    descriptor(w.output_desc.add(), "w_nz", 3, nz_shape, "FRACTAL_NZ")
    w.attr["value"].t.desc.CopyFrom(w.output_desc[0])
    w.attr["value"].t.data = nz
    # Match normalize_ac_constants: GE shape inference otherwise interprets
    # the already-packed four-dimensional Const as logical ND and tiles it again.
    w.attr["_out_shape_locked"].b = True
    s = graph.op.add(name="scale", type="Const")
    descriptor(s.output_desc.add(), "s", 2, [k // 128, n])
    s.attr["value"].t.desc.CopyFrom(s.output_desc[0])
    s.attr["value"].t.data = gn
    gate = graph.op.add(name=OLD_OP, type=OLD_OP)
    for v in (x, w, s):
        gate.input.append(v.name + ":0")
        gate.input_desc.add().CopyFrom(v.output_desc[0])
    descriptor(gate.output_desc.add(), "y", 2, [m, n])
    out = graph.op.add(name="NetOutput", type="NetOutput")
    out.input.append(gate.name + ":0")
    out.input_desc.add().CopyFrom(gate.output_desc[0])
    graph.input.append("x:0")
    graph.output.append(gate.name + ":0")
    for i, node in enumerate(graph.op):
        node.id = i
    return msg
