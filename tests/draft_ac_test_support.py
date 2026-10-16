"""Extended FileConstant test protobuf; historical fixture stays hash-locked."""
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory


def graph_type():
    proto = descriptor_pb2.FileDescriptorProto(name="weight_quant_layout_test.proto")
    enum = proto.enum_type.add(name="DType")
    for i, name in enumerate(("DT_UNDEFINED", "DT_FLOAT16", "DT_INT8", "DT_INT32", "DT_INT64")):
        enum.value.add(name=name, number=i)
    def msg(name, fields):
        item = proto.message_type.add(name=name)
        for number, (key, kind, label, typename) in enumerate(fields, 1):
            field = item.field.add(name=key, number=number, type=kind, label=label)
            if typename:
                field.type_name = "." + typename
        return item
    msg("Shape", [("dim", 3, 3, None)])
    desc = msg("Desc", [("name", 9, 1, None), ("dtype", 14, 1, "DType"), ("shape", 11, 1, "Shape"),
                        ("layout", 9, 1, None), ("attr", 11, 3, "Desc.Entry")])
    entry = desc.nested_type.add(name="Entry"); entry.options.map_entry = True
    entry.field.add(name="key", number=1, type=9, label=1)
    entry.field.add(name="value", number=2, type=11, type_name=".Attr", label=1)
    msg("Tensor", [("desc", 11, 1, "Desc"), ("data", 12, 1, None)])
    msg("Ints", [("i", 3, 3, None), ("val_type", 3, 1, None), ("s", 12, 3, None)])
    msg("Attr", [("i", 3, 1, None), ("b", 8, 1, None), ("t", 11, 1, "Tensor"),
                 ("list", 11, 1, "Ints"), ("s", 12, 1, None), ("dt", 3, 1, None)])
    op = msg("Op", [("name", 9, 1, None), ("type", 9, 1, None), ("input", 9, 3, None),
                    ("input_desc", 11, 3, "Desc"), ("output_desc", 11, 3, "Desc"),
                    ("attr", 11, 3, "Op.Entry")])
    entry = op.nested_type.add(name="Entry"); entry.options.map_entry = True
    entry.field.add(name="key", number=1, type=9, label=1)
    entry.field.add(name="value", number=2, type=11, type_name=".Attr", label=1)
    msg("Graph", [("op", 11, 3, "Op")])
    pool = descriptor_pool.DescriptorPool(); pool.Add(proto)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName("Graph"))
