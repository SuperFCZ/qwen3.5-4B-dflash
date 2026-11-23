"""Copy the real single-Gate graph, replacing its sole NZ operand with a typed ND tile constant."""

import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys

from layout import K, M, N, NZ, TILE, OLD_OP, OP, pack, require, sha


def protobuf_classes(descriptor):
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

    desc = descriptor_pb2.FileDescriptorSet.FromString(descriptor)
    pool = descriptor_pool.DescriptorPool()
    for f in desc.file:
        pool.Add(f)

    def cls(name):
        kind = pool.FindMessageTypeByName("ge.proto." + name)
        if hasattr(message_factory, "GetMessageClass"):
            return message_factory.GetMessageClass(kind)
        return message_factory.MessageFactory(pool).GetPrototype(kind)

    return cls("ModelDef"), cls("GraphDef")


def classes(cann, work):
    # The stand-alone Conda benchmark compiles the installed CANN schema without TorchAir.
    if importlib.util.find_spec("grpc_tools"):
        schema = cann / "include/proto/ge_ir.proto"
        path = work / "ge.desc"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "grpc_tools.protoc",
                "-I" + str(schema.parent),
                "--descriptor_set_out=" + str(path),
                schema.name,
            ],
            cwd=schema.parent,
            check=True,
        )
        return protobuf_classes(path.read_bytes())
    protoc = shutil.which("protoc")
    torch = importlib.util.find_spec("torch")
    if not protoc and torch and torch.origin:
        bundled = Path(torch.origin).parent / "bin/protoc"
        if bundled.is_file():
            protoc = str(bundled)
    if protoc:
        schema = cann / "include/proto/ge_ir.proto"
        path = work / "ge.desc"
        subprocess.run(
            [protoc, "-I" + str(schema.parent), "--descriptor_set_out=" + str(path), schema.name],
            cwd=schema.parent,
            check=True,
        )
        return protobuf_classes(path.read_bytes())
    # TorchAir installs this generated GE schema in the same venv used for the accepted export.
    npu = importlib.util.find_spec("torch_npu")
    if npu and npu.origin:
        sys.path.insert(0, str(Path(npu.origin).parent / "dynamo"))
    from torchair._ge_concrete_graph.ge_ir_pb2 import ModelDef, GraphDef

    return ModelDef, GraphDef


def parse_graph(data, types):
    from google.protobuf.message import DecodeError

    for cls in types:
        msg = cls()
        try:
            msg.ParseFromString(data)
        except DecodeError:
            continue
        graphs = msg.graph if hasattr(msg, "graph") else [msg]
        if len(graphs) == 1 and sum(n.type == OLD_OP for n in graphs[0].op) == 1:
            return msg, graphs[0]
    raise ValueError("expected the frozen single-Gate GE ModelDef or GraphDef")


def tile_desc(desc):
    dtype = desc.DESCRIPTOR.fields_by_name["dtype"].enum_type
    require(dtype.values_by_number[desc.dtype].name == "DT_INT8", "Gate weight is not INT8")
    require(
        list(desc.shape.dim) == list(NZ) and desc.layout == "FRACTAL_NZ",
        f"unexpected original Gate weight descriptor: layout={desc.layout!r}, shape={list(desc.shape.dim)}",
    )
    # Build a plain ND descriptor. Logical and physical dimensions of this private port are identical.
    out = type(desc)()
    out.name = "w_tiles"
    out.dtype = desc.dtype
    out.layout = "ND"
    out.shape.dim.extend(TILE)
    out.attr["format_for_int"].i = 2
    out.attr["origin_format_for_int"].i = 2
    out.attr["origin_shape"].list.val_type = 2
    out.attr["origin_shape"].list.i.extend(TILE)
    out.attr["origin_shape_initialized"].b = True
    out.attr["origin_format_is_set"].b = True
    return out


def transform(original, dest, nz, scale, types):
    require(
        not dest.exists() and not (dest.parent / "weights.tile-nd.bin").exists(),
        "candidate AIR/weight output requires a fresh directory",
    )
    msg, graph = parse_graph(original.read_bytes(), types)
    nodes = {n.name: n for n in graph.op}
    require(len(nodes) == len(graph.op), "duplicate source node")
    op = next(n for n in graph.op if n.type == OLD_OP)
    require(len(op.input) == len(op.input_desc) == 3, "Gate arity changed")
    weight = nodes[op.input[1].rsplit(":", 1)[0]]
    gn = nodes[op.input[2].rsplit(":", 1)[0]]
    require(
        weight.type == "Const"
        and not weight.input
        and len(weight.output_desc) == 1
        and sum(list(n.input).count(op.input[1]) for n in graph.op) == 1
        and weight.attr["value"].t.data == nz,
        "AIR weight differs from frozen Layer0 NZ",
    )
    # Make existing FileConstant references independent of the candidate ATC working directory.
    for node in graph.op:
        if node.type == "FileConstant":
            rel = Path(node.attr["file_path"].s.decode())
            actual = (original.parent / rel).resolve(strict=True)
            require(actual.is_file(), "missing original FileConstant: " + str(actual))
            node.attr["file_path"].s = str(actual).encode()
    if gn.type == "Const":
        require(gn.attr["value"].t.data == scale, "AIR scale differs from frozen GN")
    else:
        require(
            gn.type == "FileConstant"
            and ("offset" not in gn.attr or gn.attr["offset"].i == 0)
            and Path(gn.attr["file_path"].s.decode()).read_bytes() == scale,
            "unexpected scale FileConstant",
        )
    require(
        list(op.input_desc[0].shape.dim) == [M, K] and list(op.input_desc[2].shape.dim) == [K // 128, N],
        "wrong activation or scale shape",
    )
    packed, ms = pack(nz)
    for desc in (weight.output_desc[0], weight.attr["value"].t.desc, op.input_desc[1]):
        desc.CopyFrom(tile_desc(desc))
    # TorchAir may serialize a name-to-port table as well as the ordered input edges.
    for node, key in ((op, "_input_name_key"), (weight, "_output_name_key")):
        if key in node.attr:
            names = node.attr[key].list.s
            for i, value in enumerate(names):
                if value == b"w_nz":
                    names[i] = b"w_tiles"
    op.type = OP
    weight.attr["value"].t.data = packed
    (dest.parent / "weights.tile-nd.bin").write_bytes(packed)
    dest.write_bytes(msg.SerializeToString())
    return dict(
        name=op.name,
        weight_node=weight.name,
        layout="ND",
        shape=list(TILE),
        bytes=len(packed),
        nz_sha256=sha(nz),
        tile_sha256=sha(packed),
        scale_sha256=sha(scale),
        pack_ms=ms,
    )


if __name__ == "__main__":
    import json

    original, dest, nz, scale, cann = map(Path, sys.argv[1:])
    info = transform(original, dest, nz.read_bytes(), scale.read_bytes(), classes(cann, dest.parent))
    (dest.parent / "pack.json").write_text(json.dumps(info, indent=2) + "\n")
