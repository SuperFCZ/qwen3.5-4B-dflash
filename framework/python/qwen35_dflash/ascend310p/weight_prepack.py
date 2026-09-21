"""Value-preserving, export-time INT8 NZ constants for native W8 Drafts.

Only immutable TorchAir weight bindings may be folded. Runtime inputs and W4
unpacking stay on their existing path. GE Const carries both the actual NZ
bytes and its physical/logical descriptors; no format-only relabel is used.
"""
from __future__ import annotations

import copy
import hashlib
from math import prod
from pathlib import Path

import torch

from .weight_quant_layout import GE_OP, POLICY, _dtype, _nz_shape, _source
from .utils import contained_path, load_json_object, sha256_file


PREPACK_POLICY = "w8-int8-nz-const-v1"
# Offline bytes remain v1. AIR stores a GeTensor, not a normalized public
# Tensor argument: its value and output must describe the SAME physical bytes.
# GE CreateConstOp can recreate an output directly from the value descriptor.
CONST_DESC_POLICY = "locked-physical-nz-value-and-output-v4"
# ge_attr_define.cc: ATTR_NAME_OUT_SHAPE_LOCKED (leading underscore required).
CONST_SHAPE_LOCK = "_out_shape_locked"


def _const_value_desc(physical_desc):
    """Keep the serialized GeTensor value in the output's physical format.

    Logical [N,K] lives in origin_shape, physical [K1,N1,16,32] in shape.
    TensorAdapter::NormalizeGeTensorDesc is for public Tensor arguments; its
    logical-ND plus storage-attrs encoding must NOT be used as Const.value
    alongside a physical-NZ output. Const cloning/folding consumes the raw
    GeTensor descriptor, not those public-API storage attributes.
    """
    logical = list(physical_desc.attr["origin_shape"].list.i)
    storage = list(physical_desc.shape.dim)
    if (physical_desc.layout != "FRACTAL_NZ" or _dtype(physical_desc) != "DT_INT8"
            or len(logical) != 2 or min(logical) <= 0 or storage != _nz_shape(*logical)):
        raise ValueError("W8 Const value requires a valid logical/physical NZ descriptor")
    if (physical_desc.attr["format_for_int"].i != 29
            or physical_desc.attr["origin_format_for_int"].i != 2
            or not physical_desc.attr["origin_shape_initialized"].b
            or not physical_desc.attr["origin_format_is_set"].b
            or "storage_shape" in physical_desc.attr or "storage_format" in physical_desc.attr):
        raise ValueError("W8 Const value requires an unnormalized physical NZ descriptor")
    return copy.deepcopy(physical_desc)


def pack_int8_nz(weight):
    if weight.dtype != torch.int8 or weight.ndim != 2 or min(weight.shape) <= 0:
        raise ValueError("NZ prepack requires a nonempty INT8 matrix [N,K]")
    matrix = weight.detach().cpu().contiguous()
    n, k = matrix.shape
    k1, n1, n0, k0 = _nz_shape(n, k)
    padded = torch.zeros(n1 * n0, k1 * k0, dtype=torch.int8)
    padded[:n, :k] = matrix
    packed = padded.view(n1, n0, k1, k0).permute(2, 0, 1, 3).contiguous()
    # Check the carrier itself, including any alignment lanes. This is a byte
    # permutation, without casts, arithmetic, scale folding or requantization.
    restored = packed.permute(1, 2, 0, 3).reshape(n1 * n0, k1 * k0)
    if not torch.equal(restored, padded):
        raise ValueError("INT8 NZ prepack failed its byte-exact round trip")
    return packed


def load_prepacked_weights(manifest_path):
    path = Path(manifest_path).expanduser().resolve()
    manifest = load_json_object(path)
    if (manifest.get("policy") != PREPACK_POLICY or manifest.get("status") != "PASS"
            or manifest.get("schema_version") != 1 or not isinstance(manifest.get("weights"), list)
            or not manifest["weights"] or manifest.get("weight_count") != len(manifest["weights"])):
        raise ValueError("invalid offline W8 NZ weight manifest")
    indexed = {}
    for record in manifest["weights"]:
        shape = record.get("logical_shape", [])
        if (record.get("dtype") != "int8" or record.get("format") != "FRACTAL_NZ"
                or len(shape) != 2 or any(type(d) is not int or d <= 0 for d in shape)
                or record.get("storage_shape") != _nz_shape(*shape)
                or record.get("bytes") != prod(record["storage_shape"])):
            raise ValueError("offline W8 NZ weight shape/dtype differs")
        key = (tuple(shape), record["logical_sha256"])
        if key in indexed and indexed[key]["sha256"] != record["sha256"]:
            raise ValueError("conflicting offline NZ carriers for the same logical weight")
        indexed[key] = record
    return {"root": path.parent, "weights": indexed, "sha256": sha256_file(path)}


def _cached_weight(matrix, cache):
    logical_hash = hashlib.sha256(matrix.numpy().tobytes()).hexdigest()
    record = cache["weights"].get((tuple(matrix.shape), logical_hash))
    if record is None:
        raise ValueError("offline NZ cache does not contain this exact W8 checkpoint weight")
    path = contained_path(cache["root"], record["path"])
    if not path.is_file() or path.stat().st_size != record["bytes"]:
        raise ValueError("offline NZ weight payload size differs")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != record["sha256"]:
        raise ValueError("offline NZ weight payload hash differs")
    k1, n1, n0, k0 = record["storage_shape"]
    carrier = torch.frombuffer(bytearray(data), dtype=torch.int8).reshape(k1, n1, n0, k0)
    restored = carrier.permute(1, 2, 0, 3).reshape(n1 * n0, k1 * k0)
    n, k = matrix.shape
    if (not torch.equal(restored[:n, :k], matrix)
            or torch.count_nonzero(restored[n:, :]) or torch.count_nonzero(restored[:n, k:])):
        raise ValueError("offline NZ carrier does not restore the exact weight with zero padding")
    return data, logical_hash, record


def prepack_weight_quant_constants(graph, layout_audit, immutable_weights, cache):
    """Replace audited weight TransData with byte-exact NZ Const at AIR save.

    ``immutable_weights`` maps original GE weight edges to exporter-bound
    tensors, captured BEFORE TorchAir changes those Data nodes into constants.
    Validate every source first. Preserve any original ND constant still used
    by another consumer, including control edges.
    """
    if layout_audit.get("policy") != POLICY or not layout_audit.get("nodes"):
        raise ValueError("W8 NZ prepack requires the production WeightQuant layout audit")
    nodes = {node.name: node for node in graph.op}
    plans = []
    for record in layout_audit["nodes"]:
        op = nodes[record["name"]]
        conversion, port = _source(nodes, op.input[1])
        if (op.type != GE_OP or port != 0 or conversion.type != "TransData"
                or len(conversion.input) != 1 or len(conversion.output_desc) != 1
                or record["format_conversion"] != conversion.name):
            raise ValueError("W8 NZ prepack requires the audited weight TransData")
        edge = conversion.input[0]
        source, source_port = _source(nodes, edge)
        value = immutable_weights.get(edge)
        if (source.type not in {"Const", "FileConstant"} or source_port != 0
                or source.input or value is None or value.dtype != torch.int8
                or list(value.shape) != record["weight_shape"]
                or _dtype(source.output_desc[0]) != "DT_INT8"):
            raise ValueError(f"W8 NZ prepack source {edge} is not an immutable INT8 weight")
        if (conversion.output_desc[0].layout != "FRACTAL_NZ"
                or list(conversion.output_desc[0].shape.dim) != _nz_shape(*value.shape)
                or list(conversion.output_desc[0].attr["origin_shape"].list.i) != list(value.shape)):
            raise ValueError("W8 NZ prepack descriptor does not match weight storage")
        plans.append((record, conversion, edge, value))

    records, candidates = [], set()
    for record, conversion, edge, value in plans:
        matrix = value.detach().cpu().contiguous()
        # Read the reusable offline carrier; export never repacks the weight.
        data, logical_hash, cached = _cached_weight(matrix, cache)
        desc = copy.deepcopy(conversion.output_desc[0])
        name = conversion.name
        # Both the raw value and its edge describe the actual NZ bytes.
        # The logical dimensions are retained separately in origin_shape.
        conversion.Clear()
        conversion.name, conversion.type = name, "Const"
        conversion.output_desc.add().CopyFrom(desc)
        conversion.attr["value"].t.desc.CopyFrom(_const_value_desc(desc))
        conversion.attr["value"].t.data = data
        # InferShapePass::CallInferShapeFunc honors this on Const and on the
        # Data proxy created by MultiBatchClonePass (which preserves attrs).
        # The constant's exact shape is known from its verified offline bytes;
        # never put this flag on the WeightQuant consumer or a public input.
        conversion.attr[CONST_SHAPE_LOCK].b = True
        item = {"name": name, "source": edge, "dtype": "int8", "format": "FRACTAL_NZ",
                "logical_shape": list(matrix.shape), "storage_shape": cached["storage_shape"],
                "descriptor_policy": CONST_DESC_POLICY,
                "output_shape_locked": True,
                "value_shape": cached["storage_shape"], "value_format": "FRACTAL_NZ",
                "value_origin_shape": list(matrix.shape), "value_origin_format": "ND",
                "logical_bytes": matrix.numel(), "storage_bytes": len(data),
                "logical_sha256": logical_hash,
                "storage_sha256": hashlib.sha256(data).hexdigest(),
                "roundtrip": "BIT_EXACT", "padding": "ZERO"}
        records.append(item)
        record.update(format_conversion=None, weight_storage=PREPACK_POLICY, prepacked_constant=item)
        candidates.add(edge.rpartition(":")[0])
    used = {edge.rpartition(":")[0] for node in graph.op for edge in node.input if edge}
    removed = candidates - used
    for index in range(len(graph.op) - 1, -1, -1):
        if graph.op[index].name in removed:
            del graph.op[index]
    # Match TorchAir's 2 GiB protobuf limit with its 200 MiB reserve. Its other
    # large parameters can remain FileConstant; this guard never relabels or
    # silently falls back to a different weight representation.
    if graph.ByteSize() > (2048 - 200) * 1024 * 1024:
        raise ValueError("NZ constants exceed TorchAir protobuf budget; retain runtime weight conversion")
    layout_audit["prepack"] = {"policy": PREPACK_POLICY, "status": "PASS",
        "descriptor_policy": CONST_DESC_POLICY,
        "offline_manifest_sha256": cache["sha256"],
        "node_count": len(records), "removed_weight_transdata": [r["name"] for r in records],
        "removed_nd_constants": sorted(removed), "constants": records}
    layout_audit["inserted_transdata"] = []
    validate_prepacked_graph(graph, layout_audit)
    return layout_audit


def validate_prepacked_graph(graph, audit):
    """Check every locked output against its bytes and consumer before save.

    This validates the exporter, not the installed CANN implementation. It
    deliberately does not repair conflicting descriptors or suppress any
    WeightQuant shape/tiling error.
    """
    nodes = {node.name: node for node in graph.op}
    expected = {r["name"] for r in audit["prepack"]["constants"]}
    locked = {n.name for n in graph.op if CONST_SHAPE_LOCK in n.attr
              and n.attr[CONST_SHAPE_LOCK].b}
    if locked != expected:
        raise ValueError("NZ shape lock must apply only to audited immutable weight constants")
    for record in audit["nodes"]:
        op = nodes[record["name"]]
        node, port = _source(nodes, op.input[1])
        entry = record["prepacked_constant"]
        if (op.type != GE_OP or not op.attr["transpose_weight"].b
                or op.attr["transpose_x"].b or op.attr["antiquant_group_size"].i != 128
                or node.type != "Const" or node.input or port != 0
                or len(node.output_desc) != 1 or "value" not in node.attr
                or not valid_prepacked_record(record)):
            raise ValueError("invalid locked NZ weight/consumer contract")
        desc = node.output_desc[0]
        value = node.attr["value"].t
        if (list(desc.shape.dim) != record["weight_storage_shape"]
                or list(desc.attr["origin_shape"].list.i) != record["weight_shape"]
                or desc.layout != "FRACTAL_NZ" or _dtype(desc) != "DT_INT8"
                or desc.attr["format_for_int"].i != 29
                or desc.attr["origin_format_for_int"].i != 2
                or not desc.attr["origin_shape_initialized"].b
                or value.desc != _const_value_desc(desc)
                or len(value.data) != entry["storage_bytes"]
                or hashlib.sha256(value.data).hexdigest() != entry["storage_sha256"]):
            raise ValueError("locked NZ constant descriptor or payload differs from offline weight")
        # Descriptor names may describe different ports, all shape/format
        # fields on the edge must otherwise agree.
        consumer = copy.deepcopy(op.input_desc[1])
        consumer.name = desc.name
        if consumer != desc:
            raise ValueError("WeightQuant input differs from locked NZ constant output")


def valid_prepacked_record(node):
    record = node.get("prepacked_constant", {})
    shape = node.get("weight_shape", [])
    storage = node.get("weight_storage_shape", [])
    def digest(key):
        value = record.get(key)
        return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    return (node.get("weight_storage") == PREPACK_POLICY and node.get("format_conversion") is None
            and record.get("name") == node.get("name", "") + "_weight_nz"
            and node.get("weight") == record["name"] + ":0"
            and record.get("dtype") == "int8" and record.get("format") == "FRACTAL_NZ"
            and record.get("descriptor_policy") == CONST_DESC_POLICY
            and record.get("output_shape_locked") is True
            and record.get("value_shape") == storage and record.get("value_format") == "FRACTAL_NZ"
            and record.get("value_origin_shape") == shape and record.get("value_origin_format") == "ND"
            and record.get("logical_shape") == shape and record.get("storage_shape") == storage
            and record.get("logical_bytes") == prod(shape) and record.get("storage_bytes") == prod(storage)
            and record.get("roundtrip") == "BIT_EXACT" and record.get("padding") == "ZERO"
            and digest("logical_sha256") and digest("storage_sha256"))
