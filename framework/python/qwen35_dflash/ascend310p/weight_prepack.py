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
        # Const's TensorDef is the storage authority. Keep physical NZ shape
        # AND logical [N,K] origin on its value and outgoing edge descriptors.
        conversion.Clear()
        conversion.name, conversion.type = name, "Const"
        conversion.output_desc.add().CopyFrom(desc)
        conversion.attr["value"].t.desc.CopyFrom(desc)
        conversion.attr["value"].t.data = data
        item = {"name": name, "source": edge, "dtype": "int8", "format": "FRACTAL_NZ",
                "logical_shape": list(matrix.shape), "storage_shape": cached["storage_shape"],
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
        "offline_manifest_sha256": cache["sha256"],
        "node_count": len(records), "removed_weight_transdata": [r["name"] for r in records],
        "removed_nd_constants": sorted(removed), "constants": records}
    layout_audit["inserted_transdata"] = []
    return layout_audit


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
            and record.get("logical_shape") == shape and record.get("storage_shape") == storage
            and record.get("logical_bytes") == prod(shape) and record.get("storage_bytes") == prod(storage)
            and record.get("roundtrip") == "BIT_EXACT" and record.get("padding") == "ZERO"
            and digest("logical_sha256") and digest("storage_sha256"))
