"""Immutable FileConstant descriptor extension, preserved from 5e5b0d7.

The canonical historical weight_prepack.py and its hash remain unchanged.
"""
import torch
from .weight_quant_layout import GE_OP, POLICY, _dtype, _resolved_desc, _source


def restore_fileconstant_weight_descriptors(graph, layout_audit, immutable_weights, tensor_metadata):
    """Restore only audited INT8 FileConstant outputs from immutable bindings.

    TorchAir may replace a typed Data with a FileConstant whose output desc is
    undefined/empty. The layout pass resolves a COPY using FX metadata, while
    prepack deliberately validates the source node itself. Reconcile the FX
    record and any concrete GE fields with the exporter-bound tensor before
    filling that node. Runtime inputs, constant payloads and file attributes
    are never changed; the prepack immutability/hash/shape gates still run.
    """
    if layout_audit.get("policy") != POLICY or not layout_audit.get("nodes"):
        raise ValueError("FileConstant repair requires the production WeightQuant layout audit")
    nodes = {node.name: node for node in graph.op}
    plans, records = [], []
    for record in layout_audit["nodes"]:
        op = nodes[record["name"]]
        conversion, port = _source(nodes, op.input[1])
        if (op.type != GE_OP or port != 0 or conversion.type != "TransData"
                or len(conversion.input) != 1 or record["format_conversion"] != conversion.name):
            raise ValueError("FileConstant repair requires the audited weight TransData")
        edge = conversion.input[0]
        source, source_port = _source(nodes, edge)
        if source.type != "FileConstant":
            continue  # Const and runtime inputs keep the existing prepack checks.
        value = immutable_weights.get(edge)
        if (source_port != 0 or len(source.output_desc) != 1 or source.input
                or not isinstance(value, torch.Tensor) or value.device.type == "meta"
                or value.dtype != torch.int8 or value.ndim != 2
                or any(type(d) is not int or d <= 0 for d in value.shape)
                or list(value.shape) != record["weight_shape"]):
            raise ValueError(f"FileConstant {edge} lacks a matching immutable INT8 weight binding")
        expected = {"dtype": "DT_INT8", "shape": list(value.shape)}
        metadata = tensor_metadata.get(edge)
        if metadata is not None and (metadata.get("dtype") != expected["dtype"] or
                                     metadata.get("shape") != expected["shape"]):
            raise ValueError(f"FileConstant FX metadata/immutable tensor conflict at {edge}: "
                             f"metadata={metadata}, immutable={expected}")
        # Reuse the existing resolver to reject conflicting concrete GE fields.
        # Only the proven immutable tensor can supply missing descriptor fields.
        resolved = _resolved_desc(nodes, edge, {edge: expected})
        before = source.output_desc[0]
        if _dtype(resolved) != "DT_INT8" or list(resolved.shape.dim) != expected["shape"]:
            raise ValueError(f"FileConstant {edge} descriptor differs from its immutable tensor")
        if resolved != before:
            plans.append((before, resolved))
            records.append({"source": edge, "before_dtype": _dtype(before),
                            "before_shape": list(before.shape.dim), **expected,
                            "fx_metadata_checked": metadata is not None})
    # Validate every candidate before changing any producer descriptor.
    for before, resolved in plans:
        before.CopyFrom(resolved)
    return {"policy": "immutable-int8-fileconstant-descriptor-v1", "status": "PASS",
            "node_count": len(records), "nodes": records}
