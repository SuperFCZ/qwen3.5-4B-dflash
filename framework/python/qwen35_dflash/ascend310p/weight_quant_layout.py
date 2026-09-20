"""Lower Draft weight-only MatMul to the built-in 310P WeightNz template.

This is an AIR serialization rewrite of the built-in WeightQuantBatchMatmulV2,
not a new NPU kernel. The public [N,K] integer weight and [N,K/128] scale ABI,
group size and precision remain unchanged. A real TransData converts the
transient INT8 weight to NZ; no persistent FP16 weight or custom kernel is used.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import importlib
import struct

import torch


POLICY = "weight-quant-nk-nz-weight-gn-scale-v3"
PROBE_POLICY = "weight-quant-synthetic-support-probe-v3"
GE_OP = "WeightQuantBatchMatmulV2"


def validate_probe_config(config):
    """The synthetic probe may vary support controls, never checkpoint scales."""
    if (not isinstance(config, dict) or set(config) != {"weight_layout", "group_size", "weight_format"}
            or config["weight_layout"] not in {"nk", "kn"}
            or config["weight_format"] not in {"nd", "nz"}
            or (config["weight_format"] == "nz" and config["weight_layout"] != "nk")
            or type(config["group_size"]) is not int or config["group_size"] not in (0, 128)):
        raise ValueError("WeightQuant synthetic probe requires group_size=0/128, "
                         "weight_format=nd/nz and weight_layout=nk/kn (NZ requires NK on 310P)")
    return config


def _dtype(desc):
    enum = desc.DESCRIPTOR.fields_by_name["dtype"].enum_type
    value = enum.values_by_number.get(desc.dtype)
    return value.name if value is not None else f"UNKNOWN({desc.dtype})"


@contextmanager
def capture_weight_quant_metadata(enabled):
    """Retain typed FX shapes, which TorchAir does not put in every GE desc.

    Tensor.set_meta sets dtype/symsize but may leave output_desc.shape empty.
    Snapshot only names, shapes and dtypes while the existing export lock is
    held. Never keep device tensors or specialize a symbolic dimension, and
    leave TorchAir's descriptors and converter registration untouched.
    """
    metadata = {}
    if not enabled:
        yield metadata
        return
    tensor_type = importlib.import_module("torchair.ge._ge_graph").Tensor
    original = tensor_type.set_meta

    def record(tensor, meta_output, *args, **kwargs):
        result = original(tensor, meta_output, *args, **kwargs)
        if isinstance(meta_output, torch.Tensor):
            metadata[tensor.tensor] = {
                "shape": [dim if type(dim) is int else -1 for dim in meta_output.shape],
                "dtype": _dtype(tensor.desc),
            }
        return result

    tensor_type.set_meta = record
    try:
        yield metadata
    finally:
        tensor_type.set_meta = original


def _source(nodes, edge):
    name, _, index = edge.rpartition(":")
    node = nodes.get(name)
    if node is None or not index.isdigit() or int(index) >= len(node.output_desc):
        raise ValueError(f"WeightQuant AIR has an invalid tensor edge: {edge!r}")
    return node, int(index)


def _resolved_desc(nodes, edge, metadata):
    node, port = _source(nodes, edge)
    desc = copy.deepcopy(node.output_desc[port])
    meta = metadata.get(edge)
    if meta is None:
        return desc
    shape = list(desc.shape.dim)
    actual = meta["shape"]
    dtype = _dtype(desc)
    if (dtype not in ("DT_UNDEFINED", meta["dtype"]) or
            (shape not in ([], [-2]) and
             (len(shape) != len(actual) or any(
                 a >= 0 and b >= 0 and a != b for a, b in zip(shape, actual))))):
        raise ValueError(f"WeightQuant descriptor/FX metadata conflict at {edge}: "
                         f"descriptor={shape}/{dtype}, metadata={actual}/{meta['dtype']}")
    # Empty intermediate shapes are unresolved descriptors at this boundary,
    # not evidence for scalar tensors. Typed metadata supplies the rank.
    resolved = actual if shape in ([], [-2]) else [
        a if a >= 0 else b for a, b in zip(shape, actual)]
    desc.shape.dim[:] = resolved
    if dtype == "DT_UNDEFINED":
        enum = desc.DESCRIPTOR.fields_by_name["dtype"].enum_type
        desc.dtype = enum.values_by_name[meta["dtype"]].number
    return desc


def _permutation(nodes, transpose):
    if any(edge.endswith(":-1") for edge in transpose.input):
        raise ValueError("WeightQuant transpose has control dependencies; refusing to drop them")
    if transpose.type == "TransposeD":
        return list(transpose.attr["perm"].list.i)
    if transpose.type != "Transpose" or len(transpose.input) != 2:
        raise ValueError("WeightQuant AIR requires a 2-D Transpose or TransposeD")
    perm, index = _source(nodes, transpose.input[1])
    if perm.type != "Const" or index != 0 or "value" not in perm.attr:
        raise ValueError("WeightQuant transpose permutation must be constant")
    tensor = perm.attr["value"].t
    formats = {"DT_INT32": "i", "DT_INT64": "q"}
    fmt = formats.get(_dtype(tensor.desc))
    if list(tensor.desc.shape.dim) != [2] or fmt is None:
        raise ValueError("WeightQuant transpose permutation must have two integers")
    if len(tensor.data) != struct.calcsize("<2" + fmt):
        raise ValueError("WeightQuant transpose permutation has an invalid byte count")
    return list(struct.unpack("<2" + fmt, tensor.data))


def _untranspose(nodes, edge, metadata):
    transpose, index = _source(nodes, edge)
    if index != 0 or _permutation(nodes, transpose) != [1, 0]:
        raise ValueError("WeightQuant AIR only folds the exact [1,0] permutation")
    before = _resolved_desc(nodes, transpose.input[0], metadata)
    after = _resolved_desc(nodes, edge, metadata)
    if (len(before.shape.dim) != 2 or list(before.shape.dim)[::-1] != list(after.shape.dim)
            or before.dtype != after.dtype):
        raise ValueError(f"WeightQuant transpose {edge} does not match source {transpose.input[0]}: "
                         f"before={list(before.shape.dim)}/{_dtype(before)}, "
                         f"after={list(after.shape.dim)}/{_dtype(after)}; "
                         "requires matching typed FX metadata or resolved GE descriptors")
    return transpose.input[0], before, transpose.name


def _nz_shape(n, k):
    # INT8 FRACTAL_NZ for logical [N,K]: K1,N1,N0,K0. K0 is 32 bytes.
    return [(k + 31) // 32, (n + 15) // 16, 16, 32]


def _storage_desc(desc, *, layout, shape, origin_shape):
    result = copy.deepcopy(desc)
    result.layout = layout
    result.shape.dim[:] = shape
    # GE reads these integer format attributes when deserializing TorchAir IR.
    # Keep logical/origin shape separate from the four-dimensional NZ storage.
    result.attr["format_for_int"].i = 29 if layout == "FRACTAL_NZ" else 2
    result.attr["origin_format_for_int"].i = 2
    result.attr["origin_shape"].list.val_type = 2
    result.attr["origin_shape"].list.i[:] = origin_shape
    result.attr["origin_shape_initialized"].b = True
    result.attr["origin_format_is_set"].b = True
    return result


def _weight_nz_node(op, edge, desc):
    """Materialize the format conversion, never relabel ND bytes as NZ."""
    n, k = desc.shape.dim
    if (desc.layout not in ("", "ND") or
            ("format_for_int" in desc.attr and desc.attr["format_for_int"].i != 2)):
        raise ValueError("WeightQuant TransData requires a real ND source")
    node = type(op)()
    node.name, node.type = op.name + "_weight_nz", "TransData"
    node.input.append(edge)
    node.attr["src_format"].s = b"ND"
    node.attr["dst_format"].s = b"FRACTAL_NZ"
    node.attr["src_subformat"].i = node.attr["dst_subformat"].i = 0
    node.attr["groups"].i = 1
    src = _storage_desc(desc, layout="ND", shape=[n, k], origin_shape=[n, k])
    dst = _storage_desc(desc, layout="FRACTAL_NZ", shape=_nz_shape(n, k), origin_shape=[n, k])
    src.name, dst.name = "src", "dst"
    node.input_desc.add().CopyFrom(src)
    node.output_desc.add().CopyFrom(dst)
    return node


def _existing_weight_nz(nodes, op, metadata):
    """Validate a previously lowered conversion before treating it as idempotent."""
    node, port = _source(nodes, op.input[1])
    if node.type != "TransData" or port != 0 or len(node.input) != 1:
        raise ValueError("WeightQuant NZ input must come from the audited ND-to-NZ TransData")
    desc = _resolved_desc(nodes, node.input[0], metadata)
    if len(desc.shape.dim) != 2 or _dtype(desc) != "DT_INT8":
        raise ValueError("WeightQuant NZ source must be INT8 weight[N,K]")
    expected = _weight_nz_node(op, node.input[0], desc)
    if node.SerializeToString(deterministic=True) != expected.SerializeToString(deterministic=True):
        raise ValueError("WeightQuant NZ TransData descriptor/attributes changed")
    return desc, node


def normalize_weight_quant_layout(graph, tensor_metadata=None, *, probe_config=None):
    """Use NZ w[N,K] with transpose_weight=true and ND scale[G,N].

    Explicit synthetic probes may retain ND weights. Per-channel scales use [N].
    CANN's transpose_weight attribute applies ONLY to weight. Its group scale
    axes remain [K/group_size,N], even with transposed weight. Keep the scale
    transpose and its values; never replace it with a reshape. Validate all
    candidates before rewiring. Optional input slots are preserved.
    """
    nodes = {op.name: op for op in graph.op}
    metadata = tensor_metadata or {}
    probe = validate_probe_config(probe_config) if probe_config is not None else None
    layout = probe["weight_layout"] if probe is not None else "nk"
    weight_format = probe["weight_format"] if probe is not None else "nz"
    plans = []
    for op in graph.op:
        if op.type != GE_OP:
            continue
        if len(op.input) < 3 or len(op.input_desc) < 3:
            raise ValueError("WeightQuant AIR is missing required inputs")
        if op.attr["transpose_x"].b or op.attr["inner_precise"].i != 0:
            raise ValueError("Draft WeightQuant AIR requires transpose_x=false, inner_precise=0")
        group = op.attr["antiquant_group_size"].i
        if group not in (0, 128):
            raise ValueError("Draft WeightQuant AIR requires the original group-128 scales")
        if probe is not None and group != probe["group_size"]:
            raise ValueError("WeightQuant synthetic probe group differs from its declared control")
        folded = bool(op.attr["transpose_weight"].b)
        conversion = None
        insert_conversion = False
        replacements = []
        # The project uses symmetric quantization and no bias/output quantization.
        # Reject unrecognized numerics rather than guessing their layout.
        if any(edge and not edge.endswith(":-1") for edge in op.input[3:]):
            raise ValueError("Draft WeightQuant AIR expects absent optional quantization inputs")
        if layout == "kn":
            parent, _ = _source(nodes, op.input[1])
            if folded or parent.type in {"Transpose", "TransposeD"}:
                raise ValueError("KN probe requires physical [K,N] weights without a transpose")
            replacements.append((1, op.input[1], _resolved_desc(nodes, op.input[1], metadata), None))
        elif folded:
            parent, _ = _source(nodes, op.input[1])
            if parent.type in {"Transpose", "TransposeD"}:
                raise ValueError("WeightQuant AIR is already transposed twice")
            if parent.type == "TransData" and weight_format == "nz":
                desc, conversion = _existing_weight_nz(nodes, op, metadata)
                replacements.append((1, conversion.input[0], desc, None))
            else:
                replacements.append((1, op.input[1], _resolved_desc(nodes, op.input[1], metadata), None))
        else:
            edge, desc, removed = _untranspose(nodes, op.input[1], metadata)
            replacements.append((1, edge, desc, removed))
        # Still check any scale transpose against its source/typed metadata,
        # but do not remove or redirect it: the tiler requires [G,N].
        scale_parent, _ = _source(nodes, op.input[2])
        if scale_parent.type in {"Transpose", "TransposeD"}:
            _untranspose(nodes, op.input[2], metadata)
        replacements.append((2, op.input[2], _resolved_desc(nodes, op.input[2], metadata), None))
        weight, scale = replacements[0][2], replacements[1][2]
        x = _resolved_desc(nodes, op.input[0], metadata)
        n, k = weight.shape.dim if len(weight.shape.dim) == 2 else (0, 0)
        if layout == "kn":
            k, n = n, k
        groups = 1 if group == 0 else k // group
        expected_scale = [n] if group == 0 else [groups, n]
        if (n <= 0 or k <= 0 or k % 128 or (probe is None and group == 0 and k != 128)
                or list(scale.shape.dim) != expected_scale
                or len(x.shape.dim) != 2 or x.shape.dim[1] != k
                or [_dtype(x), _dtype(weight), _dtype(scale)] != ["DT_FLOAT16", "DT_INT8", "DT_FLOAT16"]):
            raise ValueError(f"WeightQuant {op.name} requires FP16 x[M,K], INT8 w[{layout.upper()}], "
                             f"FP16 scale{expected_scale} for group_size={group} "
                             f"(transpose_weight does not transpose scales); actual shapes/dtypes="
                             f"{[(list(d.shape.dim), _dtype(d)) for d in (x, weight, scale)]}")
        if weight_format == "nz":
            if conversion is None:
                conversion = _weight_nz_node(op, replacements[0][1], weight)
                if conversion.name in nodes:
                    raise ValueError(f"WeightQuant TransData name already exists: {conversion.name}")
                insert_conversion = True
            slot, edge, _, removed = replacements[0]
            replacements[0] = (slot, conversion.name + ":0", conversion.output_desc[0], removed)
        plans.append((op, replacements, folded, conversion, insert_conversion, [n, k]))
    records, candidates = [], set()
    insertions = {}
    for op, replacements, folded, conversion, insert_conversion, logical_shape in plans:
        for slot, edge, desc, removed in replacements:
            name = op.input_desc[slot].name
            op.input[slot] = edge
            op.input_desc[slot].CopyFrom(desc)
            op.input_desc[slot].name = name
            if removed:
                candidates.add(removed)
        op.attr["transpose_weight"].b = layout == "nk"
        if insert_conversion:
            insertions[op.name] = conversion
        records.append({"name": op.name, "weight": op.input[1], "scale": op.input[2],
                        "transpose_weight": layout == "nk", "already_folded": folded,
                        "weight_layout": layout.upper(),
                        "scale_layout": "N" if op.attr["antiquant_group_size"].i == 0 else "GN",
                        "group_size": op.attr["antiquant_group_size"].i,
                        "weight_shape": logical_shape if layout == "nk" else logical_shape[::-1],
                        "weight_format": "FRACTAL_NZ" if weight_format == "nz" else "ND",
                        "weight_storage_shape": list(op.input_desc[1].shape.dim),
                        "format_conversion": conversion.name if conversion is not None else None,
                        "scale_shape": list(op.input_desc[2].shape.dim)})
    # Preserve topological order and leave all public inputs/packed bytes intact.
    for i in range(len(graph.op) - 1, -1, -1):
        if graph.op[i].name in insertions:
            graph.op.insert(i, insertions[graph.op[i].name])
    # Preserve transposes shared by other consumers, including control edges.
    used = ({edge.rpartition(":")[0] for op in graph.op for edge in op.input if edge}
            if candidates else set())
    removed = candidates - used
    for i in range(len(graph.op) - 1, -1, -1):
        if graph.op[i].name in removed:
            del graph.op[i]
    return {"policy": PROBE_POLICY if probe is not None else POLICY,
            "probe_config": probe, "status": "PASS", "scope": "torchair-before-ge-save",
            "metadata_source": "torchair.Tensor.set_meta" if metadata else "GE descriptors",
            "node_count": len(records), "nodes": records, "removed_transposes": sorted(removed),
            "inserted_transdata": [node.name for node in insertions.values()]}


def weight_quant_layout_failure(graph, metadata, error):
    """Small failure report: descriptors/metadata only, never weight payloads."""
    nodes = {op.name: op for op in graph.op}

    def describe(edge, depth=1):
        record = {"edge": edge}
        try:
            node, port = _source(nodes, edge)
        except ValueError:
            return record
        desc = node.output_desc[port]
        record.update(op_type=node.type, shape=list(desc.shape.dim), dtype=_dtype(desc), layout=desc.layout,
                      fx_metadata=metadata.get(edge))
        if depth and node.type in {"Transpose", "TransposeD", "TransData"}:
            record["inputs"] = [describe(e, depth - 1) for e in node.input]
        return record

    records = [{"name": op.name, "inputs": [describe(e) for e in op.input]}
               for op in graph.op if op.type == GE_OP]
    return {"policy": POLICY, "status": "FAIL", "scope": "torchair-before-ge-save",
            "error": str(error), "node_count": len(records), "nodes": records}


def validate_weight_quant_layout(graph):
    """Do not send a known old transpose graph back to ATC."""
    count = sum(item.get("ge_node_occurrences", 0) for item in graph.get("custom_op_audit", [])
                if item.get("ge_op_type") == GE_OP)
    if not count:
        return
    abi = graph.get("runtime_input_abi", {})
    # This marker is accepted only with an explicit fake compiler by the ABI gate.
    if abi.get("status") == "NOT_APPLICABLE_EXPLICIT_TEST_DOUBLE":
        return
    audit = abi.get("weight_quant_layout", {})
    probe = graph.get("metadata", {}).get("weight_quant_probe")
    if probe is not None:
        validate_probe_config(probe)
        if graph.get("name") != "weight_quant_probe" or graph.get("role") != "diagnostic" or count != 1:
            raise ValueError("WeightQuant probe controls are only valid for a single synthetic diagnostic graph")
    expected_policy = PROBE_POLICY if probe is not None else POLICY
    expected_transpose = probe is None or probe["weight_layout"] == "nk"
    expected_format = "FRACTAL_NZ" if probe is None or probe["weight_format"] == "nz" else "ND"
    from .weight_prepack import PREPACK_POLICY, valid_prepacked_record
    storage = graph.get("metadata", {}).get("draft_weight_storage")
    prepack = audit.get("prepack")
    if storage is not None or prepack is not None:
        if (storage != PREPACK_POLICY or probe is not None
                or graph.get("metadata", {}).get("draft_quantization") != "w8a16"
                or not isinstance(prepack, dict) or prepack.get("policy") != PREPACK_POLICY
                or prepack.get("status") != "PASS" or prepack.get("node_count") != count
                or prepack.get("constants") != [n.get("prepacked_constant") for n in audit.get("nodes", [])]
                or prepack.get("removed_weight_transdata") != [n.get("name", "") + "_weight_nz" for n in audit.get("nodes", [])]
                or audit.get("inserted_transdata") != []):
            raise ValueError("WeightQuant offline W8 NZ constant audit differs; re-export AIR")

    def valid_storage(node):
        shape = node.get("weight_shape", [])
        if len(shape) != 2 or any(type(dim) is not int or dim <= 0 for dim in shape):
            return False
        if node.get("weight_format") != expected_format:
            return False
        if expected_format == "ND":
            return node.get("weight_storage_shape") == shape and node.get("format_conversion") is None
        if storage == PREPACK_POLICY:
            return node.get("weight_storage_shape") == _nz_shape(*shape) and valid_prepacked_record(node)
        return (node.get("weight_storage_shape") == _nz_shape(*shape)
                and node.get("format_conversion") == node.get("name", "") + "_weight_nz"
                and node.get("weight") == node["format_conversion"] + ":0")

    if (audit.get("policy") != expected_policy or audit.get("status") != "PASS"
            or audit.get("probe_config") != probe
            or audit.get("node_count") != count or len(audit.get("nodes", [])) != count
            or any(node.get("transpose_weight") is not expected_transpose
                   or not valid_storage(node)
                   or node.get("scale_layout") != ("N" if node.get("group_size") == 0 else "GN")
                   or (node.get("group_size") == 0 and len(node.get("scale_shape", [])) != 1)
                   or (probe is not None and node.get("group_size") != probe["group_size"])
                   for node in audit.get("nodes", []))):
        raise ValueError("WeightQuant AIR lacks the NZ conversion and group-specific weight/scale audit; "
                         "re-export AIR with the current source. Recompiling old AIR or disabling "
                         "fusion cannot apply this fix.")
