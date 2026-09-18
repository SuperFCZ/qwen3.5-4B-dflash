"""Fold the Draft's exact 2-D transposes before CANN's graph fusion.

This is an AIR serialization rewrite of the built-in WeightQuantBatchMatmulV2,
not a new NPU kernel. The public [N,K] integer weight and [N,K/128] scale ABI,
group size and precision remain unchanged. No weight is copied or dequantized.
"""
from __future__ import annotations

import struct


POLICY = "weight-quant-nk-transpose-attr-v1"
GE_OP = "WeightQuantBatchMatmulV2"


def _dtype(desc):
    enum = desc.DESCRIPTOR.fields_by_name["dtype"].enum_type
    return enum.values_by_number[desc.dtype].name


def _source(nodes, edge):
    name, _, index = edge.rpartition(":")
    node = nodes.get(name)
    if node is None or not index.isdigit() or int(index) >= len(node.output_desc):
        raise ValueError(f"WeightQuant AIR has an invalid tensor edge: {edge!r}")
    return node, int(index)


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


def _untranspose(nodes, edge):
    transpose, index = _source(nodes, edge)
    if index != 0 or _permutation(nodes, transpose) != [1, 0]:
        raise ValueError("WeightQuant AIR only folds the exact [1,0] permutation")
    node, port = _source(nodes, transpose.input[0])
    before, after = node.output_desc[port], transpose.output_desc[0]
    if (len(before.shape.dim) != 2 or list(before.shape.dim)[::-1] != list(after.shape.dim)
            or before.dtype != after.dtype):
        raise ValueError("WeightQuant transpose descriptor does not match its source")
    return transpose.input[0], before, transpose.name


def normalize_weight_quant_layout(graph):
    """Validate all candidates first, then fold weight/scale as one operation.

    GE interprets per-group scale in the same orientation as weight. Folding
    weight alone would silently swap the group/channel axes. Optional inputs
    keep their slots (including empty edges); they are never removed/relinked.
    """
    nodes = {op.name: op for op in graph.op}
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
        folded = bool(op.attr["transpose_weight"].b)
        replacements = []
        # The project uses symmetric quantization and no bias/output quantization.
        # Reject unrecognized numerics rather than guessing their layout.
        if any(edge and not edge.endswith(":-1") for edge in op.input[3:]):
            raise ValueError("Draft WeightQuant AIR expects absent optional quantization inputs")
        for slot in (1, 2):
            if folded:
                parent, port = _source(nodes, op.input[slot])
                if parent.type in {"Transpose", "TransposeD"}:
                    raise ValueError("WeightQuant AIR is already transposed twice")
                replacements.append((slot, op.input[slot], parent.output_desc[port], None))
            else:
                edge, desc, removed = _untranspose(nodes, op.input[slot])
                replacements.append((slot, edge, desc, removed))
        weight, scale = replacements[0][2], replacements[1][2]
        xnode, xport = _source(nodes, op.input[0])
        x = xnode.output_desc[xport]
        n, k = weight.shape.dim if len(weight.shape.dim) == 2 else (0, 0)
        groups = 1 if group == 0 else k // group
        if (n <= 0 or k <= 0 or k % 128 or (group == 0 and k != 128)
                or list(scale.shape.dim) != [n, groups]
                or len(x.shape.dim) != 2 or x.shape.dim[1] != k
                or [_dtype(x), _dtype(weight), _dtype(scale)] != ["DT_FLOAT16", "DT_INT8", "DT_FLOAT16"]):
            raise ValueError("WeightQuant AIR does not match FP16 x[M,K], INT8 w[N,K], FP16 s[N,K/128]")
        plans.append((op, replacements, folded))
    records, candidates = [], set()
    for op, replacements, folded in plans:
        for slot, edge, desc, removed in replacements:
            name = op.input_desc[slot].name
            op.input[slot] = edge
            op.input_desc[slot].CopyFrom(desc)
            op.input_desc[slot].name = name
            if removed:
                candidates.add(removed)
        op.attr["transpose_weight"].b = True
        records.append({"name": op.name, "weight": op.input[1], "scale": op.input[2],
                        "transpose_weight": True, "already_folded": folded,
                        "group_size": op.attr["antiquant_group_size"].i,
                        "weight_shape": list(op.input_desc[1].shape.dim),
                        "scale_shape": list(op.input_desc[2].shape.dim)})
    # Preserve transposes shared by other consumers, including control edges.
    used = ({edge.rpartition(":")[0] for op in graph.op for edge in op.input if edge}
            if candidates else set())
    removed = candidates - used
    for i in range(len(graph.op) - 1, -1, -1):
        if graph.op[i].name in removed:
            del graph.op[i]
    return {"policy": POLICY, "status": "PASS", "scope": "torchair-before-ge-save",
            "node_count": len(records), "nodes": records, "removed_transposes": sorted(removed)}


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
    if (audit.get("policy") != POLICY or audit.get("status") != "PASS"
            or audit.get("node_count") != count or len(audit.get("nodes", [])) != count
            or any(node.get("transpose_weight") is not True for node in audit.get("nodes", []))):
        raise ValueError("Quantized Draft AIR lacks the NK transpose-attribute audit; re-export AIR "
                         "with the current source. Recompiling old AIR or disabling fusion cannot apply this fix.")
