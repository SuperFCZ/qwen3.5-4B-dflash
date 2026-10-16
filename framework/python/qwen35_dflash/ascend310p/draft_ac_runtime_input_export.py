"""Explicit diagnostic ABI/FileConstant hook preserved from 5e5b0d7.

Reuses historical ABI helpers and lock, without modifying their module globals.
The TorchAir conversion/save hook lifecycle and all hash checks are unchanged.
"""
from __future__ import annotations
from contextlib import contextmanager
import importlib
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
import torch
from .utils import atomic_write_json
from .runtime_input_export import (
    _PATCH_LOCK, _order_weight_conversion_inputs, _public_bindings,
    _normalize_public_nodes, _gdr_output_dtype_audit, _verify_discard_output_audit,
    _indexed_graph_inputs,
)


@contextmanager
def canonical_runtime_input_abi(
    torchair: Any, *, public_inputs: Sequence[torch.Tensor],
    public_names: Sequence[str], explicit_test_double: bool = False,
    require_static_shapes: bool = False,
    dynamic_input_axes: Mapping[str, Sequence[int]] | None = None,
    public_output_names: Sequence[str] = (),
    verify_discard_output_names: Sequence[str] = (),
    capture_weight_quant_shapes: bool = False,
    weight_quant_probe: Mapping[str, Any] | None = None,
    weight_prepack_manifest: str | None = None,
    weight_prepack_output: Path | None = None,
    draft_ac: Mapping[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    if weight_prepack_manifest and weight_prepack_output is not None:
        raise ValueError("select automatic NZ packing or an existing offline manifest")
    audit: dict[str, Any] = {
        "policy": "public-tensor-storage-identity-v1",
        "status": "ARMED",
        "python_float_policy": "dynamo-specialize-float",
        "logical_input_names": list(public_names),
        "bindings": [],
        "calls": 0,
    }
    if explicit_test_double:
        audit["status"] = "NOT_APPLICABLE_EXPLICIT_TEST_DOUBLE"
        yield audit
        return
    if len(public_inputs) != len(public_names) or not public_names:
        raise ValueError("AIR public names must describe every positional tensor input")
    if len(set(public_names)) != len(public_names):
        raise ValueError("AIR public input names must be unique")
    if torchair is not importlib.import_module("torchair"):
        raise RuntimeError("public AIR ABI normalization requires canonical TorchAir")
    export_utils = importlib.import_module("torchair._utils.export_utils")
    dynamo_config = importlib.import_module("torch._dynamo.config")
    if not hasattr(dynamo_config, "specialize_float"):
        raise RuntimeError("this TorchDynamo lacks the required specialize_float policy")

    from .weight_quant_layout import (
        capture_weight_quant_metadata, normalize_weight_quant_layout, weight_quant_layout_failure,
    )
    from .weight_prepack import (
        load_prepacked_weights, prepack_weight_quant_constants,
    )
    from .draft_ac_weight_prepack import restore_fileconstant_weight_descriptors
    prepacked = load_prepacked_weights(weight_prepack_manifest) if weight_prepack_manifest else None

    with _PATCH_LOCK, capture_weight_quant_metadata(capture_weight_quant_shapes) as tensor_metadata:
        original = export_utils._convert_data_to_const
        ac_payload = None
        if draft_ac is not None:
            from .draft_ac_payload import PayloadLifecycle
            ac_payload = PayloadLifecycle(draft_ac, export_utils, original)

        def convert(inputs, export_graph, file_path, weight_name):
            reordered = _order_weight_conversion_inputs(inputs, export_graph)
            bindings = _public_bindings(
                inputs, export_graph, weight_name, public_inputs, public_names,
            )
            records = [
                {"index": index, "logical_name": public_names[index],
                 "data_node_name": node.name,
                 "dtype": str(public_inputs[index].dtype).removeprefix("torch."),
                 "example_shape": list(public_inputs[index].shape)}
                for index, node in bindings
            ]
            audit["weight_conversion"] = {
                "policy": "runtime-input-index-order-v1",
                "reordered": reordered,
                "input_count": len(inputs),
                "weight_count": sum(id(value) in weight_name for value in inputs),
            }
            immutable_weights = {export_graph.op[i].name + ":0": value
                                 for i, value in enumerate(inputs) if id(value) in weight_name}
            if ac_payload is not None:
                ac_payload.capture(export_graph, immutable_weights, file_path, inputs, weight_name)
            result = original(inputs, export_graph, file_path, weight_name)
            _normalize_public_nodes(export_graph, bindings)
            try:
                weight_quant = normalize_weight_quant_layout(export_graph, tensor_metadata,
                                                              probe_config=weight_quant_probe)
                if prepacked is not None or weight_prepack_output is not None:
                    weight_quant["fileconstant_descriptors"] = restore_fileconstant_weight_descriptors(
                        export_graph, weight_quant, immutable_weights, tensor_metadata)
                    weight_quant = prepack_weight_quant_constants(
                        export_graph, weight_quant, immutable_weights, prepacked,
                        output_dir=weight_prepack_output)
            except ValueError as error:
                failure = weight_quant_layout_failure(export_graph, tensor_metadata, error)
                audit["weight_quant_layout"] = failure
                report = atomic_write_json(Path(file_path) / "weight-quant-layout.json", failure)
                raise ValueError(f"{error}; report={report}") from error
            if weight_quant["node_count"]:
                audit["weight_quant_layout"] = weight_quant
                atomic_write_json(Path(file_path) / "weight-quant-layout.json", weight_quant)
                from .atc_diagnostics import weight_quant_snapshot
                atomic_write_json(Path(file_path) / "weight-quant-descriptors.json", {
                    "scope": "torchair-before-ge-save; not CANN InferShape evidence",
                    "nodes": weight_quant_snapshot(export_graph),
                })
                node = weight_quant["nodes"][0]
                print(f"[export-air] WeightQuantBatchMatmulV2 layout={node['weight_layout']} "
                      f"weight_format={node['weight_format']} storage_shape={node['weight_storage_shape']} "
                      f"scale_layout={node['scale_layout']} scale_shape={node['scale_shape']} "
                      f"transpose_weight={str(node['transpose_weight']).lower()} "
                      f"group_size={node['group_size']} nodes={weight_quant['node_count']}", flush=True)
                if "prepack" in weight_quant:
                    print(f"[export-air] W8 offline NZ constants={weight_quant['node_count']} "
                          "weight_transdata=0 roundtrip=BIT_EXACT "
                          f"const_value_shape={node['prepacked_constant']['value_shape']} "
                          f"const_value_format={node['prepacked_constant']['value_format']} "
                          f"const_value_origin={node['prepacked_constant']['value_origin_shape']} "
                          "const_output_shape_locked=true weightquant_inference=enabled "
                          f"descriptor={weight_quant['prepack']['descriptor_policy']}", flush=True)
            if draft_ac is not None:
                try:
                    ac_payload.after_convert(export_graph,immutable_weights,file_path,result,inputs,weight_name,audit)
                except (ValueError,OSError) as error:
                    atomic_write_json(Path(file_path)/'ac-constants.json',dict(status='FAIL',error=str(error),
                        nodes=[dict(name=n.name,type=n.type,inputs=list(n.input),attrs=list(n.attr)) for n in export_graph.op
                               if n.type in ('DFlashGroupQuantLinear','DFlashDraftLmHeadTop1','DFlashDraftLmHeadTop1Audit','FileConstant')]))
                    raise
            gdr_dtypes = _gdr_output_dtype_audit(export_graph)
            if gdr_dtypes["node_count"]:
                audit["gdr_output_dtypes"] = gdr_dtypes
                dtype_report = atomic_write_json(
                    Path(file_path) / "gdr-output-dtypes.json", gdr_dtypes,
                )
                if gdr_dtypes["status"] != "PASS":
                    failed = next(node for node in gdr_dtypes["nodes"] if node["status"] != "PASS")
                    raise RuntimeError(
                        f"ChunkGatedDeltaRule {failed['name']!r} GE output dtype mismatch "
                        f"before AIR save: {failed['outputs']}; expected core_attn DT_FLOAT16 "
                        f"and last_recurrent_state DT_FLOAT; report={dtype_report}"
                    )
            if verify_discard_output_names:
                discard = _verify_discard_output_audit(
                    export_graph, public_output_names, verify_discard_output_names,
                )
                audit["verify_discard_outputs"] = discard
                report = atomic_write_json(
                    Path(file_path) / "verify-discard-outputs.json", discard,
                )
                if discard["status"] != "PASS":
                    raise RuntimeError(
                        "raw first-pass GDR states must reach separate OM outputs; "
                        f"report={report}"
                    )
            if require_static_shapes or dynamic_input_axes:
                final_nodes = _indexed_graph_inputs(export_graph)
                for record in records:
                    index = record["index"]
                    node = final_nodes[index]
                    if len(node.output_desc) != 1:
                        raise RuntimeError("static AIR Data must have one tensor output")
                    shape = list(node.output_desc[0].shape.dim)
                    expected = list(public_inputs[index].shape)
                    for axis in (dynamic_input_axes or {}).get(public_names[index], ()):
                        expected[axis] = -1
                    if shape != expected:
                        raise RuntimeError(
                            f"static AIR input / declared dynamic axis {public_names[index]} has unexpected "
                            f"serialized shape {shape}; expected {expected}"
                        )
                    record["serialized_shape"] = shape
            audit["calls"] += 1
            audit["bindings"] = records
            return result

        export_utils._convert_data_to_const = convert
        if ac_payload is not None and callable(ac_payload.original_save):
            export_utils._save_weight2file = ac_payload.save
        try:
            # Specializes Python model attributes, not tensor dimensions or
            # data-derived values. Tensor/SymInt shapes remain dynamic.
            with dynamo_config.patch(specialize_float=True):
                yield audit
            if ac_payload is not None:
                ac_payload.finish()
            if audit["calls"] != 1:
                raise RuntimeError("expected one canonical public ABI serialization per AIR graph")
            audit["status"] = "PASS"
        finally:
            export_utils._convert_data_to_const = original
            if ac_payload is not None and callable(ac_payload.original_save):
                export_utils._save_weight2file = ac_payload.original_save
