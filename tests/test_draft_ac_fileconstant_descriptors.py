"""Retained descriptor regressions; CPU-only, not NPU acceptance."""
import copy
import hashlib
import json
import sys
from types import ModuleType, SimpleNamespace
import pytest
import torch
from test_weight_prepack import cached_weight, const
from test_weight_quant_layout import fixture_graph, fake_ge_tensor
from qwen35_dflash.ascend310p.weight_prepack import pack_int8_nz, load_prepacked_weights, prepack_weight_quant_constants
from qwen35_dflash.ascend310p.weight_quant_layout import normalize_weight_quant_layout
from qwen35_dflash.ascend310p.draft_ac_weight_prepack import restore_fileconstant_weight_descriptors
from qwen35_dflash.ascend310p.draft_ac_runtime_input_export import canonical_runtime_input_abi


def sparse_fileconstant(node):
    """Receiver regression: FX knows the weight, FileConstant desc does not."""
    node.type = "FileConstant"
    node.ClearField("input")
    node.attr.clear()
    node.attr["file_id"].s = b"immutable-weight.bin"
    node.attr["offset"].i = 512
    desc = node.output_desc[0]
    desc.dtype = desc.DESCRIPTOR.fields_by_name["dtype"].enum_type.values_by_name["DT_UNDEFINED"].number
    desc.shape.dim[:] = []


def sparse_weight_graph(n=64, k=256):
    graph = fixture_graph(shared=True)
    nodes = {node.name: node for node in graph.op}
    for name, shape in (("x", [16, k]), ("w", [n, k]), ("wt", [k, n]),
                        ("s", [n, k // 128]), ("st", [k // 128, n])):
        nodes[name].output_desc[0].shape.dim[:] = shape
    sparse_fileconstant(nodes["w"])
    metadata = {"w:0": {"dtype": "DT_INT8", "shape": [n, k]}}
    audit = normalize_weight_quant_layout(graph, metadata)
    q = torch.zeros(n, k, dtype=torch.int8)
    return graph, audit, {"w:0": q}, metadata


@pytest.mark.parametrize("n,k", [(64, 256), (19456, 2560), (2560, 9728)])
@pytest.mark.parametrize("use_fx", [False, True])
def test_fileconstant_restore_checks_real_shapes_without_changing_binding(n, k, use_fx):
    graph, audit, immutable, metadata = sparse_weight_graph(n, k)
    nodes = {node.name: node for node in graph.op}
    before = {name: node.SerializeToString(deterministic=True) for name, node in nodes.items() if name != "w"}
    original = copy.deepcopy(nodes["w"])
    report = restore_fileconstant_weight_descriptors(graph, audit, immutable, metadata if use_fx else {})
    weight = nodes["w"]
    assert weight.type == "FileConstant" and not weight.input
    assert weight.attr == original.attr  # Keep the exact file binding/offset.
    assert len(weight.output_desc) == 1 and list(weight.output_desc[0].shape.dim) == [n, k]
    enum = weight.output_desc[0].DESCRIPTOR.fields_by_name["dtype"].enum_type
    assert enum.values_by_number[weight.output_desc[0].dtype].name == "DT_INT8"
    assert before == {name: node.SerializeToString(deterministic=True) for name, node in nodes.items() if name != "w"}
    assert report["node_count"] == 1 and report["nodes"][0]["fx_metadata_checked"] is use_fx
    snapshot = graph.SerializeToString(deterministic=True)
    assert restore_fileconstant_weight_descriptors(graph, audit, immutable, metadata)["node_count"] == 0
    assert graph.SerializeToString(deterministic=True) == snapshot


@pytest.mark.parametrize("damage", ["missing_binding", "binding_dtype", "binding_shape", "meta_tensor",
                                    "fx_dtype", "fx_shape", "fx_symbolic", "ge_dtype", "ge_shape",
                                    "dependency", "multiple_outputs"])
def test_fileconstant_restore_does_not_override_conflicts_or_guess_immutability(damage):
    graph, audit, immutable, metadata = sparse_weight_graph()
    weight = next(node for node in graph.op if node.name == "w")
    if damage == "missing_binding": immutable.clear()
    elif damage == "binding_dtype": immutable["w:0"] = immutable["w:0"].half()
    elif damage == "binding_shape": immutable["w:0"] = immutable["w:0"].t()
    elif damage == "meta_tensor": immutable["w:0"] = torch.empty(64, 256, dtype=torch.int8, device="meta")
    elif damage == "fx_dtype": metadata["w:0"]["dtype"] = "DT_FLOAT16"
    elif damage == "fx_shape": metadata["w:0"]["shape"] = [256, 64]
    elif damage == "fx_symbolic": metadata["w:0"]["shape"] = [-1, 256]
    elif damage == "ge_dtype": weight.output_desc[0].dtype = 1  # Test protobuf DT_FLOAT16.
    elif damage == "ge_shape": weight.output_desc[0].shape.dim[:] = [63, 256]
    elif damage == "dependency": weight.input.append("x:-1")
    else: weight.output_desc.add().CopyFrom(weight.output_desc[0])
    before = graph.SerializeToString(deterministic=True)
    with pytest.raises(ValueError):
        restore_fileconstant_weight_descriptors(graph, audit, immutable, metadata)
    assert graph.SerializeToString(deterministic=True) == before


def test_fileconstant_restore_validates_all_candidates_before_mutation():
    graph, audit, immutable, metadata = sparse_weight_graph()
    nodes = {node.name: node for node in graph.op}
    for name in ("w", "quant_weight_nz", "quant"):
        clone = graph.op.add(); clone.CopyFrom(nodes[name]); clone.name = "other_" + name
        if name == "quant_weight_nz": clone.input[0] = "other_w:0"
        if name == "quant": clone.input[1] = "other_quant_weight_nz:0"
    audit["nodes"].append(dict(audit["nodes"][0], name="other_quant", format_conversion="other_quant_weight_nz"))
    immutable["other_w:0"] = immutable["w:0"]
    metadata["other_w:0"] = {"dtype": "DT_FLOAT16", "shape": [64, 256]}
    before = graph.SerializeToString(deterministic=True)
    with pytest.raises(ValueError, match="FX metadata/immutable tensor conflict"):
        restore_fileconstant_weight_descriptors(graph, audit, immutable, metadata)
    assert graph.SerializeToString(deterministic=True) == before


@pytest.mark.parametrize("damage", ["runtime_input", "weight_changed", "payload_changed", "rehashed_wrong_layout"])
@pytest.mark.parametrize("externalized", [False, True])
def test_cannot_use_runtime_or_corrupt_offline_weights(tmp_path, damage, externalized):
    g = fixture_graph()
    q = (torch.arange(64 * 256).reshape(64, 256) % 256 - 128).to(torch.int8)
    manifest = cached_weight(tmp_path, q)
    metadata = {"w:0": {"dtype": "DT_INT8", "shape": [64, 256]}}
    if damage != "runtime_input":
        weight = next(n for n in g.op if n.name == "w")
        if externalized: sparse_fileconstant(weight)
        else: const(weight, q)
    audit = normalize_weight_quant_layout(g, metadata)
    before_restore = g.SerializeToString(deterministic=True)
    restore_fileconstant_weight_descriptors(g, audit, {"w:0": q}, metadata)
    if damage == "runtime_input":
        assert g.SerializeToString(deterministic=True) == before_restore
    if damage == "weight_changed": q = q.clone(); q[0, 0] += 1
    if damage in {"payload_changed", "rehashed_wrong_layout"}:
        wrong = q.numpy().tobytes()
        (tmp_path / "weight.bin").write_bytes(wrong)
        if damage == "rehashed_wrong_layout":
            data = json.loads(manifest.read_text()); data["weights"][0]["sha256"] = hashlib.sha256(wrong).hexdigest()
            manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        prepack_weight_quant_constants(g, audit, {"w:0": q}, load_prepacked_weights(manifest))


@pytest.mark.parametrize("automatic", [False, True])
def test_actual_air_save_hook_repairs_sparse_fileconstant_before_strict_prepack(monkeypatch, tmp_path, automatic):
    graph = fixture_graph(shared=True)
    graph.op[0].output_desc[0].shape.dim[0] = 16
    x = torch.zeros(16, 256).half()
    q = ((torch.arange(64).reshape(-1, 1) * 17 + torch.arange(256) * 13) % 256 - 128).to(torch.int8)
    s = torch.ones(64, 2).half() / 32
    torchair = ModuleType("torchair")
    tensor_type = fake_ge_tensor(monkeypatch)
    original_set_meta = tensor_type.set_meta

    def original(inputs, graph, path, weight_name):
        for i, value in enumerate(inputs):
            if id(value) in weight_name:
                if value.dtype == torch.int8: sparse_fileconstant(graph.op[i])
                else: const(graph.op[i], value)
        return True, len(weight_name)

    module = SimpleNamespace(_convert_data_to_const=original)
    monkeypatch.setitem(sys.modules, "torchair", torchair)
    monkeypatch.setitem(sys.modules, "torchair._utils.export_utils", module)
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    options = ({"weight_prepack_output": tmp_path / "cache"} if automatic else
               {"weight_prepack_manifest": str(cached_weight(tmp_path / "cache", q))})
    with canonical_runtime_input_abi(torchair, public_inputs=[x], public_names=["x"],
                                     capture_weight_quant_shapes=True, **options) as audit:
        # Capture typed FX metadata BEFORE TorchAir externalizes the weight and
        # replaces its output descriptor with DT_UNDEFINED / shape=[].
        tensor_type(next(node for node in graph.op if node.name == "w")).set_meta(q)
        module._convert_data_to_const([x, q, s], graph, str(tmp_path), {id(q): "qweight", id(s): "scales"})
    layout = audit["weight_quant_layout"]
    restoration = layout["fileconstant_descriptors"]
    assert restoration["node_count"] == 1
    assert restoration["nodes"][0] == {"source": "w:0", "before_dtype": "DT_UNDEFINED", "before_shape": [],
                                        "dtype": "DT_INT8", "shape": [64, 256], "fx_metadata_checked": True}
    # A shared original producer stays a correctly described FileConstant,
    # while WeightQuant receives the actual hash-verified NZ Const bytes.
    weight = next(node for node in graph.op if node.name == "w")
    assert weight.type == "FileConstant" and list(weight.output_desc[0].shape.dim) == [64, 256]
    assert weight.output_desc[0].dtype == 2  # Test protobuf DT_INT8.
    assert weight.attr["file_id"].s == b"immutable-weight.bin" and weight.attr["offset"].i == 512
    assert [node.name for node in graph.op if node.type in {"Data", "RefData"}] == ["x"]
    assert layout["prepack"]["status"] == "PASS" and not layout["inserted_transdata"]
    packed = next(node for node in graph.op if node.name == "quant_weight_nz").attr["value"].t
    assert packed.data == pack_int8_nz(q).numpy().tobytes()
    assert layout["prepack"]["constants"][0]["logical_sha256"] == hashlib.sha256(q.numpy().tobytes()).hexdigest()
    report = json.loads((tmp_path / "weight-quant-layout.json").read_text())
    assert report["fileconstant_descriptors"] == restoration
    assert module._convert_data_to_const is original and tensor_type.set_meta is original_set_meta
