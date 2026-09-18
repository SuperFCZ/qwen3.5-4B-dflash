"""Protobuf graph/numerical checks; these do not execute CANN fusion."""
import copy
import json
import struct
from types import ModuleType, SimpleNamespace
import sys

import pytest
import torch
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

from qwen35_dflash.ascend310p.weight_quant_layout import (
    GE_OP, POLICY, capture_weight_quant_metadata, normalize_weight_quant_layout, validate_weight_quant_layout,
)
from qwen35_dflash.ascend310p.runtime_input_export import canonical_runtime_input_abi


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
    msg("Desc", [("name", 9, 1, None), ("dtype", 14, 1, "DType"), ("shape", 11, 1, "Shape")])
    msg("Tensor", [("desc", 11, 1, "Desc"), ("data", 12, 1, None)])
    msg("Ints", [("i", 3, 3, None)])
    msg("Attr", [("i", 3, 1, None), ("b", 8, 1, None), ("t", 11, 1, "Tensor"),
                 ("list", 11, 1, "Ints")])
    op = msg("Op", [("name", 9, 1, None), ("type", 9, 1, None), ("input", 9, 3, None),
                    ("input_desc", 11, 3, "Desc"), ("output_desc", 11, 3, "Desc"),
                    ("attr", 11, 3, "Op.Entry")])
    entry = op.nested_type.add(name="Entry"); entry.options.map_entry = True
    entry.field.add(name="key", number=1, type=9, label=1)
    entry.field.add(name="value", number=2, type=11, type_name=".Attr", label=1)
    msg("Graph", [("op", 11, 3, "Op")])
    pool = descriptor_pool.DescriptorPool(); pool.Add(proto)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName("Graph"))


def fixture_graph(*, trans_type="Transpose", perm_dtype="DT_INT64", shared=False):
    graph = graph_type()()
    def node(name, kind, shape, dtype, edges=()):
        op = graph.op.add(name=name, type=kind, input=edges)
        desc = op.output_desc.add(name="y", dtype=dtype); desc.shape.dim.extend(shape)
        return op
    x = node("x", "Data", [-1, 256], "DT_FLOAT16"); x.attr["index"].i = 0
    w = node("w", "Data", [64, 256], "DT_INT8"); w.attr["index"].i = 1
    s = node("s", "Data", [64, 2], "DT_FLOAT16"); s.attr["index"].i = 2
    perm = node("perm", "Const", [2], perm_dtype)
    perm.attr["value"].t.desc.CopyFrom(perm.output_desc[0])
    perm.attr["value"].t.data = struct.pack("<2q" if perm_dtype == "DT_INT64" else "<2i", 1, 0)
    for source in (w, s):
        edges = [source.name + ":0"] + (["perm:0"] if trans_type == "Transpose" else [])
        t = node(source.name + "t", trans_type, list(source.output_desc[0].shape.dim)[::-1],
                 source.output_desc[0].dtype, edges)
        t.attr["perm"].list.i.extend([1, 0])
    op = node("quant", GE_OP, [-1, 64], "DT_FLOAT16",
              ["x:0", "wt:0", "st:0", "", "", "", ""])
    for name, source in zip(("x", "weight", "antiquant_scale"), (x, graph.op[4], graph.op[5])):
        op.input_desc.add().CopyFrom(source.output_desc[0]); op.input_desc[-1].name = name
    op.attr["antiquant_group_size"].i = 128
    op.attr["inner_precise"].i = 0
    op.attr["transpose_x"].b = op.attr["transpose_weight"].b = False
    node("out", "NetOutput", [-1, 64], "DT_FLOAT16", ["quant:0"] + (["wt:0"] if shared else []))
    return graph


@pytest.mark.parametrize("trans_type", ["Transpose", "TransposeD"])
@pytest.mark.parametrize("perm_dtype", ["DT_INT64", "DT_INT32"])
def test_fold_both_axes_keep_optional_slots_and_preserve_group_math(trans_type, perm_dtype):
    g = fixture_graph(trans_type=trans_type, perm_dtype=perm_dtype)
    before = {op.name: op.SerializeToString() for op in g.op if op.type == "Data"}
    report = normalize_weight_quant_layout(g)
    op = next(op for op in g.op if op.type == GE_OP)
    assert report["policy"] == POLICY and report["node_count"] == 1
    assert report["removed_transposes"] == ["st", "wt"]
    assert list(op.input) == ["x:0", "w:0", "s:0", "", "", "", ""]
    assert op.attr["transpose_weight"].b and not op.attr["transpose_x"].b
    assert list(op.input_desc[1].shape.dim) == [64, 256]
    assert list(op.input_desc[2].shape.dim) == [64, 2]
    assert [d.name for d in op.input_desc] == ["x", "weight", "antiquant_scale"]
    assert before == {op.name: op.SerializeToString() for op in g.op if op.type == "Data"}
    # Unequal scales for every group/channel catch a weight-only transpose fold.
    x = ((torch.arange(16 * 256).reshape(16, 256) % 7) - 3).float() / 16
    q = ((torch.arange(64 * 256).reshape(64, 256) % 11) - 5).float()
    scale = (1 + torch.arange(64 * 2).reshape(64, 2) % 4).float() / 32
    original = x @ (q.t() * scale.t().repeat_interleave(128, 0))
    folded = x @ (q * scale.repeat_interleave(128, 1)).t()
    assert torch.equal(original, folded)
    snapshot = g.SerializeToString()
    repeated = normalize_weight_quant_layout(g)
    assert repeated["nodes"][0]["already_folded"]
    assert repeated["removed_transposes"] == [] and g.SerializeToString() == snapshot


def test_shared_transpose_retained_for_other_consumers():
    g = fixture_graph(shared=True)
    report = normalize_weight_quant_layout(g)
    assert report["removed_transposes"] == ["st"]
    assert next(op for op in g.op if op.name == "out").input[1] == "wt:0"


@pytest.mark.parametrize("damage", ["perm", "runtime_perm", "dtype", "shape", "scale", "optional", "precision", "control"])
def test_unrecognized_patterns_fail_before_any_rewiring(damage):
    g = fixture_graph(); nodes = {op.name: op for op in g.op}
    if damage == "perm": nodes["perm"].attr["value"].t.data = struct.pack("<2q", 0, 1)
    elif damage == "runtime_perm": nodes["perm"].type = "Data"
    elif damage == "dtype": nodes["w"].output_desc[0].dtype = 1
    elif damage == "shape": nodes["w"].output_desc[0].shape.dim[1] = 128
    elif damage == "scale": nodes["st"].input[0] = "w:0"
    elif damage == "optional": nodes["quant"].input[3] = "st:0"
    elif damage == "precision": nodes["quant"].attr["inner_precise"].i = 1
    elif damage == "control": nodes["wt"].input.append("x:-1")
    original_edges = [list(op.input) for op in g.op]
    with pytest.raises(ValueError): normalize_weight_quant_layout(g)
    assert original_edges == [list(op.input) for op in g.op]
    assert not nodes["quant"].attr["transpose_weight"].b


def test_air_save_boundary_records_normalized_graph(monkeypatch, tmp_path):
    g = fixture_graph(); g.op[0].output_desc[0].shape.dim[0] = 16
    public = [torch.zeros(16, 256).half(), torch.zeros(64, 256, dtype=torch.int8), torch.zeros(64, 2).half()]
    torchair = ModuleType("torchair")
    original = lambda *args: (False, 0)
    module = SimpleNamespace(_convert_data_to_const=original)
    monkeypatch.setitem(sys.modules, "torchair", torchair)
    monkeypatch.setitem(sys.modules, "torchair._utils.export_utils", module)
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    with canonical_runtime_input_abi(torchair, public_inputs=public, public_names=["x", "w", "s"]) as audit:
        module._convert_data_to_const(public, g, str(tmp_path), {})
    assert module._convert_data_to_const is original
    assert audit["weight_quant_layout"]["node_count"] == 1
    assert (tmp_path / "weight-quant-layout.json").is_file()
    entry = {"custom_op_audit": [{"ge_op_type": GE_OP, "ge_node_occurrences": 1}], "runtime_input_abi": audit}
    validate_weight_quant_layout(entry)
    broken = copy.deepcopy(entry); del broken["runtime_input_abi"]["weight_quant_layout"]
    with pytest.raises(ValueError, match="re-export AIR"): validate_weight_quant_layout(broken)
    broken = copy.deepcopy(entry); broken["runtime_input_abi"]["weight_quant_layout"]["node_count"] = 2
    with pytest.raises(ValueError): validate_weight_quant_layout(broken)


def test_unrelated_graph_unchanged():
    g = graph_type()(); g.op.add(name="matmul", type="MatMul")
    before = g.SerializeToString()
    assert normalize_weight_quant_layout(g)["node_count"] == 0
    assert g.SerializeToString() == before


def fake_ge_tensor(monkeypatch):
    # Mirrors the real Tensor.set_meta contract: it records dtype and symsize,
    # but does NOT populate output_desc.shape for intermediate ops.
    class Tensor:
        def __init__(self, node):
            self.tensor = node.name + ":0"
            self.desc = node.output_desc[0]

        def set_meta(self, meta_output, ge_outputs=None):
            self.symsize = list(meta_output.size())
            enum = self.desc.DESCRIPTOR.fields_by_name["dtype"].enum_type
            name = "DT_INT8" if meta_output.dtype == torch.int8 else "DT_FLOAT16"
            self.desc.dtype = enum.values_by_name[name].number
            return "original-result"

    module = ModuleType("torchair.ge._ge_graph"); module.Tensor = Tensor
    monkeypatch.setitem(sys.modules, "torchair.ge._ge_graph", module)
    return Tensor


@pytest.mark.parametrize("empty_shape", [[], [-2]])
def test_sparse_intermediate_descriptors_use_typed_metadata(monkeypatch, empty_shape):
    g = fixture_graph()
    tensor_type = fake_ge_tensor(monkeypatch)
    original = tensor_type.set_meta
    with capture_weight_quant_metadata(True) as metadata:
        for node in g.op:
            if node.name not in ("x", "w", "s", "wt", "st"):
                continue
            shape = [16 if d == -1 else d for d in node.output_desc[0].shape.dim]
            dtype = torch.int8 if node.name in ("w", "wt") else torch.float16
            node.output_desc[0].shape.dim[:] = empty_shape
            assert tensor_type(node).set_meta(torch.empty(shape, dtype=dtype), None) == "original-result"
        # A symbolic M axis is retained, not replaced by the sample's M=16.
        metadata["x:0"]["shape"][0] = -1
        untouched = {node.name: node.SerializeToString() for node in g.op if node.type == "Data"}
        report = normalize_weight_quant_layout(g, metadata)
        assert report["metadata_source"] == "torchair.Tensor.set_meta"
        assert report["nodes"][0]["weight_shape"] == [64, 256]
        assert report["nodes"][0]["scale_shape"] == [64, 2]
        assert untouched == {node.name: node.SerializeToString() for node in g.op if node.type == "Data"}
        snapshot = g.SerializeToString()
        assert normalize_weight_quant_layout(g, metadata)["nodes"][0]["already_folded"]
        assert snapshot == g.SerializeToString()
    assert tensor_type.set_meta is original


def test_capture_symbolic_dimension_never_concretizes_or_leaks_patch(monkeypatch):
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv
    tensor_type = fake_ge_tensor(monkeypatch)
    original = tensor_type.set_meta
    g = fixture_graph(); env = ShapeEnv(); m = env.create_unbacked_symint()
    with FakeTensorMode(shape_env=env):
        x = torch.empty(m, 256, dtype=torch.float16)
    with pytest.raises(RuntimeError, match="abort export"):
        with capture_weight_quant_metadata(True) as metadata:
            tensor_type(g.op[0]).set_meta(x)
            assert metadata["x:0"] == {"shape": [-1, 256], "dtype": "DT_FLOAT16"}
            raise RuntimeError("abort export")
    assert tensor_type.set_meta is original


@pytest.mark.parametrize("damage", ["metadata_shape", "metadata_dtype", "transpose_shape", "no_metadata"])
def test_sparse_metadata_does_not_mask_real_conflicts(monkeypatch, damage):
    g = fixture_graph(); nodes = {op.name: op for op in g.op}
    metadata = {"w:0": {"shape": [64, 256], "dtype": "DT_INT8"}}
    if damage == "metadata_shape": metadata["w:0"]["shape"] = [32, 256]
    elif damage == "metadata_dtype": metadata["w:0"]["dtype"] = "DT_FLOAT16"
    elif damage == "transpose_shape": nodes["wt"].output_desc[0].shape.dim[:] = [64, 256]
    else:
        metadata = {}; nodes["w"].output_desc[0].shape.dim[:] = []
    before = g.SerializeToString()
    with pytest.raises(ValueError): normalize_weight_quant_layout(g, metadata)
    assert g.SerializeToString() == before


@pytest.mark.parametrize("conflict", [False, True])
def test_export_boundary_captures_sparse_fx_shapes_and_retains_failure_report(monkeypatch, tmp_path, conflict):
    g = fixture_graph(); g.op[0].output_desc[0].shape.dim[0] = 16
    tensor_type = fake_ge_tensor(monkeypatch); original_set_meta = tensor_type.set_meta
    public = [torch.zeros(16, 256).half(), torch.zeros(64, 256, dtype=torch.int8), torch.zeros(64, 2).half()]
    torchair = ModuleType("torchair")
    original = lambda *args: (False, 0)
    module = SimpleNamespace(_convert_data_to_const=original)
    monkeypatch.setitem(sys.modules, "torchair", torchair)
    monkeypatch.setitem(sys.modules, "torchair._utils.export_utils", module)
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    def export():
        with canonical_runtime_input_abi(torchair, public_inputs=public, public_names=["x", "w", "s"],
                                         capture_weight_quant_shapes=True) as audit:
            for node in g.op:
                if node.name in ("wt", "st"):
                    shape = list(node.output_desc[0].shape.dim)
                    if conflict and node.name == "st": shape = [2, 32]
                    dtype = torch.int8 if node.name == "wt" else torch.float16
                    node.output_desc[0].shape.dim[:] = []
                    tensor_type(node).set_meta(torch.empty(shape, dtype=dtype))
            module._convert_data_to_const(public, g, str(tmp_path), {})
        return audit
    if conflict:
        with pytest.raises(ValueError, match="report=.*weight-quant-layout.json"): export()
    else:
        assert export()["weight_quant_layout"]["metadata_source"] == "torchair.Tensor.set_meta"
    report = json.loads((tmp_path / "weight-quant-layout.json").read_text())
    assert report["status"] == ("FAIL" if conflict else "PASS")
    if conflict:
        assert report["nodes"][0]["inputs"][2]["fx_metadata"]["shape"] == [2, 32]
        assert not next(op for op in g.op if op.type == GE_OP).attr["transpose_weight"].b
    assert module._convert_data_to_const is original and tensor_type.set_meta is original_set_meta
