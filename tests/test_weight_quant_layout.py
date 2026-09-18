"""Protobuf graph/numerical checks; these do not execute CANN fusion."""
import copy
import struct
from types import ModuleType, SimpleNamespace
import sys

import pytest
import torch
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

from qwen35_dflash.ascend310p.weight_quant_layout import (
    GE_OP, POLICY, normalize_weight_quant_layout, validate_weight_quant_layout,
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
