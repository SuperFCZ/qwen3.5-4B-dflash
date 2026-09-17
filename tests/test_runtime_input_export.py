from __future__ import annotations

import sys
import copy
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from rms_norm_test_support import adn_rms_norm_cpu  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/python"))

from qwen35_dflash.ascend310p.runtime_input_export import (
    _gdr_output_dtype_audit, _normalize_public_nodes, _order_weight_conversion_inputs,
    _public_bindings, canonical_runtime_input_abi,
    _verify_discard_output_audit,
    validated_runtime_input_abi as _validated_runtime_input_abi,
)


def _gdr_node(core_dtype="DT_FLOAT16", state_dtype="DT_FLOAT"):
    # Use protobuf descriptors, whose dtype numbers differ from ge.DataType.
    # No NPU/TorchAir import or kernel execution is represented by this fixture.
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
    proto = descriptor_pb2.FileDescriptorProto(name="gdr_audit_test.proto")
    dtype = proto.enum_type.add(name="DataType")
    for number, name in enumerate(("DT_UNDEFINED", "DT_FLOAT", "DT_FLOAT16")):
        dtype.value.add(name=name, number=number)
    shape = proto.message_type.add(name="Shape")
    shape.field.add(name="dim", number=1, type=3, label=3)
    desc = proto.message_type.add(name="Desc")
    desc.field.add(name="name", number=1, type=9, label=1)
    desc.field.add(name="dtype", number=2, type=14, type_name=".DataType", label=1)
    desc.field.add(name="shape", number=3, type=11, type_name=".Shape", label=1)
    pool = descriptor_pool.DescriptorPool()
    pool.Add(proto)
    tensor_desc = message_factory.GetMessageClass(pool.FindMessageTypeByName("Desc"))
    return SimpleNamespace(
        name="ChunkGatedDeltaRule_23", type="ChunkGatedDeltaRule",
        input_desc=[tensor_desc(name="query", dtype="DT_FLOAT16")],
        output_desc=[tensor_desc(name="core_attn", dtype=core_dtype),
                     tensor_desc(name="last_recurrent_state", dtype=state_dtype)],
    )


@pytest.mark.parametrize("core,state,passed", [
    ("DT_FLOAT16", "DT_FLOAT", True),
    ("DT_FLOAT", "DT_FLOAT", False),
    ("DT_UNDEFINED", "DT_FLOAT", False),
    ("DT_FLOAT16", "DT_FLOAT16", False),
])
def test_gdr_audit_reads_both_physical_output_dtypes(core, state, passed):
    node = _gdr_node(core, state)
    # Verify's commit pass may not consume core_attn, but its descriptor still
    # participates in ATC kernel selection and must pass the same check.
    graph = _Graph([node, _Op("consume_state_only", "Cast", inputs=(node.name + ":1",))])
    audit = _gdr_output_dtype_audit(graph)
    assert audit["node_count"] == 1
    assert audit["status"] == ("PASS" if passed else "FAIL")
    assert [o["dtype"] for o in audit["nodes"][0]["outputs"]] == [core, state]


@pytest.mark.parametrize("damage", ["missing_state", "swapped_names"])
def test_gdr_audit_rejects_missing_or_reordered_outputs(damage):
    node = _gdr_node()
    if damage == "missing_state":
        node.output_desc.pop()
    else:
        node.output_desc.reverse()
    assert _gdr_output_dtype_audit(_Graph([node]))["status"] == "FAIL"


def _discard_graph(damage=None):
    first, second = _gdr_node(), _gdr_node()
    first.name, second.name = "verify_gdr", "commit_gdr"
    for node in (first, second):
        node.input = []
        node.attr = {"output_final_state": SimpleNamespace(b=True)}
    core = _Op("consume_core", "Mul", inputs=("verify_gdr:0",))
    committed = _Op("committed", "Cast", inputs=("commit_gdr:1",))
    raw = _Op("raw_state", "Identity", inputs=("verify_gdr:1",))
    if damage == "cast":
        raw.type = "Cast"
    elif damage == "commit":
        raw.input[0] = "commit_gdr:1"
    elif damage == "core":
        raw.input[0] = "verify_gdr:0"
    elif damage == "outstate_false":
        first.attr["output_final_state"].b = False
    elif damage == "unused_core":
        core.input.clear()
    output = _Op("NetOutput", "NetOutput", inputs=("committed:0", "raw_state:0"))
    if damage == "missing":
        output.input.pop()
    return _Graph([first, core, second, committed, raw, output])


@pytest.mark.parametrize("damage", [None, "cast", "commit", "core", "outstate_false", "unused_core", "missing"])
def test_discard_audit_requires_raw_first_pass_state_in_netoutput(damage):
    report = _verify_discard_output_audit(
        _discard_graph(damage), ["t0_recurrent", "verify_discard_t0_recurrent"],
        ["verify_discard_t0_recurrent"],
    )
    assert report["status"] == ("PASS" if damage is None else "FAIL")
    if damage is None:
        assert report["outputs"][0]["source"] == "verify_gdr:1"
        assert report["outputs"][0]["output_index"] == 1


@pytest.mark.parametrize("damage", [None, "cast", "outstate_false", "missing"])
def test_serialization_checks_discard_liveness_and_retains_report(monkeypatch, tmp_path, damage):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    torchair = ModuleType("torchair")
    public = torch.zeros(1)
    graph = _discard_graph(damage)
    graph.op.insert(0, _Op("input", "Data", index=0))
    original = lambda *args: (False, 0)
    module = SimpleNamespace(_convert_data_to_const=original)
    monkeypatch.setitem(sys.modules, "torchair", torchair)
    monkeypatch.setitem(sys.modules, "torchair._utils.export_utils", module)
    try:
        with canonical_runtime_input_abi(
            torchair, public_inputs=[public], public_names=["x"],
            public_output_names=["t0_recurrent", "verify_discard_t0_recurrent"],
            verify_discard_output_names=["verify_discard_t0_recurrent"],
        ) as audit:
            module._convert_data_to_const([public], graph, str(tmp_path), {})
    except RuntimeError as error:
        assert damage and "raw first-pass GDR states" in str(error)
    else:
        assert damage is None
        assert audit["verify_discard_outputs"]["status"] == "PASS"
    report = json.loads((tmp_path / "verify-discard-outputs.json").read_text())
    assert report["status"] == ("PASS" if damage is None else "FAIL")
    assert module._convert_data_to_const is original


@pytest.mark.parametrize("damage", [False, True])
def test_serialization_gdr_audit_retains_evidence_and_restores_hook(monkeypatch, tmp_path, damage):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    torchair = ModuleType("torchair")
    public = torch.zeros(1)
    gdr = _gdr_node()
    graph = _Graph([_Op("input", "Data", index=0), gdr])

    def original(*args):
        if damage:
            gdr.output_desc[0].dtype = 1  # Proto DT_FLOAT, not GE DT_FLOAT16.
        return False, 0

    module = SimpleNamespace(_convert_data_to_const=original)
    monkeypatch.setitem(sys.modules, "torchair", torchair)
    monkeypatch.setitem(sys.modules, "torchair._utils.export_utils", module)
    try:
        with canonical_runtime_input_abi(
            torchair, public_inputs=[public], public_names=["x"],
        ) as audit:
            module._convert_data_to_const([public], graph, str(tmp_path), {})
    except RuntimeError as error:
        assert damage and "GE output dtype mismatch before AIR save" in str(error)
    else:
        assert not damage
        assert audit["gdr_output_dtypes"]["status"] == "PASS"
    report = json.loads((tmp_path / "gdr-output-dtypes.json").read_text())
    assert report["scope"] == "torchair-before-ge-save"
    assert report["status"] == ("FAIL" if damage else "PASS")
    assert module._convert_data_to_const is original


class _Attr:
    def __init__(self, value: int) -> None:
        self.i = value


class _Op:
    def __init__(
        self,
        name: str,
        op_type: str,
        *,
        index: int | None = None,
        inputs: tuple[str, ...] = (),
    ) -> None:
        self.name = name
        self.type = op_type
        self.attr = {} if index is None else {"index": _Attr(index)}
        self.input = list(inputs)

    def Clear(self) -> None:
        self.name = ""
        self.type = ""
        self.attr = {}
        self.input = []
        self.output_desc = []

    def MergeFrom(self, other: "_Op") -> None:
        self.name = other.name
        self.type = other.type
        self.attr = dict(other.attr)
        self.input = list(other.input)
        self.output_desc = copy.deepcopy(getattr(other, "output_desc", []))


class _Graph:
    def __init__(self, ops: list[_Op]) -> None:
        self.op = ops

    @staticmethod
    def ByteSize() -> int:
        return 1


def test_public_input_order_comes_from_identity_not_placeholder_order_or_shape():
    first = torch.zeros(1, dtype=torch.int32)
    second = torch.zeros(1, dtype=torch.int32)
    graph = _Graph([
        _Op("arg9", "Data", index=0), _Op("helper", "Gather"),
        _Op("arg3", "Data", index=1),
    ])
    bindings = _public_bindings([second, first], graph, {}, [first, second], ["a", "b"])
    assert [(i, n.name) for i, n in bindings] == [(0, "arg3"), (1, "arg9")]
    _normalize_public_nodes(graph, bindings)
    assert [op.name for op in graph.op] == ["arg3", "helper", "arg9"]
    assert [op.attr["index"].i for op in graph.op if op.type == "Data"] == [0, 1]


def test_weight_slots_and_shape_helpers_are_not_public_inputs():
    weight = torch.ones(4)
    public = torch.zeros(2)
    graph = _Graph([
        _Op("helper", "Gather"), _Op("weight", "Data", index=0),
        _Op("public", "Data", index=1),
    ])
    bindings = _public_bindings([weight, public], graph, {id(weight): "w"}, [public], ["x"])
    graph.op[1].type = "FileConstant"
    _normalize_public_nodes(graph, bindings)
    assert graph.op[0].type == "Gather"
    assert graph.op[1].type == "FileConstant"
    assert graph.op[2].attr["index"].i == 0


def _positional_weight_conversion(inputs, graph, file_path, weight_name):
    """Match TorchAir's graph.op[i] lookup, not an index-aware substitute."""
    del file_path
    for index, value in enumerate(inputs):
        file_id = weight_name.get(id(value))
        if file_id is None:
            continue
        node = graph.op[index]
        constant = _Op(node.name, "FileConstant")
        constant.attr["file_id"] = file_id
        node.Clear()
        node.MergeFrom(constant)
    # TorchAir resets Data then RefData indexes after converting the weights.
    index = 0
    for kind in ("Data", "RefData"):
        for node in graph.op:
            if node.type == kind:
                node.attr["index"].i = index
                index += 1
    return True, len(weight_name)


@pytest.mark.parametrize("cache_kind", ["Data", "RefData"])
@pytest.mark.parametrize("reverse_data_order", [False, True])
def test_weight_conversion_preserves_interleaved_shape_nodes(
    monkeypatch, cache_kind, reverse_data_order,
):
    torchair = ModuleType("torchair")
    feature, cache, weight = torch.zeros(1, 64, 4), torch.zeros(4), torch.ones(4, 4)
    nodes = [
        _Op("feature", "Data", index=0),
        _Op("projection", "Data", index=1),
        _Op("cache", cache_kind, index=2),
    ]
    if reverse_data_order:
        nodes.reverse()
    shape = _Op("feature_shape", "Shape", inputs=("feature:0",))
    output = _Op("result", "NetOutput", inputs=("feature_shape:0", "projection:0", "cache:0"))
    graph = _Graph([nodes[0], shape, *nodes[1:], output])
    module = SimpleNamespace(_convert_data_to_const=_positional_weight_conversion)
    monkeypatch.setitem(sys.modules, "torchair", torchair)
    monkeypatch.setitem(sys.modules, "torchair._utils.export_utils", module)
    with canonical_runtime_input_abi(
        torchair, public_inputs=[cache, feature], public_names=["cache", "features"],
    ) as audit:
        result = module._convert_data_to_const(
            [feature, weight, cache], graph, "unused", {id(weight): "projection.weight"},
        )
    assert result == (True, 1)
    by_name = {node.name: node for node in graph.op}
    assert len(by_name) == 5
    assert by_name["projection"].type == "FileConstant"
    assert by_name["projection"].attr["file_id"] == "projection.weight"
    assert by_name["feature_shape"].type == "Shape"
    assert by_name["feature_shape"].input == ["feature:0"]
    assert by_name["result"].input == ["feature_shape:0", "projection:0", "cache:0"]
    public = [node for node in graph.op if node.type in {"Data", "RefData"}]
    assert [(node.name, node.attr["index"].i) for node in public] == [("cache", 0), ("feature", 1)]
    assert audit["status"] == "PASS"
    assert audit["weight_conversion"] == {
        "policy": "runtime-input-index-order-v1", "reordered": True,
        "input_count": 3, "weight_count": 1,
    }
    assert module._convert_data_to_const is _positional_weight_conversion


def test_weight_input_sort_on_protobuf_preserves_edges_and_public_bindings():
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
    proto = descriptor_pb2.FileDescriptorProto(name="input_order_test.proto")
    attr = proto.message_type.add(name="Attr")
    attr.field.add(name="i", number=1, type=3, label=1)
    op = proto.message_type.add(name="Op")
    op.field.add(name="name", number=1, type=9, label=1)
    op.field.add(name="type", number=2, type=9, label=1)
    op.field.add(name="input", number=3, type=9, label=3)
    entry = op.nested_type.add(name="AttrEntry")
    entry.options.map_entry = True
    entry.field.add(name="key", number=1, type=9, label=1)
    entry.field.add(name="value", number=2, type=11, label=1, type_name=".Attr")
    op.field.add(name="attr", number=4, type=11, label=3, type_name=".Op.AttrEntry")
    graph_desc = proto.message_type.add(name="Graph")
    graph_desc.field.add(name="op", number=1, type=11, label=3, type_name=".Op")
    pool = descriptor_pool.DescriptorPool()
    pool.Add(proto)
    graph = message_factory.GetMessageClass(pool.FindMessageTypeByName("Graph"))()
    graph.op.add(name="x", type="Data").attr["index"].i = 1
    graph.op.add(name="shape", type="Shape", input=["x:0"])
    graph.op.add(name="w", type="Data").attr["index"].i = 0
    graph.op.add(name="s", type="RefData").attr["index"].i = 2
    graph.op.add(name="out", type="NetOutput", input=["shape:0", "s:0", "w:0"])
    helper_bytes = [node.SerializeToString() for node in graph.op if node.type in {"Shape", "NetOutput"}]
    weight, feature, state = torch.ones(4), torch.zeros(4), torch.zeros(4)
    inputs = [weight, feature, state]
    assert _order_weight_conversion_inputs(inputs, graph)
    assert [node.name for node in graph.op] == ["w", "x", "s", "shape", "out"]
    bindings = _public_bindings(inputs, graph, {id(weight): "w"}, [state, feature], ["s", "x"])
    graph.op[0].Clear()
    graph.op[0].name, graph.op[0].type = "w", "Const"
    _normalize_public_nodes(graph, bindings)
    assert [node.name for node in graph.op] == ["w", "s", "x", "shape", "out"]
    assert [graph.op[i].attr["index"].i for i in (1, 2)] == [0, 1]
    assert helper_bytes == [node.SerializeToString() for node in graph.op if node.type in {"Shape", "NetOutput"}]


@pytest.mark.parametrize("indexes", [[0], [0, 2], [0, -1]])
def test_weight_conversion_rejects_incomplete_or_invalid_runtime_index_map(indexes):
    graph = _Graph([_Op(f"input{i}", "Data", index=i) for i in indexes])
    with pytest.raises(RuntimeError, match="indexes do not cover weight-conversion inputs"):
        _order_weight_conversion_inputs([torch.zeros(1), torch.ones(1)], graph)


def test_remaining_scalar_is_rejected_without_guessing_its_value():
    public = torch.zeros(2)
    lifted = torch.tensor(0.125, dtype=torch.float64)
    graph = _Graph([_Op("arg1", "Data", index=0), _Op("arg7", "Data", index=1)])
    with pytest.raises(RuntimeError, match="unbound AIR runtime input.*arg7.*float64"):
        _public_bindings([public, lifted], graph, {}, [public], ["x"])


def test_missing_and_aliased_public_inputs_are_rejected():
    public = torch.zeros(2)
    with pytest.raises(RuntimeError, match="disappeared"):
        _public_bindings([], _Graph([]), {}, [public], ["x"])
    with pytest.raises(RuntimeError, match="distinct"):
        _public_bindings([], _Graph([]), {}, [public, public], ["x", "y"])


@pytest.mark.parametrize("index", [-1, 1])
def test_invalid_data_index_cannot_alias_a_public_tensor(index):
    public = torch.zeros(2)
    graph = _Graph([_Op("public", "Data", index=index)])
    with pytest.raises(RuntimeError, match="outside runtime inputs"):
        _public_bindings([public], graph, {}, [public], ["x"])


@pytest.mark.parametrize("fail", [False, True])
def test_normalization_context_restores_converter_and_float_policy(monkeypatch, fail):
    torchair = ModuleType("torchair")
    public = [torch.zeros(1), torch.ones(2)]
    graph = _Graph([_Op("second", "Data", index=0), _Op("first", "Data", index=1)])

    def original(inputs, export_graph, file_path, weight_name):
        if fail:
            raise RuntimeError("conversion failure")
        return False, 0

    module = SimpleNamespace(_convert_data_to_const=original)
    monkeypatch.setitem(sys.modules, "torchair", torchair)
    monkeypatch.setitem(sys.modules, "torchair._utils.export_utils", module)
    with torch._dynamo.config.patch(specialize_float=False):
        try:
            with canonical_runtime_input_abi(
                torchair, public_inputs=public, public_names=["a", "b"],
            ) as audit:
                assert torch._dynamo.config.specialize_float is True
                module._convert_data_to_const(public[::-1], graph, "unused", {})
        except RuntimeError as error:
            assert fail and str(error) == "conversion failure"
        assert torch._dynamo.config.specialize_float is False
    assert module._convert_data_to_const is original
    if not fail:
        assert audit["status"] == "PASS"
        assert [r["logical_name"] for r in audit["bindings"]] == ["a", "b"]


def test_dynamo_specializes_model_float_without_freezing_dynamic_tensor_rows():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = 0.125

        def forward(self, x):
            return x * self.scale + x.shape[0]

    examples = []

    def backend(graph, args):
        examples.append(args)
        return graph.forward

    model = Model()
    torch._dynamo.reset()
    try:
        with torch._dynamo.config.patch(specialize_float=True):
            compiled = torch.compile(model, backend=backend, dynamic=True, fullgraph=True)
            for rows in (3, 7, 16, 64):
                x = torch.arange(rows * 4, dtype=torch.float32).reshape(rows, 4)
                torch.testing.assert_close(compiled(x), model(x), rtol=0, atol=0)
        assert examples
        for args in examples:
            assert not any(isinstance(arg, (float, torch.SymFloat)) for arg in args)
            assert not any(isinstance(arg, torch.Tensor) and arg.ndim == 0
                           and arg.dtype == torch.float64 for arg in args)
        assert any(isinstance(arg, torch.SymInt) for args in examples for arg in args)
    finally:
        torch._dynamo.reset()


@pytest.mark.parametrize("serialized", [[1, 64, 4], [1, -1, 4], [1, 16, 4]])
@pytest.mark.parametrize("change_during_conversion", [False, True])
def test_static_export_audits_real_data_descriptors_and_restores_patch(
    monkeypatch, serialized, change_during_conversion,
):
    torchair = ModuleType("torchair")
    public = torch.zeros(1, 64, 4)
    node = _Op("feature", "Data", index=0)
    node.output_desc = [SimpleNamespace(shape=SimpleNamespace(
        dim=[1, 64, 4] if change_during_conversion else serialized,
    ))]
    graph = _Graph([node])

    def original(*args):
        if change_during_conversion:
            node.output_desc[0].shape.dim = serialized
        return False, 0

    module = SimpleNamespace(_convert_data_to_const=original)
    monkeypatch.setitem(sys.modules, "torchair", torchair)
    monkeypatch.setitem(sys.modules, "torchair._utils.export_utils", module)
    if serialized == [1, 64, 4]:
        with canonical_runtime_input_abi(
            torchair, public_inputs=[public], public_names=["features"], require_static_shapes=True,
        ) as audit:
            module._convert_data_to_const([public], graph, "unused", {})
        assert audit["bindings"][0]["serialized_shape"] == serialized
    else:
        with pytest.raises(RuntimeError, match="static AIR input"):
            with canonical_runtime_input_abi(
                torchair, public_inputs=[public], public_names=["features"], require_static_shapes=True,
            ):
                module._convert_data_to_const([public], graph, "unused", {})
    assert module._convert_data_to_const is original


@pytest.mark.parametrize("damage", [None, "missing", "scalar", "order", "duplicate", "calls"])
def test_compiler_checks_canonical_input_audit(damage):
    graph = {"input_names": ["x", "state"], "runtime_input_abi": {
        "policy": "public-tensor-storage-identity-v1", "status": "PASS", "calls": 1,
        "python_float_policy": "dynamo-specialize-float",
        "logical_input_names": ["x", "state"],
        "bindings": [
            {"index": 0, "logical_name": "x", "data_node_name": "arg4"},
            {"index": 1, "logical_name": "state", "data_node_name": "arg1"},
        ],
    }}
    graph = copy.deepcopy(graph)
    if damage == "missing":
        del graph["runtime_input_abi"]
    elif damage == "scalar":
        graph["runtime_input_abi"]["bindings"].append({"name": "arg7"})
    elif damage == "order":
        graph["runtime_input_abi"]["bindings"].reverse()
    elif damage == "duplicate":
        graph["runtime_input_abi"]["bindings"][1]["data_node_name"] = "arg4"
    elif damage == "calls":
        graph["runtime_input_abi"]["calls"] = 0
    if damage is None:
        assert _validated_runtime_input_abi(graph, required=True)["status"] == "PASS"
    else:
        with pytest.raises(ValueError, match="runtime_input_abi"):
            _validated_runtime_input_abi(graph, required=True)


@pytest.mark.parametrize("field", ["dtype", "example_shape", "serialized_shape"])
def test_static_audit_must_match_chunk_tensor_contract(field):
    tensor = {"name": "start_position", "dtype": "int64", "shape": [1]}
    binding = {"index": 0, "logical_name": tensor["name"], "data_node_name": "arg7",
               "dtype": tensor["dtype"], "example_shape": [1], "serialized_shape": [1]}
    graph = {"input_names": [tensor["name"]], "metadata": {"tensor_abi": {"inputs": [tensor]}},
             "runtime_input_abi": {
                 "status": "PASS", "policy": "public-tensor-storage-identity-v1", "calls": 1,
                 "python_float_policy": "dynamo-specialize-float",
                 "logical_input_names": [tensor["name"]], "bindings": [binding],
             }}
    assert _validated_runtime_input_abi(graph, required=True)["status"] == "PASS"
    binding[field] = "int16" if field == "dtype" else [-1]
    with pytest.raises(ValueError, match="tensor descriptor"):
        _validated_runtime_input_abi(graph, required=True)


@pytest.mark.parametrize("verify_gdr", ["chunk", "mtp"])
def test_incremental_exports_normalize_actual_dynamo_input_order(
    tmp_path, monkeypatch, adn_rms_norm_cpu, verify_gdr,
):
    """Real Dynamo capture + host serializer fixture; no AIR/device claim."""
    import json
    from test_incremental_air_om import specs
    from qwen35_dflash.ascend310p.exporter import export_air_bundle
    from qwen35_dflash.ascend310p.runtime_input_export import _tensor_identity

    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    torchair = ModuleType("torchair")
    values = specs(verify_gdr=verify_gdr)
    by_name = {spec.name: spec for spec in values}
    raw_orders = {}

    original = _positional_weight_conversion
    export_utils = SimpleNamespace(_convert_data_to_const=original)

    def dynamo_export(*public, model, export_path, export_name, **kwargs):
        spec = by_name[export_name]
        names = {_tensor_identity(t): n for n, t in zip(spec.input_names, public)}
        weights = {id(t): n for n, t in (*model.named_parameters(), *model.named_buffers())}

        def backend(fx, example_args):
            # Model TorchAir's _optimize_sym_input and dead-Data removal:
            # shape symbols become sym_size(tensor, axis), never constants or
            # additional public scalar inputs. Use symbolic provenance, not
            # the concrete value 64. This remains a host serializer fixture.
            placeholders = [n for n in fx.graph.nodes if n.op == "placeholder"]
            symbolic = [n.meta.get("example_value", arg) for n, arg in zip(placeholders, example_args)]
            for node, value in zip(placeholders, symbolic):
                if not isinstance(value, torch.SymInt):
                    continue
                source = next((
                    (tensor_node, axis)
                    for tensor_node, tensor in zip(placeholders, symbolic)
                    if isinstance(tensor, torch.Tensor)
                    for axis, dim in enumerate(tensor.shape)
                    if isinstance(dim, torch.SymInt) and str(dim.node.expr) == str(value.node.expr)
                ), None)
                assert source is not None, f"unbound shape symbol {node.name}={value} in Draft capture"
                with fx.graph.inserting_after(source[0]):
                    shape = fx.graph.call_function(torch.ops.aten.sym_size.int, source)
                    node.replace_all_uses_with(shape)
            fx.graph.lint()
            fx.recompile()
            live = [i for i, n in enumerate(placeholders) if n.users]
            assert all(isinstance(symbolic[i], torch.Tensor) for i in live)

            def run(*actual):
                raw_orders[export_name] = [names[_tensor_identity(t)] for t in actual
                                          if _tensor_identity(t) in names]
                serialized = tuple(actual[i] for i in live)
                nodes = []
                for index, raw_index in enumerate(live):
                    shape = [-1 if isinstance(d, torch.SymInt) else int(d)
                             for d in symbolic[raw_index].shape]
                    node = _Op(f"arg{raw_index}", "Data", index=index)
                    node.output_desc = [SimpleNamespace(shape=SimpleNamespace(dim=shape))]
                    nodes.append(node)
                # A consumer keeps its input edges when Data nodes move.
                consumer = _Op("consumer", "Add", inputs=tuple(n.name + ":0" for n in nodes))
                consumer_edges = tuple(consumer.input)
                graph = _Graph([*nodes, consumer])
                if export_name == "draft":
                    feature_node = next(
                        node for node in nodes
                        if names.get(_tensor_identity(serialized[node.attr["index"].i])) == "features"
                    )
                    assert feature_node.output_desc[0].shape.dim[1] == -1
                    # Dynamic GE lowering may emit this helper immediately
                    # after features, ahead of the remaining input/weight Data.
                    helper = _Op("feature_shape", "Shape", inputs=(feature_node.name + ":0",))
                    graph.op.insert(graph.op.index(feature_node) + 1, helper)
                if export_name == "target_verify" and verify_gdr == "chunk":
                    # Synthetic GE nodes for the serializer fixture. Actual
                    # opaque-op torch.export output liveness has its own test.
                    discard_graph = _discard_graph()
                    discard_graph.op[-1].input = (
                        ["consumer:0"] * (len(spec.output_names) - 1) + ["raw_state:0"]
                    )
                    graph.op.extend(discard_graph.op)
                export_utils._convert_data_to_const(serialized, graph, export_path, weights)
                assert tuple(consumer.input) == consumer_edges
                if export_name == "draft":
                    actual_helper = next(n for n in graph.op if n.name == "feature_shape")
                    assert actual_helper.type == "Shape"
                    assert actual_helper.input == [feature_node.name + ":0"]
                normalized = [n for n in graph.op if n.type == "Data"]
                assert [n.attr["index"].i for n in normalized] == list(range(len(public)))
                for index, node in enumerate(normalized):
                    captured = actual[int(node.name.removeprefix("arg"))]
                    assert _tensor_identity(captured) == _tensor_identity(public[index])
                Path(export_path, export_name + ".air").write_text(json.dumps({
                    "host_serializer_fixture": True,
                    "public_nodes": [n.name for n in normalized],
                }))
                return fx.forward(*actual)
            return run

        captured = torch.compile(model, backend=backend, dynamic=kwargs["dynamic"], fullgraph=True)(*public)
        eager = model(*public)
        torch.testing.assert_close(captured, eager, rtol=0, atol=0)

    torchair.dynamo_export = dynamo_export
    monkeypatch.setitem(sys.modules, "torchair", torchair)
    monkeypatch.setitem(sys.modules, "torchair._utils.export_utils", export_utils)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch._dynamo.reset()
    try:
        report = export_air_bundle(lambda config: values, {}, tmp_path / "bundle")
    finally:
        torch._dynamo.reset()
        torch.set_num_threads(previous_threads)
    assert raw_orders["draft"] != list(by_name["draft"].input_names)
    assert raw_orders["draft"][:3] == ["features", "valid_rows", "start_position"]
    for graph in report["graphs"]:
        audit = _validated_runtime_input_abi(graph, required=True)
        assert audit["status"] == "PASS" and audit["calls"] == 1
        assert [b["logical_name"] for b in audit["bindings"]] == graph["input_names"]
        assert audit["weight_conversion"]["reordered"] == (graph["name"] == "draft")
    assert export_utils._convert_data_to_const is original
