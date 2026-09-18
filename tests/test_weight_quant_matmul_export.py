"""Native-op export/ABI tests use explicit CPU shims, not CANN measurements."""
import json
from types import SimpleNamespace

import pytest
import torch

from weight_quant_test_support import weight_quant_cpu
from test_draft_variant_bundles import variant_builder
from test_incremental_air_om import small_threads
from rms_norm_test_support import adn_rms_norm_cpu
from models.dflash_v1.weight_quant_matmul import TORCH_OP, GE_OP, preflight_weight_quant_matmul, weight_quant_linear
from qwen35_dflash.ascend310p.contracts import CustomOpExportSpec
from qwen35_dflash.ascend310p import custom_op_export as exports
from qwen35_dflash.ascend310p.compiler import _validated_custom_op_audit

pytestmark = pytest.mark.usefixtures("weight_quant_cpu", "small_threads", "adn_rms_norm_cpu")


@pytest.mark.parametrize("bits", [4, 8])
def test_fused_contract_survives_bundle_and_compile(bits, variant_builder):
    build, _ = variant_builder
    path = build(f"w{bits}a16", "chunk", extra={"draft_layers": 5, "draft_quant_matmul": "weight_quant"})
    data = json.loads(path.read_text())
    for graph in data["graphs"]:
        audit = _validated_custom_op_audit(graph)
        native = [item for item in audit if item["ge_op_type"] == GE_OP]
        if graph["name"] == "draft":
            assert len(native) == 1
            assert native[0]["minimum_occurrences"] == native[0]["ge_node_occurrences"] == 26
            assert native[0]["converter_policy"] == "torchair-builtin"
            assert len(graph["constant_inputs"]) == 52
            assert sum(len(t["shape"]) for t in graph["metadata"]["tensor_abi"]["inputs"]) == 99
            broken = dict(graph, custom_op_audit=[dict(item, ge_node_occurrences=0) for item in audit])
            with pytest.raises((ValueError, RuntimeError)):
                _validated_custom_op_audit(broken)
        else:
            assert not native


def test_meta_validates_actual_gate_up_shape_and_retains_builtin_converter(weight_quant_cpu):
    session = exports.prepare_custom_op_export(CustomOpExportSpec(TORCH_OP, GE_OP), object())
    assert session.converter_policy == "torchair-builtin"
    exports._validate_npu_weight_quant_matmul_meta(weight_quant_cpu)


def test_bad_dtype_or_meta_result_is_rejected():
    with pytest.raises(ValueError, match="A16W8"):
        exports._fake_npu_weight_quant_matmul(torch.empty(16, 256).half(), torch.empty(256, 64).half(),
                                             torch.empty(2, 64).half(), antiquant_group_size=128)
    with pytest.raises(RuntimeError):
        exports._validate_npu_weight_quant_matmul_meta(lambda x, w, s, **kw: x.new_empty(16, 19456, dtype=torch.int8))
    with pytest.raises(ValueError, match="NPU"):
        preflight_weight_quant_matmul("cpu")


@pytest.mark.parametrize("shape", [(1, 64), (64, 1)])
def test_perchannel_meta_requires_vector_scale(shape):
    x = torch.empty(16, 256, dtype=torch.float16)
    w = torch.empty(256, 64, dtype=torch.int8)
    with pytest.raises(ValueError, match=r"per-channel scale.*\[N\]"):
        exports._fake_npu_weight_quant_matmul(x, w, torch.empty(shape).half(), antiquant_group_size=0)
    result = exports._fake_npu_weight_quant_matmul(x, w, torch.empty(64).half(), antiquant_group_size=0)
    assert result.shape == (16, 64) and result.dtype == torch.float16


def test_single_group_linear_preserves_values_with_perchannel_vector():
    q = ((torch.arange(64 * 128).reshape(64, 128) % 7) - 3).to(torch.int8)
    x = ((torch.arange(16 * 128).reshape(16, 128) % 5) - 2).half() / 16
    scale = (1 + torch.arange(64).reshape(64, 1) % 4).half() / 32
    expected = (x.float() @ (q.float() * scale.float()).t()).half()
    assert torch.equal(weight_quant_linear(x, q, scale), expected)


def test_new_optional_dtype_schema_is_accepted_but_other_drift_is_rejected(weight_quant_cpu):
    original = str(weight_quant_cpu._schema)
    adapter = exports._ADAPTERS[TORCH_OP]
    new = original.replace("int inner_precise=0)", "int inner_precise=0, int? weight_dtype=None)")
    exports._validate_schema(SimpleNamespace(_schema=torch._C.parse_schema(new)), adapter)
    broken = new.replace("int? weight_dtype=None", "int? weight_dtype=1")
    with pytest.raises(RuntimeError, match="schema drifted"):
        exports._validate_schema(SimpleNamespace(_schema=torch._C.parse_schema(broken)), adapter)


def test_target_reuse_allows_draft_only_matmul_change(variant_builder):
    build, calls = variant_builder
    fp16 = build("fp16", "chunk")
    calls.clear()
    build("w8a16", "chunk", reuse_target=fp16, extra={"draft_quant_matmul": "weight_quant"})
    assert calls == [("export", "draft"), ("compile", "draft")]
