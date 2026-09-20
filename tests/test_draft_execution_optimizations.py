"""Equivalent execution formulas and export/memory contracts; CPU evidence only."""
import weakref

import pytest
import torch

from test_incremental_air_om import TinyTarget, draft_model, small_threads
from test_custom_op_export import _ensure_target_test_schema, _FakeTorchAir
from rms_norm_test_support import adn_rms_norm_cpu
from qwen35_dflash.ascend310p.contracts import CustomOpExportSpec
from qwen35_dflash.ascend310p.custom_op_export import (
    NPU_FUNCTIONAL_SCATTER_ND_UPDATE_TORCH_OP,
    prepare_custom_op_export,
)
from qwen35_dflash.ascend310p.incremental import DraftGraph, copy_draft_cache_rows
from qwen35_dflash.ascend310p.quant_factory import AirDFlashOps, _repeat_kv

pytestmark = pytest.mark.usefixtures("small_threads", "adn_rms_norm_cpu")
_LIBRARIES = []


@pytest.fixture
def row_update():
    operation = _ensure_target_test_schema("npu_scatter_nd_update")
    if not torch._C._dispatch_has_kernel_for_dispatch_key(
            NPU_FUNCTIONAL_SCATTER_ND_UPDATE_TORCH_OP, "CPU"):
        library = torch.library.Library("npu", "IMPL", "CPU")
        library.impl("npu_scatter_nd_update", lambda x, i, u:
                     torch.index_copy(x, 0, i[:, 0].long(), u))
        _LIBRARIES.append(library)
    prepare_custom_op_export(CustomOpExportSpec(
        NPU_FUNCTIONAL_SCATTER_ND_UPDATE_TORCH_OP, "ScatterNdUpdate"), _FakeTorchAir())
    return operation


@pytest.mark.parametrize("rows,start", [(16, 0), (16, 13), (16, 112), (64, 7), (64, 64)])
def test_native_row_write_preserves_input_heads_and_untouched_rows(row_update, rows, start):
    cache = torch.randn(2, 4, 128, 16).half()
    before = cache.clone()
    # Noncontiguous new rows, exactly as the K/V projections deliver them.
    updates = torch.randn(2, rows, 4, 16).half().transpose(1, 2)
    expected = cache.clone()
    expected[:, :, start:start + rows] = updates
    actual = copy_draft_cache_rows(cache, torch.arange(start, start + rows), updates, row_update)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(cache, before, rtol=0, atol=0)
    assert actual.untyped_storage().data_ptr() != cache.untyped_storage().data_ptr()


@pytest.mark.parametrize("groups", [1, 2, 4])
@pytest.mark.parametrize("mode", ["causal", "sliding", "noncausal", "per_head", "none"])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_grouped_rows_match_explicit_kv_repetition(groups, mode, dtype):
    torch.manual_seed(17)
    ops = AirDFlashOps(attention_matmul_dtype=dtype)
    q = torch.randn(2, 16, 2 * groups, 16).half().transpose(1, 2)
    k = torch.randn(2, 80, 2, 16).half().transpose(1, 2)
    v = torch.randn_like(k)
    distance = torch.arange(16)[:, None] + 28 - torch.arange(80)
    visible = (torch.arange(80) < 44)[None].expand(16, -1)
    if mode in ("causal", "sliding"):
        visible = visible & (distance >= 0)
    if mode == "sliding":
        visible = visible & (distance < 12)
    mask = visible[None, None]
    if mode == "per_head":
        mask = mask.expand(2, 2 * groups, -1, -1).clone()
        mask[:, 1::2, :, :10] = False
    if mode == "none":
        mask = None
    scores = ops._attention_matmul(q, _repeat_kv(k, groups).transpose(-2, -1)) * 0.25
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    probabilities = torch.softmax(scores, -1, dtype=torch.float32)
    expected = ops._attention_matmul(probabilities, _repeat_kv(v, groups)).to(q.dtype)
    actual = ops.attention(q, k, v, mask, 0.25, groups)
    # Group folding changes GEMM tiling; FP16 is not a bitwise contract.
    # Reuse the existing AirDFlashOps vs decomposed-golden FP16 tolerance.
    # FP32 remains exact for these inputs; all original model gates stay intact.
    tolerance = 2e-3 if dtype == "float16" else 0
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


def test_packing_replaces_weights_without_retaining_original_banks():
    draft, target = draft_model(), TinyTarget().eval()
    original_numel = sum(p.numel() for p in draft.parameters())
    original_parameters = []
    for layer in draft.layers:
        original_parameters.extend(weakref.ref(module.weight) for module in (
            layer.self_attn.k_proj, layer.self_attn.v_proj,
            layer.mlp.gate_proj, layer.mlp.up_proj))
    graph = DraftGraph(draft, target.embedding, target.head, consume_source=True)
    head_and_embedding = {id(p) for m in (target.embedding, target.head) for p in m.parameters()}
    assert sum(p.numel() for p in graph.parameters() if id(p) not in head_and_embedding) == original_numel
    assert all(ref() is None for ref in original_parameters)
    assert not any(name.endswith(("k_proj.weight", "v_proj.weight", "gate_proj.weight", "up_proj.weight"))
                   for name in graph.state_dict())
    assert not hasattr(graph, "draft")


def test_optimized_graph_retains_row_ops_and_reduces_matmuls(row_update):
    draft, target = draft_model(), TinyTarget().eval()
    graph = DraftGraph(draft, target.embedding, target.head, row_update=row_update)
    state = tuple(torch.zeros(1, 1, 192, 16).half() for _ in range(4))
    args = (torch.randn(1, 64, 64).half(), torch.tensor([13]),
            torch.tensor([28], dtype=torch.int16), torch.tensor([4]),
            torch.tensor([15], dtype=torch.int16), *state)
    shapes = {"features": {1: torch.export.Dim("context_rows", min=16, max=64)},
              "start_position": {}, "valid_rows": {}, "anchor": {}, "proposal_count": {},
              "state": tuple({} for _ in state)}
    program = torch.export.export(graph, args, dynamic_shapes=shapes, strict=True)
    nodes = list(program.graph.nodes)
    assert sum(n.target == torch.ops.aten.linear.default for n in nodes) == 2 + 5 * len(graph.layers)
    assert sum(n.target == row_update for n in nodes) == 2 * len(graph.layers)
    assert not any(str(n.target) in ("aten.repeat.default", "aten.scatter.src", "aten.index_copy.default") for n in nodes)
    exported = program.module()
    with torch.inference_mode():
        for rows in (1, 16, 17, 28, 64):
            inputs = (args[0][:, :16 if rows <= 16 else 64], args[1],
                      torch.tensor([rows], dtype=torch.int16), *args[3:])
            for expected, actual in zip(graph(*inputs), exported(*inputs)):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
