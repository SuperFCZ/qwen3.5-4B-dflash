"""Quantization arithmetic, shape/identity failures and real export data flow."""
from __future__ import annotations

from dataclasses import replace
import copy
import json
import os
from pathlib import Path
import sys

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from rms_norm_test_support import adn_rms_norm_cpu
from test_incremental_air_om import small_threads

pytestmark = pytest.mark.usefixtures("adn_rms_norm_cpu", "small_threads")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/python"))

from models.dflash_v1.draft_quantization import (
    GroupQuantLinear, configure_target_for_draft, pack_device_weight,
    unpack_packed_int32, validate_quantization_config, require_draft_checkpoint,
)
from models.dflash_v1.dflash_config import Qwen35DFlashConfig
from models.dflash_v1.dflash_target_features import DFlashFeatureCollector
from models.dflash_v1.modeling_dflash import DFlashDraftModel, WeightOnlyLinear
from test_draft_execution_optimizations import row_update
from qwen35_dflash.ascend310p.quant_factory import AirDFlashOps
from qwen35_dflash.ascend310p.contracts import AirGraphSpec
from qwen35_dflash.ascend310p.draft_constants import (
    expose_draft_constants, write_constant_inputs, verify_constant_inputs,
)


def config():
    return Qwen35DFlashConfig(
        hidden_size=128, intermediate_size=256, vocab_size=32,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=32, num_target_layers=4, target_layer_ids=(0, 2),
        layer_types=("full_attention",) * 2, block_size=4, mask_token_id=31,
        rms_norm_eps=1e-6, rope_theta=1e7, max_position_embeddings=64,
        sliding_window=None, use_sliding_window=False, attention_bias=False,
        attention_dropout=0.0, hidden_act="silu", dtype="bfloat16")


@pytest.mark.parametrize("bits", [4, 8])
def test_packed_words_match_independent_scalar_bit_reader(bits):
    # Covers signed int32 words, offset-binary extrema and a non-word-aligned tail.
    words = [-1, -2147483648, 0x12345678, 0]
    k = len(words) * (32 // bits) - 1
    reference = []
    for word in words:
        for shift in range(0, 32, bits):
            reference.append(((word & 0xffffffff) >> shift & ((1 << bits)-1)) - (1 << (bits-1)))
    actual = unpack_packed_int32(torch.tensor([words], dtype=torch.int32), bits, (1, k))
    assert actual.tolist() == [reference[:k]]


@pytest.mark.parametrize("bits", [4, 8])
def test_group_linear_matches_dense_fp16_and_keeps_compressed_storage(bits):
    torch.manual_seed(417)
    q = torch.randint(-(1 << (bits-1)), 1 << (bits-1), (6, 256), dtype=torch.int8)
    scales = torch.tensor([[0.003, 0.09]] * 6, dtype=torch.float16)
    linear = GroupQuantLinear(pack_device_weight(q, bits), scales, bits=bits,
                              in_features=256, ops=AirDFlashOps())
    # Build the reference independently, one group at a time in FP32.
    expected_weight = torch.cat([(q[:, :128].float() * scales[:, :1].float()).half(),
                                 (q[:, 128:].float() * scales[:, 1:].float()).half()], dim=1)
    inputs = torch.randn(1, 3, 256, dtype=torch.float16)
    torch.testing.assert_close(linear(inputs), F.linear(inputs, expected_weight), rtol=0, atol=0)
    assert list(linear.parameters()) == []
    assert linear.qweight.numel() * linear.qweight.element_size() == 6 * 256 * bits // 8
    assert not hasattr(linear, "weight")
    with pytest.raises(ValueError, match="float16"):
        linear(inputs.float())


@pytest.mark.parametrize("bits", [4, 8])
def test_complete_quantized_draft_matches_dense_and_cached_attention(bits):
    torch.manual_seed(591)
    draft = DFlashDraftModel(config(), ops=AirDFlashOps(), dtype=torch.float16).eval()
    with torch.no_grad():
        for p in draft.parameters(): p.uniform_(-0.05, 0.05)
    dense = copy.deepcopy(draft)
    for path, module in list(draft.named_modules()):
        if not isinstance(module, WeightOnlyLinear): continue
        q = torch.randint(-7, 8, module.weight.shape, dtype=torch.int8)
        scale = torch.full((module.out_features, module.in_features // 128), 0.005, dtype=torch.float16)
        quant = GroupQuantLinear(pack_device_weight(q, bits), scale, bits=bits,
                                 in_features=module.in_features, ops=draft.ops)
        # Independent vectorized reference uses integer q, not the unpacker.
        weight = (q.float() * scale.float().repeat_interleave(128, dim=1)).half()
        with torch.no_grad(): dense.get_submodule(path).weight.copy_(weight)
        parent, attr = path.rsplit('.', 1) if '.' in path else ('', path)
        setattr(draft.get_submodule(parent), attr, quant)
    hidden = torch.randn(1, 3, config().feature_size, dtype=torch.float16)
    noise = torch.randn(1, 4, 128, dtype=torch.float16)
    positions = torch.arange(7).view(1, -1)
    with torch.inference_mode():
        expected = dense(hidden, noise, positions)
        torch.testing.assert_close(draft(hidden, noise, positions), expected, rtol=0, atol=0)
        projected = draft.project_target_hidden(hidden)
        cache = draft.new_kv_cache(max_length=32)
        actual = draft.forward_cached_projected(projected, noise, positions, cache)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert cache.committed_length == 3


class LinearGraph(nn.Module):
    def __init__(self, bits):
        super().__init__()
        self.linear = GroupQuantLinear(pack_device_weight(torch.ones(2, 128, dtype=torch.int8), bits),
            torch.full((2, 1), 0.5, dtype=torch.float16), bits=bits, in_features=128, ops=AirDFlashOps())

    def forward(self, ids, mask):
        value = (ids * mask).half()
        return self.linear(value)


@pytest.mark.parametrize("bits", [4, 8])
def test_exported_graph_consumes_runtime_compressed_weights(bits, tmp_path):
    base = AirGraphSpec(name="draft", role="generation-recompute", model=LinearGraph(bits),
                        example_args=(torch.ones(1, 128, dtype=torch.long), torch.ones(1, 128, dtype=torch.long)),
                        input_names=("input_ids", "attention_mask"), output_names=("result",))
    reference = base.model(*base.example_args)
    spec = expose_draft_constants(base)
    assert spec.input_names[2:] == ("draft_weight_000", "draft_weight_001")
    assert spec.model.model.linear.qweight.numel() == 0
    # Actual torch.export, not a fake exporter: external weights must affect output
    # after capture and must not be folded into dense FP16 state.
    exported = torch.export.export(spec.model, spec.example_args).module()
    torch.testing.assert_close(exported(*spec.example_args), reference, rtol=0, atol=0)
    args = (*spec.example_args[:-1], spec.example_args[-1] * 2)
    torch.testing.assert_close(exported(*args), reference * 2, rtol=0, atol=0)
    assert all(t.numel() == 0 for t in exported.buffers())
    graph_dir = tmp_path / "draft"
    graph_dir.mkdir()
    metadata = write_constant_inputs(spec, graph_dir, tmp_path)
    graph = {"input_names": list(spec.input_names), **metadata}
    assert verify_constant_inputs(graph, tmp_path) is not None
    path = tmp_path / metadata["constant_inputs"][0]["path"]
    path.write_bytes(bytes(path.stat().st_size))
    with pytest.raises(ValueError, match="integrity"):
        verify_constant_inputs(graph, tmp_path)


def test_feature_binding_selects_checkpoint_layers_per_instance():
    target = nn.Sequential(nn.Identity())
    configure_target_for_draft(target, config())
    spec = target[0].dflash_target_feature_spec
    collector = DFlashFeatureCollector(spec, enabled=True)
    for i in range(4): collector.capture(i, torch.full((1, 2, 128), float(i)))
    result = collector.finalize()
    assert result.shape == (1, 2, 256)
    assert result[..., :128].eq(0).all() and result[..., 128:].eq(2).all()
    other = nn.Sequential(nn.Identity())
    assert not hasattr(other, "dflash_target_feature_spec")
    with pytest.raises(ValueError, match="different"):
        configure_target_for_draft(target, replace(config(), target_layer_ids=(1, 3)))


def test_unknown_or_wrong_checkpoint_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unsupported"):
        require_draft_checkpoint(tmp_path, "gptq-fake")
    (tmp_path / "config.json").write_text('{}')
    with pytest.raises(ValueError, match="hash mismatch"):
        require_draft_checkpoint(tmp_path, "w4a16")


def test_cli_version_selection():
    from qwen35_dflash.ascend310p.cli import build_parser, _factory_config
    om = build_parser().parse_args(['export-air', '--factory', 'x:y', '--bundle-dir', 'b',
                                   '--draft-quantization', 'w8a16'])
    assert _factory_config(om)['draft_quantization'] == 'w8a16'


@pytest.mark.parametrize("bits", [4, 8])
def test_incremental_quantized_draft_shared_features_and_dynamic_export(bits, row_update):
    from qwen35_dflash.ascend310p.incremental import DraftGraph

    torch.manual_seed(174)
    cfg = replace(config(), block_size=16)
    draft = DFlashDraftModel(cfg, ops=AirDFlashOps(), dtype=torch.float16).eval()
    with torch.no_grad():
        for p in draft.parameters():
            p.uniform_(-0.05, 0.05)
    dense = copy.deepcopy(draft)
    for path, module in list(draft.named_modules()):
        if not isinstance(module, WeightOnlyLinear):
            continue
        q = torch.randint(-7, 8, module.weight.shape, dtype=torch.int8)
        scale = torch.full((module.out_features, module.in_features // 128), 0.004, dtype=torch.float16)
        with torch.no_grad():
            dense.get_submodule(path).weight.copy_((q.float() * scale.float().repeat_interleave(128, dim=1)).half())
        quant = GroupQuantLinear(pack_device_weight(q, bits), scale, bits=bits, in_features=module.in_features, ops=draft.ops)
        parent, attr = path.rsplit('.', 1) if '.' in path else ('', path)
        setattr(draft.get_submodule(parent), attr, quant)
    embedding, head = nn.Embedding(32, 128).half(), nn.Linear(128, 32, bias=False).half()
    expected = DraftGraph(dense, embedding, head, row_update=row_update)
    current = DraftGraph(draft, embedding, head, row_update=row_update, consume_source=True, feature_layers=(0, 1, 2))
    args = (torch.randn(1, 64, 384).half(), torch.tensor([4]), torch.tensor([28], dtype=torch.int16),
            torch.tensor([3]), torch.tensor([15], dtype=torch.int16),
            *(torch.zeros(1, 2, 192, 32).half() for _ in range(4)))
    spec = expose_draft_constants(AirGraphSpec(name="draft", role="draft", model=current, example_args=args,
        input_names=("features", "start_position", "valid_rows", "anchor", "proposal_count", "d0_key", "d0_value", "d1_key", "d1_value"),
        output_names=("draft_top1", "d0_key", "d0_value", "d1_key", "d1_value")))
    assert len(spec.constant_input_names) == 22  # FC + five packed projections/layer.
    assert not any(n.endswith(("qweight", "scales")) and t.numel() for n, t in spec.model.named_buffers())
    # *inputs is one tuple argument in torch.export's dynamic-shape schema.
    shapes = ({1: torch.export.Dim("context_rows", min=16, max=64)}, *({} for _ in spec.example_args[1:]))
    exported = torch.export.export(spec.model, spec.example_args, dynamic_shapes=(shapes,), strict=True).module()
    with torch.inference_mode():
        for rows, valid, proposals in ((16, 1, 1), (16, 13, 15), (64, 28, 15), (64, 64, 7)):
            x = args[0][:, :rows]
            inputs = (x, args[1], torch.tensor([valid], dtype=torch.int16), args[3], torch.tensor([proposals], dtype=torch.int16), *args[5:])
            reference = expected(torch.cat((x[..., :128], x[..., 256:]), -1), *inputs[1:])
            actual = exported(*inputs, *spec.example_args[len(args):])
            for left, right in zip(reference, actual):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert not any("weight_quant" in str(n.target) for n in exported.graph.nodes)
