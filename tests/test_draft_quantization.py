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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/python"))

from models.dflash_v1.draft_quantization import (
    GroupQuantLinear, configure_target_for_draft, pack_device_weight,
    unpack_packed_int32, validate_quantization_config, require_draft_checkpoint,
)
from models.dflash_v1.dflash_config import Qwen35DFlashConfig
from models.dflash_v1.dflash_target_features import DFlashFeatureCollector
from models.dflash_v1.modeling_dflash import DFlashDraftModel, WeightOnlyLinear
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
    from models.dflash_v1.run_npu import _parser
    from qwen35_dflash.ascend310p.cli import build_parser, _factory_config
    npu = _parser().parse_args(['--target-dir', 't', '--draft-dir', 'd', '--prompt', 'x',
                               '--kv-cache-max-len', '128', '--draft-quantization', 'w4a16'])
    assert npu.draft_quantization == 'w4a16'
    om = build_parser().parse_args(['export-air', '--factory', 'x:y', '--bundle-dir', 'b',
                                   '--draft-quantization', 'w8a16'])
    assert _factory_config(om)['draft_quantization'] == 'w8a16'


@pytest.mark.parametrize("bits", [4, 8])
def test_cpp_launch_binds_compressed_manifest_and_report(bits, tmp_path, monkeypatch):
    from qwen35_dflash.ascend310p.cpp_runtime import run_cpp_pair, validate_cpp_runner_report
    from qwen35_dflash.ascend310p.utils import file_record

    runner = os.environ.get('DFLASH_FAKE_CPP_RUNNER')
    if not runner:
        pytest.skip('set DFLASH_FAKE_CPP_RUNNER to the CMake-built host test runner')
    monkeypatch.setenv('AI_RUN_DIR', str(tmp_path))
    monkeypatch.setenv('QWEN35_FAKE_DRAFT_CONSTANTS', 'w4a16' if bits == 4 else 'w8a16')
    names = ('draft_weight_000', 'draft_weight_001')
    weight = torch.full((1, 64 if bits == 4 else 128), 0x89 if bits == 4 else 1,
                        dtype=torch.uint8 if bits == 4 else torch.int8)
    args = (torch.zeros(1, 32, dtype=torch.int64), torch.zeros(1, 32, dtype=torch.int64),
            weight, torch.full((1, 1), 0.5, dtype=torch.float16))
    spec = AirGraphSpec(name='quant_dflash_recompute', role='generation-recompute',
                        model=nn.Identity(), example_args=args,
                        input_names=('input_ids', 'attention_mask', *names),
                        output_names=('target_top1', 'draft_top1'), constant_input_names=names)
    graph_dir = tmp_path / 'air'
    graph_dir.mkdir()
    constants = write_constant_inputs(spec, graph_dir, tmp_path)
    om = tmp_path / 'test.om'
    om.write_bytes(b'fake-om')
    air = tmp_path / 'air-manifest.json'
    air.write_text('{}')
    manifest = tmp_path / 'deployment-manifest.json'
    manifest.write_text(json.dumps({
        'artifact_kind': 'qwen35-dflash-ascend310p-om-bundle', 'status': 'PASS',
        'air_manifest': file_record(air, relative_to=tmp_path),
        'graphs': [{'name': spec.name, 'role': spec.role,
                    'input_names': list(spec.input_names), 'output_names': list(spec.output_names),
                    'om': file_record(om, relative_to=tmp_path), **constants}]}))
    report = run_cpp_pair(deployment_manifest=manifest, runner=runner,
        runner_options={'device_model': 'fake-ACL-host-test', 'cann': 'fake', 'driver': 'fake',
                        'firmware': 'fake', 'runtime': 'fake-ACL-host-test'},
        prompt_token_ids=[1,2], eos_token_ids=[], device_id=0, max_new_tokens=8, max_draft_tokens=3,
        raw_output=tmp_path/'result.json', log_output=tmp_path/'runner.log')
    assert report['constant_input_count'] == 2
    assert report['ordinary_parity']['token_id_mismatches'] == 0
    report['constant_inputs_sha256'] = '0' * 64
    with pytest.raises(RuntimeError, match='identity'):
        validate_cpp_runner_report(report, prompt_token_ids=[1,2], om_sha256=file_record(om,relative_to=tmp_path)['sha256'],
            device_id=0, max_new_tokens=8, max_draft_tokens=3, constant_input_names=names,
            constant_inputs_sha256=constants['constant_inputs_table']['sha256'])
