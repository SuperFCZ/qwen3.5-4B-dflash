#!/usr/bin/env python3
"""CPU check for DynamicCache rollback and bounded commit replay."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import nn
from transformers import PreTrainedConfig


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from models.dflash_v1.dflash_rollback_adapter import (  # noqa: E402
    FrameworkDFlashRollbackTarget,
    _restore_framework_cache,
    _snapshot_framework_cache,
)


def _state(layer, name):
    value = getattr(layer, name)
    return value[0] if isinstance(value, dict) else value


class TinyHybridConfig(PreTrainedConfig):
    model_type = "tiny_hybrid_rollback_test"

    def __init__(self) -> None:
        super().__init__()
        self.num_hidden_layers = 2
        self.layer_types = ["full_attention", "linear_attention"]


class TinyHybridTarget(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = TinyHybridConfig()
        self.embedding = nn.Embedding(128, 2)
        self.head = nn.Linear(2, 128, bias=False)

    def get_input_embeddings(self) -> nn.Module:
        return self.embedding

    def get_output_embeddings(self) -> nn.Module:
        return self.head

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        past_key_values,
        output_dflash_features: bool,
        **kwargs,
    ):
        del kwargs
        values = input_ids.to(torch.float32)
        kv = values.view(1, 1, -1, 1)
        past_key_values.update(kv, -kv, 0)

        linear = past_key_values.layers[1]
        delta = values.sum().view(1, 1, 1)
        conv = _state(linear, "conv_states")
        recurrent = _state(linear, "recurrent_states")
        if conv is None:
            conv = torch.zeros_like(delta)
            recurrent = torch.zeros((1, 1, 1, 1))
        past_key_values.update_conv_state(conv + delta, 1)
        past_key_values.update_recurrent_state(
            recurrent + delta.view(1, 1, 1, 1) * 10, 1,
        )

        token_ids = ((input_ids + 1) % 128).to(torch.long)
        logits = torch.full((1, input_ids.shape[1], 128), -10.0)
        logits.scatter_(2, token_ids.unsqueeze(-1), 10.0)
        result = {
            "logits": logits,
            "past_key_values": past_key_values,
        }
        if output_dflash_features:
            result["dflash_features"] = values.unsqueeze(-1).repeat(1, 1, 3)
        return result


def main() -> None:
    target = TinyHybridTarget().eval()
    controller = FrameworkDFlashRollbackTarget(target).eval()
    controller.begin_rollback(torch.tensor([[1, 2]], dtype=torch.long))
    assert controller._cache is not None
    assert controller._cache.get_seq_length() == 2
    linear = controller._cache.layers[1]
    assert float(_state(linear, "conv_states").item()) == 3.0
    assert float(_state(linear, "recurrent_states").item()) == 30.0

    # The rejected tail is deliberately large, making a failed state restore
    # obvious. accepted=1 must commit only rows [3,4], never row 99.
    controller.verify_rollback(torch.tensor([[3, 4, 99]], dtype=torch.long))
    assert controller._cache.get_seq_length() == 5
    assert float(_state(linear, "conv_states").item()) == 109.0
    committed = controller.commit_rollback(1)
    assert controller._cache.get_seq_length() == 4
    assert float(_state(linear, "conv_states").item()) == 10.0
    assert float(_state(linear, "recurrent_states").item()) == 100.0
    assert tuple(committed["logits"].shape) == (1, 2, 128)
    assert tuple(committed["dflash_features"].shape) == (1, 2, 3)
    audit = controller.dflash_rollback_audit
    assert audit["historical_prefix_replay_during_verify"] is False
    assert audit["rollback_commit_transactions"] == 1
    assert audit["rollback_commit_replay_calls"] == 2
    assert audit["cache_sequence_length"] == 4

    failed = FrameworkDFlashRollbackTarget(TinyHybridTarget().eval()).eval()
    failed.begin_rollback(torch.tensor([[1, 2]], dtype=torch.long))
    failed.verify_rollback(torch.tensor([[3, 99]], dtype=torch.long))
    failed.abort_rollback()
    failed_audit = failed.dflash_rollback_audit
    assert failed_audit["pending_transaction"] is False
    assert failed_audit["cache_sequence_length"] is None
    assert failed_audit["rollback_aborts"] == 1
    print("PASS: framework KV/GDN restore and anchor+accepted commit replay")


def test_hybrid_cache_transaction():
    main()


def test_snapshot_preserves_per_state_flags_and_tensor_addresses():
    from transformers.cache_utils import DynamicCache

    config = TinyHybridConfig()
    config.number_of_conv_states = 2
    cache = DynamicCache(config=config)
    layer = cache.layers[1]
    if not isinstance(layer.conv_states, dict):
        pytest.skip("per-state dictionary ABI requires Transformers 5.14.1")
    cache.update(torch.zeros(1, 1, 2, 1), torch.zeros(1, 1, 2, 1), 0)
    cache.update_conv_state(torch.ones(1, 2, 4), 1)
    cache.update_recurrent_state(torch.ones(1, 2, 2, 2), 1)
    addresses = (layer.conv_states[0].data_ptr(), layer.recurrent_states[0].data_ptr())
    snapshot = _snapshot_framework_cache(cache)
    layer.conv_states[0].add_(99)
    layer.recurrent_states[0].mul_(42)
    cache.update_conv_state(torch.ones(1, 2, 4), 1, state_idx=1)
    cache.update_recurrent_state(torch.ones(1, 2, 2, 2), 1, state_idx=1)
    cache.update(torch.zeros(1, 1, 3, 1), torch.zeros(1, 1, 3, 1), 0)
    _restore_framework_cache(cache, snapshot)
    assert cache.get_seq_length() == 2
    assert layer.has_previous_state == {0: True, 1: False}
    assert layer.is_conv_states_initialized == {0: True, 1: False}
    assert layer.is_recurrent_states_initialized == {0: True, 1: False}
    assert layer.conv_states[1] is None and layer.recurrent_states[1] is None
    assert addresses == (layer.conv_states[0].data_ptr(), layer.recurrent_states[0].data_ptr())
    assert torch.equal(layer.conv_states[0], torch.ones(1, 2, 4))
    assert torch.equal(layer.recurrent_states[0], torch.ones(1, 2, 2, 2))
    # Restoring again must not alias the private snapshot to mutable cache data.
    layer.conv_states[0].zero_()
    _restore_framework_cache(cache, snapshot)
    assert torch.equal(layer.conv_states[0], torch.ones(1, 2, 4))


@pytest.mark.parametrize("accepted", [0, 1, 2])
def test_real_qwen_hybrid_cache_matches_ordinary_after_rejected_tail(accepted):
    import transformers
    if transformers.__version__ != "5.14.1":
        pytest.skip("packaged target is locked to Transformers 5.14.1")
    from models.dflash_v1.configuration_qwen3_5 import Qwen3_5TextConfig
    from models.dflash_v1.modeling_qwen3_5_dflash import Qwen3_5ForCausalLM
    from models.dflash_v1.dflash_target_features import DFlashTargetFeatureSpec

    torch.manual_seed(42)
    config = Qwen3_5TextConfig(
        vocab_size=128, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, linear_num_key_heads=2, linear_num_value_heads=2,
        linear_key_head_dim=8, linear_value_head_dim=8,
        layer_types=["linear_attention", "full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 1.0, "mrope_section": [2, 1, 1]},
    )
    config._attn_implementation = "eager"
    raw = Qwen3_5ForCausalLM(config).eval()
    raw.model.dflash_target_feature_spec = DFlashTargetFeatureSpec((0, 1), 32, 2)
    ordinary = FrameworkDFlashRollbackTarget(raw).eval()
    rollback = FrameworkDFlashRollbackTarget(raw).eval()
    prompt, block = torch.tensor([[1, 2, 3]]), torch.tensor([[4, 5, 99]])
    with torch.inference_mode():
        ordinary.begin_ordinary(prompt)
        rollback.begin_rollback(prompt)
        expected = [ordinary.advance_ordinary(block[:, i:i+1]).logits
                    for i in range(accepted + 1)]
        rollback.verify_rollback(block)
        actual = rollback.commit_rollback(accepted)
        torch.testing.assert_close(actual["logits"], torch.cat(expected, 1), rtol=0, atol=0)
        for name in ("conv_states", "recurrent_states"):
            torch.testing.assert_close(
                _state(ordinary._cache.layers[0], name),
                _state(rollback._cache.layers[0], name), rtol=0, atol=0,
            )
        assert ordinary._cache.get_seq_length() == rollback._cache.get_seq_length()


def test_packaged_target_rejects_wrong_runtime_before_weight_load():
    from models.dflash_v1.dflash_qwen_adapter_v1 import _load_target
    with patch("transformers.__version__", "5.13.0"):
        with pytest.raises(RuntimeError, match="requires transformers==5.14.1"):
            _load_target("/absent-target", target_loader=None, device="cpu",
                         dtype=torch.float16, allow_download=False, trust_remote_code=False)


if __name__ == "__main__":
    main()
