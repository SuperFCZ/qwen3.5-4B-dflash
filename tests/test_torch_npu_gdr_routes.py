"""CPU dispatch/transaction checks of native-route source; no device evidence."""
import ast
import copy
import __future__
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from models.dflash_v1 import run_npu, run_rollback, benchmark_npu
from models.internal_dflash_bridge import InternalDFlashTarget
from test_incremental_air_om import gdr, mtp_gdr
from test_internal_dflash_bridge_rollback import FakeHIAIModel, FakeWrapper


@pytest.fixture
def native_source():
    source = Path(__file__).resolve().parents[1] / "models/modeling_qwen3_5_hiai_nd_dflash_rollback.py"
    names = {
        "require_gdr_mtp", "run_dflash_mtp_gdr", "torch_dflash_causal_conv1d_chunk",
        "select_dflash_chunk_commit_state", "run_dflash_chunk_gdr_commit",
        "torch_causal_conv1d_update", "Qwen3_5RMSNormGated", "Qwen3_5GatedDeltaNet",
    }
    tree = ast.parse(source.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    model_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Qwen3_5ForCausalLM")
    nodes.append(next(n for n in model_class.body if isinstance(n, ast.FunctionDef)
                      and n.name == "configure_dflash_verify_gdr"))
    calls = []

    def chunk(*args, **kwargs):
        calls.append(("chunk", args, kwargs))
        return gdr(*args, **kwargs)

    def mtp(*args, **kwargs):
        calls.append(("mtp", args, kwargs))
        assert args[5].dtype == torch.float32 and args[6].dtype == torch.int8
        assert torch.equal(args[6], torch.zeros_like(args[6]))
        return mtp_gdr(*args, **kwargs)

    def rms(x, gamma, epsilon):
        rstd = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + epsilon)
        return (x.float() * rstd * gamma.float()).to(x.dtype), rstd

    operations = SimpleNamespace(npu_chunk_gated_delta_rule=chunk, npu_gated_delta_rule_mtp=mtp,
                                 adn_rms_norm=rms)
    scope = dict(torch=torch, nn=nn, F=F, torch_npu=operations, ACT2FN={"silu": F.silu},
                 causal_conv1d_fn=None, causal_conv1d_update=None, DFLASH_MAX_VERIFY_TOKENS=16)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec",
                 flags=__future__.annotations.compiler_flag), scope)
    config = SimpleNamespace(hidden_size=32, linear_num_value_heads=1, linear_num_key_heads=1,
                             linear_key_head_dim=16, linear_value_head_dim=16,
                             linear_conv_kernel_dim=4, hidden_act="silu", rms_norm_eps=1e-6,
                             layer_types=["linear_attention"])
    torch.manual_seed(93)
    block = scope["Qwen3_5GatedDeltaNet"](config, 0).half().eval()
    block.dflash_verify_gdr = "mtp"
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield block, calls, scope
    torch.set_num_threads(previous)


@pytest.mark.parametrize("rows,accepted", [(1, 0), (2, 0), (2, 1), (4, 0), (4, 2), (4, 3),
                                           (16, 0), (16, 1), (16, 14), (16, 15)])
def test_native_mtp_commits_exact_prefix_without_second_gdr(native_source, rows, accepted):
    block, calls, _ = native_source
    reference = copy.deepcopy(block)
    reference.dflash_verify_gdr = "chunk"
    hidden = torch.randn(1, rows, 32).half()
    state = (torch.randn(1, 48, 4).half(), torch.randn(1, 1, 16, 16) * 0.03)
    valid = torch.tensor([rows], dtype=torch.int16)
    with torch.inference_mode():
        actual, _ = block(hidden, tuple(t.clone() for t in state), dflash_chunk_verify=True,
                          gdr_effective_length=valid)
        committed = block.compute_dflash_chunk_commit(accepted + 1)
        assert [c[0] for c in calls] == ["mtp"]
        assert block.dflash_pending_chunk_rows == rows
        assert committed[1].dtype == torch.float32
        for t in committed:
            assert t.untyped_storage().nbytes() == t.numel() * t.element_size()
        block.discard_dflash_chunk_commit()
        assert block.dflash_pending_chunk_rows is None
        expected_output, _ = reference(hidden, tuple(t.clone() for t in state), dflash_chunk_verify=True,
                                       gdr_effective_length=valid)
        expected = reference.compute_dflash_chunk_commit(accepted + 1)
        torch.testing.assert_close(actual, expected_output, rtol=0, atol=0)
        for got, wanted in zip(committed, expected):
            torch.testing.assert_close(got, wanted, rtol=0, atol=0)
        # Changing rejected future rows must not change the selected prefix.
        other = hidden.clone()
        other[:, accepted+1:] = torch.randn_like(other[:, accepted+1:])
        block(other, tuple(t.clone() for t in state), dflash_chunk_verify=True, gdr_effective_length=valid)
        for got, wanted in zip(block.compute_dflash_chunk_commit(accepted+1), committed):
            torch.testing.assert_close(got, wanted, rtol=0, atol=0)
        block.discard_dflash_chunk_commit()
        # A new round may use another K; selected state remains valid input.
        block(torch.randn(1, 3, 32).half(), committed, dflash_chunk_verify=True,
              gdr_effective_length=torch.tensor([3], dtype=torch.int16))
        assert block.compute_dflash_chunk_commit(1)[1].dtype == torch.float32


def test_mtp_does_not_change_ordinary_dispatch_or_rounding(native_source):
    block, calls, _ = native_source
    other = copy.deepcopy(block)
    other.dflash_verify_gdr = "chunk"
    state = (torch.randn(1, 48, 4).half(), torch.randn(1, 1, 16, 16).half())
    hidden = torch.randn(1, 1, 32).half()
    valid = torch.ones(1, dtype=torch.int16)
    with torch.inference_mode():
        a = block(hidden, tuple(t.clone() for t in state), gdr_effective_length=valid)
        b = other(hidden, tuple(t.clone() for t in state), gdr_effective_length=valid)
    assert [c[0] for c in calls] == ["chunk", "chunk"]
    assert a[1][1].dtype == torch.float16
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_mtp_operator_missing_is_not_a_chunk_fallback(native_source):
    _, calls, scope = native_source
    scope["torch_npu"] = SimpleNamespace()
    scope["torch"] = SimpleNamespace(ops=SimpleNamespace(npu=SimpleNamespace()))
    with pytest.raises(RuntimeError, match="requires registered"):
        scope["require_gdr_mtp"]()
    assert not calls


def test_configure_route_refuses_pending_transactions(native_source):
    block, _, scope = native_source
    model = SimpleNamespace(language_model=SimpleNamespace(_dflash_gdn_layers=lambda: [block]))
    configure = scope["configure_dflash_verify_gdr"]
    configure(model, "chunk")
    assert block.dflash_verify_gdr == "chunk"
    configure(model, "mtp")
    assert model.dflash_verify_gdr == block.dflash_verify_gdr == "mtp"
    assert "mtp-bank" in model.language_model.dflash_state_contract_id
    block._dflash_chunk_commit_capsule = (torch.zeros(1, 3, 48, 4), torch.zeros(1, 3, 1, 16, 16))
    with pytest.raises(RuntimeError, match="pending verification"):
        configure(model, "chunk")
    assert block.dflash_verify_gdr == "mtp"


class MtpBridgeModel(FakeHIAIModel):
    def configure_dflash_verify_gdr(self, route):
        assert route == "mtp"

    def commit_dflash_chunk_state(self, rows):
        result = super().commit_dflash_chunk_state(rows)
        conv, recurrent = result[0]
        return {0: (conv, recurrent + 0.00123)}


def test_mtp_bridge_preserves_fp32_and_reports_selection_after_zero_acceptance():
    model = MtpBridgeModel()
    bridge = InternalDFlashTarget(FakeWrapper(model), device=torch.device("cpu"),
                                 dtype=torch.float16, kv_cache_max_len=192,
                                 rollback_enabled=True, verify_gdr="mtp").eval()
    bridge.begin_rollback(torch.ones(1, 63, dtype=torch.long))
    for rows, accepted in ((4, 0), (16, 14), (3, 2)):
        before = bridge._persistent_cursor
        bridge.verify_rollback(torch.ones(1, rows, dtype=torch.long))
        bridge.commit_rollback(accepted)
        assert bridge._persistent_cursor == before + accepted + 1
        state = bridge._persistent_state[0][1]
        assert state.dtype == torch.float32
        assert not torch.equal(state, state.half().float())
    audit = bridge.dflash_rollback_audit
    assert audit["verify_gdr"] == "mtp" and audit["custom_gdr_mtp_required"]
    assert audit["rollback_gdr_commit_layer_calls"] == 0
    assert audit["rollback_mtp_state_select_layer_calls"] == 3
    bridge.verify_rollback(torch.ones(1, 4, dtype=torch.long))
    bridge.abort_rollback()
    assert bridge._persistent_state is None and model._pending_chunk is None
    bridge.begin_ordinary(torch.ones(1, 2, dtype=torch.long))
    assert bridge._persistent_state[0][1].dtype == torch.float16


@pytest.mark.parametrize("route", ["chunk", "mtp"])
def test_native_cli_forwards_route(monkeypatch, route):
    received = []
    monkeypatch.setattr(run_npu, "_adapter_main", lambda argv: received.extend(argv) or 0)
    assert run_npu.main(["--target-dir", "/model/target", "--draft-dir", "/model/draft",
                         "--prompt-ids", "1,2", "--kv-cache-max-len", "192", "--verify-gdr", route]) == 0
    assert received[received.index("--verify-gdr") + 1] == route
    assert run_rollback._parser().parse_args(received).verify_gdr == route
    options = benchmark_npu._parser().parse_args([
        "--target-dir", "/model/target", "--draft-dir", "/model/draft", "--prompt-ids", "1,2",
        "--mode", "dflash", "--kv-cache-max-len", "192", "--max-new-tokens", "128",
        "--report", "/unused/report.json", "--verify-gdr", route])
    assert options.verify_gdr == route


def test_cpu_mtp_request_rejected_before_loading():
    with pytest.raises(ValueError, match="Torch-NPU"):
        run_rollback._load_transactional_target(SimpleNamespace(device="cpu", verify_gdr="mtp"),
                                                dtype=torch.float16)


@pytest.mark.parametrize("reported_route", ["chunk", "mtp"])
def test_loader_passes_selection_and_rejects_factory_route_mismatch(monkeypatch, reported_route):
    class Target(nn.Module):
        dflash_rollback_audit = {"enabled": True, "historical_prefix_replay_during_verify": False,
                                  "verify_gdr": reported_route}
    for name in ("begin_ordinary", "advance_ordinary", "begin_rollback", "verify_rollback",
                 "commit_rollback", "abort_rollback"):
        setattr(Target, name, lambda *a: None)
    observed = []
    def factory(path, **kwargs):
        observed.append(kwargs)
        return Target().eval()
    monkeypatch.setattr(run_rollback._legacy, "_load_callable", lambda spec: factory)
    monkeypatch.setattr(run_rollback, "torch", SimpleNamespace(device=lambda name: name))
    args = SimpleNamespace(device="npu:0", verify_gdr="mtp", target_factory=None, target_dir="unused")
    if reported_route == "mtp":
        run_rollback._load_transactional_target(args, dtype=torch.float16)
    else:
        with pytest.raises(RuntimeError, match="route differs"):
            run_rollback._load_transactional_target(args, dtype=torch.float16)
    assert observed == [{"device": "npu:0", "dtype": torch.float16, "verify_gdr": "mtp"}]
