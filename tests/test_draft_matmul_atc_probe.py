"""Check probe data flow and dynamic capture; CPU oracle is not CANN evidence."""
from pathlib import Path
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from probe_draft_matmul_atc import make_spec, parser
from weight_quant_test_support import weight_quant_cpu
from test_incremental_air_om import small_threads

pytestmark = pytest.mark.usefixtures("weight_quant_cpu", "small_threads")


@pytest.mark.parametrize("bits", [4, 8])
def test_probe_keeps_flat_compressed_inputs_and_dynamic_gears(bits):
    spec = make_spec(bits, "tiny", "cpu")
    x, packed, scales = spec.example_args
    assert packed.shape == (64 * 256 * bits // 8,) and scales.shape == (64 * 2,)
    assert not any(buffer.numel() for buffer in spec.model.buffers())
    exported = torch.export.export(spec.model, spec.example_args, dynamic_shapes=(
        {0: torch.export.Dim("rows", min=16, max=64)}, None, None)).module()
    for rows in (16, 64):
        value = torch.ones(rows, 256).half() / 16
        # Runtime inputs must remain live and cannot be frozen into the AIR.
        output = exported(value, packed, scales)
        torch.testing.assert_close(output, spec.model(value, packed, scales), rtol=0, atol=0)
        zeros = torch.full_like(packed, 0x88 if bits == 4 else 0)
        assert torch.count_nonzero(exported(value, zeros, scales)) == 0
        assert torch.count_nonzero(output) > 0


def test_probe_cli_defaults_to_small_two_bitwidth_test():
    args = parser().parse_args(["--output-dir", "probe", "--atc", "/bin/true", "--soc-version", "Ascend310P3"])
    assert args.bits == [4, 8] and args.projection == ["tiny"]


def test_probe_reports_export_stage_and_traceback_without_invoking_atc(monkeypatch, tmp_path):
    import probe_draft_matmul_atc as probe
    npu = ModuleType("torch_npu"); npu.__version__ = "test-double"
    monkeypatch.setitem(sys.modules, "torch_npu", npu)
    monkeypatch.setitem(sys.modules, "torchair", ModuleType("torchair"))
    monkeypatch.setattr(torch, "npu", SimpleNamespace(set_device=lambda _: None,
                        get_device_name=lambda _: "fake", empty_cache=lambda: None), raising=False)
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    monkeypatch.setattr(probe, "make_spec", lambda bits, *args: bits)
    calls = []
    def export(factory, config, directory):
        if factory(config) == (4,):
            raise ValueError("descriptor conflict")
        return {"manifest_path": "air.json", "graphs": [{"runtime_input_abi": {"weight_quant_layout": {}}}]}
    def compile_air(*args, **kwargs):
        calls.append(args)
        return {"manifest_path": "deployment.json"}
    monkeypatch.setattr(probe, "export_air_bundle", export)
    monkeypatch.setattr(probe, "compile_air_bundle", compile_air)
    output = tmp_path / "probe"
    assert probe.main(["--output-dir", str(output), "--atc", "/bin/true", "--soc-version", "Ascend310P3"]) == 1
    report = json.loads((output / "summary.json").read_text())
    assert report["status"] == "FAIL" and len(calls) == 1
    assert [(c["status"], c["phase"]) for c in report["cases"]] == [("FAIL", "export"), ("PASS", "complete")]
    assert "ValueError: descriptor conflict" in Path(report["cases"][0]["traceback"]).read_text()
