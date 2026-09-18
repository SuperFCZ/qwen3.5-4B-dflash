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
@pytest.mark.parametrize("group_size", [0, 128])
@pytest.mark.parametrize("layout", ["nk", "kn"])
def test_probe_keeps_flat_compressed_inputs_and_dynamic_gears(bits, group_size, layout):
    spec = make_spec(bits, "tiny", "cpu", group_size, layout)
    x, packed, scales = spec.example_args
    groups = 256 // group_size if group_size else 1
    assert packed.shape == (64 * 256 * bits // 8,) and scales.shape == (64 * groups,)
    assert spec.metadata["synthetic_control"] is (group_size == 0)
    assert spec.metadata["weight_quant_probe"] == {"group_size": group_size, "weight_layout": layout}
    assert not any(buffer.numel() for buffer in spec.model.buffers())
    generator = torch.Generator().manual_seed(812 + bits)
    q = torch.randint(-(1 << (bits - 1)), 1 << (bits - 1), (64, 256),
                      generator=generator, dtype=torch.int8)
    dense = (q.float() * scales.reshape(64, groups).float().repeat_interleave(group_size or 256, 1)).half()
    exported = torch.export.export(spec.model, spec.example_args, dynamic_shapes=(
        {0: torch.export.Dim("rows", min=16, max=64)}, None, None)).module()
    # A [1,N] row broadcasts numerically on CPU but is rejected by the
    # receiver's per-channel GE tiler. Assert the actual captured op operand.
    native = next(node for node in exported.graph.nodes if node.op == "call_function"
                  and node.target == torch.ops.npu.npu_weight_quant_batchmatmul.default)
    assert tuple(native.args[2].meta["val"].shape) == ((groups, 64) if group_size else (64,))
    for rows in (16, 64):
        value = torch.ones(rows, 256).half() / 16
        # Runtime inputs must remain live and cannot be frozen into the AIR.
        output = exported(value, packed, scales)
        torch.testing.assert_close(output, value @ dense.t(), rtol=0, atol=0)
        torch.testing.assert_close(exported(value, packed, scales * 2), value @ (dense * 2).t(), rtol=0, atol=0)
        zeros = torch.full_like(packed, 0x88 if bits == 4 else 0)
        assert torch.count_nonzero(exported(value, zeros, scales)) == 0
        assert torch.count_nonzero(output) > 0


def test_probe_cli_defaults_to_small_two_bitwidth_test():
    args = parser().parse_args(["--output-dir", "probe", "--atc", "/bin/true", "--soc-version", "Ascend310P3"])
    assert args.bits == [4, 8] and args.projection == ["tiny"]
    assert args.group_size == [128] and args.weight_layout == ["nk"]


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


@pytest.mark.parametrize("group,layout", [(1, "nk"), (128, "nz")])
def test_probe_rejects_unknown_controls(group, layout):
    with pytest.raises(ValueError, match="probe controls"):
        make_spec(8, "tiny", "cpu", group, layout)


def test_probe_runs_every_selected_control_after_compile_failure(monkeypatch, tmp_path, capsys):
    import probe_draft_matmul_atc as probe
    npu = ModuleType("torch_npu"); npu.__version__ = "test-double"
    monkeypatch.setitem(sys.modules, "torch_npu", npu)
    monkeypatch.setitem(sys.modules, "torchair", ModuleType("torchair"))
    monkeypatch.setattr(torch, "npu", SimpleNamespace(set_device=lambda _: None,
                        get_device_name=lambda _: "fake", empty_cache=lambda: None), raising=False)
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    monkeypatch.setattr(probe, "make_spec", lambda bits, projection, device, group, layout: (group, layout))
    directories = []
    def export(factory, config, directory):
        group, layout = factory(config)[0]
        directories.append(directory)
        return {"manifest_path": f"{group}-{layout}", "graphs": [{"runtime_input_abi": {"weight_quant_layout": {}}}]}
    def compile_air(manifest, **kwargs):
        if manifest.startswith("128-"):
            raise RuntimeError("no valid template is found")
        return {"manifest_path": "deployment.json"}
    monkeypatch.setattr(probe, "export_air_bundle", export)
    monkeypatch.setattr(probe, "compile_air_bundle", compile_air)
    output = tmp_path / "probe"
    assert probe.main(["--output-dir", str(output), "--atc", "/bin/true", "--soc-version", "Ascend310P3",
                       "--bits", "8", "--group-size", "0", "128", "--weight-layout", "nk", "kn"]) == 1
    report = json.loads((output / "summary.json").read_text())
    assert len(directories) == len(set(directories)) == 4
    assert [(c["group_size"], c["weight_layout"], c["status"]) for c in report["cases"]] == [
        (0, "nk", "PASS"), (0, "kn", "PASS"), (128, "nk", "FAIL"), (128, "kn", "FAIL")]
    assert all(c["phase"] == "compile" and Path(c["traceback"]).is_file()
               for c in report["cases"] if c["status"] == "FAIL")
    assert "| 8 | tiny | 128 | KN | FAIL | compile |" in capsys.readouterr().out


def test_model_export_cannot_enable_synthetic_perchannel_control(monkeypatch, tmp_path):
    from dataclasses import replace
    from qwen35_dflash.ascend310p.exporter import export_air_bundle
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    diagnostic = make_spec(8, "tiny", "cpu", 0, "nk")
    for name, role in (("draft", "diagnostic"), ("weight_quant_probe", "draft")):
        spec = replace(diagnostic, name=name, role=role)
        with pytest.raises(ValueError, match="cannot be applied to model graphs"):
            export_air_bundle(lambda _: (spec,), {}, tmp_path / name,
                              torchair_module=SimpleNamespace())
