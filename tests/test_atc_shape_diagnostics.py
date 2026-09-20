"""Compiler-dump parsing and subprocess isolation; no CANN simulation claim."""
import json
import os
import subprocess
import sys

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory, text_format
import pytest

from qwen35_dflash.ascend310p.atc_diagnostics import (
    AtcShapeDiagnostics, diagnostic_summary, weight_quant_inference_lines, weight_quant_snapshot,
)
from qwen35_dflash.ascend310p.compiler import _atc_failure_detail
from test_weight_quant_layout import fixture_graph


def dump_model():
    """Synthetic GE-shaped protobuf dump, explicitly not produced by ATC."""
    graph = fixture_graph()
    descriptor = descriptor_pb2.FileDescriptorProto.FromString(graph.DESCRIPTOR.file.serialized_pb)
    graph_def = next(msg for msg in descriptor.message_type if msg.name == "Graph")
    graph_def.field.add(name="name", number=2, type=9, label=1)
    model = descriptor.message_type.add(name="ModelDef")
    model.field.add(name="graph", number=1, type=11, type_name=".Graph", label=3)
    pool = descriptor_pool.DescriptorPool(); pool.Add(descriptor)
    model_type = message_factory.GetMessageClass(pool.FindMessageTypeByName("ModelDef"))
    model = model_type()
    subgraph = model.graph.add()
    subgraph.ParseFromString(graph.SerializeToString()); subgraph.name = "Batch_0"
    node = next(op for op in subgraph.op if op.name == "quant")
    node.name = "WeightQuantBatchMatmulV2_ascend_mbatch_batch_0"
    # Deliberately retain a conflicting origin shape to verify that the
    # collector reports it rather than repairing it or trusting a sidecar.
    node.input_desc[1].shape.dim[:] = [8, 4, 16, 32]
    node.input_desc[1].layout = "FRACTAL_NZ"
    node.input_desc[1].attr["origin_shape"].list.i[:] = [8, 4, 16, 32]
    weight = next(op for op in subgraph.op if op.name == "wt")
    weight.type = "Const"
    weight.attr["_out_shape_locked"].b = True
    weight.output_desc[0].CopyFrom(node.input_desc[1])
    weight.attr["value"].t.desc.shape.dim[:] = [64, 256]
    weight.attr["value"].t.desc.layout = "ND"
    weight.attr["value"].t.data = b"MUST_NOT_APPEAR_IN_DIAGNOSTIC_JSON"
    return model


def test_snapshot_preserves_conflicting_metadata_and_omits_payloads():
    model = dump_model()
    before = model.SerializeToString(deterministic=True)
    nodes = weight_quant_snapshot(model.graph[0])
    assert nodes[0]["inputs"][1]["origin_shape"] == [8, 4, 16, 32]
    assert nodes[0]["inputs"][1]["dtype"] == "DT_INT8"
    producer = next(node for node in nodes[0]["producers"] if node["name"] == "wt")
    assert producer["value_descriptor"]["shape"] == [64, 256]
    assert producer["_out_shape_locked"] is True
    assert "_out_shape_locked" not in nodes[0]
    assert "MUST_NOT_APPEAR" not in json.dumps(nodes)
    assert model.SerializeToString(deterministic=True) == before


def test_primary_error_before_generic_atc_summary_is_not_lost():
    stdout = ("[ERROR] WeightQuantBatchMatmulV2 Ka[256] != Kb[32]\n" + "noise\n" * 100
              + "[PID:1] Inner_Error_Compile_Fail(E90003): The Shape Check failed\n"
              + "Call InferShapeForWeightQuantBatchMatmulV2 failed\n"
              + "[ERROR][PID:2] WeightQuantBatchMatmulV2 Ka[256] != Kb[32]\n")
    assert weight_quant_inference_lines(stdout) == ["Ka[256] != Kb[32]"]
    detail = _atc_failure_detail(stdout, {})
    assert detail.index("Ka[256] != Kb[32]") < detail.index("ATC diagnostic:")
    assert "diagnose-prepack" not in detail
    detail = _atc_failure_detail("WeightQuantBatchMatmulV2: The Shape Check failed", {})
    assert "diagnose-prepack" in detail
    assert "Kb[32]" not in detail  # absence of evidence is not a diagnosed shape


@pytest.mark.parametrize("pass_log", [
    "Register graph fusion pass WeightQuantBatchMatmulV2TransposeNZFusionPass",
    "Run graph fusion pass [WeightQuantBatchMatmulV2TransposeNZFusionPass] successfully",
    "WeightQuantBatchMatmulV2TransposeNZFusionPass is off",
])
def test_debug_pass_mentions_do_not_misdiagnose_shape_failure(pass_log):
    log = (pass_log + "\n[ERROR] WeightQuantBatchMatmulV2 Ka[256] != Kb[32]\n"
           "x_shape: [16, 256], weight_shape: [8, 4, 16, 32]\n"
           "InferShapeForWeightQuantBatchMatmulV2: The Shape Check failed")
    detail = _atc_failure_detail(log, {})
    assert "Ka[256] != Kb[32]" in detail
    assert "transpose/NZ graph fusion failed" not in detail
    assert "off switch" not in detail


@pytest.mark.parametrize("failure", [
    "Graph fusion pass WeightQuantBatchMatmulV2TransposeNZFusionPass failed.",
    "Failed to run graph fusion pass [WeightQuantBatchMatmulV2TransposeNZFusionPass, built-in-ai-core-graph-pass]",
])
def test_actual_fusion_failure_is_still_diagnosed(failure):
    assert "transpose/NZ graph fusion failed" in _atc_failure_detail(failure, {})


def test_real_subprocess_scopes_debug_env_and_retains_failed_graph(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("DUMP_GE_GRAPH", "3")
    monkeypatch.setenv("DUMP_GRAPH_PATH", str(tmp_path / "untouched"))
    monkeypatch.setenv("NPU_COLLECT_PATH", str(tmp_path / "untouched"))
    monkeypatch.delenv("IGNORE_INFER_ERROR", raising=False)
    parent_env = dict(os.environ)
    model = dump_model()
    dump_text = text_format.MessageToString(model)
    code = ("import os, pathlib, sys\n"
            "assert os.environ['DUMP_GE_GRAPH'] == '2'\n"
            "assert 'NPU_COLLECT_PATH' not in os.environ\n"
            "assert sys.argv[-1] == '--log=debug'\n"
            "p = pathlib.Path(os.environ['DUMP_GRAPH_PATH']) / 'pid_123_deviceid_0'\n"
            "p.mkdir()\n"
            f"(p/'ge_proto_00001_InferShapeBlackBox.txt').write_text({dump_text!r})\n"
            "print('WeightQuantBatchMatmulV2 Ka[256] != Kb[32]')\n"
            "sys.exit(255)\n")
    diag = AtcShapeDiagnostics(tmp_path / "case")
    result = diag([sys.executable, "-c", code], tmp_path)
    assert result.returncode == 255 and "Ka[256]" in result.stdout
    assert dict(os.environ) == parent_env
    assert not (tmp_path / "untouched").exists()
    report = diag.collect(model_type=type(model))
    assert report["status"] == "CAPTURED"
    assert report["snapshots"][0]["failure_dump"] is True
    case = dict(name="test", status="FAIL", phase="compile", shape_diagnostics=report,
                shape_diagnostics_path=str(diag.report_path))
    summary = diagnostic_summary(case)
    assert "origin=[8, 4, 16, 32]" in summary and "value: shape=[64, 256] format=ND" in summary
    assert "shape_locked=True" in summary and "shape_locked=False" in summary
    assert "MUST_NOT_APPEAR" not in diag.report_path.read_text()
    assert (diag.root / "command.json").is_file()
    with pytest.raises(ValueError, match="separate directory"):
        diag([sys.executable, "-c", "pass"], tmp_path)


def test_unparseable_dump_is_retained_and_cannot_be_reported_as_captured(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    diag = AtcShapeDiagnostics(tmp_path / "case")
    (diag.root / "graphs").mkdir()
    raw = diag.root / "graphs/ge_proto_InferShapeBlackBox.txt"
    raw.write_text("not a protobuf descriptor")
    report = diag.collect(model_type=type(dump_model()))
    assert report["status"] == "NO_WEIGHTQUANT_SNAPSHOT"
    assert report["graph_files"][0]["status"] == "PARSE_FAILED"
    assert raw.read_text() == "not a protobuf descriptor"
    assert not report["snapshots"]


@pytest.mark.parametrize("value", ["1", "0", "false"])
def test_diagnostic_runner_keeps_inference_errors_enabled(monkeypatch, tmp_path, value):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("IGNORE_INFER_ERROR", value)
    def unexpected(*args, **kwargs):
        pytest.fail("must not invoke ATC with inference validation disabled")
    monkeypatch.setattr(subprocess, "run", unexpected)
    diag = AtcShapeDiagnostics(tmp_path / "case")
    with pytest.raises(ValueError, match="unset IGNORE_INFER_ERROR"):
        diag(["atc"], tmp_path)
