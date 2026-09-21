"""Local metadata extraction, including non-disclosure on parser failures."""
import json

from google.protobuf import text_format

from test_atc_shape_diagnostics import dump_model
from tools import extract_atc_node as extract


SECRET = "PRIVATE_TENSOR_BYTES_MUST_STAY_LOCAL"


def conversion_model():
    model = dump_model()
    graph = model.graph[0]
    const = next(node for node in graph.op if node.name == "wt")
    const.attr["value"].t.data = SECRET.encode()
    const.attr["unrelated_attribute"].s = SECRET.encode()
    const.output_desc[0].attr["private_attribute"].s = SECRET.encode()
    graph.op.add(name="UNRELATED_MODEL_NODE", type="SomePrivateOp")
    node = graph.op.add(name="trans_TransData_1", type="TransData", input=["wt:0"])
    node.input_desc.add().CopyFrom(const.output_desc[0])
    node.input_desc[0].shape.dim[:] = [64, 256]  # reproduce observed rank mismatch
    node.output_desc.add().CopyFrom(node.input_desc[0])
    node.output_desc[0].layout = "ND"
    node.attr["src_format"].s = b"FRACTAL_NZ"
    node.attr["dst_format"].s = b"ND"
    consumer = next(node for node in graph.op if node.type == "WeightQuantBatchMatmulV2")
    consumer.input[1] = "trans_TransData_1:0"
    return model


def test_neighborhood_keeps_shape_conflict_without_payload_or_whole_graph():
    model = conversion_model()
    before = model.SerializeToString(deterministic=True)
    snap = extract.node_snapshot(model.graph[0], "trans_TransData_1")
    assert snap["node"]["inputs"]["0"]["shape"] == [64, 256]
    assert snap["node"]["src_format"] == "FRACTAL_NZ"
    assert snap["node"]["dst_format"] == "ND"
    producer = snap["producers"][0]
    assert producer["to_input"] == 0 and producer["from_output"] == 0
    assert producer["node"]["outputs"]["0"]["shape"] == [8, 4, 16, 32]
    assert producer["node"]["value_descriptor"]["shape"] == [64, 256]
    assert producer["node"]["_out_shape_locked"] is True
    consumer = snap["consumers"][0]
    assert consumer["to_input"] == 1
    assert set(consumer["node"]["inputs"]) == {"1"}
    encoded = json.dumps(snap)
    assert SECRET not in encoded and "UNRELATED_MODEL_NODE" not in encoded
    assert "private_attribute" not in encoded and "unrelated_attribute" not in encoded
    assert "perm" not in encoded  # no recursive producer traversal
    assert model.SerializeToString(deterministic=True) == before


def test_repeated_stages_are_compacted_and_limit_keeps_latest_change(tmp_path):
    graphs = tmp_path / "graphs"
    graphs.mkdir()
    model = conversion_model()
    first = text_format.MessageToString(model)
    (graphs / "ge_proto_00001_Before.txt").write_text(first)
    node = next(node for node in model.graph[0].op if node.type == "TransData")
    node.input_desc[0].shape.dim[:] = [8, 4, 16, 32]
    after = text_format.MessageToString(model)
    (graphs / "ge_proto_00002_After.txt").write_text(after)
    (graphs / "ge_proto_00003_Final.txt").write_text(after)
    report = extract.collect(tmp_path, node.name, limit=1, model_type=type(model))
    assert report["status"] == "CAPTURED" and report["parsed"] == 3
    assert len(report["snapshots"]) == 1 and report["omitted_snapshots"] == 1
    snapshot = report["snapshots"][0]
    assert snapshot["first_stage"] == "ge_proto_00002_After.txt"
    assert snapshot["last_stage"] == "ge_proto_00003_Final.txt"
    assert snapshot["node"]["inputs"]["0"]["shape"] == [8, 4, 16, 32]
    assert SECRET not in extract.format_report(report)
    assert (graphs / "ge_proto_00001_Before.txt").read_text() == first


def test_malformed_dump_never_exports_parser_source_line(tmp_path):
    (tmp_path / "graphs").mkdir()
    (tmp_path / "graphs/ge_proto_bad.txt").write_text('graph { name: "' + SECRET)
    report = extract.collect(tmp_path, "trans_TransData_1", model_type=type(conversion_model()))
    assert report["status"] == "NO_PARSED_DUMPS"
    assert report["errors"] == [{"stage": "ge_proto_bad.txt", "error_type": "ParseError"}]
    assert SECRET not in json.dumps(report) + extract.format_report(report)


def test_missing_node_reports_names_only_without_falling_back_to_full_graph(tmp_path):
    (tmp_path / "graphs").mkdir()
    model = conversion_model()
    (tmp_path / "graphs/ge_proto_model.txt").write_text(text_format.MessageToString(model))
    report = extract.collect(tmp_path, "absent", model_type=type(model))
    assert report["status"] == "NODE_NOT_FOUND" and report["snapshots"] == []
    assert report["transdata_names"] == ["trans_TransData_1"]
    text = extract.format_report(report)
    assert SECRET not in text and "UNRELATED_MODEL_NODE" not in text


def test_large_fanout_is_explicitly_limited():
    model = conversion_model()
    for index in range(10):
        node = model.graph[0].op.add(name=f"extra_{index}", type="Identity", input=["trans_TransData_1:0"])
        node.input_desc.add().shape.dim[:] = [64, 256]
    snap = extract.node_snapshot(model.graph[0], "trans_TransData_1")
    assert len(snap["consumers"]) == 4 and snap["consumers_omitted"] == 7
    assert "extra_9" not in json.dumps(snap)


def test_cli_failure_does_not_echo_exception_payload(monkeypatch, capsys, tmp_path):
    def fail(*args, **kwargs):
        raise ValueError(SECRET)
    monkeypatch.setattr(extract, "collect", fail)
    assert extract.main(["--diagnostics-dir", str(tmp_path)]) == 1
    out = capsys.readouterr()
    assert SECRET not in out.err + out.out
    assert "ValueError" in out.err
