"""Cross-route ordinary reuse; fake ACL establishes scheduling, not performance."""
import argparse
import copy
import json
from pathlib import Path

import pytest

from tools import benchmark_gdr_lengths as matrix, benchmark_prompts as suite, ordinary_baseline as baseline
from qwen35_dflash.ascend310p.cpp_runtime import validate_cpp_runner_report
from test_gdr_length_matrix import matrix_args, small_threads, adn_rms_norm_cpu  # noqa: F401

pytestmark = pytest.mark.usefixtures("adn_rms_norm_cpu")


def execute(args, monkeypatch):
    monkeypatch.delenv("ASCEND310P_SIMULATION_ONLY", raising=False)
    monkeypatch.delenv("PROFILING_MODE", raising=False)
    monkeypatch.setenv("QWEN35_FAKE_ACCEPT", "15")
    assert matrix.run(args) == 1  # Fake ACL must never pass device measurement gates.
    root, = args.run_dir.glob("gdr-lengths-*")
    return root, json.loads((root / "summary.json").read_text())


@pytest.mark.parametrize("low_memory", [True, False])
@pytest.mark.parametrize("matrix_args", [128, 256], indirect=True)
def test_two_routes_execute_ordinary_once_per_budget(matrix_args, monkeypatch, low_memory):
    matrix_args.low_memory = low_memory
    matrix_args.lengths = [20, 32]
    matrix_args.warmup, matrix_args.repetitions = 1, 3
    events, loads = matrix_args.run_dir / "events.jsonl", matrix_args.run_dir / "loads.jsonl"
    monkeypatch.setenv("QWEN35_FAKE_EVENT_LOG", str(events))
    monkeypatch.setenv("QWEN35_FAKE_WORKSPACE_LOG", str(loads))
    root, summary = execute(matrix_args, monkeypatch)
    # Includes warmups: a second ordinary loop would double this count.
    calls = [json.loads(line)[0] for line in events.read_text().splitlines()]
    assert calls.count("target_decode") == (19 + 31) * 4
    assert [json.loads(line)[0] for line in loads.read_text().splitlines()].count("target_decode") == 2
    for first, second in zip(summary["cells"][::2], summary["cells"][1::2]):
        assert first["ordinary_baseline_reused"] is False
        assert second["ordinary_baseline_reused"] is True
        assert first["ordinary_baseline"] == second["ordinary_baseline"]
        first_root, second_root = Path(first["summary"]).parent, Path(second["summary"]).parent
        before = (first_root / "runner-batch.json.cases/p.json").read_bytes()
        ordinary = json.loads(before)["ordinary"]
        request = json.loads((second_root / "request.json").read_text())
        report = json.loads((second_root / "runner-batch.json.cases/p.json").read_text())
        raw = json.loads((second_root / "runner-dflash.json.cases/p.json").read_text())
        assert "--mode" in request["command"] and baseline.argument(request, "--mode") == "dflash"
        assert "ordinary" not in raw and raw["ordinary_parity"]["status"] == "NOT_RUN"
        assert report["ordinary"] == ordinary
        assert report["dflash"] == raw["dflash"]
        stats = suite.summarize_prompt(report)
        assert stats["ordinary_total_measured_ms"] == sum(
            m["latency_ms"]["model_total"] for m in ordinary["measurements"])
        assert suite.stage_timings(report)["ordinary_decode"]["calls"] == (second["max_new_tokens"] - 1) * 3
        assert report["protocol"]["order"] == baseline.ORDER
        assert report["protocol"]["max_resident_models"] == 3
        assert report["formal_latency_evidence"] is False
        assert not any("target_decode" in m["stage_ms"] for m in raw["dflash"]["measurements"])
        validate_cpp_runner_report(report, prompt_token_ids=[4, 5],
            om_sha256=baseline.argument(request, "--model-sha256"), device_id=0,
            max_new_tokens=second["max_new_tokens"], max_draft_tokens=15, chunk_abi=True,
            low_memory=low_memory, verify_gdr="mtp", warmup=1, repetitions=3)
        baseline.validate_saved(second_root / "runner-batch.json", request)
        assert (first_root / "runner-batch.json.cases/p.json").read_bytes() == before
    # Reanalysis retains the comparison and still refuses fake device evidence.
    last = Path(summary["cells"][-1]["summary"]).parent
    options = argparse.Namespace(run_dir=matrix_args.run_dir, summarize_existing=last / "runner-batch.json",
                                model_dir=None, allow_output_differences=True)
    assert suite.summarize_existing(options) == 1
    # Input, protocol and ordinary graph identities cannot be mixed.
    request = json.loads((last / "request.json").read_text())
    source = baseline.read_locked(request["ordinary_baseline"]["request"])
    derived_path = last / "runner-batch.json.cases/p.json"
    saved = derived_path.read_bytes()
    damaged_report = json.loads(saved)
    damaged_report["ordinary"]["measurements"][0]["latency_ms"]["model_total"] += 1
    derived_path.write_text(json.dumps(damaged_report))
    with pytest.raises(ValueError, match="comparison differs from source reports"):
        baseline.validate_saved(last / "runner-batch.json", request)
    derived_path.write_bytes(saved)
    for key in ("max_new_tokens", "repetitions", "warmup", "eos_token_ids", "runtime_identity",
                "ordinary_contract", "runner", "tokenizer_source"):
        damaged = copy.deepcopy(request)
        damaged[key] = None
        with pytest.raises(ValueError, match="ordinary baseline differs"):
            baseline.compatible(source, damaged)
    damaged = copy.deepcopy(request)
    damaged["prompts"][0]["prompt_token_ids"] = [8, 9]
    with pytest.raises(ValueError, match="prompt selection"):
        baseline.compatible(source, damaged)
    # Saved evidence is immutable, including the ordinary timing used for speedup.
    ref = request["ordinary_baseline"]["reports"]["p"]
    Path(ref["path"]).write_bytes(Path(ref["path"]).read_bytes() + b" ")
    with pytest.raises(ValueError, match="source changed"):
        baseline.validate_saved(last / "runner-batch.json", request)


def test_ordinary_reused_despite_first_route_parity_failure(matrix_args, monkeypatch):
    matrix_args.lengths = [20]
    matrix_args.warmup, matrix_args.repetitions = 0, 1
    monkeypatch.setenv("QWEN35_FAKE_VERIFY_DRIFT_ROW", "1")
    _, summary = execute(matrix_args, monkeypatch)
    first, second = [Path(c["summary"]).parent for c in summary["cells"]]
    a = json.loads((first / "runner-batch.json.cases/p.json").read_text())
    b = json.loads((second / "runner-batch.json.cases/p.json").read_text())
    assert a["failure_stage"] == b["failure_stage"] == "ordinary_dflash_parity"
    assert a["ordinary"] == b["ordinary"]
    assert b["ordinary_parity"]["first_difference"]["ordinary_location"] is not None
    assert b["ordinary_parity"]["first_difference"]["dflash_location"] is not None


def test_baseline_drift_is_preserved(matrix_args, monkeypatch):
    matrix_args.lengths = [20]
    matrix_args.warmup, matrix_args.repetitions = 0, 3
    monkeypatch.setenv("QWEN35_FAKE_PREFILL_DRIFT_CALL", "2")
    _, summary = execute(matrix_args, monkeypatch)
    first, second = [Path(c["summary"]).parent for c in summary["cells"]]
    a = json.loads((first / "runner-batch.json.cases/p.json").read_text())
    b = json.loads((second / "runner-batch.json.cases/p.json").read_text())
    assert a["ordinary"] == b["ordinary"]
    assert b["ordinary"]["repeatability"]["status"] == "DRIFT_OBSERVED"
    assert b["status"] == "PASS_WITH_OBSERVATIONS"


def test_different_ordinary_om_fails_before_execution(matrix_args, monkeypatch):
    manifest = matrix_args.mtp_deployment_manifest
    payload = json.loads(manifest.read_text())
    graph = next(g for g in payload["graphs"] if g["name"] == "target_decode")
    path = (manifest.parent / graph["om"]["path"]).resolve()
    path.write_bytes(path.read_bytes() + b"\n")
    graph["om"].update(sha256=baseline.sha256_file(path), bytes=path.stat().st_size)
    manifest.write_text(json.dumps(payload))
    monkeypatch.setattr(suite, "run", lambda *_: pytest.fail("ordinary mismatch must fail preflight"))
    with pytest.raises(ValueError, match="ordinary Prefill/Decode OM"):
        matrix.run(matrix_args)


def test_missing_first_baseline_never_reruns_ordinary(matrix_args, monkeypatch):
    matrix_args.lengths = [20]
    calls = []
    def failed_run(args):
        calls.append(args.verify_gdr)
        raise RuntimeError("ordinary process failed before measurements")
    monkeypatch.setattr(suite, "run", failed_run)
    assert matrix.run(matrix_args) == 1
    assert calls == ["chunk"]
    root, = matrix_args.run_dir.glob("gdr-lengths-*")
    summary = json.loads((root / "summary.json").read_text())
    assert "refusing to rerun ordinary" in summary["cells"][1]["error"]


def test_ordinary_survives_first_route_dflash_runtime_failure(matrix_args, monkeypatch):
    matrix_args.lengths = [20]
    matrix_args.warmup, matrix_args.repetitions = 0, 1
    original = suite.run
    def run(options):
        if options.verify_gdr == "chunk":
            monkeypatch.setenv("QWEN35_FAKE_FAIL_GRAPH", "draft")
        else:
            monkeypatch.delenv("QWEN35_FAKE_FAIL_GRAPH")
        return original(options)
    monkeypatch.setattr(suite, "run", run)
    _, summary = execute(matrix_args, monkeypatch)
    first, second = [Path(c["summary"]).parent for c in summary["cells"]]
    a = json.loads((first / "runner-batch.json.cases/p.json").read_text())
    b = json.loads((second / "runner-batch.json.cases/p.json").read_text())
    assert a["failure_stage"] == "dflash_benchmark"
    assert b["status"] == "PASS" and b["ordinary"] == a["ordinary"]


def test_offline_questions_share_baseline_and_keep_early_eos(matrix_args, monkeypatch):
    data = matrix_args.run_dir / "questions.jsonl"
    data.write_text("".join(json.dumps({"question": str(i)}) + "\n" for i in range(66)))
    matrix_args.dataset_files, matrix_args.prompts = [data], None
    matrix_args.lengths, matrix_args.eos_token_id = [20], [7]
    matrix_args.warmup, matrix_args.repetitions = 0, 1
    events = matrix_args.run_dir / "events.jsonl"
    monkeypatch.setenv("QWEN35_FAKE_EVENT_LOG", str(events))
    _, summary = execute(matrix_args, monkeypatch)
    assert len(summary["prompts"]) == 66
    assert [json.loads(line)[0] for line in events.read_text().splitlines()].count("target_decode") == 66
    first, second = [Path(c["summary"]).parent for c in summary["cells"]]
    for p in summary["prompts"]:
        name = p["id"]
        a = json.loads((first / f"runner-batch.json.cases/{name}.json").read_text())
        b = json.loads((second / f"runner-batch.json.cases/{name}.json").read_text())
        assert a["ordinary"] == b["ordinary"]
        assert b["ordinary"]["totals"]["generated_tokens"] == 2
        assert b["ordinary"]["stable_stop_reason"] == "eos"
    request = json.loads((second / "request.json").read_text())
    assert request["datasets"][0]["selected_samples"] == 66
