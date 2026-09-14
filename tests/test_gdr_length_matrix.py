"""Length/route orchestration, counting and fake-ACL rejection; host evidence only."""
import argparse
import copy
import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools import benchmark_gdr_lengths as matrix
from qwen35_dflash.ascend310p.utils import sha256_file
from test_incremental_air_om import chunk_bundle, small_threads  # noqa: F401
from rms_norm_test_support import adn_rms_norm_cpu  # noqa: F401

pytestmark = pytest.mark.usefixtures("adn_rms_norm_cpu")


@pytest.fixture
def matrix_args(tmp_path, monkeypatch, request):
    from qwen35_dflash.ascend310p import workflow
    import test_incremental_air_om as fixtures

    capacity = getattr(request, "param", 128)
    original_specs = fixtures.incremental_graph_specs
    def sized_attention(*, query, key, value, atten_mask, **kwargs):
        q = query.reshape(1, 2, query.shape[-2], 16)
        k, v = (item[0].permute(0, 2, 1, 3).reshape(1, 1, -1, 16).expand(1, 2, -1, 16)
                for item in (key, value))
        scores = q.float() @ k.float().transpose(-1, -2) * .25 + atten_mask.float()
        return (scores.softmax(-1) @ v.float()).half().reshape_as(query)
    def sized_specs(target, *args, **kwargs):
        target.kv_cache_max_len = capacity + 64
        attention = target.dflash_execution_model.language_model.layers[1].self_attn
        attention.kv_max_len = capacity + 64
        attention.block_table = torch.arange((capacity + 64) // 64, dtype=torch.int32)[None]
        original_cache = target._fresh_hybrid_cache
        def sized_cache(batch_size):
            cache = original_cache(batch_size)
            cache[1] = tuple(t.new_zeros(((capacity + 64) // 64, *t.shape[1:])) for t in cache[1])
            return cache
        target._fresh_hybrid_cache = sized_cache
        return original_specs(target, *args, **dict(kwargs, capacity=capacity, attention=sized_attention))
    monkeypatch.setattr(fixtures, "incremental_graph_specs", sized_specs)

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return [4, 5]

        def decode(self, tokens, **kwargs):
            return str(tokens)

    runner = os.environ.get("QWEN35_CPP_TEST_RUNNER")
    if not runner:
        pytest.skip("fake ACL binary required for orchestration checks")
    manifests = {}
    for route in matrix.ROUTES:
        output = tmp_path / route
        output.mkdir()
        manifests[route] = chunk_bundle.__wrapped__(output, monkeypatch, SimpleNamespace(param=route))
    config = tmp_path / "runner.json"
    config.write_text(json.dumps(dict(device_model="Ascend310P3-host-fixture", cann="fake",
                                     driver="fake", firmware="fake", runtime="fake-acl")))
    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps([{"id": "p", "prompt": "test"}]))
    monkeypatch.setattr(workflow, "load_tokenizer", lambda **kwargs: (Tokenizer(), "host-test-tokenizer"))
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    return argparse.Namespace(
        run_dir=tmp_path, runner=Path(runner), runner_config=config, model_dir=tmp_path,
        chunk_deployment_manifest=manifests["chunk"], mtp_deployment_manifest=manifests["mtp"],
        lengths=[32, 64], prompts=prompts, prompt_id=None, chat=True, eos_token_id=[63],
        device_id=0, max_draft_tokens=15, low_memory=True, allow_output_differences=True,
        plan_only=False)


def no_execution(*args):
    raise AssertionError("preflight must not launch any device suite")


def test_default_grid_includes_512_and_1024():
    args = matrix.parser().parse_args([
        "--run-dir", "/run", "--runner", "/runner", "--model-dir", "/model",
        "--chunk-deployment-manifest", "/chunk.json", "--mtp-deployment-manifest", "/mtp.json"])
    assert args.lengths == [32, 64, 128, 256, 512, 1024]


def test_plan_only_checks_both_routes_and_all_budgets(matrix_args, monkeypatch):
    matrix_args.plan_only = True
    monkeypatch.setattr(matrix.suite, "run", no_execution)
    assert matrix.run(matrix_args) == 0
    output, = matrix_args.run_dir.glob("gdr-lengths-*")
    report = json.loads((output / "summary.json").read_text())
    assert report["status"] == "PREPARED"
    assert [(r["verify_gdr"], r["max_new_tokens"]) for r in report["cells"]] == [
        ("chunk", 32), ("mtp", 32), ("chunk", 64), ("mtp", 64)]
    assert report["bundles"]["chunk"]["abi"] != report["bundles"]["mtp"]["abi"]
    assert all(c["status"] == "NOT_RUN" for c in report["cells"])
    assert report["formal_latency_evidence"] is False


@pytest.mark.parametrize("matrix_args", [2048], indirect=True)
def test_2048_capacity_prepares_full_grid_through_1024(matrix_args, monkeypatch):
    matrix_args.plan_only = True
    matrix_args.lengths = list(matrix.DEFAULT_LENGTHS)
    monkeypatch.setattr(matrix.suite, "run", no_execution)
    assert matrix.run(matrix_args) == 0
    root, = matrix_args.run_dir.glob("gdr-lengths-*")
    report = json.loads((root / "request.json").read_text())
    assert len(report["cells"]) == 12
    assert report["minimum_required_capacity"] == 1088  # 1024 + 2 prompt rows, aligned to 64.
    assert all(bundle["capacity"] == 2048 for bundle in report["bundles"].values())


def test_unsupported_later_budget_rejected_before_first_job(matrix_args, monkeypatch):
    matrix_args.lengths = [32, 128]  # Logical capacity=128 and prompt length=2.
    monkeypatch.setattr(matrix.suite, "run", no_execution)
    with pytest.raises(ValueError, match="exceeds capacity 128; no device jobs started"):
        matrix.run(matrix_args)


@pytest.mark.parametrize("matrix_args", [1088, 2048], indirect=True)
def test_1k_context_grid_checks_input_plus_output_before_any_device_job(matrix_args, monkeypatch):
    from qwen35_dflash.ascend310p import workflow

    tokenizer = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: [4] * 1024)
    monkeypatch.setattr(workflow, "load_tokenizer", lambda **kwargs: (tokenizer, "host-1k-token-fixture"))
    monkeypatch.setattr(matrix.suite, "run", no_execution)
    matrix_args.prompts = matrix.REPO / "config/prompts_long_1k.json"
    matrix_args.lengths = [128, 512, 1024]
    matrix_args.plan_only = True
    capacity = json.loads(matrix_args.chunk_deployment_manifest.read_text())["graphs"][0]["metadata"]["incremental_contract"]["capacity"]
    if capacity < 2048:
        with pytest.raises(ValueError, match=r"prompt 1024 \+ budget 128 exceeds capacity 1088"):
            matrix.run(matrix_args)
        return
    assert matrix.run(matrix_args) == 0
    root, = matrix_args.run_dir.glob("gdr-lengths-*")
    report = json.loads((root / "request.json").read_text())
    frozen = json.loads((root / "prompts.json").read_text())
    assert report["minimum_required_capacity"] == 2048
    assert len(report["cells"]) * len(frozen) == 24
    assert frozen == report["prompts"]
    assert all(p["input_tokens"] == len(p["prompt_token_ids"]) == 1024 for p in frozen)
    assert "| long_zh_qa | 1024 |" in (root / "summary.md").read_text()


def test_wrong_route_is_not_silently_relabelled(matrix_args, monkeypatch):
    matrix_args.mtp_deployment_manifest = matrix_args.chunk_deployment_manifest
    monkeypatch.setattr(matrix.suite, "run", no_execution)
    with pytest.raises(ValueError, match="requested verify_gdr=mtp"):
        matrix.run(matrix_args)


@pytest.mark.parametrize("lengths", [[0, 32], [-1], [32, 32]])
def test_invalid_length_grid_fails_before_preparation(matrix_args, monkeypatch, lengths):
    matrix_args.lengths = lengths
    monkeypatch.setattr(matrix.suite, "run", no_execution)
    with pytest.raises(ValueError, match="distinct positive"):
        matrix.run(matrix_args)


def test_all_lengths_and_routes_execute_even_when_fake_acl_is_rejected(matrix_args, monkeypatch):
    # Deliberately exercise the fake runner. Its reports must remain rejected
    # as device evidence regardless of --allow-output-differences.
    monkeypatch.delenv("ASCEND310P_SIMULATION_ONLY", raising=False)
    monkeypatch.delenv("PROFILING_MODE", raising=False)
    monkeypatch.setenv("QWEN35_FAKE_ACCEPT", "15")
    assert matrix.run(matrix_args) == 1
    root, = matrix_args.run_dir.glob("gdr-lengths-*")
    report = json.loads((root / "summary.json").read_text())
    assert report["status"] == "FAIL_OR_INCOMPLETE"
    assert len(report["cells"]) == 4
    csv_rows = list(csv.DictReader((root / "cases.csv").open()))
    assert len(csv_rows) == 4 and all(row["input_tokens"] == "2" for row in csv_rows)
    for cell in report["cells"]:
        assert cell["suite_exit_code"] == 1
        assert cell["aggregate"]["measured_prompts"] == 0
        assert "fake ACL" in cell["cases"][0]["error"]
        raw = json.loads(Path(cell["summary"]).read_text())
        request = json.loads(Path(raw["request"]).read_text())
        assert request["verify_gdr"] == cell["verify_gdr"]
        assert request["max_new_tokens"] == cell["max_new_tokens"]
        assert request["allow_output_differences"] is True
        assert "--low-memory" in request["command"]
    assert os.environ["AI_RUN_DIR"] == str(matrix_args.run_dir)


def measured_row(name, accepted, proposed, elapsed, tokens=20):
    timings = {stage: {"available": True, "calls": 2, "total_ms": 6.0, "mean_ms": 3.0}
               for stage in matrix.STAGES}
    return dict(id=name, status="PASS_WITH_DIFFERENCES", drafted_tokens=proposed,
                accepted_draft_tokens=accepted, ordinary_total_measured_ms=100,
                dflash_total_measured_ms=elapsed, speculative_rounds=2,
                tokens_emitted_in_speculative_rounds=19, ordinary_measured_tokens=tokens,
                dflash_measured_tokens=tokens, generated_tokens=tokens, ordinary_generated_tokens=tokens,
                max_new_tokens=32, stage_timings=timings)


def test_weighted_metrics_use_actual_eos_tokens_and_calls():
    rows = [measured_row("a", 9, 10, 50), measured_row("b", 1, 90, 150)]
    totals = matrix.aggregate_cases(rows)
    assert totals["weighted_acceptance_rate"] == .1
    assert totals["total_model_time_speedup"] == 1
    assert totals["dflash_tokens_per_second"] == 200  # 40 actual tokens / 200 ms.
    assert totals["both_modes_reached_budget"] == 0  # EOS before budget=32.
    assert totals["stage_ms_per_call"]["verify"]["calls"] == 4
    assert totals["stage_ms_per_call"]["verify"]["mean_ms"] == 3
    rows[1]["stage_timings"]["draft"] = {"available": False}
    assert matrix.aggregate_cases(rows)["stage_ms_per_call"]["draft"]["available"] is False


def test_cross_route_comparison_uses_only_matched_prompts():
    a, b = measured_row("a", 1, 10, 100), measured_row("b", 2, 10, 500)
    faster = measured_row("a", 1, 10, 25, tokens=10)
    cells = [
        dict(verify_gdr="chunk", max_new_tokens=32, cases=[a, b]),
        dict(verify_gdr="mtp", max_new_tokens=32, cases=[faster, dict(id="b", status="FAIL")]),
    ]
    result, = matrix.compare_routes(cells, [32])
    assert result["matched_prompt_ids"] == ["a"]
    assert result["mtp_over_chunk_model_time_speedup"] == 4
    assert result["mtp_over_chunk_throughput"] == 2  # MTP produced half as many tokens.


def test_stage_timing_excludes_warmup_and_distinguishes_missing_from_zero_calls():
    report = {
        "ordinary": {"warmups": [{"stage_ms": {"target_decode": [9999]}}],
                     "measurements": [{"stage_ms": {"target_decode": [2, 4]}}]},
        "dflash": {"measurements": [{"stage_ms": {"draft": [], "target_verify": [3, 9]}}]},
    }
    times = matrix.stage_timings(report)
    assert times["ordinary_decode"]["mean_ms"] == 3
    assert times["verify"]["median_ms"] == 6
    assert times["draft"] == dict(available=True, calls=0, total_ms=0.0, mean_ms=None, median_ms=None)
    assert times["ordinary_prefill"] == {"available": False}
    for bad in (-1, float("nan"), True):
        broken = copy.deepcopy(report)
        broken["dflash"]["measurements"][0]["stage_ms"]["draft"] = [bad]
        with pytest.raises(ValueError, match="invalid stage timing"):
            matrix.stage_timings(broken)


def test_cell_hash_mismatch_cannot_enter_aggregates(tmp_path):
    raw = tmp_path / "raw.json"
    raw.write_text("{}")
    row = measured_row("p", 1, 10, 50)
    row.update(raw_report=str(raw), raw_report_sha256=sha256_file(raw), prompt_token_ids=[4, 5])
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps(dict(protocol={"verify_gdr": "mtp"}, cases=[row])))
    raw.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="changed after suite validation"):
        matrix.read_cell(summary, "mtp", 32, [{"id": "p", "prompt_token_ids": [4, 5]}])
