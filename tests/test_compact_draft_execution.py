"""Execution-gear equivalence and real C++ scheduling with a fake ACL device."""
import copy
import json
import os
import re
import subprocess

import pytest
import torch
import torch.nn.functional as F

from test_incremental_air_om import (
    TinyTarget, chunk_bundle, draft_model, manifest_graphs, small_threads, specs,
)
from test_msprof_stage_script import sandbox  # noqa: F401
from rms_norm_test_support import adn_rms_norm_cpu
from qwen35_dflash.ascend310p.incremental import DraftContextGraph, DraftGraph, DraftProposeGraph
from qwen35_dflash.ascend310p.incremental_plan import validate_incremental_bundle, write_incremental_plan
from qwen35_dflash.ascend310p.cpp_runtime import run_cpp_pair
from qwen35_dflash.ascend310p.utils import sha256_file
from tools.benchmark_prompts import stage_timings, render_timings

pytestmark = pytest.mark.usefixtures("adn_rms_norm_cpu", "small_threads")


@pytest.mark.parametrize("prefix", [1, 16, 17, 63, 64, 65, 80, 81, 127])
@pytest.mark.parametrize("proposals", [1, 7, 15])
def test_default_draft_matches_separate_context_and_proposals_across_commits(prefix, proposals):
    torch.manual_seed(7)
    draft, target = draft_model(), TinyTarget().eval()
    reference_draft = copy.deepcopy(draft)
    reference_context = DraftContextGraph(reference_draft, 64)
    reference_propose = DraftProposeGraph(reference_draft, target.embedding, target.head)
    def reference(features, start, rows, anchor, count, *state):
        visible = torch.arange(64) < rows.long()
        masked = torch.where(visible[None, :, None], features, torch.zeros_like(features))
        updated = reference_context(masked, start, *state)
        return (*reference_propose(anchor, start + rows.long(), count, *updated), *updated)
    compact = DraftGraph(draft, target.embedding, target.head)
    baseline = tuple(torch.zeros(1, 1, 256, 16).half() for _ in range(4))
    optimized = tuple(t.clone() for t in baseline)
    seen_rows = []
    hook = draft.fc.register_forward_pre_hook(lambda _, args: seen_rows.append(args[0].shape[1]))
    start = 0

    def padded(rows):
        return F.pad(torch.randn(1, rows, 64).half(), (0, 0, 0, 64 - rows), value=float("nan"))

    def call(graph, features, pos, rows, state, count=proposals):
        return graph(features, torch.tensor([pos]), torch.tensor([rows], dtype=torch.int16),
                     torch.tensor([4]), torch.tensor([count], dtype=torch.int16), *state)

    with torch.inference_mode():
        while start < prefix:
            rows = min(64, prefix - start)
            features = padded(rows)
            ref = call(reference, features, start, rows, baseline)
            baseline = ref[1:]
            for part in range(0, rows, 16):
                count = min(16, rows - part)
                # Match the runtime's in-place use of the shared feature buffer.
                if part:
                    features[:, :16] = features[:, part:part + 16].clone()
                actual = call(compact, features, start + part, count, optimized,
                              proposals if start + part + count == prefix else 1)
                optimized = actual[1:]
                assert seen_rows[-1] == 16
                for left, right in zip(baseline, optimized):
                    end = start + part + count
                    torch.testing.assert_close(left[:, :, :end], right[:, :, :end], rtol=0, atol=0)
            start += rows
            for left, right in zip(baseline, optimized):
                torch.testing.assert_close(left[:, :, :start], right[:, :, :start], rtol=0, atol=0)
        assert seen_rows[-1] == 16
        torch.testing.assert_close(actual[0], ref[0], rtol=0, atol=0)
        # Includes a full acceptance and a zero-accept round (one committed anchor).
        for rows in (1, 4, 16, 1):
            features = padded(rows)
            ref = call(reference, features, start, rows, baseline)
            actual = call(compact, features, start, rows, optimized)
            assert seen_rows[-1] == 16
            torch.testing.assert_close(actual[0], ref[0], rtol=0, atol=0)
            baseline, optimized = ref[1:], actual[1:]
            start += rows
            for left, right in zip(baseline, optimized):
                torch.testing.assert_close(left[:, :, :start], right[:, :, :start], rtol=0, atol=0)
    hook.remove()


@pytest.mark.parametrize("route", ["chunk", "mtp"])
def test_compact_contract_requires_one_draft_and_subchunk_policy(route):
    graphs = manifest_graphs(specs(verify_gdr=route))
    assert validate_incremental_bundle(graphs)["draft_context_rows"] == 16
    assert {g["name"] for g in graphs} == {"target_prefill", "target_decode", "draft", "target_verify"}
    extra = copy.deepcopy(graphs[-1])
    extra["name"] = "draft_context"
    with pytest.raises(ValueError, match="one draft"):
        validate_incremental_bundle(graphs + [extra])
    for g in graphs:
        g["metadata"]["incremental_contract"]["draft_context_rows"] = 64
    with pytest.raises(ValueError, match="draft_context"):
        validate_incremental_bundle(graphs)


@pytest.mark.parametrize("chunk_bundle", ["chunk", "mtp"], indirect=True)
@pytest.mark.parametrize("prefix,accepted,low_memory,eos", [
    (1, 0, True, []), (16, 3, False, []), (17, 15, True, []),
    (63, 0, False, []), (64, 0, True, []), (65, 3, True, []),
    (80, 15, False, [7]), (81, 3, True, []),
])
def test_compact_cpp_schedule_and_phase_accounting(chunk_bundle, tmp_path, monkeypatch,
                                                   prefix, accepted, low_memory, eos):
    runner = os.environ.get("QWEN35_CPP_TEST_RUNNER")
    if not runner:
        pytest.skip("set QWEN35_CPP_TEST_RUNNER to the fake ACL runner")
    monkeypatch.setenv("QWEN35_FAKE_ACCEPT", str(accepted))
    result = run_cpp_pair(
        deployment_manifest=chunk_bundle, runner=runner,
        runner_options={"device_model": "host-fixture", "cann": "fake",
                        "driver": "fake", "firmware": "fake", "runtime": "fake-acl"},
        prompt_token_ids=[4] * prefix, eos_token_ids=eos, device_id=0,
        max_new_tokens=32, max_draft_tokens=15,
        raw_output=tmp_path / "cpp.json", log_output=tmp_path / "cpp.log",
        low_memory=low_memory, trace_rounds=True,
    )
    assert result["abi"]["draft_context_rows"] == 16
    assert result["abi"]["graph_count"] == 4
    assert result["abi"]["draft_prefill_policy"] == "single_draft16_subchunks"
    assert result["protocol"]["max_resident_models"] == (3 if low_memory else 4)
    assert result["ordinary_parity"]["token_id_mismatches"] == 0
    context_calls = (prefix - 1) // 16
    for measurement in result["dflash"]["measurements"]:
        times = measurement["stage_ms"]
        assert "draft_context" not in times
        assert len(times["draft"]) == len(times["target_verify"]) + context_calls
        assert len(times["draft"]) == measurement["counters"]["decode_iterations"] + context_calls
    assert all("draft_context" not in m["stage_ms"] for m in result["ordinary"]["measurements"])
    timings = stage_timings(result)
    text = render_timings([{"id": "compact", "status": "PASS", "stage_timings": timings}])
    assert "Draft Context" not in text


def test_export_defaults_need_no_draft_mode_switch():
    from qwen35_dflash.ascend310p.cli import build_parser, _factory_config
    args = build_parser().parse_args([
        "export-air", "--factory", "example:factory", "--bundle-dir", "/unused",
    ])
    assert "draft_context_rows" not in _factory_config(args)
    assert not hasattr(args, "draft_context_rows")
    with pytest.raises(SystemExit):
        build_parser().parse_args([
            "export-air", "--factory", "example:factory", "--bundle-dir", "/unused",
            "--draft-context-rows", "64",
        ])


@pytest.mark.parametrize("rows", [16, 64])
def test_removed_factory_option_fails_before_loading_models(rows):
    from qwen35_dflash.ascend310p.quant_factory import create_quant_incremental_graphs
    with pytest.raises(ValueError, match="remove it from the factory config"):
        create_quant_incremental_graphs({"max_sequence_length": 128, "draft_context_rows": rows})


@pytest.mark.parametrize("chunk_bundle", ["chunk", "mtp"], indirect=True)
def test_native_runner_rejects_plan_without_subchunk_policy_before_execution(chunk_bundle, tmp_path, monkeypatch):
    runner = os.environ.get("QWEN35_CPP_TEST_RUNNER")
    if not runner:
        pytest.skip("fake ACL runner required")
    plan, _, _ = write_incremental_plan(chunk_bundle, tmp_path / "plan.txt")
    rows = [line for line in plan.read_text().splitlines()
            if not line.startswith("draft_prefill_policy ")]
    plan.write_text("\n".join(rows) + "\n")
    events, report = tmp_path / "calls.jsonl", tmp_path / "result.json"
    monkeypatch.setenv("QWEN35_FAKE_EVENT_LOG", str(events))
    result = subprocess.run([
        runner, "--model-kind", "chunk", "--model", str(plan),
        "--model-sha256", sha256_file(plan), "--prompt-token-ids", "4",
        "--output", str(report),
    ], capture_output=True, text=True)
    assert result.returncode != 0
    assert "single_draft16_subchunks; regenerate the plan" in result.stderr
    assert not report.exists() and not events.exists()


@pytest.mark.parametrize("name", ["draft"])
def test_compact_graph_captures_without_graph_breaks(name):
    spec = next(s for s in specs() if s.name == name)
    captured = torch.export.export(spec.model, spec.example_args, strict=True).module()
    args = list(spec.example_args)
    with torch.inference_mode():
        for valid in (0, 1, 16):
            args[2] = torch.tensor([valid], dtype=torch.int16)
            actual, expected = captured(*args), spec.model(*args)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("chunk_bundle", ["chunk", "mtp"], indirect=True)
def test_compact_cpp_profile_keeps_cache_setup_outside_capture(chunk_bundle, sandbox):
    from test_cpp_stage_profile import test_cpp_stage_windows
    test_cpp_stage_windows(chunk_bundle, sandbox, "dflash", "all", "", 15, 32)


@pytest.mark.parametrize("chunk_bundle", [
    {"verify_gdr": "chunk", "capacity": 2048}, {"verify_gdr": "mtp", "capacity": 2048},
], indirect=True)
def test_single_draft_prefill_reuses_allocations_and_loads_only_three_models(
        chunk_bundle, tmp_path, monkeypatch):
    runner = os.environ.get("QWEN35_CPP_TEST_RUNNER")
    if not runner:
        pytest.skip("fake ACL runner required")
    plan, _, _ = write_incremental_plan(chunk_bundle, tmp_path / "memory-plan.txt", mode="dflash")
    allocated = []
    for prefix in (1, 96, 1024, 1984):
        cleanup, loads = tmp_path / f"cleanup-{prefix}.json", tmp_path / f"loads-{prefix}.jsonl"
        report = tmp_path / f"memory-{prefix}.json"
        monkeypatch.setenv("QWEN35_FAKE_CLEANUP_LOG", str(cleanup))
        monkeypatch.setenv("QWEN35_FAKE_WORKSPACE_LOG", str(loads))
        proc = subprocess.run([
            runner, "--model-kind", "chunk", "--mode", "dflash", "--model", str(plan),
            "--model-sha256", sha256_file(plan),
            "--prompt-token-ids", ",".join(["4"] * prefix),
            "--max-new-tokens", "32", "--max-draft-tokens", "15",
            "--warmup", "0", "--repetitions", "1", "--output", str(report),
        ], capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        loaded = [json.loads(line) for line in loads.read_text().splitlines()]
        assert [row[0] for row in loaded] == ["draft", "target_prefill", "target_verify"]
        assert max(row[3] for row in loaded) == 3
        audit = json.loads(cleanup.read_text())
        assert audit["allocations_after_execute"] == 0
        assert all(value == 0 for key, value in audit.items() if key.startswith("live_"))
        allocated.append(int(re.findall(r"allocated_device_bytes=(\d+)", proc.stderr)[-1]))
        measurement, = json.loads(report.read_text())["benchmark"]["measurements"]
        assert len(measurement["stage_ms"]["draft"]) == (
            (prefix - 1) // 16 + measurement["counters"]["decode_iterations"])
        # Priming candidates are discarded and never inflate acceptance counters.
        assert measurement["counters"]["drafted_tokens"] <= 15 * measurement["counters"]["decode_iterations"]
    assert len(set(allocated)) == 1
