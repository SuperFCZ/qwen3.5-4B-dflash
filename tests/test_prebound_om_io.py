"""Reusable OM bindings: real C++ scheduling, fake ACL only (no device timing)."""
import json
import os
import re
import subprocess

import pytest

from test_incremental_air_om import chunk_bundle, small_threads, assert_cpp_resources_released
from rms_norm_test_support import adn_rms_norm_cpu
from qwen35_dflash.ascend310p.cpp_runtime import run_cpp_pair
from qwen35_dflash.ascend310p.incremental_plan import write_incremental_plan
from qwen35_dflash.ascend310p.utils import sha256_file

pytestmark = pytest.mark.usefixtures("adn_rms_norm_cpu", "small_threads")


def runner():
    value = os.environ.get("QWEN35_CPP_TEST_RUNNER")
    if not value:
        pytest.skip("fake ACL runner required")
    return value


def logical_results(report):
    return {mode: [{key: row[key] for key in
                    ("generated_token_ids", "stop_reason", "rounds", "counters")}
                   for row in report[mode]["measurements"]]
            for mode in ("ordinary", "dflash")}


@pytest.mark.parametrize("chunk_bundle", [
    {"verify_gdr": route, "draft_context_rows": rows}
    for route in ("chunk", "mtp") for rows in (64, 16)
], indirect=True)
@pytest.mark.parametrize("accepted,eos", [(0, []), (3, []), (15, []), (15, [7])])
@pytest.mark.parametrize("low_memory", [True, False])
def test_prebound_banks_preserve_rounds_and_do_not_rebind(
        chunk_bundle, tmp_path, monkeypatch, accepted, eos, low_memory):
    monkeypatch.setenv("QWEN35_FAKE_ACCEPT", str(accepted))
    trace, cleanup = tmp_path / "io.jsonl", tmp_path / "cleanup.json"
    monkeypatch.setenv("QWEN35_FAKE_IO_LOG", str(trace))
    monkeypatch.setenv("QWEN35_FAKE_CLEANUP_LOG", str(cleanup))
    options = dict(
        deployment_manifest=chunk_bundle,
        runner_options={"device_model": "host-fixture", "cann": "fake", "driver": "fake",
                        "firmware": "fake", "runtime": "fake-acl"},
        prompt_token_ids=[4] * 65, eos_token_ids=eos, device_id=0,
        max_new_tokens=40, max_draft_tokens=15, low_memory=low_memory, trace_rounds=True,
    )
    report = run_cpp_pair(runner=runner(), raw_output=tmp_path / "result.json",
                          log_output=tmp_path / "runner.log", **options)
    assert report["ordinary_parity"]["token_id_mismatches"] == 0
    assert report["protocol"]["om_io_binding"] == "prebound_ping_pong"
    assert_cpp_resources_released(cleanup, (tmp_path / "runner.log").read_text())
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    assert not any(e[0] == "update_buffer" for e in events)
    created = {e[2] for e in events if e[0] == "create_dataset"}
    assert created == {e[2] for e in events if e[0] == "destroy_dataset"}
    calls = [e for e in events if e[0] == "execute"]
    assert all(e[2] in created and e[3] in created and e[2] != e[3] for e in calls)
    # These two graphs execute repeatedly, across resets and partial commits.
    for role in ("target_decode", "target_verify"):
        assert len({(e[2], e[3]) for e in calls if e[1] == role}) == 2
    # Optionally compare with a separately built previous revision, never use
    # its fake timings as a performance result.
    reference = os.environ.get("QWEN35_CPP_REFERENCE_RUNNER")
    if reference:
        original = run_cpp_pair(runner=reference, raw_output=tmp_path / "reference.json",
                                log_output=tmp_path / "reference.log", **options)
        assert logical_results(report) == logical_results(original)
        sizes = [re.findall(r"allocated_device_bytes=(\d+)", (tmp_path / name).read_text())[-1]
                 for name in ("runner.log", "reference.log")]
        assert sizes[0] == sizes[1]  # Only host descriptors are duplicated.


@pytest.mark.parametrize("variable,call", [
    ("QWEN35_FAKE_FAIL_DATASET_CREATE_CALL", 1),
    ("QWEN35_FAKE_FAIL_DATASET_CREATE_CALL", 3),
    ("QWEN35_FAKE_FAIL_DATASET_CREATE_CALL", 4),
    ("QWEN35_FAKE_FAIL_BUFFER_CREATE_CALL", 1),
    ("QWEN35_FAKE_FAIL_BUFFER_CREATE_CALL", 15),
    ("QWEN35_FAKE_FAIL_BUFFER_CREATE_CALL", 18),
])
def test_partial_binding_construction_releases_all_resources(
        chunk_bundle, tmp_path, monkeypatch, variable, call):
    plan, _, _ = write_incremental_plan(chunk_bundle, tmp_path / "plan.txt")
    cleanup = tmp_path / "cleanup.json"
    monkeypatch.setenv(variable, str(call))
    monkeypatch.setenv("QWEN35_FAKE_CLEANUP_LOG", str(cleanup))
    result = subprocess.run([
        runner(), "--model-kind", "chunk", "--model", str(plan),
        "--model-sha256", sha256_file(plan), "--prompt-token-ids", "4",
        "--max-new-tokens", "8", "--output", str(tmp_path / "result.json"),
    ], text=True, capture_output=True)
    assert result.returncode != 0 and not (tmp_path / "result.json").exists()
    assert "returned null" in result.stderr
    assert_cpp_resources_released(cleanup, result.stderr)
