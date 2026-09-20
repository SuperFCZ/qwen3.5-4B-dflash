"""Host-only regression for nonfatal drift; these fixtures are not device evidence."""
import copy
import json
from pathlib import Path
import subprocess

import pytest

from test_incremental_air_om import chunk_bundle, small_threads  # noqa: F401
from rms_norm_test_support import adn_rms_norm_cpu  # noqa: F401
from test_prompt_suite import batch_command
from test_prompt_analysis import saved_report, write_saved_suite, offline_args
from qwen35_dflash.ascend310p.cpp_runtime import validate_cpp_runner_report
from qwen35_dflash.ascend310p.repeatability import repeatability_observation
from qwen35_dflash.ascend310p.utils import sha256_file
from tools import benchmark_prompts as suite, benchmark_gdr_lengths as matrix


def observation_fields(benchmark):
    observation = repeatability_observation(benchmark)
    first = benchmark["measurements"][0]
    drift = bool(observation["differences"])
    benchmark.update(
        status="PASS_WITH_OBSERVATIONS" if drift else "PASS",
        representative_repetition=0,
        representative_generated_token_ids=copy.deepcopy(first["generated_token_ids"]),
        representative_stop_reason=first["stop_reason"],
        stable_generated_token_ids=None if drift else copy.deepcopy(first["generated_token_ids"]),
        stable_stop_reason=None if drift else first["stop_reason"],
        repeatability=observation,
    )


def drifting_report():
    report = saved_report(1, 3)
    draft = report["dflash"]
    # Repetition 1: same length, one changed accepted token and matching trace.
    m = draft["measurements"][1]
    m["generated_token_ids"][-1] = 48
    for key in ("proposed_token_ids", "accepted_draft_token_ids", "emitted_token_ids"):
        m["rounds"][-1][key][-1] = 48
    m["rounds"][-1]["target_token_ids"][3] = 48
    # Repetition 2: valid early EOS; actual tokens and work must be counted.
    m = draft["measurements"][2]
    m["generated_token_ids"] = [6, 7, 8, 63]
    m["stop_reason"] = "eos"
    m["rounds"] = m["rounds"][:2]
    r = m["rounds"][-1]
    r["emitted_token_ids"][-1] = r["fallback_token_id"] = 63
    r["target_token_ids"][2] = 63
    m["counters"].update(drafted_tokens=3, accepted_draft_tokens=2, rejected_draft_tokens=1)
    m["latency_ms"]["model_total"] = 1.0
    observation_fields(draft)
    observation_fields(report["ordinary"])
    return report


def validate(report, **kwargs):
    validate_cpp_runner_report(report, prompt_token_ids=[4, 5],
        om_sha256=report["model"]["sha256"], device_id=0,
        max_new_tokens=8, max_draft_tokens=15, chunk_abi=True,
        warmup=1, repetitions=3, allow_output_differences=True, **kwargs)


def test_drift_keeps_tokens_eos_and_all_measurements(tmp_path, capsys):
    report = drifting_report()
    validate(report)
    with pytest.raises(RuntimeError, match="token-stable/EOS-stable"):
        validate(report, require_repeatability=True)
    observation = report["dflash"]["repeatability"]
    assert observation["status"] == "DRIFT_OBSERVED"
    assert [d["repetition"] for d in observation["differences"]] == [1, 2]
    assert [d["token_id_mismatches"] for d in observation["differences"]] == [1, 5]
    index = write_saved_suite(tmp_path / "saved", report)
    before = {p: p.read_bytes() for p in index.parent.rglob("*") if p.is_file()}
    assert suite.summarize_existing(offline_args(tmp_path, index)) == 0
    output, = tmp_path.glob("prompt-summary-*/summary.json")
    summary = json.loads(output.read_text())
    assert summary["status"] == "PASS_WITH_OBSERVATIONS"
    assert summary["repeatability"] == "DRIFT_OBSERVED"
    assert summary["formal_latency_evidence"] is False
    assert summary["aggregate"]["drift_observed_prompts"] == 1
    assert summary["aggregate"]["allowed_difference_prompts"] == 1
    row, = summary["cases"]
    assert row["repeatability"]["dflash"] == observation
    assert row["dflash_measured_tokens"] == 20
    assert row["dflash_tokens_per_second"] == 4000  # 20 tokens / 5ms, not 8*3 / 5ms.
    assert row["generated_tokens_range"] == [4, 8]
    assert row["generated_tokens"] == pytest.approx(20 / 3)
    assert row["acceptance_rate"] == 14 / 17
    text = capsys.readouterr().out
    assert "DRIFT_OBSERVED" in text and "repeat 1: token[7] 46 -> 48" in text
    assert "4..8" in text
    assert "passed independent" not in text
    assert all(p.read_bytes() == contents for p, contents in before.items())
    decoded = suite.decode_outputs(report, type("Tokenizer", (), {"decode": lambda self, ids, **kw: str(ids)})())
    assert decoded["dflash"]["available"] and decoded["dflash"]["token_count"] == 8
    assert decoded["dflash"]["measurement_repetition"] == 0


@pytest.mark.parametrize("damage", ["hidden_drift", "empty", "negative", "bad_eos", "too_long", "missing"])
def test_observation_policy_still_rejects_invalid_evidence(damage):
    report = drifting_report()
    m = report["dflash"]["measurements"][2]
    if damage == "hidden_drift":
        report["dflash"]["repeatability"]["differences"] = []
    elif damage == "empty":
        m["generated_token_ids"] = []
    elif damage == "negative":
        m["generated_token_ids"][0] = -1
    elif damage == "bad_eos":
        m["generated_token_ids"][-1] = 62
    elif damage == "too_long":
        m["generated_token_ids"] = list(range(20))
    else:
        report["dflash"]["measurements"].pop()
    with pytest.raises(RuntimeError):
        validate(report)


@pytest.mark.parametrize("route", ["chunk", "mtp"])
def test_matrix_keeps_drift_metrics_and_displays_observations(tmp_path, route):
    report = drifting_report()
    row = dict(suite.summarize_prompt(report), id="p", status="PASS_WITH_OBSERVATIONS",
               max_new_tokens=8, group="short", output_difference="outputs differ",
               verify_gdr=route, prompt_token_ids=[4, 5])
    cell = dict(verify_gdr=route, max_new_tokens=8, status=row["status"], cases=[row],
                aggregate=matrix.aggregate_cases([row]))
    result = dict(cells=[cell], prompts=[dict(id="p", group="short", prompt_token_ids=[4, 5])],
                  lengths=[8], route_comparison=[], protocol=dict(warmup=1, repetitions=3))
    assert suite.measured_status([cell]) == "PASS_WITH_OBSERVATIONS"
    assert cell["aggregate"]["measured_prompts"] == 1
    assert cell["aggregate"]["dflash_tokens_per_second"] == 4000
    assert cell["aggregate"]["weighted_acceptance_rate"] == 14 / 17
    assert "DRIFT_OBSERVED" in matrix.render(result)
    assert f"{route}/8/p" in matrix.render(result)


@pytest.mark.usefixtures("adn_rms_norm_cpu")
@pytest.mark.parametrize("low_memory", [False, True])
def test_cpp_batch_continues_after_repetition_drift(chunk_bundle, tmp_path, monkeypatch, low_memory):
    # Change repetition 1 of the first DFlash prompt. Repetition 0 still matches ordinary.
    monkeypatch.setenv("QWEN35_FAKE_PREFILL_DRIFT_CALL", "11" if low_memory else "5")
    command, output, plan = batch_command(
        chunk_bundle, tmp_path, [("drifting", [4, 5]), ("next", [8, 9])], low_memory)
    command[command.index("--warmup") + 1] = "1"
    command[command.index("--repetitions") + 1] = "3"
    proc = subprocess.run(command, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    index = json.loads(output.read_text())
    assert index["status"] == "PASS_WITH_OBSERVATIONS"
    assert [c["status"] for c in index["cases"]] == ["PASS_WITH_OBSERVATIONS", "PASS"]
    report = json.loads(Path(index["cases"][0]["report"]).read_text())
    validate_cpp_runner_report(report, prompt_token_ids=[4, 5], om_sha256=sha256_file(plan),
        device_id=0, max_new_tokens=20, max_draft_tokens=15, chunk_abi=True,
        low_memory=low_memory, warmup=1, repetitions=3)
    assert report["dflash"]["stable_generated_token_ids"] is None
    assert report["dflash"]["repeatability"]["status"] == "DRIFT_OBSERVED"
    assert report["dflash"]["repeatability"]["differences"][0]["repetition"] == 1
    assert len(report["dflash"]["measurements"]) == 3
    assert report["dflash"]["totals"]["generated_tokens"] == 60
    assert "DRIFT_OBSERVED" in proc.stderr


@pytest.mark.usefixtures("adn_rms_norm_cpu")
def test_cpp_single_mode_counts_actual_tokens_with_variable_eos(chunk_bundle, tmp_path, monkeypatch):
    monkeypatch.setenv("QWEN35_FAKE_PREFILL_DRIFT_CALL", "3")
    command, output, _ = batch_command(chunk_bundle, tmp_path, [("p", [4, 5])])
    for key in ("--prompt-batch", "--prompt-batch-sha256"):
        index = command.index(key)
        del command[index:index + 2]
    command[command.index("--mode") + 1] = "dflash"
    command[command.index("--warmup") + 1] = "1"
    command[command.index("--repetitions") + 1] = "3"
    command += ["--prompt-token-ids", "4,5", "--eos-token-ids", "10"]
    proc = subprocess.run(command, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    report = json.loads(output.read_text())
    assert report["status"] == "PASS_WITH_OBSERVATIONS"
    b = report["benchmark"]
    assert [len(m["generated_token_ids"]) for m in b["measurements"]] == [5, 20, 5]
    assert b["totals"]["generated_tokens"] == 30
    assert b["generated_tokens_per_second"] == pytest.approx(
        30000 / sum(m["latency_ms"]["model_total"] for m in b["measurements"]))
    assert b["repeatability"] == repeatability_observation(b)
    assert b["repeatability"]["differences"][0]["stop_reason_changed"]
