"""Decode denominators across measured repetitions, grouping and saved reports."""
import copy
import json
import statistics
import subprocess
from pathlib import Path

import pytest

from qwen35_dflash import decode_metrics as metrics
from qwen35_dflash.ascend310p.workflow import _decode_speedup
from tools import benchmark_prompts as suite, benchmark_draft_variants as variants
from tools import benchmark_gdr_lengths as matrix
from test_gdr_length_matrix import measured_row
from test_prompt_analysis import saved_report, write_saved_suite, offline_args


def report_with_phases():
    report = saved_report(warmup=1, repetitions=3)
    # The outlier makes sums, medians and mean per-run ratios all different.
    for mode, decodes, prefill in (("ordinary", [10, 20, 90], 1),
                                   ("dflash", [4, 5, 6], 100)):
        benchmark = report[mode]
        for measurement, decode in zip(benchmark["measurements"], decodes):
            measurement["latency_ms"].update(prefill=prefill, decode=decode,
                                               model_total=prefill + decode)
        benchmark["latency_ms"]["model_total"]["median"] = prefill + statistics.median(decodes)
        benchmark["warmups"] = [{"latency_ms": {"decode": 123456789}}]
    return report


def test_prompt_speedup_uses_sum_decode_excludes_prefill_and_warmup():
    report = report_with_phases()
    row = suite.summarize_prompt(report)
    assert row["speedup_scope"] == "decode_loop"
    assert row["ordinary_decode_measured_ms"] == 120
    assert row["dflash_decode_measured_ms"] == 15
    assert row["decode_time_speedup"] == _decode_speedup(report["ordinary"], report["dflash"]) == 8
    assert row["generation_throughput_ratio"] == pytest.approx(123 / 315)
    assert row["decode_time_speedup"] != statistics.mean([10/4, 20/5, 90/6])
    assert "speedup" not in row and "throughput_speedup" not in row
    changed = copy.deepcopy(report)
    for mode in ("ordinary", "dflash"):
        for measurement in changed[mode]["measurements"]:
            measurement["latency_ms"]["prefill"] *= 1000
            measurement["latency_ms"]["model_total"] += 10000
    assert suite.summarize_prompt(changed)["decode_time_speedup"] == 8


@pytest.mark.parametrize("bad", [None, -1, float("nan"), float("inf"), True])
def test_incomplete_or_invalid_decode_is_not_replaced_with_model_total(bad):
    report = report_with_phases()
    measurement = report["dflash"]["measurements"][1]
    if bad is None:
        del measurement["latency_ms"]["decode"]
        row = suite.summarize_prompt(report)
        assert row["dflash_decode_measured_ms"] is None
        assert row["decode_time_speedup"] is None
    else:
        measurement["latency_ms"]["decode"] = bad
        with pytest.raises(ValueError, match="invalid decode timing"):
            suite.summarize_prompt(report)


def test_aggregate_is_ratio_of_complete_matched_sums_not_average_ratios():
    a, b = measured_row("a", 3, 10, 150), measured_row("b", 2, 20, 50)
    a.update(ordinary_decode_measured_ms=10, dflash_decode_measured_ms=5)
    b.update(ordinary_decode_measured_ms=90, dflash_decode_measured_ms=15)
    rows = [a, b, dict(id="failed", status="FAIL", ordinary_decode_measured_ms=99999)]
    assert suite.aggregate_metrics(rows)["decode_time_speedup"] == 5
    del b["dflash_decode_measured_ms"]
    assert suite.aggregate_metrics(rows)["decode_time_speedup"] is None
    # Legacy phase totals are usable; no fallback to model total or mean.
    b["phase_timings"] = {"dflash_decode": {"available": True, "total_ms": 15}}
    assert suite.aggregate_metrics(rows)["decode_time_speedup"] == 5
    b["dflash_decode_measured_ms"] = None
    assert suite.aggregate_metrics(rows)["decode_time_speedup"] is None
    assert suite.aggregate_metrics([])["decode_time_speedup"] is None


def test_matched_draft_and_route_speedup_ignore_prefill_and_unmatched_rows():
    base, other = measured_row("a", 1, 10, 100), measured_row("extra", 2, 10, 999)
    quant = measured_row("a", 1, 10, 200)
    base["dflash_decode_measured_ms"] = 80
    quant["dflash_decode_measured_ms"] = 20
    cells = [dict(draft_quantization=v, verify_gdr="chunk", max_new_tokens=32, cases=rows)
             for v, rows in (("fp16", [base, other]), ("w8a16", [quant]))]
    result, = variants.compare(cells)
    assert result["matched_prompt_ids"] == ["a"]
    assert result["decode_time_speedup_vs_fp16"] == 4
    assert result["throughput_vs_fp16"] == .5
    route_cells = [dict(verify_gdr=r, max_new_tokens=32, cases=rows)
                   for r, rows in (("chunk", [base, other]), ("mtp", [quant]))]
    result, = matrix.compare_routes(route_cells, [32])
    assert result["mtp_over_chunk_decode_time_speedup"] == 4
    assert result["mtp_over_chunk_throughput"] == .5
    quant["dflash_decode_measured_ms"] = None
    assert variants.compare(cells)[0]["decode_time_speedup_vs_fp16"] is None
    assert matrix.compare_routes(route_cells, [32])[0]["mtp_over_chunk_decode_time_speedup"] is None


@pytest.mark.parametrize("a,b", [(None, 1), (1, None), (0, 0), (0, 1), (1, 0)])
def test_no_decode_is_na(a, b):
    assert metrics.time_ratio(a, b) is None


def test_saved_report_reanalysis_emits_decode_ratio_without_touching_raw_evidence(tmp_path):
    report = report_with_phases()
    # Match fixture warmup protocol: real warmup records are not needed here.
    report["ordinary"].pop("warmups")
    report["dflash"].pop("warmups")
    index = write_saved_suite(tmp_path / "saved", report)
    before = {p: p.read_bytes() for p in index.parent.rglob("*") if p.is_file()}
    assert suite.summarize_existing(offline_args(tmp_path, index)) == 0
    output, = tmp_path.glob("prompt-summary-*/summary.json")
    summary = json.loads(output.read_text())
    assert summary["cases"][0]["decode_time_speedup"] == summary["aggregate"]["decode_time_speedup"] == 8
    text = output.with_suffix(".md").read_text()
    assert "Decode speedup" in text and "DFlash gen tok/s" in text
    assert all(p.read_bytes() == value for p, value in before.items())


def test_native_prefill_only_report_has_no_decode_speedup(chunk_bundle, tmp_path, monkeypatch):
    from test_prompt_suite import batch_command

    monkeypatch.setenv("QWEN35_FAKE_ACCEPT", "15")
    command, output, _ = batch_command(chunk_bundle, tmp_path, [("one", [4, 5])])
    command[command.index("--max-new-tokens") + 1] = "1"
    proc = subprocess.run(command, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    index = json.loads(output.read_text())
    report = json.loads(Path(index["cases"][0]["report"]).read_text())
    assert report["dflash_decode_time_speedup"] is None
    assert report["speedup_scope"] == "decode_loop"
    assert suite.summarize_prompt(report)["decode_time_speedup"] is None


# Import shared fake-ACL model fixtures; they do not establish device performance.
from test_incremental_air_om import chunk_bundle, small_threads  # noqa: E402,F401
from rms_norm_test_support import adn_rms_norm_cpu  # noqa: E402,F401
pytestmark = pytest.mark.usefixtures("small_threads", "adn_rms_norm_cpu")
