"""Offline file selection, provenance and OM reporting; all evidence is host-only."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

import pytest

from tools import benchmark_prompts as suite
from tools import benchmark_gdr_lengths as matrix
from tools import offline_datasets as offline
from test_gdr_length_matrix import matrix_args, measured_row, small_threads  # noqa: F401
from rms_norm_test_support import adn_rms_norm_cpu  # noqa: F401

pytestmark = pytest.mark.usefixtures("adn_rms_norm_cpu")


def args(**kwargs):
    return argparse.Namespace(dataset_dir=None, dataset_files=None, num_questions=None,
                              dataset_field="question", **kwargs)


def jsonl(path, questions):
    path.write_text("\n".join(json.dumps({"question": q}, ensure_ascii=False) for q in questions) + "\n", encoding="utf-8")
    return path


def test_directory_matches_gpu_format_limit_per_file_and_preserves_text(tmp_path):
    a = jsonl(tmp_path / "gsm8k.jsonl", ["题目\n保留换行", "second", "third"])
    b = jsonl(tmp_path / "math500.jsonl", ["one", "two"])
    (tmp_path / "unrelated.txt").write_text("ignored")
    options = args()
    options.dataset_dir, options.num_questions = tmp_path, 2
    prompts, datasets = suite.load_inputs(options)
    assert [d["name"] for d in datasets] == [a.name, b.name]
    assert [d["selected_samples"] for d in datasets] == [2, 2]
    assert [d["total_samples"] for d in datasets] == [3, 2]
    assert datasets[0]["sha256"] == hashlib.sha256(a.read_bytes()).hexdigest()
    assert [p["prompt"] for p in prompts] == ["题目\n保留换行", "second", "one", "two"]
    assert len({p["id"] for p in prompts}) == 4
    assert [p["dataset_line"] for p in prompts] == [1, 2, 1, 2]
    # Names/IDs remain stable when selecting a single file, so reports can be joined.
    options.dataset_dir, options.dataset_files = None, [b]
    single, _ = suite.load_inputs(options)
    assert single == prompts[2:]


def test_explicit_files_json_array_custom_field_bom_blank_lines(tmp_path):
    a = tmp_path / "data.json"
    a.write_text(json.dumps([{"prompt": "hello", "answer": "must not enter input"}, "world"]))
    b = tmp_path / "例子.jsonl"
    b.write_text('\ufeff\n{"prompt":"中文"}\n\n{"prompt":"English"}\n', encoding="utf-8")
    options = args()
    options.dataset_files, options.dataset_field = [b, a], "prompt"
    prompts, datasets = offline.load(options)
    assert [p["prompt"] for p in prompts] == ["中文", "English", "hello", "world"]
    assert [p["dataset_line"] for p in prompts] == [2, 4, 1, 2]
    assert datasets[1]["line_unit"] == "array item (1-based)"


@pytest.mark.parametrize("content,match", [
    ('{"question":"ok"}\nnot-json\n', r"bad.jsonl:2: invalid"),
    ('{"question":"ok"}\n{"answer":"only"}\n', r"bad.jsonl:2: expected"),
    ('{"question":["not string"]}\n', r"bad.jsonl:1: expected"),
    ('{"question":"  "}\n', r"bad.jsonl:1: expected"),
    ('\n\n', "empty dataset"),
])
def test_bad_file_is_never_silently_skipped(tmp_path, content, match):
    path = tmp_path / "bad.jsonl"
    path.write_text(content)
    options = args()
    options.dataset_files, options.num_questions = [path], 1
    with pytest.raises(ValueError, match=match):
        suite.load_inputs(options)


@pytest.mark.parametrize("option,value", [("num_questions", 0), ("num_questions", -1),
                                         ("prompts", Path("x.json")), ("prompt_group", "short"),
                                         ("prompt_id", ["p"]), ("dataset_field", "")])
def test_incompatible_or_invalid_selection(tmp_path, option, value):
    options = args()
    options.dataset_files = [jsonl(tmp_path / "a.jsonl", ["hello"])]
    setattr(options, option, value)
    with pytest.raises(ValueError):
        suite.load_inputs(options)


def test_duplicate_names_and_empty_directory_are_rejected(tmp_path):
    options = args()
    options.dataset_dir = tmp_path
    with pytest.raises(ValueError, match="no offline"):
        offline.load(options)
    a = jsonl(tmp_path / "a.jsonl", ["hello"])
    other = tmp_path / "other"
    other.mkdir()
    b = jsonl(other / a.name, ["different"])
    options.dataset_dir, options.dataset_files = None, [a, b]
    with pytest.raises(ValueError, match="distinct paths and filenames"):
        offline.load(options)


def test_file_aggregation_weights_counters_and_actual_eos_work(tmp_path, monkeypatch):
    options = args()
    options.dataset_files = [jsonl(tmp_path / "a.jsonl", ["a", "b", "bad"]),
                             jsonl(tmp_path / "b.jsonl", ["c"])]
    prompts, datasets = offline.load(options)
    rows = [dict(measured_row("a", 9, 10, 50, tokens=10), **prompts[0]),
            dict(measured_row("b", 1, 90, 150, tokens=30), **prompts[1]),
            dict(prompts[2], status="FAIL"),
            dict(measured_row("c", 1, 2, 100), **prompts[3])]
    rows[1]["status"] = "PASS_WITH_OBSERVATIONS"
    rows[1]["stage_timings"]["verify"].update(calls=4, total_ms=20)
    summary = dict(datasets=datasets, cases=rows, max_new_tokens=32,
                   protocol={"verify_gdr": "chunk", "warmup": 1, "repetitions": 3})
    a, b = suite.dataset_results(summary)
    assert a["status"] == "FAIL_OR_INCOMPLETE" and a["measured_prompts"] == 2
    assert a["weighted_acceptance_rate"] == .1  # Not mean(90%, 1/90).
    assert a["dflash_tokens_per_second"] == 200 and a["decode_time_speedup"] == 1
    assert a["stage_ms_per_call"]["verify"]["mean_ms"] == pytest.approx(26 / 6)
    assert a["stage_ms_per_call"]["verify"]["mean_total_ms_per_generation"] == 13
    assert a["drift_observed_prompts"] == 1
    assert b["weighted_acceptance_rate"] == .5
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    suite.write_dataset_reports(tmp_path, summary)
    saved = json.loads((tmp_path / "datasets" / datasets[0]["id"] / "summary.json").read_text())
    assert len(saved["cases"]) == 3 and saved["dataset"]["sha256"] == datasets[0]["sha256"]
    csv_rows = list(csv.DictReader((tmp_path / "datasets.csv").open()))
    assert len(csv_rows) == 2 and csv_rows[0]["weighted_acceptance_rate"] == "0.1"
    assert float(csv_rows[0]["verify_ms_per_call"]) == pytest.approx(26 / 6)


def test_directory_preflight_and_frozen_snapshot(matrix_args, monkeypatch):
    data = matrix_args.run_dir / "data"
    data.mkdir()
    file = jsonl(data / "gsm8k.jsonl", [f"q{i}" for i in range(70)])
    matrix_args.dataset_dir, matrix_args.prompts, matrix_args.plan_only = data, None, True
    monkeypatch.setattr(matrix.suite, "run", lambda *a: pytest.fail("plan-only must not execute"))
    assert matrix.run(matrix_args) == 0
    root, = matrix_args.run_dir.glob("gdr-lengths-*")
    request = json.loads((root / "request.json").read_text())
    assert len(request["prompts"]) == 70 and request["datasets"][0]["selected_samples"] == 70
    assert len(list(csv.DictReader((root / "datasets.csv").open()))) == 4  # 2 routes x 2 budgets.
    file.unlink()  # Source files are not required after freezing or for reanalysis.
    options = copy.copy(matrix_args)
    options._prepared_inputs = request["prompts"], request["datasets"]
    frozen, _ = suite.load_inputs(options)
    assert frozen == request["prompts"]
    frozen[0]["prompt"] = "changed"
    assert request["prompts"][0]["prompt"] == "q0"


def test_over_capacity_names_file_and_line_before_execution(matrix_args, monkeypatch):
    from qwen35_dflash.ascend310p import workflow
    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return [4] * 100
    monkeypatch.setattr(workflow, "load_tokenizer", lambda **kw: (Tokenizer(), "test"))
    monkeypatch.setattr(matrix.suite, "run", lambda *a: pytest.fail("must preflight first"))
    matrix_args.dataset_files = [jsonl(matrix_args.run_dir / "long.jsonl", ["long"])]
    matrix_args.prompts = None
    with pytest.raises(ValueError, match=r"long.jsonl:1 .*exceeds capacity 128"):
        matrix.run(matrix_args)


def test_full_offline_matrix_supports_over_64_questions_without_fake_metrics(matrix_args, monkeypatch):
    monkeypatch.delenv("ASCEND310P_SIMULATION_ONLY", raising=False)
    monkeypatch.delenv("PROFILING_MODE", raising=False)
    monkeypatch.setenv("QWEN35_FAKE_ACCEPT", "15")
    matrix_args.dataset_files = [jsonl(matrix_args.run_dir / "many.jsonl", [str(i) for i in range(65)]),
                                 jsonl(matrix_args.run_dir / "second.jsonl", ["another"])]
    matrix_args.prompts, matrix_args.verify_gdr, matrix_args.lengths = None, "chunk", [2]
    matrix_args.warmup, matrix_args.repetitions = 0, 1
    assert matrix.run(matrix_args) == 1  # Simulation can exercise scheduling, never claim NPU metrics.
    root, = matrix_args.run_dir.glob("gdr-lengths-*")
    summary = json.loads((root / "summary.json").read_text())
    cell, = summary["cells"]
    index = json.loads(Path(json.loads(Path(cell["summary"]).read_text())["runner_index"]).read_text())
    assert index["fake_acl"] and len(index["cases"]) == 66
    assert index["models_reused_across_prompts"]
    assert all(row["status"] == "PASS" for row in index["cases"])
    assert len(summary["dataset_results"]) == 2
    assert all(r["measured_prompts"] == 0 and r["weighted_acceptance_rate"] is None for r in summary["dataset_results"])
    assert [r["selected_samples"] for r in summary["dataset_results"]] == [65, 1]
    rows = list(csv.DictReader((root / "cases.csv").open()))
    assert len(rows) == 66 and rows[-1]["dataset_file"] == "second.jsonl"


def test_both_cli_entrypoints_expose_gpu_compatible_selection(monkeypatch):
    import sys
    base = ["--run-dir", "/run", "--runner", "/runner", "--model-dir", "/model"]
    selection = ["--dataset-files", "/data/gsm8k.jsonl", "/data/math500.jsonl", "--num-questions", "10"]
    parsed = matrix.parser().parse_args(base + selection + ["--verify-gdr", "mtp", "--lengths", "128"])
    assert [p.name for p in parsed.dataset_files] == ["gsm8k.jsonl", "math500.jsonl"]
    assert parsed.num_questions == 10 and parsed.dataset_field == "question"
    assert (parsed.warmup, parsed.repetitions) == (1, 3)
    calls = []
    monkeypatch.setattr(suite, "run", lambda args: calls.append(args) or 0)
    monkeypatch.setattr(sys, "argv", ["benchmark_prompts.py"] + base + selection + ["--max-new-tokens", "128"])
    assert suite.main() == 0 and len(calls[0].dataset_files) == 2


def test_reanalysis_keeps_large_dataset_provenance_without_original_files(tmp_path):
    from test_prompt_analysis import saved_report, write_saved_suite, offline_args
    from qwen35_dflash.ascend310p.utils import sha256_file
    source = jsonl(tmp_path / "saved-questions.jsonl", [str(i) for i in range(65)])
    options = args()
    options.dataset_files = [source]
    prompts, datasets = offline.load(options)
    index = write_saved_suite(tmp_path / "saved", saved_report(1, 3))
    request_path = index.parent / "request.json"
    request = json.loads(request_path.read_text())
    raw = json.loads(index.read_text())
    sample = Path(raw["cases"][0]["report"]).read_text()
    batch = index.parent / "prompts.txt"
    batch.write_text("QWEN35_PROMPT_BATCH_V1\n" + "".join(f'{p["id"]} "4,5"\n' for p in prompts))
    request["command"][request["command"].index("--prompt-batch-sha256") + 1] = sha256_file(batch)
    request.update(prompts=[dict(p, prompt_token_ids=[4, 5]) for p in prompts], datasets=datasets)
    raw.update(prompt_batch_sha256=sha256_file(batch), cases=[])
    for p in prompts:
        case = Path(str(index) + ".cases") / (p["id"] + ".json")
        case.write_text(sample)
        raw["cases"].append(dict(id=p["id"], status="FAIL", report=str(case)))
    index.write_text(json.dumps(raw))
    request_path.write_text(json.dumps(request))
    before = sha256_file(index), sha256_file(request_path)
    source.unlink()
    assert suite.run(offline_args(tmp_path, index)) == 0
    out, = tmp_path.glob("prompt-summary-*/summary.json")
    summary = json.loads(out.read_text())
    result, = summary["dataset_results"]
    assert result["status"] == "PASS_WITH_DIFFERENCES" and result["measured_prompts"] == 65
    assert result["accepted_draft_tokens"] == 65 * 3 * 6  # Excludes one warmup/question.
    assert result["dataset_sha256"] == datasets[0]["sha256"]
    assert summary["reanalysis"]["device_execution"] == "NOT_RUN"
    assert before == (sha256_file(index), sha256_file(request_path))
    # The original token/EOS comparison gate still controls metric admission.
    assert suite.run(offline_args(tmp_path, index, allow=False)) == 1
