"""Public benchmark entry: mixed inputs, precision selection and saved baselines."""
import csv
import json
from pathlib import Path

import pytest

from tools import benchmark_gdr_lengths as matrix
from tools import benchmark_prompts as suite
from tools import ordinary_baseline
from test_gdr_length_matrix import matrix_args, small_threads, adn_rms_norm_cpu
from test_unified_draft_bundle import unified_export, variant_builder
from qwen35_dflash.ascend310p.compiler import compile_air_bundle

pytestmark = pytest.mark.usefixtures("small_threads", "adn_rms_norm_cpu")


@pytest.fixture
def unified_args(matrix_args, unified_export, monkeypatch):
    export, atc, _ = unified_export
    air = export()
    result = compile_air_bundle(air["manifest_path"], atc_bin="/bin/true", soc_version="Ascend310P3",
                               runner=atc, atc_identity="fake-atc")
    matrix_args.bundle_dir = Path(result["manifest_path"]).parent
    matrix_args.draft_quantizations = ["fp16", "w4a16", "w8a16"]
    matrix_args.lengths = [4]
    matrix_args.verify_gdr = "both"
    matrix_args.warmup, matrix_args.repetitions = 0, 1
    monkeypatch.delenv("ASCEND310P_SIMULATION_ONLY", raising=False)
    monkeypatch.delenv("PROFILING_MODE", raising=False)
    return matrix_args


def test_one_run_combines_short_long_and_offline_files(unified_args, monkeypatch):
    args = unified_args
    from qwen35_dflash.ascend310p import workflow
    tokenizer, source = workflow.load_tokenizer(model_dir=args.model_dir)
    original_template = tokenizer.apply_chat_template
    def no_thinking(messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return original_template(messages, **kwargs)
    tokenizer.apply_chat_template = no_thinking
    monkeypatch.setattr(workflow, "load_tokenizer", lambda **kw: (tokenizer, source))
    data = args.run_dir / "datasets"
    data.mkdir()
    for name in ("gsm8k", "humaneval"):
        (data / (name + ".jsonl")).write_text('{"question":"one"}\n{"question":"two"}\n')
    args.dataset_dir, args.num_questions = data, 1
    args.include_builtin_prompts, args.prompts, args.prompt_group = True, None, "all"
    events = args.run_dir / "combined-events.jsonl"
    monkeypatch.setenv("QWEN35_FAKE_EVENT_LOG", str(events))
    assert matrix.run(args) == 1  # Fake ACL must remain rejected as device evidence.
    root, = args.run_dir.glob("gdr-lengths-*")
    saved = json.loads((root / "fp16/request.json").read_text())
    assert len(saved["prompts"]) == 22
    assert sum(p.get("group") == "short" for p in saved["prompts"]) == 8
    assert sum(p.get("group") == "long" for p in saved["prompts"]) == 12
    assert len(saved["datasets"]) == 2
    assert saved["protocol"]["enable_thinking"] is False
    calls = [json.loads(line)[0] for line in events.read_text().splitlines()]
    assert calls.count("target_decode") == 22 * 3
    result = json.loads((root / "summary.json").read_text())
    assert len(result["cells"]) == 6
    assert [c["ordinary_baseline_reused"] for c in result["cells"]] == [False, True, True, True, True, True]
    with (root / "datasets.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 12
    assert {r["draft_quantization"] for r in rows} == {"fp16", "w4a16", "w8a16"}
    assert "| Draft | Dataset file | GDR |" in (root / "summary.md").read_text()
    assert len(list((root / "datasets").glob("*/summary.json"))) == 2


def test_saved_matrix_reuse_runs_no_ordinary_decode(unified_args, monkeypatch):
    args = unified_args
    assert matrix.run(args) == 1
    root, = args.run_dir.glob("gdr-lengths-*")
    args.ordinary_baseline = root / "summary.json"
    # A fresh call must not retain the first invocation's prepared runtime objects.
    events = args.run_dir / "reuse-events.jsonl"
    monkeypatch.setenv("QWEN35_FAKE_EVENT_LOG", str(events))
    assert matrix.run(args) == 1
    calls = [json.loads(line)[0] for line in events.read_text().splitlines()]
    assert "target_decode" not in calls
    assert "draft" in calls
    roots = set(args.run_dir.glob("gdr-lengths-*")) - {root}
    result = json.loads((roots.pop() / "summary.json").read_text())
    assert all(c["ordinary_baseline_reused"] for c in result["cells"])
    # Reuse must reject a changed thinking mode before any device work, even
    # when this test tokenizer happens to produce the same IDs for both modes.
    events.unlink()
    args.enable_thinking = True
    with pytest.raises(ValueError, match="enable_thinking"):
        matrix.run(args)
    assert not events.exists()


def test_saved_baseline_incompatible_selection_fails_before_run(unified_args, monkeypatch):
    args = unified_args
    assert matrix.run(args) == 1
    root, = args.run_dir.glob("gdr-lengths-*")
    args.ordinary_baseline = root
    args.lengths = [8]
    monkeypatch.setattr(suite, "run", lambda _: pytest.fail("must not launch inference"))
    with pytest.raises(ValueError, match="missing output budgets"):
        matrix.run(args)


def test_plain_matrix_can_reuse_existing_baseline(matrix_args, monkeypatch):
    args = matrix_args
    args.lengths = [4]
    args.warmup, args.repetitions = 0, 1
    monkeypatch.delenv("ASCEND310P_SIMULATION_ONLY", raising=False)
    monkeypatch.delenv("PROFILING_MODE", raising=False)
    assert matrix.run(args) == 1
    root, = args.run_dir.glob("gdr-lengths-*")
    args.ordinary_baseline = root / "summary.json"
    events = args.run_dir / "plain-reuse-events.jsonl"
    monkeypatch.setenv("QWEN35_FAKE_EVENT_LOG", str(events))
    assert matrix.run(args) == 1
    assert "target_decode" not in [json.loads(line)[0] for line in events.read_text().splitlines()]


def test_public_parser_supports_combined_inputs_precision_and_baseline():
    args = matrix.parser().parse_args(["--run-dir", "/run", "--runner", "/runner", "--model-dir", "/model",
        "--bundle-dir", "/bundle", "--draft-quantization", "fp16", "w8a16",
        "--dataset-dir", "/datasets", "--include-builtin-prompts", "--prompt-group", "long",
        "--ordinary-baseline", "/saved/summary.json", "--lengths", "128"])
    assert args.draft_quantizations == ["fp16", "w8a16"]
    assert args.include_builtin_prompts and args.prompt_group == "long"
    assert args.ordinary_baseline == Path("/saved/summary.json")
