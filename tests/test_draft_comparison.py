"""Variant selection, shared baseline scheduling and own-output denominators."""
import argparse
import json
from pathlib import Path

import pytest

from tools import benchmark_draft_variants as comparison
from tools import build_draft_variants as builder
from tools import benchmark_gdr_lengths as matrix
from test_gdr_length_matrix import matrix_args, measured_row, small_threads, adn_rms_norm_cpu
from test_draft_variant_bundles import variant_builder
from qwen35_dflash.ascend310p.draft_variants import compose_draft_variant
from qwen35_dflash.ascend310p.utils import sha256_file

pytestmark = pytest.mark.usefixtures("small_threads", "adn_rms_norm_cpu")


@pytest.fixture
def comparison_args(matrix_args, variant_builder):
    build, _ = variant_builder
    fp16 = build("fp16", "chunk")
    mtp = build("fp16", "mtp", reuse_common=fp16)
    bundles = {"fp16": {"chunk": fp16, "mtp": mtp}}
    for v in ("w4a16", "w8a16"):
        chunk = build(v, "chunk", reuse_target=fp16)
        other = compose_draft_variant(target_manifest=mtp, draft_manifest=chunk,
                                      bundle_dir=matrix_args.run_dir / v / "mtp")
        bundles[v] = dict(chunk=chunk, mtp=other)
    index = matrix_args.run_dir / "draft-variants.json"
    index.write_text(json.dumps(dict(artifact_kind="qwen35-draft-variants",
        bundles={v: {r: dict(manifest=str(p), manifest_sha256=sha256_file(p), status="PASS",
                            checkpoint={"fixture": True, "variant": v}) for r, p in routes.items()}
                 for v, routes in bundles.items()})))
    matrix_args.draft_variants_manifest = index
    matrix_args.draft_quantizations = ["fp16", "w4a16", "w8a16"]
    matrix_args.verify_gdr = "both"
    matrix_args.lengths = [20, 32]
    matrix_args.warmup, matrix_args.repetitions = 1, 2
    return matrix_args


def test_three_variants_two_routes_measure_ordinary_once_per_length(comparison_args, monkeypatch):
    monkeypatch.delenv("ASCEND310P_SIMULATION_ONLY", raising=False)
    monkeypatch.delenv("PROFILING_MODE", raising=False)
    events = comparison_args.run_dir / "events.jsonl"
    monkeypatch.setenv("QWEN35_FAKE_EVENT_LOG", str(events))
    assert comparison.run(comparison_args) == 1  # Fake ACL remains invalid device evidence.
    assert [json.loads(s)[0] for s in events.read_text().splitlines()].count("target_decode") == (19 + 31) * 3
    root, = comparison_args.run_dir.glob("draft-comparison-*")
    summary = json.loads((root / "summary.json").read_text())
    assert len(summary["cells"]) == 12
    for length in (20, 32):
        cells = [c for c in summary["cells"] if c["max_new_tokens"] == length]
        assert [c["ordinary_baseline_reused"] for c in cells] == [False, True, True, True, True, True]
        assert len({c["ordinary_baseline"] for c in cells}) == 1
    assert (root / "cases.csv").read_text().startswith("draft_quantization,")
    assert summary["quality_evaluation"] == "NOT_RUN"


def test_missing_baseline_does_not_trigger_ordinary_for_other_variants(comparison_args, monkeypatch):
    calls = []
    def failed(args):
        calls.append(str(args.deployment_manifest))
        raise RuntimeError("ordinary unavailable")
    monkeypatch.setattr(matrix.suite, "run", failed)
    assert comparison.run(comparison_args) == 1
    assert len(calls) == len(comparison_args.lengths)
    assert all("fp16/chunk" in p for p in calls)


def test_plan_only_and_selected_precision(comparison_args, monkeypatch):
    comparison_args.draft_quantizations = ["w8a16"]
    comparison_args.verify_gdr = "mtp"
    comparison_args.plan_only = True
    monkeypatch.setattr(matrix.suite, "run", lambda _: pytest.fail("plan must not execute"))
    assert comparison.run(comparison_args) == 0
    root, = comparison_args.run_dir.glob("draft-comparison-*")
    summary = json.loads((root / "summary.json").read_text())
    assert summary["status"] == "PREPARED"
    assert {c["draft_quantization"] for c in summary["cells"]} == {"w8a16"}


def test_comparison_uses_matched_prompts_and_actual_output_lengths():
    original = [measured_row("a", 9, 10, 100), measured_row("b", 1, 10, 200)]
    quant = [measured_row("a", 1, 10, 50, tokens=10), dict(id="b", status="FAIL")]
    cells = [dict(draft_quantization=v, verify_gdr="chunk", max_new_tokens=32, cases=rows)
             for v, rows in (("fp16", original), ("w4a16", quant))]
    row, = comparison.compare(cells)
    assert row["matched_prompt_ids"] == ["a"]
    assert row["decode_time_speedup_vs_fp16"] == 2
    assert row["throughput_vs_fp16"] == 1
    assert row["acceptance_delta_pp"] == -80


def test_prepare_build_creates_seven_compile_jobs_and_selected_weights(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    monkeypatch.setattr(builder, "require_draft_checkpoint", lambda directory, v: {"variant": v})
    def lock(**kwargs):
        kwargs["output"].write_text('{}')
    monkeypatch.setattr(builder, "build_quant_input_manifest", lock)
    cfg = tmp_path / "factory.json"
    cfg.write_text(json.dumps(dict(target_dir="t", quant_config="q", receiver_models_dir="r")))
    args = argparse.Namespace(factory_config=cfg, output=tmp_path / "build", draft_quantizations=list(builder.VARIANTS),
        verify_gdr="both", fp16_draft_dir=tmp_path / "FP16", w4a16_draft_dir=tmp_path / "W4", w8a16_draft_dir=tmp_path / "W8",
        atc=Path("/bin/true"), soc_version="Ascend310P3")
    path = builder.prepare(args)
    plan = json.loads(path.read_text())
    # Four compile invocations: 4 FP16/Chunk graphs, 1 MTP Verify, 1 Draft each for W4/W8.
    assert len([j for j in plan["jobs"] if j["kind"] == "compile"]) == 4
    assert len([j for j in plan["jobs"] if j["kind"] == "compose"]) == 2
    w4 = json.loads((args.output / "config/w4a16-chunk.json").read_text())
    assert w4["draft_dir"] == str(tmp_path / "W4")
    assert w4["draft_quantization"] == "w4a16" and w4["shared_draft_features"] is True


def test_dataset_summary_keeps_precision_table_contiguous(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    root = tmp_path / "summary"
    root.mkdir()
    prepared = {}
    for v in ("fp16", "w4a16", "w8a16"):
        row = dict(measured_row("p", 2, 10, 60), dataset_id="file")
        cell = dict(status="PASS_WITH_DIFFERENCES", verify_gdr="chunk", max_new_tokens=32,
                    cases=[row], aggregate=matrix.aggregate_cases([row]))
        summary = dict(cells=[cell], prompts=[dict(id="p")], bundles={}, protocol={},
                       datasets=[dict(id="file", name="gsm8k.jsonl", sha256="fixture", total_samples=1, selected_samples=1)])
        prepared[v] = argparse.Namespace(_prepared_summary=summary, plan_only=False)
    rendered = comparison.save(root, prepared)
    prefix = rendered.split("Draft: fp16")[0]
    assert all(f"| {v} | chunk |" in prefix for v in prepared)
    report = json.loads((root / "summary.json").read_text())
    assert [r["draft_quantization"] for r in report["datasets"]] == list(prepared)
    assert all(r["weighted_acceptance_rate"] == .2 for r in report["datasets"])
