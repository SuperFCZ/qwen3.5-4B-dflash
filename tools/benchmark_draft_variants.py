#!/usr/bin/env python3
"""Compare published FP16/W4A16/W8A16 Drafts using one ordinary baseline."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "framework/python"), str(REPO)]
from tools import benchmark_gdr_lengths as matrix
from tools import benchmark_prompts as suite
from tools.build_draft_variants import VARIANTS
from qwen35_dflash.ascend310p.utils import atomic_write_json, load_json_object, sha256_file


def prepare(args, root):
    if len(set(args.draft_quantizations)) != len(args.draft_quantizations):
        raise ValueError("duplicate Draft selections")
    index = load_json_object(args.draft_variants_manifest)
    if index.get("artifact_kind") != "qwen35-draft-variants":
        raise ValueError("not a compiled Draft index; run export-air and compile-om with --draft-quantizations")
    routes = matrix.ROUTES if args.verify_gdr == "both" else (args.verify_gdr,)
    result, ordinary, prompts = {}, None, None
    for variant in args.draft_quantizations:
        directory = root / variant
        directory.mkdir()
        options = argparse.Namespace(**vars(args))
        for route in routes:
            entry = index.get("bundles", {}).get(variant, {}).get(route, {})
            if entry.get("status") != "PASS":
                raise ValueError(f"{variant}/{route} has no compiled bundle")
            manifest = Path(entry["manifest"])
            if not manifest.is_absolute():
                manifest = args.draft_variants_manifest.parent / manifest
            if sha256_file(manifest) != entry["manifest_sha256"]:
                raise ValueError(f"{variant}/{route} manifest changed after build")
            deployment = load_json_object(manifest)
            draft = next(g for g in deployment["graphs"] if g["name"] == "draft")
            if draft["metadata"]["incremental_contract"].get("draft_quantization", "fp16") != variant:
                raise ValueError(f"{variant}/{route} Draft precision does not match the selected label")
            setattr(options, route + "_deployment_manifest", manifest)
        summary = matrix.prepare(options, directory)
        summary["draft_quantization"] = variant
        summary["protocol"]["ordinary_baseline_policy"] = "once per prompt/output budget; shared across all Drafts and GDR routes"
        for route in routes:
            bundle = summary["bundles"][route]
            contract = bundle["ordinary_contract"]
            if ordinary is not None and ordinary != contract:
                raise ValueError("Draft variants must share identical ordinary Target OMs and feature ABI; rebuild using shared_draft_features")
            ordinary = contract
            bundle["checkpoint"] = index["bundles"][variant][route]["checkpoint"]
        if prompts is not None and prompts != summary["prompts"]:
            raise ValueError("Draft variants use different prompt tokens or dataset provenance")
        prompts = summary["prompts"]
        options._prepared_summary = summary
        options._matrix_root = directory
        result[variant] = options
    if len(result) > 1:
        runner = next(iter(result.values())).runner
        help_result = subprocess.run([str(runner), "--help"], capture_output=True, text=True)
        if help_result.returncode or "dflash-only batch" not in help_result.stdout:
            raise RuntimeError("rebuild the C++ runner for ordinary baseline reuse")
    return result


def compare(cells):
    result = []
    for cell in cells:
        if cell["draft_quantization"] == "fp16":
            continue
        baseline = next((c for c in cells if c["draft_quantization"] == "fp16" and
                         c["verify_gdr"] == cell["verify_gdr"] and c["max_new_tokens"] == cell["max_new_tokens"]), None)
        if baseline is None:
            continue
        ref = {r["id"]: r for r in baseline.get("cases", []) if r["status"] in suite.MEASURED_STATUSES}
        cur = {r["id"]: r for r in cell.get("cases", []) if r["status"] in suite.MEASURED_STATUSES}
        matched = sorted(ref.keys() & cur.keys())
        row = {k: cell[k] for k in ("draft_quantization", "verify_gdr", "max_new_tokens")}
        row.update(matched_prompt_ids=matched, time_speedup_vs_fp16=None, throughput_vs_fp16=None, acceptance_delta_pp=None)
        if matched:
            a, b = (suite.aggregate_metrics([group[n] for n in matched]) for group in (ref, cur))
            row.update(time_speedup_vs_fp16=sum(ref[n]["dflash_total_measured_ms"] for n in matched) / sum(cur[n]["dflash_total_measured_ms"] for n in matched),
                       throughput_vs_fp16=b["dflash_tokens_per_second"] / a["dflash_tokens_per_second"])
            if a["weighted_acceptance_rate"] is not None and b["weighted_acceptance_rate"] is not None:
                row["acceptance_delta_pp"] = 100 * (b["weighted_acceptance_rate"] - a["weighted_acceptance_rate"])
        result.append(row)
    return result


def save(root, prepared):
    cells, datasets, timing = [], [], []
    lines = ["| Draft | GDR | Output budget | Group | Measured / selected | Acceptance | Tokens / round | Ordinary tok/s | DFlash tok/s | Speedup | Draft ms/call | Verify ms/call |",
             "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    def number(value, percent=False):
        return "N/A" if value is None else f"{value:.2%}" if percent else f"{value:.2f}"
    for variant, options in prepared.items():
        summary = options._prepared_summary
        for cell in summary["cells"]:
            cells.append(dict(cell, draft_quantization=variant))
            groups = {"all": cell.get("aggregate", {})}
            if len({p.get("group", "custom") for p in summary["prompts"]}) > 1:
                groups.update(cell.get("aggregate_by_group", {}))
            for group, a in groups.items():
                stages = a.get("stage_ms_per_call", {})
                count = sum(group == "all" or p.get("group", "custom") == group for p in summary["prompts"])
                values = [variant, cell["verify_gdr"], cell["max_new_tokens"], group,
                          f"{a.get('measured_prompts', 0)} / {count}",
                          number(a.get("weighted_acceptance_rate"), True), number(a.get("tokens_per_speculative_round")),
                          number(a.get("ordinary_tokens_per_second")), number(a.get("dflash_tokens_per_second")),
                          number(a.get("total_model_time_speedup")), number(stages.get("draft", {}).get("mean_ms")),
                          number(stages.get("verify", {}).get("mean_ms"))]
                lines.append("| " + " | ".join(map(str, values)) + " |")
                timing.append(dict(id=f"{variant}/{cell['verify_gdr']}/{cell['max_new_tokens']}/{group}",
                                   stage_timings=stages, phase_timings=a.get("phase_timings", {})))
            if cell.get("error"):
                lines.append(f"\n{variant}/{cell['verify_gdr']}/{cell['max_new_tokens']}: {cell['error']}\n")
        for d in suite.dataset_results(summary):
            datasets.append(dict(d, draft_quantization=variant))
    if datasets:
        lines += ["", suite.render_datasets(datasets).rstrip()]
    comparisons = compare(cells)
    if comparisons:
        lines += ["", "| Draft | GDR | Output budget | Matched | Acceptance change (pp) | Throughput / FP16 | Time speedup vs FP16 |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for row in comparisons:
            lines.append(f"| {row['draft_quantization']} | {row['verify_gdr']} | {row['max_new_tokens']} | "
                         f"{len(row['matched_prompt_ids'])} | {number(row['acceptance_delta_pp'])} | "
                         f"{number(row['throughput_vs_fp16'])} | {number(row['time_speedup_vs_fp16'])} |")
    lines += ["", suite.render_timings(timing, measured_only=False).rstrip(), "",
              "Ordinary runs once per prompt/output budget. Each Draft generates its own output; task quality is not evaluated.",
              "Published quantized checkpoints have five layers; the current FP16 checkpoint has six. This is not a bitwidth-only ablation.",
              "Quantized Drafts retain packed weights and FP16 activations. The AIR/deployment draft_checkpoint_audit records the MatMul backend; peak workspace requires device profiling.",
              "Per-Draft reports, cases.csv and per-file summaries are in fp16/, w4a16/, w8a16/."]
    first = next(iter(prepared.values()))._prepared_summary
    lines.insert(0, "Thinking: " + ("on" if first["protocol"].get("enable_thinking") else "off") + ".\n")
    result = dict(schema_version=1, cells=cells, datasets=datasets, comparisons_vs_fp16=comparisons,
                  prompts=first["prompts"], input_datasets=first.get("datasets", []),
                  draft_quantizations=list(prepared),
                  routes=first.get("routes", []), lengths=first.get("lengths", []),
                  formal_latency_evidence=False,
                  bundles={v: o._prepared_summary["bundles"] for v, o in prepared.items()},
                  protocol=next(iter(prepared.values()))._prepared_summary["protocol"],
                  quality_evaluation="NOT_RUN", status=("PREPARED" if all(o.plan_only for o in prepared.values()) else
                    suite.measured_status(cells) if all(c["status"] in suite.MEASURED_STATUSES for c in cells) else "FAIL_OR_INCOMPLETE"))
    atomic_write_json(root / "summary.json", result)
    suite.write_dataset_reports(root, dict(cells=cells, datasets=first.get("datasets", []),
                                           protocol=result["protocol"]))
    text = "\n".join(lines) + "\n"
    (root / "summary.md").write_text(text)
    # Existing matrix CSVs retain all phase/call latencies and dataset provenance.
    rows = []
    for variant in prepared:
        path = root / variant / "cases.csv"
        if path.exists():
            with path.open(newline="") as stream:
                rows.extend(dict(row, draft_quantization=variant) for row in csv.DictReader(stream))
    if rows:
        with (root / "cases.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["draft_quantization", *[k for k in rows[0] if k != "draft_quantization"]])
            writer.writeheader()
            writer.writerows(rows)
    return text


def run(args):
    args.draft_quantizations = getattr(args, "draft_quantizations", None) or [os.environ.get("DRAFT_QUANTIZATION", "fp16")]
    if any(v not in VARIANTS for v in args.draft_quantizations):
        raise ValueError("invalid Draft precision selection")
    if not getattr(args, "draft_variants_manifest", None):
        bundle = getattr(args, "bundle_dir", None)
        if bundle:
            args.draft_variants_manifest = Path(bundle) / "draft-variants.json"
        elif os.environ.get("DRAFT_VARIANTS_MANIFEST"):
            args.draft_variants_manifest = Path(os.environ["DRAFT_VARIANTS_MANIFEST"])
        elif getattr(args, "chunk_deployment_manifest", None):
            args.draft_variants_manifest = args.chunk_deployment_manifest.parent / "draft-variants.json"
        else:
            raise ValueError("--bundle-dir is required when selecting Draft types")
    args.draft_variants_manifest = Path(args.draft_variants_manifest).expanduser().resolve()
    args.run_dir = args.run_dir.expanduser().resolve()
    if not args.run_dir.is_dir() or args.run_dir.is_relative_to(REPO):
        raise ValueError("run-dir must exist outside the repository")
    os.environ["AI_RUN_DIR"] = str(args.run_dir)
    prefix = "gdr-lengths-" if getattr(args, "_unified_entry", False) else "draft-comparison-"
    root = Path(tempfile.mkdtemp(prefix=prefix, dir=args.run_dir))
    print(f"Output: {root}", flush=True)
    prepared = prepare(args, root)  # All variants/routes preflight before device work.
    baselines, codes = {}, []
    external_baseline = getattr(args, "ordinary_baseline", None)
    if external_baseline:
        from tools.ordinary_baseline import resolve_sources, preflight_sources
        baselines = resolve_sources(external_baseline, args.lengths)
        preflight_sources(baselines, next(iter(prepared.values()))._prepared_summary, args)
    for i, (variant, options) in enumerate(prepared.items()):
        options._shared_baselines = baselines
        options._ordinary_baseline_required = bool(external_baseline) or i != 0
        print(f"[draft-comparison] {variant} start", flush=True)
        code = matrix.run(options)
        codes.append(code)
        save(root, prepared)
        if code == 130:
            return 130
    print(save(root, prepared), flush=True)
    print(f"Summary: {root / 'summary.json'}", flush=True)
    return 1 if any(codes) else 0


def parser():
    result = matrix.parser()
    result.description = __doc__
    result.set_defaults(lengths=[128], verify_gdr="chunk")
    result.set_defaults(draft_variants_manifest=os.environ.get("DRAFT_VARIANTS_MANIFEST"),
                        draft_quantizations=list(VARIANTS))
    return result


def main():
    cli = parser()
    try:
        return run(cli.parse_args())
    except (OSError, ValueError, KeyError, RuntimeError, TypeError, IndexError) as error:
        cli.exit(2, f"draft-comparison: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
