#!/usr/bin/env python3
"""Compare legacy/batched dequantization with identical continuous timing."""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

from a2_common import checked_file, sha256, validate_execution, write_json
from compare_a3 import compatible_runs, load_run, timings
from reference import compare


def matched_variants(baseline, candidate):
    compatible_runs(baseline, candidate, allow_dequant_change=True)
    if (baseline.get("timing_protocol") != "continuous-v1" or
            candidate.get("timing_protocol") != "continuous-v1"):
        raise ValueError("A3.1 requires fresh continuous-v1 measurements for BOTH variants")
    if (baseline["build"].get("dequant_mode") != "legacy" or
            candidate["build"].get("dequant_mode") != "batched"):
        raise ValueError("baseline must be legacy and candidate must be batched")
    if baseline["build"]["core_limit"] != candidate["build"]["core_limit"]:
        raise ValueError("dequantization comparisons require the same core cap")
    for a, b in zip(baseline["launches"]["real"], candidate["launches"]["real"]):
        for key in ("block_dim", "available_cores", "m", "k", "n", "tile_n", "tile_k"):
            if a["launch"][key] != b["launch"][key]:
                raise ValueError(f"launch differs between variants: {key}")
        if a["launch"].get("dequant_mode") != "legacy" or b["launch"].get("dequant_mode") != "batched":
            raise ValueError("recorded host variant differs from the requested comparison")


def compare_variants(baseline_path, candidate_path, output, *, stage, match, labels):
    if output.exists():
        raise FileExistsError("use a new comparison output path")
    baseline, before, before_root = load_run(baseline_path, "multi-core")
    candidate, after, after_root = load_run(candidate_path, "multi-core")
    match(baseline, candidate)
    before_label, after_label = labels
    ratio_key = f"{before_label}_to_{after_label}_median_ratio"
    result = {"status": "RUNNING", "scope": f"{stage} isolated continuous benchmark; PASS means numerical gates",
              "baseline_summary": str(baseline_path), "candidate_summary": str(candidate_path),
              "baseline_summary_sha256": sha256(baseline_path), "candidate_summary_sha256": sha256(candidate_path),
              "timing_protocol": "continuous-v1", "full_draft_validation": "NOT_RUN", "decode_performance": "NOT_RUN",
              "cases": []}
    for a, b in zip(before["cases"], after["cases"]):
        shape = tuple(a["executions"]["custom"][d] for d in ("m", "k", "n"))
        dimensions = SimpleNamespace(m=shape[0], n=shape[2])
        expected = checked_file(before_root, a["outputs"]["native_om"][0]).read_bytes()
        differences = []
        protocol = a["executions"]["custom"]["timing"]
        for case, root in ((a, before_root), (b, after_root)):
            if case["status"] != "PASS":
                raise ValueError("cannot compare a failed case")
            for kind in ("custom", "native_om"):
                execution = case["executions"][kind]
                validate_execution(execution, shape, baseline["device_id"],
                                   "AscendCL ACLNN" if kind == "custom" else "AscendCL native OM",
                                   protocol["warmup"], protocol["repetitions"], "continuous-v1")
                bracket = case.get("bracket_outputs", {}).get(kind, [])
                if len(bracket) != 2:
                    raise ValueError("missing timed tail/post-poison output evidence")
                for artifact in case["outputs"][kind] + bracket:
                    differences.append(compare(expected, checked_file(root, artifact).read_bytes(), dimensions))
        valid = all(d["finite"] and d["bitwise_equal"] for d in differences)
        old, new = timings(a["executions"]["custom"]), timings(b["executions"]["custom"])
        native_before, native_after = timings(a["executions"]["native_om"]), timings(b["executions"]["native_om"])
        row = {"name": a["name"], "status": "PASS" if valid else "FAIL", "comparisons": differences,
               before_label: old, after_label: new, "native_before": native_before, "native_after": native_after,
               ratio_key: old["median_ms"] / new["median_ms"],
               f"{after_label}_to_native_median_ratio": new["median_ms"] / native_after["median_ms"],
               f"{before_label}_workspace_bytes": a["executions"]["custom"]["workspace_bytes"],
               f"{after_label}_workspace_bytes": b["executions"]["custom"]["workspace_bytes"]}
        result["cases"].append(row)
        print(f"{row['name']}: {row['status']} {before_label}={old['median_ms']:.3f} ms "
              f"{after_label}={new['median_ms']:.3f} ms native={native_after['median_ms']:.3f} ms "
              f"{before_label}/{after_label}={row[ratio_key]:.2f}x", flush=True)
    result["status"] = "PASS" if all(c["status"] == "PASS" for c in result["cases"]) else "FAIL"
    write_json(output, result)
    return 0 if result["status"] == "PASS" else 1


def run(baseline_path, candidate_path, output):
    return compare_variants(baseline_path, candidate_path, output, stage="A3.1",
                            match=matched_variants, labels=("legacy", "batched"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="fresh a31 legacy data/suite.json")
    parser.add_argument("--candidate", type=Path, required=True, help="fresh a31 batched data/suite.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return run(args.baseline.resolve(), args.candidate.resolve(), args.output.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
