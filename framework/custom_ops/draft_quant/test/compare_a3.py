#!/usr/bin/env python3
"""Compare fresh A3 single-core and multi-core runs on the same captured inputs.

This reports isolated diagnostic timing, never a full-Draft/Decode speedup.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
from types import SimpleNamespace

from a2_common import CASE_NAMES, checked_file, sha256, write_json
from a3_launch import POLICY
from reference import compare


def load_run(path, mode):
    path = Path(path).resolve()
    report = json.loads(path.read_text())
    if (report.get("abi") != "dflash-group-quant-linear-a3-v1" or report.get("policy") != POLICY or
            report.get("status") != "PASS" or report.get("mode") != mode or
            set(report.get("checks", {})) != {"a1", "real", "launches"} or
            any(c.get("status") != "PASS" for c in report["checks"].values()) or
            (mode == "multi-core" and report.get("multicore_validation") != "PASS")):
        raise ValueError(f"need a passing A3 {mode} run")
    if mode == "single-core-control" and report["build"]["core_limit"] != 1:
        raise ValueError("single-core control must be built with core cap 1")
    a1_path = checked_file(path.parent, report["checks"]["a1"]["summary"])
    real_path = checked_file(path.parent, report["checks"]["real"]["summary"])
    real = json.loads(real_path.read_text())
    if (json.loads(a1_path.read_text()).get("status") != "PASS" or real.get("status") != "PASS" or
            tuple(c["name"] for c in real.get("cases", [])) != CASE_NAMES or
            tuple(c["name"] for c in report.get("launches", {}).get("real", [])) != CASE_NAMES):
        raise ValueError("incomplete A3 constituent evidence")
    for key in ("bundle_sha256", "native_om_manifest_sha256", "device_id"):
        if report.get(key) != real.get(key):
            raise ValueError(f"A3 aggregate/real identity differs: {key}")
    for case, launched in zip(real["cases"], report["launches"]["real"]):
        cores = launched["launch"]["block_dim"]
        if (mode == "single-core-control" and cores != 1) or (mode == "multi-core" and cores < 2):
            raise ValueError("recorded launch is not the requested control/candidate")
        for kind in ("custom", "native_om"):
            if len(case["outputs"][kind]) != 2:
                raise ValueError("need both repeated outputs")
            for output in case["outputs"][kind]:
                checked_file(real_path.parent, output)
        checked_file(real_path.parent, case["comparison"])
    return report, real, real_path.parent


def compatible_runs(single, multi):
    for key in ("bundle_sha256", "native_om_manifest_sha256", "device_id"):
        if single.get(key) is None or single.get(key) != multi.get(key):
            raise ValueError(f"single/multi runs must use the same {key}")
    if not single["build"].get("source_sha256") or single["build"]["source_sha256"] != multi["build"].get("source_sha256"):
        raise ValueError("control/candidate must use the same host/kernel/runner source; only the build core cap differs")


def timings(execution):
    timing = execution["timing"]
    samples = timing["execute_sync"]["samples_ms"]
    if (timing["status"] != "MEASURED" or timing["warmup"] < 3 or timing["repetitions"] < 10 or
            len(samples) != timing["repetitions"] or any(not math.isfinite(x) or x <= 0 for x in samples)):
        raise ValueError("missing/invalid diagnostic timing samples")
    return {"samples_ms": samples, "median_ms": statistics.median(samples),
            "p95_ms": sorted(samples)[math.ceil(len(samples) * .95) - 1]}


def run(single_path, multi_path, output):
    if output.exists():
        raise FileExistsError("use a new comparison output path")
    single, single_real, single_root = load_run(single_path, "single-core-control")
    multi, multi_real, multi_root = load_run(multi_path, "multi-core")
    compatible_runs(single, multi)
    result = {"status": "RUNNING", "scope": "isolated A3 single/multi-core diagnostic; not formal performance acceptance",
              "single_summary_sha256": sha256(single_path), "multi_summary_sha256": sha256(multi_path),
              "full_draft_validation": "NOT_RUN", "decode_performance": "NOT_RUN", "cases": []}
    for a, b in zip(single_real["cases"], multi_real["cases"]):
        left, right = a["executions"]["custom"], b["executions"]["custom"]
        for key in ("m", "k", "n", "device_id"):
            if left[key] != right[key]:
                raise ValueError(f"execution identity differs: {key}")
        for key in ("scope", "warmup", "repetitions"):
            if left["timing"][key] != right["timing"][key]:
                raise ValueError(f"timing protocol differs: {key}")
        dimensions = SimpleNamespace(m=left["m"], n=left["n"])
        expected = checked_file(single_root, a["outputs"]["custom"][0]).read_bytes()
        comparisons = [compare(expected, checked_file(root, case["outputs"][kind][repeat]).read_bytes(), dimensions)
                       for root, case in ((single_root, a), (multi_root, b))
                       for kind in ("custom", "native_om") for repeat in range(2)]
        passed = all(c["finite"] and c["bitwise_equal"] for c in comparisons)
        before, after = timings(left), timings(right)
        result["cases"].append({"name": a["name"], "status": "PASS" if passed else "FAIL",
            "comparisons": comparisons, "single": before, "multi": after,
            "single_to_multi_median_ratio": before["median_ms"] / after["median_ms"],
            "single_workspace_bytes": left["workspace_bytes"], "multi_workspace_bytes": right["workspace_bytes"]})
        print(f"{a['name']}: single={before['median_ms']:.4f} ms multi={after['median_ms']:.4f} ms; "
              f"bitwise={'PASS' if passed else 'FAIL'}", flush=True)
    result["status"] = "PASS" if all(c["status"] == "PASS" for c in result["cases"]) else "FAIL"
    write_json(output, result)
    return 0 if result["status"] == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single", type=Path, required=True, help="A3 core-cap=1 data/suite.json")
    parser.add_argument("--multi", type=Path, required=True, help="A3 multi-core data/suite.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return run(args.single.resolve(), args.multi.resolve(), args.output.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
