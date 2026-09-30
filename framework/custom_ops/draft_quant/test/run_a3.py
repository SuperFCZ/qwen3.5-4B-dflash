#!/usr/bin/env python3
"""Run A1+A2 gates with an A3 OPP and verify the recorded multi-core launch plan."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from a2_common import CASE_NAMES, record, sha256, write_json
from a3_launch import POLICY, launch_evidence, load_build_config
import run_a2
import run_suite


def collect_launches(a1_root, real_root, limit):
    a1 = json.loads((a1_root / "suite.json").read_text())
    real = json.loads((real_root / "suite.json").read_text())
    if a1.get("status") != "PASS" or real.get("status") != "PASS":
        raise ValueError("A3 requires both A1 and real-projection numerical gates to pass")
    if tuple(row["name"] for row in real["cases"]) != CASE_NAMES:
        raise ValueError("incomplete real-projection inventory")
    evidence = {"a1": [], "real": []}
    for workload in a1["workloads"]:
        for case in workload["custom_cases"]:
            directory = a1_root / workload["name"] / case["name"]
            execution = json.loads((directory / "execution.json").read_text())
            plan = launch_evidence(directory / "runner.log", workload["mkn"], limit, execution["workspace_bytes"])
            evidence["a1"].append({"name": workload["name"] + "/" + case["name"], "launch": plan})
    if len(evidence["a1"]) != 81:
        raise ValueError("A3 requires all 81 A1 cases")
    for case in real["cases"]:
        execution = case["executions"]["custom"]
        plan = launch_evidence(real_root / case["name"] / "custom/runner.log",
                               [execution[d] for d in ("m", "k", "n")], limit, execution["workspace_bytes"])
        if limit != 1 and plan["block_dim"] < 2:
            raise ValueError("A3 multi-core acceptance requires at least two cores on every real projection")
        native = case["executions"]["native_om"]
        evidence["real"].append({"name": case["name"], "launch": plan,
            "custom_timing": execution["timing"], "native_om_timing": native["timing"],
            "custom_workspace_bytes": execution["workspace_bytes"], "native_om_workspace_bytes": native["workspace_bytes"]})
    return evidence, real


def run(args):
    config = load_build_config(args.build_config)
    limit = config["core_limit"]
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    report = {"abi": "dflash-group-quant-linear-a3-v1", "policy": POLICY, "status": "RUNNING",
              "mode": "single-core-control" if limit == 1 else "multi-core",
              "build": config, "build_config_sha256": sha256(args.build_config),
              "device_id": args.device_id, "checks": {}, "multicore_validation": "NOT_RUN",
              "timing_scope": "isolated diagnostic launch+sync, same A2 measurement protocol",
              "full_draft_validation": "NOT_RUN", "decode_performance": "NOT_RUN"}
    summary = root / "suite.json"
    write_json(summary, report)
    for name in ("a1", "real"):
        try:
            if name == "a1":
                passed = run_suite.run(root / name, args.runner.resolve(), args.device_id, "a1")
            else:
                real_args = SimpleNamespace(**vars(args))
                real_args.output_dir = root / name
                passed = run_a2.run(real_args) == 0
            report["checks"][name] = {"status": "PASS" if passed else "FAIL",
                                      "summary": record(root / name / "suite.json", root)}
        except Exception as error:
            report["checks"][name] = {"status": "FAIL", "error": f"{type(error).__name__}: {error}"}
        write_json(summary, report)
    try:
        launches, real = collect_launches(root / "a1", root / "real", limit)
        load_build_config(args.build_config)
        if sha256(args.build_config) != report["build_config_sha256"]:
            raise ValueError("build configuration changed during execution")
        report.update(launches=launches, bundle_sha256=real["bundle_sha256"],
                      native_om_manifest_sha256=real["native_om_manifest_sha256"],
                      multicore_validation="PASS" if limit != 1 else "NOT_RUN")
        report["checks"]["launches"] = {"status": "PASS"}
    except Exception as error:
        report["checks"]["launches"] = {"status": "FAIL", "error": f"{type(error).__name__}: {error}"}
        print(f"FAIL: A3 launch validation: {error}", flush=True)
    report["status"] = "PASS" if all(row["status"] == "PASS" for row in report["checks"].values()) else "FAIL"
    write_json(summary, report)
    print(f"{report['status']}: A3 {report['mode']}; A1 + 10 real projections; summary: {summary}", flush=True)
    print("Full Draft / decode performance NOT_RUN", flush=True)
    return 0 if report["status"] == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-config", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--native-om-manifest", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--om-runner", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    if args.device_id < 0 or not args.runner.is_file() or not args.om_runner.is_file():
        parser.error("need a valid device id and both compiled runners")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
