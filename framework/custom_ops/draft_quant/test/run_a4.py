#!/usr/bin/env python3
"""A4: full-M synthetic gates plus 26 real projections in EACH C16/C64 gear."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import a4_contract
import reference
import run_a2
import run_suite
from a2_common import case_shape, load_bundle, record, sha256, write_json
from a3_launch import launch_evidence, load_build_config


def collect_launches(root, config):
    evidence = {"synthetic": [], "c16": [], "c64": []}
    synthetic = json.loads((root / "synthetic/suite.json").read_text())
    if (synthetic.get("status") != "PASS" or
            tuple(w["name"] for w in synthetic["workloads"]) != tuple(s.name for s in reference.A4_WORKLOADS)):
        raise ValueError("A4 requires all 36 synthetic M/N/K workloads")

    def plan(directory, shape):
        execution = json.loads((directory / "execution.json").read_text())
        if (execution.get("status") != "PASS" or
                execution.get("tile_m") != shape[0] or execution.get("weight_reuse_rows") != shape[0]):
            raise ValueError("A4 runner must report full-M output/weight reuse")
        return launch_evidence(directory / "runner.log", shape, config["core_limit"], execution["workspace_bytes"],
                               config["dequant_mode"], config["pipeline_mode"], config.get("launch_version", 4), config.get("kv_m80_mode", "baseline"))

    for shape, workload in zip(reference.A4_WORKLOADS, synthetic["workloads"]):
        names = tuple(c[0] for c in reference.cases(shape))
        if (workload.get("status") != "PASS" or workload.get("mkn") != [shape.m, shape.k, shape.n] or
                tuple(c["name"] for c in workload["custom_cases"]) != names):
            raise ValueError("incomplete A4 synthetic case inventory")
        for name in names:
            directory = root / "synthetic" / shape.name / name
            evidence["synthetic"].append({"name": f"{shape.name}/{name}",
                                           "launch": plan(directory, (shape.m, shape.k, shape.n))})
    for gear in (16, 64):
        label = f"c{gear}"
        real = json.loads((root / label / "suite.json").read_text())
        if (real.get("status") != "PASS" or real.get("abi") != a4_contract.ABI or
                real.get("context_rows") != gear or
                tuple(c["name"] for c in real.get("cases", [])) != a4_contract.CASE_NAMES):
            raise ValueError(f"A4 requires all 26 real projections for C{gear}")
        for expected, case in zip(a4_contract.cases(gear), real["cases"]):
            shape = case_shape(expected, "a4")
            launch = plan(root / label / case["name"] / "custom", shape)
            if config["core_limit"] != 1 and launch["block_dim"] < 2:
                raise ValueError("A4 auto/multi-core needs at least two cores for every real projection")
            custom, native = (case["executions"][kind] for kind in ("custom", "native_om"))
            evidence[label].append({"name": case["name"], "launch": launch,
                "custom_timing": custom["timing"], "native_om_timing": native["timing"],
                "custom_workspace_bytes": custom["workspace_bytes"], "native_om_workspace_bytes": native["workspace_bytes"]})
    return evidence


def run(args):
    if args.device_id < 0 or not 3 <= args.warmup <= 100 or not 10 <= args.repetitions <= 1000:
        raise ValueError("need nonnegative device, warmup 3..100 and repetitions 10..1000")
    config = load_build_config(args.build_config)
    if config.get("tiling_abi") != "full-m-v2" or config["dequant_mode"] != "batched":
        raise ValueError("A4 needs a current full-M build with batched dequantization")
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    report = {"abi": "dflash-group-quant-linear-a4-suite-v1", "status": "RUNNING", "stage": "A4",
              "build": config, "build_config_sha256": sha256(args.build_config), "device_id": args.device_id,
              "scope": "36 synthetic shapes + 26 same-input native OM projection comparisons per C16/C64 gear",
              "timing_protocol": "continuous-v1", "checks": {}, "launches": {},
              "full_draft_validation": "NOT_RUN", "decode_performance": "NOT_RUN"}
    summary = root / "suite.json"
    write_json(summary, report)
    try:
        bundles = {gear: load_bundle(getattr(args, f"c{gear}_bundle"), "a4") for gear in (16, 64)}
        a4_contract.paired_gears(bundles[16], bundles[64])
        for gear in (16, 64):
            run_a2.validate_om_manifest(getattr(args, f"c{gear}_native_om_manifest"),
                                        getattr(args, f"c{gear}_bundle"), bundles[gear], "a4")
        report["inputs"] = {f"c{gear}": {kind: {"path": str(getattr(args, f"c{gear}_{kind}").resolve()),
                                              "sha256": sha256(getattr(args, f"c{gear}_{kind}"))}
                                       for kind in ("bundle", "native_om_manifest")} for gear in (16, 64)}
        report["checks"]["paired_gears"] = {"status": "PASS"}
    except Exception as error:
        report.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        report["checks"]["paired_gears"] = {"status": "FAIL"}
        write_json(summary, report)
        print(f"FAIL: A4 capture/native preflight: {error}; summary: {summary}", flush=True)
        return 1
    for name in ("synthetic", "c16", "c64"):
        try:
            if name == "synthetic":
                passed = run_suite.run(root / name, args.runner.resolve(), args.device_id, "a4")
            else:
                real_args = SimpleNamespace(**vars(args))
                real_args.output_dir, real_args.scope, real_args.timing_protocol = root / name, "a4", "continuous-v1"
                real_args.bundle = getattr(args, f"{name}_bundle")
                real_args.native_om_manifest = getattr(args, f"{name}_native_om_manifest")
                passed = run_a2.run(real_args) == 0
            report["checks"][name] = {"status": "PASS" if passed else "FAIL", "summary": record(root / name / "suite.json", root)}
        except Exception as error:
            report["checks"][name] = {"status": "FAIL", "error": f"{type(error).__name__}: {error}"}
        write_json(summary, report)
    try:
        report["launches"] = collect_launches(root, config)
        load_build_config(args.build_config)
        if sha256(args.build_config) != report["build_config_sha256"]:
            raise ValueError("build configuration changed during execution")
        for inputs in report["inputs"].values():
            for item in inputs.values():
                if sha256(item["path"]) != item["sha256"]:
                    raise ValueError("C16/C64 capture/native manifest changed during the full suite")
        a4_contract.paired_gears(load_bundle(args.c16_bundle, "a4"), load_bundle(args.c64_bundle, "a4"))
        report["checks"]["launches"] = {"status": "PASS"}
    except Exception as error:
        report["checks"]["launches"] = {"status": "FAIL", "error": f"{type(error).__name__}: {error}"}
    report["status"] = "PASS" if all(c["status"] == "PASS" for c in report["checks"].values()) else "FAIL"
    write_json(summary, report)
    print(f"{report['status']}: A4 M16/32/64/80; 36 synthetic shapes + C16/C64 52 real projections; summary: {summary}", flush=True)
    print("Full Draft / decode performance NOT_RUN", flush=True)
    return 0 if report["status"] == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("build-config", "c16-bundle", "c64-bundle", "c16-native-om-manifest", "c64-native-om-manifest",
                 "runner", "om-runner", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    if args.device_id < 0 or not args.runner.is_file() or not args.om_runner.is_file():
        parser.error("need valid device and both compiled runners")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
