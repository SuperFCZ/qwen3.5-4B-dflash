#!/usr/bin/env python3
"""A4.1: gate KV M80 first, then KV M32, then the unchanged full A4 suite."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import a4_contract
import run_a2
import run_a4
from a2_common import load_bundle, record, sha256, write_json
from a3_launch import launch_evidence, load_build_config

KV_CASES = tuple(f"layer-{i}-kv" for i in range(5))


def collect_kv(root, gear, config):
    real = json.loads((root / "suite.json").read_text())
    if (real.get("status") != "PASS" or real.get("context_rows") != gear or
            tuple(c["name"] for c in real.get("cases", [])) != KV_CASES or
            real.get("timing_protocol") != "continuous-v1"):
        raise ValueError("KV gate requires all five layers with continuous-v1")
    launches = []
    for case in real["cases"]:
        execution = case["executions"]["custom"]
        plan = launch_evidence(root / case["name"] / "custom/runner.log", (gear+16,2560,2048),
            config["core_limit"], execution["workspace_bytes"], config["dequant_mode"], config["pipeline_mode"],
            5, config["kv_m80_mode"])
        if config["core_limit"] != 1 and plan["block_dim"] < 2:
            raise ValueError("KV gate requires multi-core execution unless explicit single-core control")
        launches.append(dict(name=case["name"], launch=plan))
    return launches


def run(args):
    config = load_build_config(args.build_config)
    if config.get("launch_version") != 5 or config["dequant_mode"] != "batched":
        raise ValueError("A4.1 requires current v5 batched build")
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    report = dict(abi="dflash-group-quant-linear-a41-v1", status="RUNNING", stage="A4.1", build=config,
                  build_config_sha256=sha256(args.build_config), device_id=args.device_id,
                  timing_protocol="continuous-v1", stop_after=args.stop_after, checks={}, launches={},
                  phase="paired_gears",
                  full_a4_regression="NOT_RUN", full_draft_validation="NOT_RUN", decode_performance="NOT_RUN")
    path = root / "suite.json"
    write_json(path, report)
    try:
        a4_contract.paired_gears(load_bundle(args.c16_bundle,"a4"), load_bundle(args.c64_bundle,"a4"))
        for gear, label in ((64,"kv80"),(16,"kv32")):
            report["phase"] = label
            child = SimpleNamespace(**vars(args))
            child.bundle = getattr(args, f"c{gear}_bundle")
            child.native_om_manifest = getattr(args, f"c{gear}_native_om_manifest")
            child.scope, child.timing_protocol, child.case_filter = "a4", "continuous-v1", KV_CASES
            child.output_dir = root / label
            if run_a2.run(child) != 0:
                report["checks"][label] = dict(status="FAIL")
                if (child.output_dir / "suite.json").is_file():
                    report["checks"][label]["summary"] = record(child.output_dir / "suite.json", root)
                raise ValueError(f"{label} numerical gate failed; later stages were not run")
            report["launches"][label] = collect_kv(child.output_dir, gear, config)
            report["checks"][label] = dict(status="PASS", summary=record(child.output_dir / "suite.json", root))
            write_json(path, report)
            if args.stop_after == label:
                break
        if args.stop_after == "full":
            report["phase"] = "full"
            full_args = SimpleNamespace(**vars(args)); full_args.output_dir = root / "full"
            passed = run_a4.run(full_args) == 0
            report["full_a4_regression"] = "PASS" if passed else "FAIL"
            report["checks"]["full"] = dict(status=report["full_a4_regression"],
                                             summary=record(full_args.output_dir / "suite.json", root))
            if not passed: raise ValueError("full A4 regression failed")
        load_build_config(args.build_config)
        if sha256(args.build_config) != report["build_config_sha256"]:
            raise ValueError("build config changed during A4.1")
        report["status"] = "PASS"
        report["phase"] = "complete"
    except Exception as error:
        report.update(status="FAIL", error=f"{type(error).__name__}: {error}")
    write_json(path, report)
    if report["status"] == "FAIL":
        print(f"FAIL during {report['phase']}: {report['error']}", flush=True)
    print(f"{report['status']}: A4.1 {config['kv_m80_mode']} through {args.stop_after}; "
          f"full A4={report['full_a4_regression']}; summary: {path}", flush=True)
    print("Full Draft / decode performance NOT_RUN", flush=True)
    return 0 if report["status"] == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("build-config","c16-bundle","c64-bundle","c16-native-om-manifest","c64-native-om-manifest",
                 "runner","om-runner","output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--stop-after", choices=("kv80","kv32","full"), default="full")
    args = parser.parse_args()
    if args.device_id < 0 or not args.runner.is_file() or not args.om_runner.is_file():
        parser.error("need valid device and both compiled runners")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
