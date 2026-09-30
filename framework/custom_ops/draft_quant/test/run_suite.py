#!/usr/bin/env python3
"""Run isolated ACLNN/native comparisons for all A1 shapes, without the model."""
import argparse
from pathlib import Path
import subprocess
import sys
import traceback

import reference as ref


def run(root, runner, device_id, suite):
    shapes = {"a1": ref.A1_WORKLOADS, "tiny": (ref.TINY,), "a4": ref.A4_WORKLOADS}[suite]
    root.mkdir(parents=True, exist_ok=False)
    report_path = root / "suite.json"
    report = {"schema_version": 1, "suite": suite, "status": "RUNNING",
              "scope": "synthetic M/N/K tiling correctness; no model integration",
              "device_id": device_id, "cpu_fallback": False,
              "native_om_parity": "NOT_RUN", "performance": "NOT_RUN", "full_draft_validation": "NOT_RUN",
              "workloads": [{"name": s.name, "mkn": [s.m, s.k, s.n], "status": "NOT_RUN"} for s in shapes]}
    ref.write_json(report_path, report)
    for shape, result in zip(shapes, report["workloads"]):
        directory = root / shape.name
        result.update(status="RUNNING", phase="prepare")
        try:
            ref.prepare(directory, shape)
            manifest = ref.load_cases(directory)
            result["custom_cases"] = []
            result["phase"] = "custom_aclnn"
            for case in manifest["cases"]:
                case_dir = directory / case["name"]
                command = [str(runner), str(device_id), str(case_dir), str(shape.m), str(shape.k), str(shape.n)]
                log_path = case_dir / "runner.log"
                print(f"custom {shape.name}/{case['name']}", flush=True)
                with log_path.open("w") as log:
                    completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                print(log_path.read_text(), end="", flush=True)
                result["custom_cases"].append({"name": case["name"], "returncode": completed.returncode,
                                                "command": command, "log": str(log_path)})
                if completed.returncode:
                    raise RuntimeError(f"ACLNN failed for {case['name']}; see {log_path}")
            result["phase"] = "native_eager"
            # A separate process owns native NPU initialization/cleanup. The
            # custom runner resets the device, so don't retain torch NPU state.
            native_command = [sys.executable, str(Path(ref.__file__).resolve()),
                              "native", str(directory), "--device-id", str(device_id)]
            result["native_command"] = native_command
            subprocess.run(native_command, check=True)
            result["phase"] = "compare"
            passed = ref.check(directory)
            result.update(status="PASS" if passed else "FAIL", phase="complete",
                          comparison=str(directory / "comparison.json"))
        except Exception as error:
            result.update(status="FAIL", error=f"{type(error).__name__}: {error}")
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "suite-error.txt").write_text(traceback.format_exc())
            print(f"FAIL {shape.name} during {result['phase']}: {error}", flush=True)
        ref.write_json(report_path, report)
    report["status"] = "PASS" if all(w["status"] == "PASS" for w in report["workloads"]) else "FAIL"
    ref.write_json(report_path, report)
    print(f"{report['status']}: {len(shapes)} {suite} workloads; summary: {report_path}", flush=True)
    return report["status"] == "PASS"


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output-dir", type=Path, required=True)
    cli.add_argument("--runner", type=Path, required=True)
    cli.add_argument("--device-id", type=int, default=0)
    cli.add_argument("--suite", choices=("a1", "tiny", "a4"), default="a1")
    args = cli.parse_args()
    if not args.runner.is_file(): cli.error("ACLNN runner does not exist")
    if args.device_id < 0: cli.error("device-id must be nonnegative")
    return 0 if run(args.output_dir.resolve(), args.runner.resolve(), args.device_id, args.suite) else 1


if __name__ == "__main__":
    raise SystemExit(main())
