#!/usr/bin/env python3
"""Compare real A2 custom ACLNN outputs to identical-input native static OMs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from a2_common import (ABI, CASE_NAMES, case_shape, checked_file, load_bundle, record,
                       sha256, validate_execution, validate_layout, validate_native_graph, write_json)
from reference import compare


def validate_om_manifest(path, bundle_path, bundle):
    report = json.loads(path.read_text())
    if (report.get("abi") != ABI or report.get("status") != "PASS" or
            report.get("bundle_sha256") != sha256(bundle_path) or
            report.get("soc_version") != "Ascend310P3" or
            tuple(c.get("name") for c in report.get("cases", [])) != CASE_NAMES):
        raise ValueError("native OM manifest does not match this complete A2 bundle")
    for case, compiled in zip(bundle["cases"], report["cases"]):
        if compiled.get("status") != "PASS" or compiled.get("input_hashes") != {
                name: item["sha256"] for name, item in case["files"].items()}:
            raise ValueError("native OM uses different input/weight/scale bytes")
        for name in ("om", "deployment", "air_manifest"):
            checked_file(path.parent, compiled[name])
        deployment_path = checked_file(path.parent, compiled["deployment"])
        deployment = json.loads(deployment_path.read_text())
        if deployment.get("status") != "PASS" or len(deployment.get("graphs", [])) != 1:
            raise ValueError("native deployment is incomplete")
        graph = deployment["graphs"][0]
        validate_native_graph(graph, case, report["bundle_sha256"])
        if checked_file(deployment_path.parent, graph["om"]) != checked_file(path.parent, compiled["om"]):
            raise ValueError("native OM manifest points to a different deployment model")
        if (compiled.get("offline_weight", {}).get("sha256") != case["files"]["w_nz.bin"]["sha256"] or
                compiled.get("offline_weight", {}).get("logical_sha256") != case["files"]["q_nk.bin"]["sha256"]):
            raise ValueError("native OM offline weight identity differs")
    return report


def cpu_probe(root, case, actual):
    """Small full-K FP64 diagnostic; never a relaxed substitute for OM parity."""
    import numpy as np
    m, k, n = case_shape(case)
    rows = [0, 1, 7, m - 1]
    columns = sorted({0, 15, 16, 63, 64, n // 2 - 1, n // 2, n - 1})
    def array(name, dtype, shape):
        return np.memmap(checked_file(root, case["files"][name]), mode="r", dtype=dtype, shape=shape)
    q = array("q_nk.bin", "i1", (n, k))[columns]
    scales = array("s_gn.bin", "<f2", (k // 128, n))[:, columns].T
    # FP16 dequantization followed by FP64 accumulation on the CPU, sampled
    # across N tiles, both halves of gate/up, and all K groups.
    w = (q.reshape(len(columns), k // 128, 128).astype(np.float16) * scales[..., None]).reshape(len(columns), k)
    x = array("x.bin", "<f2", (m, k))[rows].astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        expected = (x @ w.astype(np.float64).T).astype("<f2")
    output = np.frombuffer(actual, dtype="<f2").reshape(m, n)[np.ix_(rows, columns)]
    return {"status": "DIAGNOSTIC_ONLY", "rows": rows, "columns": columns,
            "accumulation": "FP64 CPU full-K after FP16 dequantization",
            "comparison": compare(expected.tobytes(), output.tobytes(), SimpleNamespace(m=len(rows), n=len(columns)))}


def compare_outputs(custom, native, eager, shape):
    dims = SimpleNamespace(m=shape[0], n=shape[2])
    comparison = {
        "custom_vs_native_om": compare(native[0], custom[0], dims),
        "custom_repeat": compare(custom[0], custom[1], dims),
        "native_om_repeat": compare(native[0], native[1], dims),
        "native_om_vs_eager": compare(eager, native[0], dims),
        "custom_vs_eager": compare(eager, custom[0], dims),
    }
    gate = ("custom_vs_native_om", "custom_repeat", "native_om_repeat")
    comparison["status"] = "PASS" if all(comparison[key]["bitwise_equal"] and comparison[key]["finite"]
                                            for key in gate) else "FAIL"
    comparison["eager_policy"] = "diagnostic; native OM is the A2 numerical gate"
    return comparison


def run(args):
    timing_protocol = getattr(args, "timing_protocol", "checked-v1")
    if timing_protocol not in ("checked-v1", "continuous-v1"):
        raise ValueError("unsupported timing protocol")
    if not 3 <= args.warmup <= 100 or not 10 <= args.repetitions <= 1000 or args.device_id < 0:
        raise ValueError("need nonnegative device, warmup 3..100 and repetitions 10..1000")
    bundle_path, om_manifest = args.bundle.resolve(), args.native_om_manifest.resolve()
    bundle = load_bundle(bundle_path)
    native = validate_om_manifest(om_manifest, bundle_path, bundle)
    runners = {"custom": args.runner.resolve(), "native_om": args.om_runner.resolve()}
    runner_hashes = {name: sha256(path) for name, path in runners.items()}
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    report = {"abi": ABI, "status": "RUNNING", "scope": "10 real native-eager-captured M16 projections",
              "bundle": str(bundle_path), "bundle_sha256": sha256(bundle_path),
              "native_om_manifest": str(om_manifest), "native_om_manifest_sha256": sha256(om_manifest),
              "runners": {name: {"path": str(path), "sha256": runner_hashes[name]} for name, path in runners.items()},
              "device_id": args.device_id, "numerical_gate": "bitwise FP16 vs native OM; atol=0 rtol=0; nonfinite fails",
              "cases": [], "full_draft_validation": "NOT_RUN", "decode_performance": "NOT_RUN",
              "native_om_parity": "NOT_RUN", "isolated_timing": "NOT_RUN",
              "timing_protocol": timing_protocol,
              "process_peak_device_memory": "NOT_MEASURED"}
    summary = root / "suite.json"
    write_json(summary, report)
    for index, (case, compiled) in enumerate(zip(bundle["cases"], native["cases"])):
        row = {"name": case["name"], "status": "RUNNING", "phase": "inputs"}
        report["cases"].append(row)
        directory = root / case["name"]
        directory.mkdir()
        try:
            shape = case_shape(case)
            validate_layout(bundle_path.parent, case)
            om = checked_file(om_manifest.parent, compiled["om"])
            outputs, executions, bracket_outputs = {}, {}, {}
            # Alternate which implementation is run first across pairs.
            order = ("native_om", "custom") if index % 2 == 0 else ("custom", "native_om")
            row["execution_order"] = list(order)
            for kind in order:
                row["phase"] = kind
                dest = directory / kind
                dest.mkdir()
                for name in (("x.bin", "w_nz.bin", "s_gn.bin") if kind == "custom" else ("x.bin",)):
                    (dest / name).symlink_to(checked_file(bundle_path.parent, case["files"][name]))
                command = [str(runners[kind]), str(args.device_id), str(dest)]
                if kind == "native_om":
                    command.append(str(om))
                command += [str(v) for v in (*shape, args.warmup, args.repetitions)]
                if timing_protocol == "continuous-v1":
                    command.append("--continuous")
                write_json(dest / "command.json", command)
                with (dest / "runner.log").open("w") as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                                   timeout=args.timeout)
                execution = json.loads((dest / "execution.json").read_text())
                validate_execution(execution, shape, args.device_id,
                                   "AscendCL ACLNN" if kind == "custom" else "AscendCL native OM",
                                   args.warmup, args.repetitions, timing_protocol)
                executions[kind] = execution
                outputs[kind] = [(dest / f"actual-{i}.bin").read_bytes() for i in range(2)]
                if timing_protocol == "continuous-v1":
                    bracket_outputs[kind] = []
                    for filename in ("benchmark-last.bin", "postcheck.bin"):
                        raw = (dest / filename).read_bytes()
                        if raw != outputs[kind][0]:
                            raise ValueError(f"{kind}: benchmark tail/postcheck differs from correctness output")
                        bracket_outputs[kind].append(record(dest / filename, root))
            row["phase"] = "comparison"
            eager = checked_file(bundle_path.parent, case["files"]["eager-0.bin"]).read_bytes()
            comparison = compare_outputs(outputs["custom"], outputs["native_om"], eager, shape)
            comparison["cpu_sample"] = cpu_probe(bundle_path.parent, case, outputs["custom"][0])
            write_json(directory / "comparison.json", comparison)
            # Recheck sources after execution so stale/changed files cannot pass.
            for item in case["files"].values():
                checked_file(bundle_path.parent, item)
            checked_file(om_manifest.parent, compiled["om"])
            row.update(status=comparison["status"], phase="complete",
                       comparison=record(directory / "comparison.json", root),
                       native_om=compiled["om"], executions=executions,
                       bracket_outputs=bracket_outputs,
                       outputs={kind: [record(directory / kind / f"actual-{i}.bin", root) for i in range(2)]
                                for kind in outputs})
            bits = comparison["custom_vs_native_om"]
            print(f"{row['name']}: {row['status']} custom_vs_native_om bits={bits['bit_mismatches']} ULP={bits['max_ulp']}", flush=True)
        except Exception as error:
            row.update(status="FAIL", error=f"{type(error).__name__}: {error}")
            print(f"{row['name']}: FAIL during {row['phase']}; {row['error']}; logs: {directory}", flush=True)
        write_json(summary, report)
    sources_equal = (sha256(bundle_path) == report["bundle_sha256"] and
                     sha256(om_manifest) == report["native_om_manifest_sha256"] and
                     all(sha256(path) == runner_hashes[name] for name, path in runners.items()))
    passed = sources_equal and all(row["status"] == "PASS" for row in report["cases"])
    report.update(status="PASS" if passed else "FAIL", sources_unchanged=sources_equal,
                  native_om_parity="PASS" if passed else "FAIL_OR_INCOMPLETE",
                  isolated_timing="MEASURED" if all("executions" in r for r in report["cases"]) else "INCOMPLETE")
    write_json(summary, report)
    print(f"{report['status']}: 10 A2 real projections; summary: {summary}\nFull Draft / decode performance NOT_RUN", flush=True)
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--native-om-manifest", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--om-runner", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--timing-protocol", choices=("checked-v1", "continuous-v1"), default="checked-v1")
    parser.add_argument("--timeout", type=int, default=1800, help="seconds per isolated runner process")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
