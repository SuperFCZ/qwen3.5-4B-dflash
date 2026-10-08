#!/usr/bin/env python3
"""Paired continuous-v1 A4.1 comparison, including unchanged-shape controls."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import a4_contract
from a2_common import checked_file, sha256, validate_execution, write_json
from a3_launch import launch_evidence
from compare_a3 import timings
from reference import compare
from run_a41 import KV_CASES


def load(path, mode):
    root = path.parent
    report = json.loads(path.read_text())
    if (report.get("abi") != "dflash-group-quant-linear-a41-v1" or report.get("status") != "PASS" or
            report.get("build", {}).get("kv_m80_mode") != mode or
            report["build"].get("launch_version") != 5 or report["build"].get("dequant_mode") != "batched" or
            report.get("timing_protocol") != "continuous-v1"):
        raise ValueError("need passing v5 batched A4.1 baseline / a-ub runs")
    stages = {"kv80": ("kv80",), "kv32": ("kv80","kv32"), "full": ("kv80","kv32","full")}.get(report.get("stop_after"))
    if stages is None or set(report.get("checks", {})) != set(stages):
        raise ValueError("incomplete A4.1 stage inventory")
    if report.get("full_a4_regression") != ("PASS" if report["stop_after"] == "full" else "NOT_RUN"):
        raise ValueError("partial KV validation cannot claim full A4 regression")
    scopes = {}
    for stage in stages:
        check = report["checks"][stage]
        if check.get("status") != "PASS": raise ValueError("failed A4.1 stage")
        child_path = checked_file(root, check["summary"])
        child = json.loads(child_path.read_text())
        if child.get("status") != "PASS": raise ValueError("failed constituent report")
        if stage == "full":
            if (report.get("full_a4_regression") != "PASS" or child.get("abi") != "dflash-group-quant-linear-a4-suite-v1" or
                    child.get("build") != report["build"] or
                    set(child.get("checks", {})) != {"paired_gears","synthetic","c16","c64","launches"} or
                    any(c.get("status") != "PASS" for c in child["checks"].values())):
                raise ValueError("full A4 regression is incomplete or uses a different build")
            # Bind all constituent summaries, including synthetic correctness.
            for name in ("synthetic","c16","c64"):
                checked_file(child_path.parent, child["checks"][name]["summary"])
            for gear in (16,64):
                real_path = checked_file(child_path.parent, child["checks"][f"c{gear}"]["summary"])
                scopes[f"full/c{gear}"] = (json.loads(real_path.read_text()), real_path.parent, gear, a4_contract.CASE_NAMES)
        else:
            scopes[stage] = (child, child_path.parent, 64 if stage == "kv80" else 16, KV_CASES)
    for real, _, gear, names in scopes.values():
        if (real.get("abi") != a4_contract.ABI or real.get("status") != "PASS" or real.get("context_rows") != gear or
                tuple(c["name"] for c in real.get("cases", [])) != names or real.get("timing_protocol") != "continuous-v1"):
            raise ValueError("real projection inventory/gear/protocol differs")
        for file_key in ("bundle", "native_om_manifest"):
            if sha256(real[file_key]) != real[file_key + "_sha256"]:
                raise ValueError("capture/native identity changed after execution")
    return report, scopes


def compatible(baseline, candidate):
    for key in ("device_id","stop_after","timing_protocol"):
        if baseline.get(key) != candidate.get(key): raise ValueError(f"paired runs differ: {key}")
    for key in ("source_sha256","core_limit","dequant_mode","pipeline_mode","tiling_abi","launch_version"):
        if baseline["build"].get(key) is None or baseline["build"][key] != candidate["build"].get(key):
            raise ValueError(f"paired builds differ: {key}")


def match_plans(left, right, shape):
    for key in ("m","k","n","tile_m","tile_n","tile_k","weight_reuse_rows","block_dim","available_cores","raw_banks"):
        if left[key] != right[key]: raise ValueError(f"launch plans differ: {key}")
    if left["user_ub_bytes"] + left["matmul_ub_bytes"] != right["user_ub_bytes"] + right["matmul_ub_bytes"]:
        raise ValueError("per-core UB budget changed")
    if shape != (80,2560,2048):
        # Reject actual changes to other shapes, not merely a different label.
        for key in ("user_ub_bytes","matmul_ub_bytes","system_workspace_bytes"):
            if left[key] != right[key]: raise ValueError("non-target buffer/workspace plan changed")
        a, b = dict(left["cube"]), dict(right["cube"])
        a.pop("kv_m80_mode"); b.pop("kv_m80_mode")
        if a != b: raise ValueError("non-target Matmul plan changed")


def run(baseline_path, candidate_path, output):
    if output.exists(): raise FileExistsError("use a new comparison output")
    baseline, before = load(baseline_path, "baseline")
    candidate, after = load(candidate_path, "a-ub")
    compatible(baseline, candidate)
    result = dict(abi="dflash-a41-comparison-v1", status="RUNNING", scope="isolated continuous-v1; PASS is numerical only",
                  baseline_summary=str(baseline_path), candidate_summary=str(candidate_path),
                  baseline_sha256=sha256(baseline_path), candidate_sha256=sha256(candidate_path),
                  full_a4_regression=candidate["full_a4_regression"], full_draft_validation="NOT_RUN",
                  decode_performance="NOT_RUN", cases=[])
    for scope in before:
        a, aroot, gear, _ = before[scope]
        b, broot, _, _ = after[scope]
        for key in ("bundle_sha256","native_om_manifest_sha256","device_id"):
            if a.get(key) != b.get(key) or a.get(key) is None: raise ValueError(f"paired inputs differ: {key}")
        for old, new in zip(a["cases"], b["cases"]):
            shape = a4_contract.case_shape(next(c for c in a4_contract.cases(gear) if c["name"] == old["name"]))
            protocol = old["executions"]["custom"]["timing"]
            expected = checked_file(aroot, old["outputs"]["native_om"][0]).read_bytes()
            differences, plans = [], []
            for report, case, root in ((baseline,old,aroot), (candidate,new,broot)):
                if case.get("status") != "PASS": raise ValueError("failed projection")
                for kind in ("custom","native_om"):
                    execution = case["executions"][kind]
                    validate_execution(execution, shape, report["device_id"],
                        "AscendCL ACLNN" if kind == "custom" else "AscendCL native OM",
                        protocol["warmup"], protocol["repetitions"], "continuous-v1")
                    outputs, bracket = case["outputs"][kind], case.get("bracket_outputs", {}).get(kind, [])
                    if len(outputs) != 2 or len(bracket) != 2: raise ValueError("missing correctness/tail/postcheck evidence")
                    for item in outputs + bracket:
                        differences.append(compare(expected, checked_file(root,item).read_bytes(), SimpleNamespace(m=shape[0],n=shape[2])))
                cfg = report["build"]
                plans.append(launch_evidence(root / case["name"] / "custom/runner.log", shape, cfg["core_limit"],
                    case["executions"]["custom"]["workspace_bytes"], cfg["dequant_mode"], cfg["pipeline_mode"], 5, cfg["kv_m80_mode"]))
            match_plans(*plans, shape)
            old_time, new_time = timings(old["executions"]["custom"]), timings(new["executions"]["custom"])
            native = timings(new["executions"]["native_om"])
            passed = all(d["finite"] and d["bitwise_equal"] for d in differences)
            row = dict(scope=scope, name=old["name"], shape=shape, status="PASS" if passed else "FAIL", comparisons=differences,
                       baseline=old_time, candidate=new_time, native_before=timings(old["executions"]["native_om"]),
                       native_after=native, baseline_launch=plans[0], candidate_launch=plans[1],
                       baseline_workspace_bytes=old["executions"]["custom"]["workspace_bytes"],
                       candidate_workspace_bytes=new["executions"]["custom"]["workspace_bytes"],
                       median_speedup=old_time["median_ms"]/new_time["median_ms"],
                       candidate_native_ratio=new_time["median_ms"]/native["median_ms"],
                       candidate_slower_median=new_time["median_ms"] > old_time["median_ms"],
                       candidate_slower_p95=new_time["p95_ms"] > old_time["p95_ms"])
            result["cases"].append(row)
            print(f"{scope}/{old['name']}: {row['status']} median {old_time['median_ms']:.4f}->{new_time['median_ms']:.4f} ms; "
                  f"p95 {old_time['p95_ms']:.4f}->{new_time['p95_ms']:.4f} ms", flush=True)
    result["status"] = "PASS" if all(c["status"] == "PASS" for c in result["cases"]) else "FAIL"
    write_json(output,result)
    return 0 if result["status"] == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline","candidate","output"): parser.add_argument(f"--{name}",type=Path,required=True)
    args = parser.parse_args()
    return run(args.baseline.resolve(),args.candidate.resolve(),args.output.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
