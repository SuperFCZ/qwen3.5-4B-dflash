#!/usr/bin/env python3
"""Execute previously compiled offline-NZ W8 probe OMs against exact CPU oracles.

Synthetic MatMul validation only. Uses CANN's Python acl binding; no checkpoint,
ATC invocation, runtime weight conversion or full-Draft performance measurement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "framework/python"), str(REPO)]

import numpy as np
import torch

from tools.probe_draft_matmul_atc import SHAPES, make_spec
from qwen35_dflash.ascend310p.acl_runtime import AclOmRuntime
from qwen35_dflash.ascend310p.draft_gears import STATIC_POLICY, verify_om_files
from qwen35_dflash.ascend310p.utils import atomic_write_json, contained_path, require_run_output, sha256_file
from qwen35_dflash.ascend310p.weight_prepack import load_prepacked_weights, _cached_weight

GRAPH = "weight_quant_probe"


def record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def referenced(value, parent):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (parent / path).resolve()


def prepare(summary_path):
    """Validate the complete case/gear inventory before loading any OM."""
    summary_path = Path(summary_path).resolve()
    summary = json.loads(summary_path.read_text())
    if summary.get("status") != "PASS" or not summary.get("cases"):
        raise ValueError("provide a PASS summary from --prepack-weights --static-om-gears")
    jobs, names = [], set()
    for case in summary["cases"]:
        if (case.get("status") != "PASS" or case.get("phase") != "complete"
                or case.get("control") != "prepacked-static-gears"
                or case.get("bits") != 8 or case.get("group_size") != 128
                or case.get("weight_layout") != "nk" or case.get("weight_format") != "nz"
                or case.get("offline_weight_prepack") is not True):
            raise ValueError("only completed W8/group128/NK/NZ static prepacked probes are supported")
        projection = case["projection"]
        if projection not in SHAPES or projection in names:
            raise ValueError("unknown or duplicate projection in probe summary")
        names.add(projection)
        deployment = referenced(case["deployment_manifest"], summary_path.parent)
        manifest = json.loads(deployment.read_text())
        if (manifest.get("status") != "PASS"
                or manifest.get("artifact_kind") != "qwen35-dflash-ascend310p-om-bundle"
                or len(manifest.get("graphs", [])) != 1):
            raise ValueError("probe deployment must contain one passing OM graph")
        graph = manifest["graphs"][0]
        meta = graph.get("metadata", {})
        if (graph.get("name") != GRAPH or graph.get("input_names") != ["x"]
                or graph.get("output_names") != ["y"]
                or meta.get("synthetic_prepack") is not True
                or meta.get("draft_compile_policy") != STATIC_POLICY
                or meta.get("projection") != projection
                or meta.get("bits") != 8 or meta.get("group_size") != 128):
            raise ValueError("deployment is not the declared synthetic offline W8 probe")
        verify_om_files(graph, deployment.parent)
        if graph["static_gear_oms"] != case.get("static_gear_oms"):
            raise ValueError("probe summary and deployment static gears differ")
        offline = referenced(meta["draft_weight_prepack_manifest"], deployment.parent)
        cache = load_prepacked_weights(offline)
        audit = graph["runtime_input_abi"]["weight_quant_layout"]["prepack"]
        if audit.get("status") != "PASS" or audit.get("offline_manifest_sha256") != cache["sha256"]:
            raise ValueError("offline weight manifest differs from the exported AIR audit")
        if len(cache["weights"]) != 1:
            raise ValueError("synthetic probe must have exactly one offline weight")
        jobs.append(dict(projection=projection, deployment=deployment, graph=graph, cache=cache,
                         sources=[record(deployment), record(offline)] +
                         [record(contained_path(deployment.parent, gear["om"]["path"]))
                          for gear in graph["static_gear_oms"]] +
                         [record(contained_path(cache["root"], r["path"])) for r in cache["weights"].values()]))
    return summary, jobs


def weight_units(job):
    """Bind the r39 synthetic oracle to the audited offline weight bytes."""
    k, n = SHAPES[job["projection"]]
    spec = make_spec(8, job["projection"], "cpu")
    _, packed, scales = spec.example_args
    q = packed.reshape(n, k).contiguous()
    _cached_weight(q, job["cache"])  # hash, NZ permutation and padding must match
    # Existing probe scales are (1,2,3,4)/32; dequantized weights are exact FP16.
    factors = scales.reshape(n, k // 128).float() * 32
    if not bool(((factors >= 1) & (factors <= 4) & (factors == factors.round())).all()):
        raise ValueError("synthetic group scales no longer match the exact oracle contract")
    return q.float() * factors.repeat_interleave(128, dim=1)


def patterns(rows, k):
    zero = torch.zeros(rows, k, dtype=torch.float32)
    yield "zero", zero
    basis = zero.clone()
    indices = torch.arange(rows) * (k - 1) // (rows - 1)
    indices[:8] = torch.tensor([0, 15, 16, 31, 32, 127, 128, k - 1])
    basis[torch.arange(rows), indices] = torch.where(torch.arange(rows) % 2 == 0, 1., -1.)
    yield "group_and_tile_boundaries", basis
    for seed in (391, 817):
        generator = torch.Generator().manual_seed(seed + rows)
        units = (torch.randint(0, 2, (rows, k), generator=generator) * 2 - 1).float()
        yield f"dense_signed_{seed}", units


def vectors(units, rows):
    """Exact integer dot product followed by power-of-two division and FP16 RNE.

    |X_units|<=1, |W_units|<=512 and K<=12800: every partial integer sum
    has magnitude <2**24, so CPU FP32 multiplication/accumulation is exact.
    X=X_units/128, dequant(W)=W_units/32, hence Y=(X_units@W_units.T)/4096.
    No comparison tolerance is introduced.
    """
    n, k = units.shape
    if k * 512 >= 2**24 or not bool(torch.isfinite(units).all()):
        raise ValueError("workload exceeds the exact FP32 integer oracle bound")
    for name, x_units in patterns(rows, k):
        x = (x_units / 128).half().numpy()
        y = ((x_units @ units.t()) / 4096).half().numpy()
        yield name, x, y


def compare(expected, actual):
    if not isinstance(actual, np.ndarray) or actual.dtype != np.float16 or actual.shape != expected.shape:
        raise ValueError("OM output must have the expected FP16 [M,N] ABI")
    finite = bool(np.isfinite(actual).all())
    different = actual != expected
    flat = np.flatnonzero(different)
    first = None
    if flat.size:
        location = tuple(int(i) for i in np.unravel_index(int(flat[0]), actual.shape))
        first = {"index": list(location), "expected": float(expected[location]),
                 "actual": float(actual[location]) if np.isfinite(actual[location]) else str(actual[location])}
    return {
        "status": "PASS" if finite and not flat.size else "FAIL",
        "finite": finite, "mismatched_values": int(flat.size),
        "bitwise_equal": actual.tobytes() == expected.tobytes(),
        "max_abs_error": float(np.abs(actual.astype(np.float32) - expected.astype(np.float32)).max()) if finite else None,
        "first_difference": first,
    }


def check_abi(runtime, rows, k, n):
    if runtime.graph_names != (GRAPH,):
        raise ValueError("unexpected loaded OM graph")
    for values, expected in (
        (runtime.graph_inputs(GRAPH), ("x", [rows, k])),
        (runtime.graph_outputs(GRAPH), ("y", [rows, n])),
    ):
        if (len(values) != 1 or values[0]["name"] != expected[0]
                or values[0]["shape"] != expected[1] or values[0]["dtype"] != "float16"):
            raise ValueError(f"OM ABI differs: expected FP16 {expected}, got {values}")


def run(args):
    root = require_run_output(args.output_dir)
    if root.exists():
        raise FileExistsError("use a new output directory; existing validation evidence is retained")
    if args.device_id < 0 or args.repetitions < 1:
        raise ValueError("device-id must be nonnegative and repetitions must be positive")
    root.mkdir(parents=True)
    report = {
        "schema_version": 1, "status": "RUNNING",
        "scope": "synthetic offline-NZ W8 static OM execution and exact FP16 oracle validation",
        "probe_summary": record(args.probe_summary), "device_id": args.device_id,
        "validator": record(__file__),
        "synthetic_probe_builder": record(REPO / "tools/probe_draft_matmul_atc.py"),
        "runtime": "CANN Python acl", "cpu_fallback": False,
        "reference_device": "CPU only for expected values; all candidate execution requires ACL",
        "numerical_gate": "finite and exact FP16 values on bounded dyadic inputs; no tolerance",
        "repetitions": args.repetitions, "cases": [],
        "latency_status": "NOT_RUN", "full_draft_validation": "NOT_RUN",
    }
    try:
        summary, jobs = prepare(args.probe_summary)
        report["compiled_environment"] = summary.get("environment")
        report["compiled_soc_version"] = summary.get("soc_version")
        for job in jobs:
            k, n = SHAPES[job["projection"]]
            units = weight_units(job)
            for rows in (16, 64):
                result = dict(projection=job["projection"], rows=rows, status="RUNNING",
                              execution_status="NOT_RUN", phase="load", vectors=[], sources=job["sources"])
                report["cases"].append(result)
                try:
                    with AclOmRuntime(job["deployment"], device_id=args.device_id, static_gear_rows=rows) as runtime:
                        result["om_sha256"] = runtime.artifact_hashes()[GRAPH]
                        result["acl_module"] = getattr(runtime.acl, "__file__", None)
                        check_abi(runtime, rows, k, n)
                        result["phase"] = "execute"
                        for name, x, expected in vectors(units, rows):
                            item = dict(name=name, input_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                                        expected_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),
                                        measurements=[])
                            result["vectors"].append(item)
                            for repetition in range(args.repetitions):
                                outputs = runtime.run_graph(GRAPH, {"x": x})
                                runtime.synchronize()
                                if set(outputs) != {"y"}:
                                    raise ValueError("OM output names differ")
                                actual = outputs["y"]
                                comparison = compare(expected, actual)
                                item["measurements"].append(dict(comparison, repetition=repetition,
                                    output_sha256=hashlib.sha256(actual.tobytes()).hexdigest()))
                                result["execution_status"] = "PASS"
                            item["repeat_drift"] = len({m["output_sha256"] for m in item["measurements"]}) != 1
                            item["status"] = "PASS" if not item["repeat_drift"] and all(
                                m["status"] == "PASS" for m in item["measurements"]) else "FAIL"
                        result.update(status="PASS" if all(v["status"] == "PASS" for v in result["vectors"]) else "FAIL",
                                      phase="complete")
                except Exception as error:
                    result.update(status="FAIL", error=f"{type(error).__name__}: {error}")
                    if result["phase"] == "execute":
                        result["execution_status"] = "FAIL"
                print(f"[matmul-om] {job['projection']} M{rows} {result['status']} phase={result['phase']}", flush=True)
                if result.get("error"):
                    print(result["error"], flush=True)
                elif result["status"] == "FAIL":
                    for item in result["vectors"]:
                        if item.get("status") == "FAIL":
                            print(f"  {item['name']}: {item['measurements'][0]}", flush=True)
                atomic_write_json(root / "summary.json", report)
        for source in [report["probe_summary"]] + [s for j in jobs for s in j["sources"]]:
            if sha256_file(source["path"]) != source["sha256"]:
                raise ValueError("source probe artifacts changed during validation")
        report["status"] = "PASS" if report["cases"] and all(c["status"] == "PASS" for c in report["cases"]) else "FAIL"
    except Exception as error:
        report.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        print(report["error"], flush=True)
    atomic_write_json(root / "summary.json", report)
    print("Full Draft correctness, steady-state latency and TransData removal are NOT_RUN.")
    print(f"Report: {root / 'summary.json'}", flush=True)
    return 0 if report["status"] == "PASS" else 1


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--probe-summary", type=Path, required=True,
                     help="summary.json produced by --prepack-weights --static-om-gears")
    cli.add_argument("--output-dir", type=Path, required=True)
    cli.add_argument("--device-id", type=int, default=0)
    cli.add_argument("--repetitions", type=int, default=3)
    args = cli.parse_args(argv)
    try:
        return run(args)
    except (OSError, ValueError, RuntimeError) as error:
        cli.exit(2, f"matmul-om: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
