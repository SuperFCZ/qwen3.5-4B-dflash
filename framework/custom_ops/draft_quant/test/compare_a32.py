#!/usr/bin/env python3
"""Compare fresh A3.2 serial/prefetch builds on identical frozen inputs."""
from __future__ import annotations

import argparse
from pathlib import Path

from a2_common import CASE_NAMES
from a3_launch import pipeline_plan
from compare_a3 import compatible_runs
from compare_a31 import compare_variants


def matched_variants(baseline, candidate):
    compatible_runs(baseline, candidate, allow_pipeline_change=True)
    for report, mode in ((baseline, "serial"), (candidate, "prefetch")):
        if (report.get("stage") != "A3.2" or report.get("timing_protocol") != "continuous-v1" or
                report["build"].get("abi") != "dflash-group-quant-linear-build-v3" or
                report["build"].get("dequant_mode") != "batched" or
                report["build"].get("pipeline_mode") != mode):
            raise ValueError("need fresh A3.2 batched serial baseline / prefetch candidate with continuous-v1")
        if tuple(row["name"] for row in report["launches"]["real"]) != CASE_NAMES:
            raise ValueError("need all five gate/up controls and five down projections")
        for row in report["launches"]["real"]:
            launch = row["launch"]
            plan = pipeline_plan(launch["k"], launch["n"], mode)
            if (launch.get("version") != 3 or launch.get("dequant_mode") != "batched" or
                    any(launch.get(key) != value for key, value in plan.items())):
                raise ValueError("recorded host pipeline/buffer policy differs from the build")
    if baseline["build"]["core_limit"] != candidate["build"]["core_limit"]:
        raise ValueError("pipeline comparisons require the same core cap")
    for a, b in zip(baseline["launches"]["real"], candidate["launches"]["real"]):
        for key in ("block_dim", "available_cores", "m", "k", "n", "tile_n", "tile_k"):
            if a["launch"][key] != b["launch"][key]:
                raise ValueError(f"launch differs between variants: {key}")
        left, right = a["launch"], b["launch"]
        if (left["user_ub_bytes"] + left["matmul_ub_bytes"] !=
                right["user_ub_bytes"] + right["matmul_ub_bytes"]):
            raise ValueError("total per-core UB budget differs between variants")
        if a["name"].endswith("gate_up") and left["matmul_ub_bytes"] != right["matmul_ub_bytes"]:
            raise ValueError("gate/up control must retain its Matmul UB budget")


def run(baseline_path, candidate_path, output):
    return compare_variants(baseline_path, candidate_path, output, stage="A3.2",
                            match=matched_variants, labels=("serial", "prefetch"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="fresh a32 serial data/suite.json")
    parser.add_argument("--candidate", type=Path, required=True, help="fresh a32 prefetch data/suite.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return run(args.baseline.resolve(), args.candidate.resolve(), args.output.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
