#!/usr/bin/env python3
"""Prepare/build selectable FP16, W4A16 and W8A16 Draft OM bundles."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "framework/python"), str(REPO)]

from models.dflash_v1.draft_quantization import require_draft_checkpoint
from qwen35_dflash.ascend310p.common_reuse import FACTORY
from qwen35_dflash.ascend310p.draft_variants import compose_draft_variant
from qwen35_dflash.ascend310p.input_manifest import build_quant_input_manifest, verify_quant_input_manifest
from qwen35_dflash.ascend310p.utils import atomic_write_json, load_json_object, require_run_output, sha256_file

VARIANTS = ("fp16", "w4a16", "w8a16")


def prepare(args):
    root = require_run_output(args.output)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"use an empty output directory or --execute-plan: {root}")
    variants = args.draft_quantizations
    if len(set(variants)) != len(variants):
        raise ValueError("duplicate Draft selections")
    routes = ["chunk", "mtp"] if args.verify_gdr == "both" else [args.verify_gdr]
    config = load_json_object(args.factory_config)
    if "input_manifest" in config:
        verify_quant_input_manifest(config["input_manifest"])
    # Validate every selected checkpoint before model loading or compilation.
    audits, directories = {}, {}
    for variant in variants:
        directory = getattr(args, variant + "_draft_dir")
        if not directory:
            raise ValueError(f"set DRAFT_{variant.upper()}_DIR or --{variant}-draft-dir")
        directories[variant] = str(Path(directory).expanduser().resolve())
        print(f"[draft-build] validate {variant}: {directories[variant]}", flush=True)
        audits[variant] = require_draft_checkpoint(directories[variant], variant)
    root.mkdir(parents=True, exist_ok=True)
    bundles, jobs = {}, []
    first_variant, first_route = variants[0], routes[0]
    for variant in variants:
        inputs = root / "config" / (variant + "-inputs.json")
        inputs.parent.mkdir(exist_ok=True)
        build_quant_input_manifest(target_dir=config["target_dir"], draft_dir=directories[variant],
            quant_config=config["quant_config"], receiver_models_dir=config["receiver_models_dir"], output=inputs)
        bundles[variant] = {}
        for route in routes:
            directory = root / variant / route
            manifest = directory / "deployment-manifest.json"
            bundles[variant][route] = {"manifest": str(manifest), "status": "NOT_RUN", "checkpoint": audits[variant]}
            current = dict(config, draft_dir=directories[variant], draft_quantization=variant,
                           verify_gdr=route, input_manifest=str(inputs), shared_draft_features=True,
                           include_ordinary_decode=True)
            cfg_path = atomic_write_json(root / "config" / f"{variant}-{route}.json", current)
            common = dict(variant=variant, route=route, manifest=str(manifest), status="NOT_RUN")
            if variant != first_variant and route != first_route:
                jobs.append(dict(common, kind="compose", target=bundles[first_variant][route]["manifest"],
                                 draft=bundles[variant][first_route]["manifest"], bundle_dir=str(directory)))
                continue
            cmd = [sys.executable, "-B", "-m", "qwen35_dflash.ascend310p", "export-air",
                   "--factory", FACTORY, "--factory-config", str(cfg_path), "--bundle-dir", str(directory)]
            if variant != first_variant:
                cmd += ["--reuse-target-from", bundles[first_variant][route]["manifest"]]
            elif route != first_route:
                cmd += ["--reuse-common-from", bundles[variant][first_route]["manifest"]]
            jobs.append(dict(common, kind="export", command=cmd, output=str(directory / "air-manifest.json")))
            jobs.append(dict(common, kind="compile", command=[sys.executable, "-B", "-m",
                "qwen35_dflash.ascend310p", "compile-om", "--air-manifest", str(directory / "air-manifest.json"),
                "--atc", str(args.atc), "--soc-version", args.soc_version], output=str(manifest)))
    plan = dict(schema_version=1, artifact_kind="qwen35-draft-variants", status="PREPARED",
                draft_quantizations=variants, routes=routes, bundles=bundles, jobs=jobs,
                source_factory_config={"path": str(args.factory_config), "sha256": sha256_file(args.factory_config)},
                scope="published checkpoint variants; not a controlled bitwidth-only comparison",
                execution="compressed resident weights; group dequantization and FP16 MatMul; target validation pending")
    path = atomic_write_json(root / "draft-variants.json", plan)
    print(f"Build plan: {path}; {len(jobs)} steps; no device execution yet.", flush=True)
    return path


def execute(path):
    path = Path(path).resolve()
    require_run_output(path)
    plan = load_json_object(path)
    if plan.get("artifact_kind") != "qwen35-draft-variants":
        raise ValueError("not a Draft variant build plan")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join([str(REPO / "framework/python"), str(REPO), environment.get("PYTHONPATH", "")])
    for i, job in enumerate(plan["jobs"], 1):
        if job["status"] == "PASS":
            saved = Path(job.get("output", job["manifest"]))
            if not saved.is_file() or sha256_file(saved) != job["output_sha256"]:
                raise ValueError(f"completed build step {i} output changed; use a new output directory")
            continue
        print(f"[draft-build {i}/{len(plan['jobs'])}] {job['variant']}/{job['route']} {job['kind']} START", flush=True)
        job["status"] = "RUNNING"
        atomic_write_json(path, plan)
        try:
            if job["kind"] == "compose":
                compose_draft_variant(target_manifest=job["target"], draft_manifest=job["draft"], bundle_dir=job["bundle_dir"])
            else:
                subprocess.run(job["command"], check=True, cwd=REPO, env=environment)
            output = Path(job.get("output", job["manifest"]))
            job.update(status="PASS", output_sha256=sha256_file(output))
            if job["kind"] != "export":
                plan["bundles"][job["variant"]][job["route"]].update(status="PASS", manifest_sha256=sha256_file(output))
        except BaseException as error:
            job.update(status="FAIL", error=str(error))
            plan["status"] = "FAIL_OR_INCOMPLETE"
            atomic_write_json(path, plan)
            raise
        atomic_write_json(path, plan)
        print(f"[draft-build {i}/{len(plan['jobs'])}] {job['kind']} DONE: {output}", flush=True)
    plan["status"] = "PASS"
    atomic_write_json(path, plan)
    print(f"Draft variants ready: {path}", flush=True)
    return 0


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--factory-config", type=Path)
    result.add_argument("--output", type=Path, default=os.environ.get("DRAFT_VARIANTS_DIR"))
    result.add_argument("--draft-quantizations", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    result.add_argument("--verify-gdr", choices=("chunk", "mtp", "both"), default="both")
    for variant in VARIANTS:
        result.add_argument(f"--{variant}-draft-dir", type=Path, default=os.environ.get("DRAFT_" + variant.upper() + "_DIR"))
    result.add_argument("--atc", type=Path, default=os.environ.get("ATC_BIN"))
    result.add_argument("--soc-version", default=os.environ.get("SOC_VERSION", "Ascend310P3"))
    result.add_argument("--prepare-only", action="store_true", help="validate weights and save build plan without device execution")
    result.add_argument("--execute-plan", type=Path, help="execute a saved plan; completed steps are hash-checked and skipped")
    return result


def main():
    cli = parser()
    args = cli.parse_args()
    try:
        if args.execute_plan:
            return execute(args.execute_plan)
        if not args.factory_config or not args.output or not args.atc:
            raise ValueError("--factory-config, --output and --atc (or environment defaults) are required")
        path = prepare(args)
        return 0 if args.prepare_only else execute(path)
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.CalledProcessError) as error:
        cli.exit(2, f"draft-build: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
