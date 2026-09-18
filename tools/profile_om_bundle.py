"""Select unique OMs from the compiled Draft index and capture them serially."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

OMS = ("prefill", "decode", "draft", "draft_w4a16", "draft_w8a16", "verify_chunk", "verify_mtp")
VARIANTS = ("fp16", "w4a16", "w8a16")
ROUTES = ("chunk", "mtp")


def select_jobs(args):
    selected = args.profile_om or ["all"]
    if len(set(selected)) != len(selected) or ("all" in selected and len(selected) != 1):
        raise ValueError("select distinct OM names, or --profile-om all alone")
    if args.profile_mode != "dflash" or args.profile_stage != "all" or args.verify_gdr:
        raise ValueError("with --bundle-dir use --profile-om to select OMs; omit mode/stage/verify-gdr")
    selected = list(OMS) if selected == ["all"] else selected
    root = args.bundle_dir.expanduser().resolve()
    index_path = root / "draft-variants.json"
    index_bytes = index_path.read_bytes()
    index = json.loads(index_bytes)
    if index.get("artifact_kind") != "qwen35-draft-variants" or index.get("status") != "PASS":
        raise ValueError("--bundle-dir needs a passing compiled draft-variants.json")
    bundles, cache, jobs = index.get("bundles", {}), {}, []
    shared = {}
    for name in selected:
        if name not in OMS:
            raise ValueError(f"unknown OM: {name}")
        stage = "draft" if name.startswith("draft") else "verify" if name.startswith("verify_") else name
        wanted_variant = (name.removeprefix("draft_") if name != "draft" else "fp16") if stage == "draft" else None
        wanted_route = name.removeprefix("verify_") if stage == "verify" else None
        # Shared Target and Verify use FP16 preparation when available. A
        # partial bundle can still profile one requested OM using its Draft.
        pair = next(((v, r) for v in VARIANTS for r in ROUTES
                     if (wanted_variant is None or v == wanted_variant)
                     and (wanted_route is None or r == wanted_route)
                     and bundles.get(v, {}).get(r, {}).get("status") == "PASS"), None)
        if pair is None:
            raise ValueError(f"{name}.om has no compiled bundle; build it or select fewer OMs")
        variant, route = pair
        entry = bundles[variant][route]
        manifest = (root / entry["manifest"]).resolve()
        if manifest not in cache:
            payload = manifest.read_bytes()
            cache[manifest] = (hashlib.sha256(payload).hexdigest(), json.loads(payload))
        digest, deployment = cache[manifest]
        if digest != entry["manifest_sha256"]:
            raise ValueError(f"{variant}/{route} manifest changed after build")
        if deployment.get("status") != "PASS":
            raise ValueError(f"{variant}/{route} deployment is not PASS")
        graphs = {g["name"]: g for g in deployment["graphs"]}
        contract = graphs["draft"]["metadata"]["incremental_contract"]
        if contract.get("draft_quantization", "fp16") != variant or contract.get("verify_gdr") != route:
            raise ValueError(f"{variant}/{route} manifest contract differs from its index label")
        # A 'shared prefill/decode' row must really refer to common artifacts.
        for graph_name in ("target_prefill", "target_decode"):
            identity = graphs[graph_name]["om"]["sha256"]
            if shared.setdefault(graph_name, identity) != identity:
                raise ValueError(f"{graph_name} differs between selected bundles")
        graph = graphs["draft" if stage == "draft" else "target_" + stage]
        om = (manifest.parent / graph["om"]["path"]).resolve()
        if not om.is_file():
            raise ValueError(f"OM not found: {om}")
        jobs.append(dict(om=name, profile_mode="ordinary" if stage in {"prefill", "decode"} else "dflash",
                         profile_stage=stage, draft_quantization=variant, verify_gdr=route,
                         deployment_manifest=str(manifest), deployment_manifest_sha256=digest,
                         model=str(om), model_sha256=graph["om"]["sha256"],
                         status="NOT_RUN", graph_calls=None, profiled_elapsed_ms=None, ms_per_call=None,
                         output=None, error=None))
    return jobs, dict(path=str(index_path), sha256=hashlib.sha256(index_bytes).hexdigest())


def capture_result(output, job):
    stage = job["profile_stage"]
    capture = output / "capture"
    report_path = capture / (stage + "-stage-report.json")
    report = json.loads(report_path.read_text())
    graph = "draft" if stage == "draft" else "target_" + stage
    calls = report.get("captured_graph_calls", {})
    count = calls.get(graph)
    elapsed = report.get("profiled_elapsed_ms")
    if (report.get("status") != "PASS_CAPTURE" or report.get("profile_stage") != stage
            or report.get("profile_mode") != job["profile_mode"] or set(calls) != {graph}
            or type(count) is not int or count < 1
            or isinstance(elapsed, bool) or not isinstance(elapsed, (int, float))
            or not math.isfinite(elapsed) or elapsed < 0):
        raise ValueError(f"{job['om']}: invalid capture timing report")
    files = {"stage_report": report_path,
             "stage_summary": capture / (stage + "-stage-summary.csv"),
             "operator_types": capture / (stage + "-operator-types.csv"),
             "operator_tasks": capture / (stage + "-operator-tasks.csv"),
             "hotspots": capture / (stage + "-hotspots.txt")}
    if any(not path.is_file() for path in files.values()):
        raise ValueError(f"{job['om']}: incomplete profiling reports")
    return dict(status="PASS_CAPTURE", graph_calls=count, profiled_elapsed_ms=elapsed,
                ms_per_call=elapsed / count, runner_version=report.get("runner_version"),
                runtime=report.get("runtime"), device_id=report.get("device_id"),
                artifacts={key: str(path) for key, path in files.items()})


def save_summary(output, summary):
    jobs = summary["oms"]
    summary["status"] = "PASS_CAPTURE" if all(j["status"] == "PASS_CAPTURE" for j in jobs) else "FAIL_OR_INCOMPLETE"
    # Keep completed captures visible even if a later OM fails or is interrupted.
    temp = output / "summary.json.tmp"
    temp.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temp.replace(output / "summary.json")
    fields = ("om", "status", "graph_calls", "profiled_elapsed_ms", "ms_per_call",
              "draft_quantization", "verify_gdr", "output", "error")
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(jobs)
    lines = ["| OM | Status | Calls | Captured ms | Mean ms/call |", "|---|---|---:|---:|---:|"]
    for job in jobs:
        times = ["N/A" if job[key] is None else f"{job[key]:.3f}" for key in ("profiled_elapsed_ms", "ms_per_call")]
        lines.append(f"| {job['om']} | {job['status']} | {job['graph_calls'] or 'N/A'} | {' | '.join(times)} |")
    lines += ["", "Serial independent stage captures; these times must not be added as generation latency.",
              "Prefill covers all 64-token input chunks; other captures measure one graph call.",
              "Warmups, model loading, reset, preparation and msprof control waits are outside capture.",
              "Verify preparation uses the Draft recorded in summary.csv (FP16 preferred).",
              "Single-window msprof diagnostics, not task accuracy or unprofiled benchmark latency."]
    for job in jobs:
        if job.get("error"):
            lines += ["", f"{job['om']}: {job['error']}"]
    text = "\n".join(lines) + "\n"
    (output / "summary.md").write_text(text, encoding="utf-8")
    return text


def run_bundle_profile(args, *, run_single):
    from profile_om import REPOSITORY, prompt_and_eos

    jobs, index = select_jobs(args)
    if args.run_dir is None or args.runner is None:
        raise ValueError("Set AI_RUN_DIR and CPP_RUNNER, or pass --run-dir and --runner")
    run, runner = args.run_dir.expanduser().resolve(), args.runner.expanduser().resolve()
    if not run.is_dir() or run.is_relative_to(REPOSITORY):
        raise ValueError("--run-dir must exist outside the source repository")
    if not runner.is_file() or not os.access(runner, os.X_OK):
        raise ValueError(f"C++ runner is missing or not executable: {runner}")
    if (args.device_id < 0 or args.profile_warmup < 0 or args.profile_timeout <= 0
            or not 1 <= args.max_draft_tokens <= 15
            or args.max_new_tokens < (1 if all(j["profile_stage"] == "prefill" for j in jobs) else 2)):
        raise ValueError("invalid profiling device, warmup, timeout or token budget")
    if not shutil.which(os.path.expanduser(args.msprof_bin)):
        raise ValueError("msprof not found; load the CANN environment or set MSPROF_BIN")
    if os.environ.get("ASCEND310P_SIMULATION_ONLY") == "1":
        raise ValueError("simulation-only target cannot produce msprof device measurements")
    if args.profile_audit_draft_inputs and not any(j["profile_stage"] in {"draft", "verify"} for j in jobs):
        raise ValueError("--profile-audit-draft-inputs needs a Draft or Verify selection")
    prompt, eos, source = prompt_and_eos(args, run)
    parent = run / "msprof"
    parent.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="oms-", dir=parent))
    summary = dict(schema_version=1, status="NOT_RUN", oms=jobs, bundle_index=index,
                   runner=dict(path=str(runner), sha256=hashlib.sha256(runner.read_bytes()).hexdigest()),
                   prompt_source=source, prompt_token_ids=[int(t) for t in prompt.split(",")],
                   eos_token_ids=[int(t) for t in eos.split(",")],
                   device_id=args.device_id, aic_metrics=args.aic_metrics,
                   max_new_tokens=args.max_new_tokens, max_draft_tokens=args.max_draft_tokens,
                   profile_warmup=args.profile_warmup, formal_latency_evidence=False,
                   scope="serial independent OM diagnostic windows; no complete generation")
    save_summary(output, summary)
    print(f"Output: {output}\nSelected OMs: {', '.join(j['om'] for j in jobs)}", flush=True)
    for i, job in enumerate(jobs, 1):
        print(f"[om-profile] {i}/{len(jobs)} {job['om']} start", flush=True)
        options = argparse.Namespace(**vars(args))
        options.bundle_dir, options.profile_om = None, None
        options.profile_mode, options.profile_stage = job["profile_mode"], job["profile_stage"]
        options.deployment_manifest, options.verify_gdr = Path(job["deployment_manifest"]), job["verify_gdr"]
        options.profile_audit_draft_inputs = args.profile_audit_draft_inputs and job["profile_stage"] in {"draft", "verify"}
        # Freeze the report's tokens once for every OM; do not re-read a mutable report.
        options.prompt_token_ids, options.eos_token_ids, options.prompt_report = prompt, eos, None
        destination = output / job["om"]
        job.update(status="RUNNING", output=str(destination))
        save_summary(output, summary)
        try:
            run_single(options, output=destination)
            job.update(capture_result(destination, job))
        except KeyboardInterrupt:
            job.update(status="INTERRUPTED", error="interrupted by user")
            raise
        except (subprocess.CalledProcessError, OSError, ValueError, KeyError) as error:
            job.update(status="FAIL", error=str(error))
        finally:
            save_summary(output, summary)
        print(f"[om-profile] {job['om']} {job['status']}", flush=True)
    print(save_summary(output, summary), flush=True)
    print(f"Summary: {output / 'summary.json'}\nPer-OM timings: {output / 'summary.csv'}", flush=True)
    return output
