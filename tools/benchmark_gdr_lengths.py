#!/usr/bin/env python3
"""Run selected prompts and output budgets with Chunk, MTP, or both OM verifiers."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "framework/python"), str(REPO)]

from tools import benchmark_prompts as suite
from qwen35_dflash.ascend310p.utils import atomic_write_json, require_run_output, sha256_file

ROUTES = ("chunk", "mtp")
DEFAULT_LENGTHS = (32, 64, 128, 256, 512, 1024)
STAGES = suite.STAGES
stage_timings = suite.stage_timings


def aggregate_cases(rows):
    totals = suite.aggregate(rows)
    good = [row for row in rows if row["status"] in suite.MEASURED_STATUSES]
    rounds = sum(row["speculative_rounds"] for row in good)
    totals["tokens_per_speculative_round"] = (
        sum(row["tokens_emitted_in_speculative_rounds"] for row in good) / rounds if rounds else None)
    for mode in ("ordinary", "dflash"):
        tokens = sum(row[mode + "_measured_tokens"] for row in good)
        elapsed = sum(row[mode + "_total_measured_ms"] for row in good)
        totals[mode + "_measured_tokens"] = tokens
        totals[mode + "_tokens_per_second"] = tokens * 1000 / elapsed if elapsed else None
    ordinary_tps, dflash_tps = (totals[mode + "_tokens_per_second"] for mode in ("ordinary", "dflash"))
    totals["throughput_speedup"] = dflash_tps / ordinary_tps if ordinary_tps and dflash_tps else None
    totals["both_modes_reached_budget"] = sum(
        row["generated_tokens"] == row["ordinary_generated_tokens"] == row["max_new_tokens"] for row in good)
    totals["stage_ms_per_call"] = {}
    for stage in STAGES:
        values = [row["stage_timings"][stage] for row in good]
        missing = [row["id"] for row in good if not row["stage_timings"][stage]["available"]]
        if not values or missing:
            totals["stage_ms_per_call"][stage] = {"available": False, "missing_prompt_ids": missing}
        else:
            calls = sum(value["calls"] for value in values)
            elapsed = math.fsum(value["total_ms"] for value in values)
            measurements = sum(value["measurements"] for value in values)
            totals["stage_ms_per_call"][stage] = {
                "available": True, "calls": calls, "total_ms": elapsed,
                "mean_ms": elapsed / calls if calls else None,
                "measurements": measurements,
                "mean_total_ms_per_generation": elapsed / measurements if measurements else None,
            }
    totals["phase_timings"] = {}
    for phase in suite.PHASES:
        values = [row.get("phase_timings", {}).get(phase, {}) for row in good]
        if not values or any(not value.get("available") for value in values):
            totals["phase_timings"][phase] = {"available": False}
            continue
        measurements = sum(value["measurements"] for value in values)
        elapsed = math.fsum(value["total_ms"] for value in values)
        totals["phase_timings"][phase] = {"available": True, "measurements": measurements,
                                         "total_ms": elapsed, "mean_ms": elapsed / measurements}
    return totals


def read_cell(summary_path, route, length, prompts):
    summary = json.loads(summary_path.read_text())
    if summary["protocol"].get("verify_gdr") != route:
        raise ValueError("saved suite verification route differs from matrix request")
    if [row["id"] for row in summary["cases"]] != [p["id"] for p in prompts]:
        raise ValueError("saved suite prompt IDs differ from matrix request")
    rows, batch = [], None
    for item, expected in zip(summary["cases"], prompts):
        if item.get("prompt_token_ids") != expected["prompt_token_ids"]:
            raise ValueError("tokenized prompts differ between matrix cells")
        row = dict(item, verify_gdr=route, max_new_tokens=length,
                   input_tokens=len(expected["prompt_token_ids"]), group=expected.get("group", "custom"))
        if row["status"] in suite.MEASURED_STATUSES:
            path = Path(row["raw_report"])
            if sha256_file(path) != row["raw_report_sha256"]:
                raise ValueError("case report changed after suite validation")
            report = json.loads(path.read_text())
            # Production per-prompt reports omit fake_acl; the batch index owns it.
            index = Path(summary["runner_index"])
            if batch is None:
                if summary.get("runner_index_sha256") and sha256_file(index) != summary["runner_index_sha256"]:
                    raise ValueError("batch index changed after suite validation")
                batch = json.loads(index.read_text())
                if batch.get("fake_acl") is not False:
                    raise ValueError("fake ACL or missing batch identity cannot supply matrix device measurements")
                if [case["id"] for case in batch.get("cases", [])] != [p["id"] for p in prompts]:
                    raise ValueError("batch index prompt IDs differ from matrix request")
            case = batch["cases"][len(rows)]
            if (Path(case["report"]).resolve() != path.resolve()
                    or path.resolve() != (Path(str(index) + ".cases") / (row["id"] + ".json")).resolve()
                    or case["status"] != report.get("status")
                    or not batch.get("model_sha256")
                    or batch["model_sha256"] != report.get("model", {}).get("sha256")):
                raise ValueError("batch index/case report identity differs")
            if ("fake_acl" in report and report["fake_acl"] is not False
                    or "fake-acl" in report.get("runner_version", "")
                    or report.get("cpu_fallback") is not False):
                raise ValueError("fake ACL cannot supply matrix device measurements")
            row["stage_timings"] = stage_timings(report)
            row["phase_timings"] = suite.phase_timings(report)
            for mode in ("ordinary", "dflash"):
                row[mode + "_measured_tokens"] = sum(
                    len(m["generated_token_ids"]) for m in report[mode]["measurements"])
            if row["generated_tokens"] > length or row["ordinary_generated_tokens"] > length:
                raise ValueError("saved generation exceeds the requested budget")
        rows.append(row)
    return {
        "status": summary["status"], "summary": str(summary_path),
        "summary_sha256": sha256_file(summary_path), "cases": rows,
        "aggregate": aggregate_cases(rows),
        "aggregate_by_group": {group: aggregate_cases([r for r in rows if r["group"] == group])
                               for group in dict.fromkeys(r["group"] for r in rows)},
        "startup_ms": summary.get("startup_ms"),
        "process_wall_seconds": summary.get("process_wall_seconds"),
        "ordinary_parity": summary.get("ordinary_parity"),
    }


def compare_routes(cells, lengths):
    """Compare matched prompts only, retaining own-output token/time denominators."""
    if not set(ROUTES).issubset({cell["verify_gdr"] for cell in cells}):
        return []
    result = []
    for length in lengths:
        groups = {}
        for route in ROUTES:
            cell = next((c for c in cells if c["verify_gdr"] == route and c["max_new_tokens"] == length), {})
            groups[route] = {r["id"]: r for r in cell.get("cases", [])
                             if r["status"] in suite.MEASURED_STATUSES}
        matched = sorted(groups["chunk"].keys() & groups["mtp"].keys())
        entry = {"max_new_tokens": length, "matched_prompt_ids": matched,
                 "mtp_over_chunk_throughput": None, "mtp_over_chunk_model_time_speedup": None}
        if matched:
            totals = {}
            for route in ROUTES:
                rows = [groups[route][name] for name in matched]
                totals[route] = {
                    "tokens": sum(row["dflash_measured_tokens"] for row in rows),
                    "ms": sum(row["dflash_total_measured_ms"] for row in rows),
                }
            entry["mtp_over_chunk_model_time_speedup"] = totals["chunk"]["ms"] / totals["mtp"]["ms"]
            chunk_tps = totals["chunk"]["tokens"] / totals["chunk"]["ms"]
            mtp_tps = totals["mtp"]["tokens"] / totals["mtp"]["ms"]
            entry["mtp_over_chunk_throughput"] = mtp_tps / chunk_tps if chunk_tps else None
            entry["totals"] = totals
        result.append(entry)
    return result


def render(summary):
    def number(value, percent=False):
        return "N/A" if value is None else f"{value:.2%}" if percent else f"{value:.2f}"

    lines = ["| Prompt | Group | Input tokens |", "|---|---|---:|"]
    for prompt in summary["prompts"]:
        lines.append(f"| {prompt['id']} | {prompt.get('group', 'custom')} | {len(prompt['prompt_token_ids'])} |")
    protocol = summary.get("protocol", {})
    lines += ["", f"Warmup per mode/prompt: {protocol.get('warmup', 'N/A')}; "
              f"measured repetitions: {protocol.get('repetitions', 'N/A')}."]
    lines += ["", "Input tokens include the selected chat template; output budgets are listed separately.", "",
        "| Max new tokens | GDR | Group | Measured / prompts | Acceptance | Tokens / round | DFlash tok/s | Speedup vs ordinary |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    timing_rows = []
    for cell in summary["cells"]:
        groups = {"all": cell.get("aggregate", {})}
        if len({p.get("group", "custom") for p in summary["prompts"]}) > 1:
            groups.update(cell.get("aggregate_by_group", {}))
        for group, agg in groups.items():
            count = sum(group == "all" or p.get("group", "custom") == group for p in summary["prompts"])
            lines.append(
                f"| {cell['max_new_tokens']} | {cell['verify_gdr']} | {group} | "
                f"{agg.get('measured_prompts', 0)} / {count} | "
                f"{number(agg.get('weighted_acceptance_rate'), True)} | "
                f"{number(agg.get('tokens_per_speculative_round'))} | "
                f"{number(agg.get('dflash_tokens_per_second'))} | "
                f"{number(agg.get('total_model_time_speedup'))} |")
            timing_rows.append(dict(id=f"{cell['verify_gdr']}/{cell['max_new_tokens']}/{group}",
                                    stage_timings=agg.get("stage_ms_per_call", {}),
                                    phase_timings=agg.get("phase_timings", {})))
    lines += [
        "", "Acceptance is accepted/proposed; time speedup is sum ordinary model time / sum DFlash model time.",
        "Each row uses completed measurements admitted by the comparison policy; repeated-run drift is reported.",
        "Measured calls exclude warmups and startup. Draft includes any Draft calls during multi-chunk prefill.",
        "Verify includes commit inside the selected OM. ms/call is synchronized graph-call time, not kernel time.",
        "Budgets are upper limits: EOS can end generation early. Actual tokens and stop reasons are in cases.csv/JSON.",
    ]
    lines += ["", suite.render_timings(timing_rows, measured_only=False).rstrip()]
    observations = suite.render_repeatability([
        dict(row, id=f"{cell['verify_gdr']}/{cell['max_new_tokens']}/{row['id']}")
        for cell in summary["cells"] for row in cell.get("cases", [])
    ])
    if observations:
        lines += ["", observations.rstrip()]
    if summary["route_comparison"]:
        lines += ["", "| Max new tokens | Matched prompts | MTP/Chunk throughput | MTP speedup vs Chunk by model time |",
                  "|---:|---:|---:|---:|"]
        for row in summary["route_comparison"]:
            lines.append(
                f"| {row['max_new_tokens']} | {len(row['matched_prompt_ids'])} | "
                f"{number(row['mtp_over_chunk_throughput'])} | {number(row['mtp_over_chunk_model_time_speedup'])} |")
        lines += ["", "Cross-route comparisons use matched prompts and each route's own output. Quality is not evaluated."]
    for cell in summary["cells"]:
        if cell.get("error"):
            lines += ["", f"- {cell['verify_gdr']}/{cell['max_new_tokens']}: {cell['error']}"]
        elif cell.get("status") not in suite.MEASURED_STATUSES:
            lines += ["", f"- {cell['verify_gdr']}/{cell['max_new_tokens']}: {cell['status']}"]
    return "\n".join(lines) + "\n"


def save(root, summary):
    summary["route_comparison"] = compare_routes(summary["cells"], summary["lengths"])
    atomic_write_json(root / "summary.json", summary)
    (root / "summary.md").write_text(render(summary), encoding="utf-8")
    fields = [
        "verify_gdr", "max_new_tokens", "id", "group", "input_tokens", "status", "acceptance_rate",
        "tokens_per_speculative_round", "generated_tokens", "ordinary_generated_tokens",
        "stop_reason", "ordinary_stop_reason", "dflash_tokens_per_second",
        "ordinary_tokens_per_second", "speedup", "throughput_speedup",
        *[stage + "_ms_per_call" for stage in STAGES],
        *[stage + "_ms_per_generation" for stage in STAGES],
        *[phase + "_phase_ms_per_generation" for phase in suite.PHASES], "raw_report",
    ]
    with (root / "cases.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for cell in summary["cells"]:
            for row in cell.get("cases", []):
                record = {name: row.get(name) for name in fields}
                for stage in STAGES:
                    record[stage + "_ms_per_call"] = row.get("stage_timings", {}).get(stage, {}).get("mean_ms")
                    record[stage + "_ms_per_generation"] = row.get("stage_timings", {}).get(stage, {}).get("mean_total_ms_per_generation")
                for phase in suite.PHASES:
                    record[phase + "_phase_ms_per_generation"] = row.get("phase_timings", {}).get(phase, {}).get("mean_ms")
                writer.writerow(record)


def prepare(args, root):
    from qwen35_dflash.ascend310p.cpp_runtime import resolve_cpp_runner, validate_cpp_runner_options
    from qwen35_dflash.ascend310p.generation import tokenize_prompt
    from qwen35_dflash.ascend310p.incremental_plan import write_incremental_plan
    from qwen35_dflash.ascend310p.workflow import load_tokenizer

    if not args.lengths or any(n <= 0 for n in args.lengths) or len(set(args.lengths)) != len(args.lengths):
        raise ValueError("lengths must be distinct positive output budgets")
    if args.device_id < 0 or not 1 <= args.max_draft_tokens <= 15:
        raise ValueError("invalid device or proposal limit")
    suite.set_benchmark_counts(args)
    selection = getattr(args, "verify_gdr", "both")
    if selection not in (*ROUTES, "both"):
        raise ValueError("verify-gdr must be chunk, mtp, or both")
    routes = ROUTES if selection == "both" else (selection,)
    for route in routes:
        if getattr(args, route + "_deployment_manifest", None) is None:
            raise ValueError(f"--{route}-deployment-manifest is required for --verify-gdr {selection}")
    prompts = suite.load_prompts(args.prompts, args.prompt_id, getattr(args, "prompt_group", "all"))
    args.runner_config = (args.runner_config or args.run_dir / "runner.json").resolve()
    identity = validate_cpp_runner_options(json.loads(args.runner_config.read_text()), args.device_id)
    args.runner = resolve_cpp_runner(args.runner)
    tokenizer, tokenizer_source = load_tokenizer(model_dir=args.model_dir)
    for prompt in prompts:
        prompt["prompt_token_ids"] = tokenize_prompt(tokenizer, prompt["prompt"], chat=args.chat)
        prompt["input_tokens"] = len(prompt["prompt_token_ids"])
    required_capacity = math.ceil(
        (max(len(p["prompt_token_ids"]) for p in prompts) + max(args.lengths)) / 64) * 64
    bundles = {}
    for route in routes:
        manifest = getattr(args, route + "_deployment_manifest").expanduser().resolve()
        plan, deployment, contract = write_incremental_plan(manifest, root / (route + "-plan.txt"), verify_gdr=route)
        eos = args.eos_token_id or [248044]
        if any(token < 0 or token >= contract["vocab_size"] for token in eos):
            raise ValueError("EOS token outside vocabulary")
        for prompt in prompts:
            tokens = prompt["prompt_token_ids"]
            if not tokens or any(token < 0 or token >= contract["vocab_size"] for token in tokens):
                raise ValueError(f"invalid tokens for {prompt['id']}")
            for length in args.lengths:
                if len(tokens) + length > contract["capacity"]:
                    raise ValueError(f"{route}/{prompt['id']}: prompt {len(tokens)} + budget {length} "
                                     f"exceeds capacity {contract['capacity']}; no device jobs started. "
                                     f"Export the selected route(s) with max_sequence_length >= {required_capacity}")
        bundles[route] = {
            "manifest": str(manifest), "manifest_sha256": sha256_file(manifest),
            "plan": str(plan), "plan_sha256": sha256_file(plan),
            "abi": contract["abi"], "capacity": contract["capacity"], "vocab_size": contract["vocab_size"],
            "om_sha256": {graph["name"]: graph["om"]["sha256"] for graph in deployment["graphs"]},
            "atc_commands": {graph["name"]: graph["atc_command"] for graph in deployment["graphs"]},
        }
    if len(routes) == 2:
        if bundles["chunk"]["vocab_size"] != bundles["mtp"]["vocab_size"]:
            raise ValueError("Chunk/MTP vocabularies differ")
        if bundles["chunk"]["capacity"] != bundles["mtp"]["capacity"]:
            raise ValueError("Chunk/MTP logical KV capacities differ; export matching capacities for this comparison")
    atomic_write_json(root / "prompts.json", prompts)
    return {
        "schema_version": 1, "status": "PREPARED", "routes": list(routes),
        "lengths": args.lengths, "prompts": prompts,
        "minimum_required_capacity": required_capacity,
        "bundles": bundles, "runtime_identity": identity, "tokenizer_source": tokenizer_source,
        "runner": {"path": str(args.runner), "sha256": sha256_file(args.runner)},
        "script_sha256": sha256_file(Path(__file__)), "suite_script_sha256": sha256_file(Path(suite.__file__)),
        "protocol": {
            "warmup": args.warmup, "repetitions": args.repetitions, "max_draft_tokens": args.max_draft_tokens,
            "prompt_group": getattr(args, "prompt_group", "all"),
            "chat": args.chat, "eos_token_ids": args.eos_token_id or [248044],
            "output_comparison": "allow_output_differences" if args.allow_output_differences else "strict",
            "repeatability_policy": "observe",
            "dflash_speculation_policy": "always_on", "low_memory": args.low_memory,
            "order": f"length order, {' then '.join(routes)}; separate C++ process per cell; models reused across prompts",
        },
        "quality_evaluation": "NOT_RUN", "formal_latency_evidence": False,
        "scope": "output-budget/route experiment; each mode generates its own output; initialization excluded from model-loop metrics",
        "cells": [{"verify_gdr": route, "max_new_tokens": length, "status": "NOT_RUN"}
                  for length in args.lengths for route in routes],
    }


def run(args):
    args.run_dir = args.run_dir.expanduser().resolve()
    if not args.run_dir.is_dir() or args.run_dir.is_relative_to(REPO):
        raise ValueError("run-dir must be an existing directory outside the repository")
    os.environ["AI_RUN_DIR"] = str(args.run_dir)
    root = require_run_output(Path(tempfile.mkdtemp(prefix="gdr-lengths-", dir=args.run_dir)))
    print(f"Output: {root}", flush=True)
    summary = prepare(args, root)  # Selected routes/lengths checked before the first model load.
    print(f"Selected routes: {', '.join(summary['routes'])}; lengths: {', '.join(map(str, summary['lengths']))}; "
          f"prompts: {', '.join(p['id'] for p in summary['prompts'])}", flush=True)
    atomic_write_json(root / "request.json", summary)
    summary["status"] = "PREPARED" if args.plan_only else "RUNNING"
    save(root, summary)
    if args.plan_only:
        print(f"Prepared {len(summary['cells'])} cells, {len(summary['prompts'])} prompts each; no device execution.")
        return 0
    started = time.monotonic()
    try:
        for cell in summary["cells"]:
            route, length = cell["verify_gdr"], cell["max_new_tokens"]
            cell_root = root / f"{route}-{length}"
            cell_root.mkdir()
            cell["status"] = "RUNNING"
            print(f"[gdr-lengths] {route} max_new_tokens={length} start", flush=True)
            options = argparse.Namespace(**vars(args))
            options.run_dir = cell_root
            options.deployment_manifest = Path(summary["bundles"][route]["manifest"])
            options.verify_gdr = route
            options.max_new_tokens = length
            options.prompts = root / "prompts.json"
            options.summarize_existing = None
            try:
                if sha256_file(options.deployment_manifest) != summary["bundles"][route]["manifest_sha256"]:
                    raise ValueError("deployment manifest changed since matrix preflight")
                code = suite.run(options)
                paths = list(cell_root.glob("prompt-suite-*/summary.json"))
                if len(paths) != 1:
                    raise RuntimeError("expected exactly one saved prompt suite")
                cell.update(read_cell(paths[0], route, length, summary["prompts"]))
                cell["suite_exit_code"] = code
                if code and cell["status"] in suite.MEASURED_STATUSES:
                    raise RuntimeError("suite exit code disagrees with passing summary")
            except (OSError, ValueError, KeyError, RuntimeError, TypeError, IndexError) as error:
                cell.update(status="FAIL", error=str(error))
                cell.pop("aggregate", None)
                cell.pop("cases", None)
            finally:
                os.environ["AI_RUN_DIR"] = str(args.run_dir)
            print(f"[gdr-lengths] {route}/{length} {cell['status']}", flush=True)
            summary["process_wall_seconds"] = time.monotonic() - started
            save(root, summary)
    except KeyboardInterrupt:
        summary["status"] = "INTERRUPTED"
        for cell in summary["cells"]:
            if cell["status"] == "RUNNING":
                cell["status"] = "INTERRUPTED"
        save(root, summary)
        print(f"Interrupted; saved completed cells in {root / 'summary.json'}", flush=True)
        return 130
    good = all(cell["status"] in suite.MEASURED_STATUSES for cell in summary["cells"])
    summary["status"] = suite.measured_status(summary["cells"]) if good else "FAIL_OR_INCOMPLETE"
    save(root, summary)
    print(render(summary), flush=True)
    print(f"Summary: {root / 'summary.json'}\nPer-prompt metrics: {root / 'cases.csv'}", flush=True)
    return 0 if good else 1


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run-dir", type=Path, default=os.environ.get("AI_RUN_DIR"),
                        required=not os.environ.get("AI_RUN_DIR"))
    result.add_argument("--runner", type=Path, default=os.environ.get("CPP_RUNNER"),
                        required=not os.environ.get("CPP_RUNNER"))
    result.add_argument("--runner-config", type=Path)
    result.add_argument("--model-dir", type=Path, required=True)
    result.add_argument("--verify-gdr", choices=(*ROUTES, "both"), default="both",
                        help="select chunk, mtp, or both (default); only selected manifests are required")
    result.add_argument("--chunk-deployment-manifest", type=Path,
                        help="required when --verify-gdr is chunk or both")
    result.add_argument("--mtp-deployment-manifest", type=Path,
                        help="required when --verify-gdr is mtp or both")
    result.add_argument("--lengths", type=int, nargs="+", default=list(DEFAULT_LENGTHS),
                        help="one or more output budgets, e.g. --lengths 512")
    result.add_argument("--prompts", type=Path, help="custom JSON; default: 8 short + 12 long (~1K input) prompts")
    result.add_argument("--prompt-group", choices=("all", "short", "long"), default="all",
                        help="select all (default), short, or long inputs; custom JSON can specify group")
    result.add_argument("--prompt-id", action="append",
                        help="test only this ID from --prompts or the built-in suite; repeat to select several")
    result.add_argument("--chat", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--eos-token-id", type=int, action="append")
    result.add_argument("--device-id", type=int, default=0)
    result.add_argument("--max-draft-tokens", type=int, default=15)
    result.add_argument("--low-memory", action="store_true")
    result.add_argument("--warmup", type=int, default=1, help="warmup generations per mode and prompt (default 1)")
    result.add_argument("--repetitions", type=int, default=3, help="measured generations per mode and prompt (default 3)")
    result.add_argument("--allow-output-differences", action="store_true")
    result.add_argument("--plan-only", action="store_true", help="validate selected cells without loading/executing device models")
    return result


def main():
    cli = parser()
    try:
        return run(cli.parse_args())
    except (OSError, ValueError, KeyError, RuntimeError, TypeError, IndexError) as error:
        cli.exit(2, f"gdr-lengths: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
