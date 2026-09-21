"""Reuse a measured ordinary batch without executing or retiming it.

Raw runner reports remain immutable. Derived comparisons carry hashes of both
source reports and are checked again when an existing suite is summarized.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from qwen35_dflash.ascend310p.cpp_runtime import _validate_mode_report
from qwen35_dflash.ascend310p.repeatability import representative_output, repeatability_observation
from qwen35_dflash.ascend310p.utils import atomic_write_json, sha256_file
from qwen35_dflash.decode_metrics import SPEEDUP_SCOPE, measured_decode_ms, time_ratio

ORDER = "saved ordinary baseline then DFlash"


def resolve_sources(path, lengths):
    """Follow saved report references; do not scan runs or execute any model."""
    found, visited = {}, set()
    def visit(value):
        item = Path(value).expanduser().resolve()
        if item.is_dir():
            item = item / ("runner-batch.json" if (item / "runner-batch.json").is_file() else "summary.json")
        if item in visited:
            return
        visited.add(item)
        data = json.loads(item.read_text())
        def referenced(value):
            p = Path(value)
            return p if p.is_absolute() else item.parent / p
        if "runner_index" in data:
            ref = referenced(data["runner_index"])
            if data.get("runner_index_sha256") and sha256_file(ref) != data["runner_index_sha256"]:
                raise ValueError("ordinary baseline runner index changed")
            visit(ref)
        elif "cells" in data:
            for cell in data["cells"]:
                if cell.get("max_new_tokens") not in lengths:
                    continue
                ref = cell.get("ordinary_baseline") or cell.get("summary")
                if isinstance(ref, str):
                    visit(referenced(ref))
        elif "model_sha256" in data and "cases" in data:
            request = json.loads((item.parent / "request.json").read_text())
            if request.get("ordinary_baseline"):
                ref = request["ordinary_baseline"]["index"]
                read_locked(ref)
                visit(ref["path"])
            elif argument(request, "--mode") == "paired":
                found.setdefault(request["max_new_tokens"], item)
            else:
                raise ValueError("baseline batch has no measured ordinary mode")
        else:
            raise ValueError("ordinary baseline expects a runner-batch.json, suite/matrix summary.json, or report directory")
    visit(path)
    missing = set(lengths) - found.keys()
    if missing:
        raise ValueError(f"ordinary baseline missing output budgets {sorted(missing)}; refusing to rerun ordinary")
    return {n: found[n] for n in lengths}


def preflight_sources(sources, summary, args):
    """Validate all budgets, prompts, identities and measurements before DFlash."""
    protocol = summary["protocol"]
    identity = summary["runtime_identity"]
    for length, path in sources.items():
        flags = {"--device-id": args.device_id, "--pad-token-id": identity["pad_token_id"],
                 "--eos-token-ids": ",".join(map(str, protocol["eos_token_ids"])),
                 "--max-new-tokens": length, "--max-draft-tokens": args.max_draft_tokens,
                 "--warmup": protocol["warmup"], "--repetitions": protocol["repetitions"]}
        current = dict(prompts=summary["prompts"], ordinary_contract=next(iter(summary["bundles"].values()))["ordinary_contract"],
                       runtime_identity=identity, runner=summary["runner"],
                       tokenizer_source=summary["tokenizer_source"], chat=protocol["chat"],
                       enable_thinking=protocol.get("enable_thinking"),
                       eos_token_ids=protocol["eos_token_ids"], max_new_tokens=length,
                       max_draft_tokens=args.max_draft_tokens, warmup=protocol["warmup"],
                       repetitions=protocol["repetitions"],
                       command=[part for flag, value in flags.items() for part in (flag, str(value))])
        refs = snapshot(path, current)
        source = read_locked(refs["request"])
        for prompt in current["prompts"]:
            report = read_locked(refs["reports"][prompt["id"]])
            if (report.get("prompt_token_ids") != prompt["prompt_token_ids"]
                    or report.get("cpu_fallback") is not False
                    or report.get("model", {}).get("sha256") != argument(source, "--model-sha256")
                    or report.get("device_id") != args.device_id
                    or report.get("eos_token_ids") != current["eos_token_ids"]):
                raise ValueError(f"ordinary baseline case identity differs: {prompt['id']}")
            _validate_mode_report("ordinary", report.get("ordinary", {}),
                generation_mode="ordinary-greedy", warmup=protocol["warmup"],
                repetitions=protocol["repetitions"], max_new_tokens=length, eos_token_ids=current["eos_token_ids"])


def ordinary_contract(deployment, contract):
    return {
        "capacity": contract["capacity"], "vocab_size": contract["vocab_size"],
        "graphs": {
            name: {"om_sha256": graph["om"]["sha256"], "tensor_abi": graph["metadata"]["tensor_abi"]}
            for graph in deployment["graphs"]
            if (name := graph["name"]) in ("target_prefill", "target_decode")
        },
    }


def record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def read_locked(ref):
    path = Path(ref["path"])
    if sha256_file(path) != ref["sha256"]:
        raise ValueError(f"ordinary baseline source changed: {path}")
    return json.loads(path.read_text())


def argument(request, name):
    command = request["command"]
    if command.count(name) != 1:
        raise ValueError(f"ordinary baseline request needs one {name}")
    return command[command.index(name) + 1]


def compatible(source, current):
    if source.get("enable_thinking") != current.get("enable_thinking"):
        raise ValueError("ordinary baseline differs: enable_thinking (missing setting is unknown)")
    for key in ("ordinary_contract", "runtime_identity", "runner", "chat", "tokenizer_source",
                "eos_token_ids", "max_new_tokens", "max_draft_tokens", "warmup", "repetitions"):
        if key not in source or source[key] != current[key]:
            raise ValueError(f"ordinary baseline differs: {key}")
    fields = ("id", "prompt_token_ids", "dataset_id", "dataset_file", "dataset_line")
    if [[p.get(k) for k in fields] for p in source["prompts"]] != [
            [p.get(k) for k in fields] for p in current["prompts"]]:
        raise ValueError("ordinary baseline prompt selection/token IDs differ")
    for flag in ("--device-id", "--pad-token-id", "--eos-token-ids", "--max-new-tokens",
                 "--max-draft-tokens", "--warmup", "--repetitions"):
        if argument(source, flag) != argument(current, flag):
            raise ValueError(f"ordinary baseline differs: {flag}")


def snapshot(index_path, current):
    index_ref = record(index_path)
    request_ref = record(Path(index_path).parent / "request.json")
    index, source = read_locked(index_ref), read_locked(request_ref)
    compatible(source, current)
    if argument(source, "--mode") != "paired":
        raise ValueError("ordinary baseline must come from the original paired run")
    if (index.get("fake_acl") not in (True, False)
            or index.get("model_sha256") != argument(source, "--model-sha256")
            or index.get("prompt_batch_sha256") != argument(source, "--prompt-batch-sha256")
            or [c["id"] for c in index["cases"]] != [p["id"] for p in current["prompts"]]):
        raise ValueError("ordinary baseline batch identity differs")
    for flag in ("--model", "--prompt-batch"):
        if sha256_file(Path(argument(source, flag))) != argument(source, flag + "-sha256"):
            raise ValueError("ordinary baseline plan/batch changed")
    reports = {}
    for case in index["cases"]:
        path = Path(str(index_path) + ".cases") / (case["id"] + ".json")
        if path.resolve() != Path(case["report"]).resolve():
            raise ValueError("ordinary baseline case path differs")
        reports[case["id"]] = record(path)
    return {"request": request_ref, "index": index_ref, "reports": reports,
            "verify_gdr": source["verify_gdr"], "max_new_tokens": source["max_new_tokens"]}


def location(mode, token_index):
    offset = 0
    rounds = mode["measurements"][0].get("rounds", [])
    for i, value in enumerate(rounds):
        if offset <= token_index < offset + len(value["emitted_token_ids"]):
            return {"measurement_repetition": 0, "round_index": i,
                    "emitted_index": token_index - offset, "round": value,
                    "previous_round": rounds[i - 1] if i else None}
        offset += len(value["emitted_token_ids"])
    return None


def pair(ordinary_report, dflash_report, request, source, refs):
    report = copy.deepcopy(dflash_report)
    report["ordinary_baseline"] = refs
    report["formal_latency_evidence"] = False
    if "protocol" in report:
        report["protocol"]["order"] = ORDER
        report["protocol"]["ordinary_baseline_reused"] = True
    if report.get("status") not in ("PASS", "PASS_WITH_OBSERVATIONS"):
        return report  # Preserve the original DFlash runtime failure.
    try:
        # The ordinary mode is usable even if the first route failed parity or
        # observed repeated-run drift; its raw measurements must still validate.
        if (ordinary_report.get("model", {}).get("sha256") != argument(source, "--model-sha256")
                or ordinary_report.get("prompt_token_ids") != report.get("prompt_token_ids")
                or ordinary_report.get("device_id") != report.get("device_id")
                or ordinary_report.get("eos_token_ids") != request["eos_token_ids"]
                or ordinary_report.get("cpu_fallback") is not False):
            raise ValueError("ordinary baseline case identity differs or measurement is missing")
        ordinary = ordinary_report["ordinary"]
        _validate_mode_report("ordinary", ordinary, generation_mode="ordinary-greedy",
            warmup=request["warmup"], repetitions=request["repetitions"],
            max_new_tokens=request["max_new_tokens"], eos_token_ids=request["eos_token_ids"])
        report["ordinary"] = copy.deepcopy(ordinary)
        dflash = report["dflash"]
        _validate_mode_report("DFlash", dflash, generation_mode="dflash-strict-greedy",
            warmup=request["warmup"], repetitions=request["repetitions"],
            max_new_tokens=request["max_new_tokens"], eos_token_ids=request["eos_token_ids"])
    except (ValueError, RuntimeError, KeyError, TypeError) as error:
        report.update(status="FAIL", failure_stage="ordinary_baseline_validation", error=str(error),
                      ordinary_parity={"status": "NOT_RUN"})
        return report
    expected, expected_stop = representative_output(ordinary)
    actual, actual_stop = representative_output(dflash)
    indices = [i for i in range(max(len(expected), len(actual)))
               if i >= len(expected) or i >= len(actual) or expected[i] != actual[i]]
    eos_mismatches = int(expected_stop != actual_stop)
    difference = None
    if indices:
        i = indices[0]
        difference = {"field": "generated_token_ids", "index": i,
            "absolute_token_position": len(report["prompt_token_ids"]) + i,
            "ordinary_token_id": expected[i] if i < len(expected) else None,
            "dflash_token_id": actual[i] if i < len(actual) else None,
            "ordinary_location": location(ordinary, i), "dflash_location": location(dflash, i)}
    elif eos_mismatches:
        difference = {"field": "stop_reason", "ordinary": expected_stop, "dflash": actual_stop}
    failed = bool(indices or eos_mismatches)
    drift = any(repeatability_observation(m)["differences"] for m in (ordinary, dflash))
    report.update(scope="OM model loop; DFlash measured now, ordinary measurements reused",
        status="FAIL" if failed else "PASS_WITH_OBSERVATIONS" if drift else "PASS",
        ordinary_parity={"scope": "representative measurement 0 from each mode",
                         "status": "FAIL" if failed else "PASS", "token_id_mismatches": len(indices),
                         "eos_mismatches": eos_mismatches, "first_difference": difference},
        speedup_scope=SPEEDUP_SCOPE,
        dflash_decode_time_speedup=None if failed else
            time_ratio(measured_decode_ms(ordinary), measured_decode_ms(dflash)))
    report.pop("dflash_speedup_over_ordinary_model_total_median", None)
    if failed:
        report.update(failure_stage="ordinary_dflash_parity",
                      error="DFlash output differs from the reused ordinary greedy authority")
    return report


def comparisons(index, request, consume):
    """Reconstruct derived reports from hash-locked original runner outputs."""
    baseline = request["ordinary_baseline"]
    source, source_index = read_locked(baseline["request"]), read_locked(baseline["index"])
    compatible(source, request)
    raw_ref = request["dflash_execution"]["index"]
    raw = read_locked(raw_ref)
    if (raw.get("fake_acl") != source_index.get("fake_acl")
            or raw.get("model_sha256") != argument(request, "--model-sha256")
            or raw.get("prompt_batch_sha256") != argument(request, "--prompt-batch-sha256")
            or [c["id"] for c in raw.get("cases", [])] != [p["id"] for p in request["prompts"]]):
        raise ValueError("DFlash-only batch identity differs from ordinary baseline/request")
    result = copy.deepcopy(raw)
    result.update(order=ORDER, ordinary_baseline=baseline, dflash_execution=request["dflash_execution"])
    for case in result["cases"]:
        name = case["id"]
        ordinary_ref = baseline["reports"][name]
        dflash_ref = request["dflash_execution"]["reports"][name]
        if Path(case["report"]).resolve() != Path(dflash_ref["path"]).resolve():
            raise ValueError("DFlash-only case path differs")
        dflash = read_locked(dflash_ref)
        if case["status"] != dflash.get("status"):
            raise ValueError("DFlash-only case/index statuses differ")
        report = pair(read_locked(ordinary_ref), dflash, request, source,
                             {"ordinary_report": ordinary_ref, "dflash_report": dflash_ref,
                              "source_index": baseline["index"]})
        consume(name, report)
        case.update(status=report["status"],
                    report=str(Path(str(index) + ".cases") / (name + ".json")))
    statuses = [case["status"] for case in result["cases"]]
    result["status"] = ("FAIL" if any(s not in ("PASS", "PASS_WITH_OBSERVATIONS") for s in statuses)
                        or raw.get("error") else
                        "PASS_WITH_OBSERVATIONS" if "PASS_WITH_OBSERVATIONS" in statuses else "PASS")
    return result


def merge(index, raw_index, request):
    raw = json.loads(raw_index.read_text())
    request["dflash_execution"] = {"index": record(raw_index), "reports": {
        c["id"]: record(c["report"]) for c in raw["cases"]}}
    def save_report(name, report):
        atomic_write_json(Path(str(index) + ".cases") / (name + ".json"), report)
    result = comparisons(index, request, save_report)
    atomic_write_json(index, result)
    return result


def validate_saved(index, request):
    def check_report(name, report):
        saved = json.loads((Path(str(index) + ".cases") / (name + ".json")).read_text())
        legacy = "dflash_speedup_over_ordinary_model_total_median"
        if (legacy in saved and "dflash_decode_time_speedup" not in saved
                and "speedup_scope" not in saved and "dflash_decode_time_speedup" in report):
            # Validate the historical derived report against locked raw evidence
            # without rewriting it or trusting its old scalar speedup.
            report = copy.deepcopy(report)
            ordinary_ms = report["ordinary"]["latency_ms"]["model_total"]["median"]
            dflash_ms = report["dflash"]["latency_ms"]["model_total"]["median"]
            report[legacy] = None if report["ordinary_parity"]["status"] != "PASS" else ordinary_ms / dflash_ms
            del report["dflash_decode_time_speedup"], report["speedup_scope"]
        if saved != report:
            raise ValueError(f"reused ordinary comparison differs from source reports: {name}")
    expected = comparisons(index, request, check_report)
    if json.loads(index.read_text()) != expected:
        raise ValueError("reused ordinary comparison index differs from source reports")
