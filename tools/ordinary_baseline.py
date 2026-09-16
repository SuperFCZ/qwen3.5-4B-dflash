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

ORDER = "saved ordinary baseline then DFlash"


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
        dflash_speedup_over_ordinary_model_total_median=None if failed else
            ordinary["latency_ms"]["model_total"]["median"] / dflash["latency_ms"]["model_total"]["median"])
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
        if json.loads((Path(str(index) + ".cases") / (name + ".json")).read_text()) != report:
            raise ValueError(f"reused ordinary comparison differs from source reports: {name}")
    expected = comparisons(index, request, check_report)
    if json.loads(index.read_text()) != expected:
        raise ValueError("reused ordinary comparison index differs from source reports")
