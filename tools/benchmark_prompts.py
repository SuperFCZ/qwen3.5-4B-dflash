#!/usr/bin/env python3
"""Paired multi-prompt acceptance/latency test with models reused in one C++ process."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "framework/python"), str(REPO)]

from tools import offline_datasets

DEFAULT_PROMPTS = [
    {"id": "zh_explain", "category": "中文解释", "prompt": "请用通俗的中文解释什么是机器学习，并用一个生活中的例子说明训练和推理的区别。"},
    {"id": "zh_plan", "category": "中文规划", "prompt": "请给刚开始学习 Python 的初学者制定一周学习计划，每天列出学习内容和一个练习。"},
    {"id": "math", "category": "数学推理", "prompt": "一辆汽车前半程以每小时40公里行驶，后半程以每小时60公里行驶，全程120公里。求总用时和平均速度，并逐步解释。"},
    {"id": "code", "category": "代码生成", "prompt": "用 Python 编写二分查找函数：在升序数组中查找目标值，找到返回下标，否则返回 -1。给出代码、两个测试例子和时间复杂度。"},
    {"id": "translate", "category": "英译中", "prompt": "将下面英文翻译成自然的中文，再解释其中的技术术语：Speculative decoding uses a smaller model to propose several tokens. A larger model checks these proposals in parallel and accepts a matching prefix. The benefit depends on both acceptance and verification cost."},
    {"id": "summary", "category": "摘要", "prompt": "请将以下内容概括为三点：某社区将闲置空地改成公共花园。居民轮流浇水，孩子参与种植。花园设有雨水收集桶以节约用水。社区还每月举办一次交流活动，但需要解决工具保管和长期维护经费的问题。"},
    {"id": "en_explain", "category": "英文解释", "prompt": "Explain the difference between a process and a thread to a beginner. Include an example and discuss memory sharing and isolation."},
    {"id": "creative", "category": "中文创作", "prompt": "写一个约200字的小故事：一位图书管理员在旧书中发现一张没有署名的地图。故事要有清晰的开头、转折和结尾。"},
]

MEASURED_STATUSES = {"PASS", "PASS_WITH_DIFFERENCES", "PASS_WITH_OBSERVATIONS"}
LONG_PROMPTS = REPO / "config/prompts_long_1k.json"
STAGES = {
    "ordinary_prefill": ("ordinary", "target_prefill"),
    "ordinary_decode": ("ordinary", "target_decode"),
    "dflash_prefill": ("dflash", "target_prefill"),
    "draft": ("dflash", "draft"),
    "verify": ("dflash", "target_verify"),
}
PHASES = {f"{mode}_{phase}": (mode, phase)
          for mode in ("ordinary", "dflash") for phase in ("prefill", "decode")}


def measured_status(rows):
    if any(row["status"] == "PASS_WITH_OBSERVATIONS" for row in rows):
        return "PASS_WITH_OBSERVATIONS"
    if any(row["status"] == "PASS_WITH_DIFFERENCES" for row in rows):
        return "PASS_WITH_DIFFERENCES"
    return "PASS"


def render_repeatability(rows):
    lines = []
    for row in rows:
        for mode, observation in row.get("repeatability", {}).items():
            changes = observation["differences"]
            if not changes:
                continue
            first = changes[0]
            difference = first["first_difference"]
            detail = (f"token[{difference['index']}] "
                      f"{difference['reference_token_id']} -> {difference['token_id']}"
                      if difference else "stop reason only")
            lines.append(
                f"| {row['id']} | {mode} | {len(changes)} / {observation['compared_repetitions']} | "
                f"{sum(d['token_id_mismatches'] for d in changes)} | "
                f"repeat {first['repetition']}: {detail} |")
    if not lines:
        return ""
    return "\n".join([
        "Repeatability observations (DRIFT_OBSERVED; continuing measurements):", "",
        "| Prompt | Mode | Changed repeats / compared | Token mismatches | First change |",
        "|---|---|---:|---:|---|", *lines, "",
        "Compared with measured repetition 0; warmups excluded. JSON retains each changed repetition, "
        "token counts, stop reasons and first difference. Throughput uses actual tokens from all measurements."
    ]) + "\n"


def set_benchmark_counts(args):
    args.warmup = getattr(args, "warmup", 1)
    args.repetitions = getattr(args, "repetitions", 3)
    if (type(args.warmup) is not int or args.warmup < 0
            or type(args.repetitions) is not int or args.repetitions <= 0):
        raise ValueError("warmup must be non-negative and repetitions must be positive integers")


def checked_times(values, label):
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
        raise ValueError(f"invalid stage timing: {label}")
    return math.fsum(values)


def stage_timings(report):
    """Measured graph calls, including context-building Draft calls during prefill."""
    result = {}
    for label, (mode, stage) in STAGES.items():
        runs = report.get(mode, {}).get("measurements", [])
        groups = [m.get("stage_ms", {}).get(stage) for m in runs]
        if not groups or any(group is None for group in groups):
            result[label] = {"available": False}
            continue
        if any(not isinstance(group, list) for group in groups):
            raise ValueError(f"invalid stage timing array: {mode}/{stage}")
        values = [value for group in groups for value in group]
        total = checked_times(values, f"{mode}/{stage}")
        result[label] = {
            "available": True, "calls": len(values), "total_ms": total,
            "mean_ms": total / len(values) if values else None,
            "median_ms": statistics.median(values) if values else None,
            "measurements": len(runs), "mean_total_ms_per_generation": total / len(runs),
        }
    return result


def phase_timings(report):
    """Full prefill/decode phase elapsed time; startup/reset/warmup excluded."""
    result = {}
    for label, (mode, phase) in PHASES.items():
        values = [m.get("latency_ms", {}).get(phase)
                  for m in report.get(mode, {}).get("measurements", [])]
        if not values or any(value is None for value in values):
            result[label] = {"available": False}
            continue
        total = checked_times(values, f"{mode}/{phase}")
        result[label] = {"available": True, "measurements": len(values),
                         "total_ms": total, "mean_ms": total / len(values)}
    return result


def render_timings(rows, *, measured_only=True):
    def number(value):
        return "N/A" if value is None else f"{value:.2f}"

    good = [row for row in rows if not measured_only or row["status"] in MEASURED_STATUSES]
    if not good:
        return ""
    lines = ["Generation phases (mean ms/generation):", "",
             "| Prompt | Ordinary Prefill | Ordinary Decode loop | DFlash Prefill | DFlash Decode loop |",
             "|---|---:|---:|---:|---:|"]
    for row in good:
        timings = row.get("phase_timings", {})
        lines.append(f"| {row['id']} | " + " | ".join(
            number(timings.get(key, {}).get("mean_ms")) for key in PHASES) + " |")
    lines += ["", "Graph latency (mean ms/call / cumulative mean ms/generation):", "",
              "| Prompt | Ordinary Prefill | Ordinary Decode | DFlash Prefill | Draft | Verify |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in good:
        timings = row.get("stage_timings", {})
        cells = [number(timings.get(key, {}).get("mean_ms")) + " / " +
                 number(timings.get(key, {}).get("mean_total_ms_per_generation")) for key in STAGES]
        lines.append(f"| {row['id']} | " + " | ".join(cells) + " |")
    lines += ["", "Measured repetitions only; warmups, model loading and request reset excluded.",
              "DFlash Prefill phase includes Target Prefill and context-building Draft calls. "
              "Graph Draft totals include those calls; phase and graph tables overlap and must not be added together.",
              "Graph times are synchronized OM calls, not kernel times. Missing timings are N/A."]
    return "\n".join(lines) + "\n"


def load_prompts(path, prompt_ids=None, prompt_group="all"):
    if prompt_group not in ("all", "short", "long"):
        raise ValueError("prompt-group must be all, short, or long")
    values = (json.loads(path.read_text()) if path else
              [dict(p, group="short") for p in DEFAULT_PROMPTS] + json.loads(LONG_PROMPTS.read_text()))
    if not isinstance(values, list) or not 1 <= len(values) <= 64:
        raise ValueError("prompts must be a JSON list of 1..64 strings or {id, prompt, category} objects")
    result, seen = [], set()
    for i, value in enumerate(values):
        if not isinstance(value, (str, dict)):
            raise ValueError("each prompt must be a string or an object")
        item = {"prompt": value} if isinstance(value, str) else dict(value)
        name = item.get("id", f"prompt_{i + 1:02d}")
        prompt = item.get("prompt")
        if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name)
                or name in seen or not isinstance(prompt, str) or not prompt.strip()):
            raise ValueError("each prompt needs nonempty text and a unique safe id")
        seen.add(name)
        group = item.get("group", "custom")
        if group not in ("short", "long", "custom"):
            raise ValueError("prompt group must be short, long, or custom")
        result.append({"id": name, "category": str(item.get("category", "custom")),
                       "prompt": prompt, "group": group})
    if prompt_group != "all":
        result = [item for item in result if item["group"] == prompt_group]
        seen = {item["id"] for item in result}
    if prompt_ids:
        unknown = set(prompt_ids) - seen
        if unknown:
            raise ValueError(f"unknown prompt id(s): {', '.join(sorted(unknown))}")
        result = [item for item in result if item["id"] in prompt_ids]
    if not result:
        raise ValueError(f"no prompts in selected group {prompt_group}; custom JSON may specify group: short or long")
    return result


def load_inputs(args):
    # Matrix cells use the already frozen text/provenance, even if external files change.
    if getattr(args, "_prepared_inputs", None) is not None:
        return copy.deepcopy(args._prepared_inputs)
    if offline_datasets.selected(args):
        return offline_datasets.load(args)
    if getattr(args, "num_questions", None) is not None or getattr(args, "dataset_field", "question") != "question":
        raise ValueError("--num-questions/--dataset-field require --dataset-dir or --dataset-files")
    return load_prompts(args.prompts, getattr(args, "prompt_id", None), getattr(args, "prompt_group", "all")), []


def decode_outputs(report, tokenizer):
    """Decode saved tokens, including specials, without changing parity results."""
    from qwen35_dflash.ascend310p.repeatability import representative_output

    outputs = {}
    for mode in ("ordinary", "dflash"):
        benchmark = report.get(mode)
        # Also accept a legacy per-mode *.ordinary.json or --mode report.
        standalone = report.get("benchmark", report)
        if benchmark is None and standalone.get("generation_mode", "").startswith(mode):
            benchmark = standalone
        tokens, stop = representative_output(benchmark) if benchmark else (None, None)
        if not isinstance(tokens, list):
            outputs[mode] = {"available": False}
            continue
        if any(type(token) is not int or token < 0 for token in tokens):
            raise ValueError("invalid saved generated token IDs")
        outputs[mode] = {"available": True, "token_count": len(tokens),
                         "stop_reason": stop, "measurement_repetition": 0,
                         "text": tokenizer.decode(tokens, skip_special_tokens=False)}
        difference = report.get("ordinary_parity", {}).get("first_difference")
        if difference and difference.get("field") == "generated_token_ids":
            index = difference["index"]
            begin, end = max(0, index - 12), min(len(tokens), index + 13)
            outputs[mode]["difference_context"] = {
                "token_begin": begin, "token_end": end,
                "token_ids": tokens[begin:end],
                "text": tokenizer.decode(tokens[begin:end], skip_special_tokens=False),
            }
    return outputs


def render_outputs(rows):
    lines = []
    for row in rows:
        lines += [f"=== {row.get('id', 'report')} | {row.get('status', 'UNKNOWN')} ==="]
        if row.get("prompt"):
            lines += ["Prompt:", row["prompt"]]
        if row.get("error"):
            lines += ["Error: " + row["error"]]
        difference = row.get("first_difference")
        if difference:
            if difference.get("field") == "generated_token_ids":
                lines += [f"First different generated token (zero-based): {difference['index']}; "
                          f"ordinary={difference['ordinary_token_id']}, dflash={difference['dflash_token_id']}"]
            else:
                lines += ["First difference: " + json.dumps(difference, ensure_ascii=False)]
        for mode in ("ordinary", "dflash"):
            value = row.get("decoded_outputs", {}).get(mode, {})
            if not value.get("available"):
                lines += [f"--- {mode}: output token IDs unavailable in saved report ---"]
            else:
                lines += [f"--- {mode}: {value['token_count']} tokens, stop={value['stop_reason']} ---", value["text"]]
                context = value.get("difference_context")
                if context:
                    lines += [f"First-difference context, tokens [{context['token_begin']}, {context['token_end']}):",
                              context["text"]]
        lines += [""]
    return "\n".join(lines) + "\n"


def observed_acceptance(report):
    """Read current-verifier acceptance, independently of the pair's PASS/FAIL."""
    draft = report.get("dflash")
    standalone = report.get("benchmark", report)
    if draft is None and standalone.get("generation_mode", "").startswith("dflash"):
        draft = standalone
    if not isinstance(draft, dict):
        return {"available": False}
    measurements = draft.get("measurements")
    if measurements:
        # WriteBenchmark stores only measured calls here, not the warmup calls.
        counters = [m["counters"] for m in measurements]
        source = "measurements"
    elif isinstance(draft.get("totals"), dict):
        counters = [draft["totals"]]
        source = "totals"
    else:
        return {"available": False}
    for count in counters:
        proposed, accepted = count["drafted_tokens"], count["accepted_draft_tokens"]
        if (type(proposed) is not int or type(accepted) is not int
                or not 0 <= accepted <= proposed):
            raise ValueError("invalid saved DFlash acceptance counters")
    proposed = sum(c["drafted_tokens"] for c in counters)
    accepted = sum(c["accepted_draft_tokens"] for c in counters)
    return {"available": True, "source": source,
            "drafted_tokens": proposed, "accepted_draft_tokens": accepted,
            "acceptance_rate": accepted / proposed if proposed else None,
            "scope": "current verify decisions; measured repetitions only, warmups excluded"}


def render_acceptance(rows):
    """Display failed runs too, without admitting them into validated statistics."""
    lines = ["Observed DFlash acceptance (current verify):", "",
             "| Prompt | Status | Accepted / proposed | Acceptance |",
             "|---|---|---:|---:|"]
    available = []
    for row in rows:
        stats = row.get("observed_acceptance", {})
        if stats.get("available"):
            available.append(stats)
            ratio = stats["acceptance_rate"]
            rate = f"{ratio:.2%}" if ratio is not None else "N/A (no proposals)"
            counts = f'{stats["accepted_draft_tokens"]} / {stats["drafted_tokens"]}'
        else:
            counts, rate = "—", "N/A (not recorded)"
        lines.append(f"| {row.get('id', 'report')} | {row.get('status', 'UNKNOWN')} | {counts} | {rate} |")
    proposed = sum(s["drafted_tokens"] for s in available)
    accepted = sum(s["accepted_draft_tokens"] for s in available)
    rate = f"{accepted / proposed:.2%}" if proposed else "N/A"
    lines += ["", f"Observed weighted acceptance: {rate}; accepted={accepted}, proposed={proposed}; "
              f"reports with counters={len(available)}/{len(rows)}.",
              "Counts sum measured repetitions and exclude warmups.",
              "FAIL remains FAIL. These are current verify decisions, including failed parity runs; "
              "they do not establish ordinary parity or speedup."]
    return "\n".join(lines) + "\n"


def acceptance_by_position(report, window=32):
    """Attribute whole speculative rounds to their next output token's index."""
    from qwen35_dflash.ascend310p.compare_rounds import _rounds_by_prefix

    bins = {}
    prompt = report["prompt_token_ids"]
    for measurement in report["dflash"]["measurements"]:
        generated = measurement["generated_token_ids"]
        rounds = measurement.get("rounds")
        if not rounds:
            raise ValueError("position analysis requires complete saved round traces")
        _rounds_by_prefix(rounds, prompt, generated)
        proposed_total = accepted_total = 0
        for row in rounds:
            proposed = len(row["proposed_token_ids"])
            if not proposed:
                continue  # Prefill and historical target-only continuation.
            accepted = len(row["accepted_draft_token_ids"])
            offset = row["committed_prefix_length"] - len(prompt)
            begin = offset // window * window
            bucket = bins.setdefault(begin, {
                "generated_index_begin": begin, "generated_index_end": begin + window,
                "rounds": 0, "zero_accept_rounds": 0, "drafted_tokens": 0,
                "accepted_draft_tokens": 0, "emitted_tokens": 0,
            })
            bucket["rounds"] += 1
            bucket["zero_accept_rounds"] += int(accepted == 0)
            bucket["drafted_tokens"] += proposed
            bucket["accepted_draft_tokens"] += accepted
            bucket["emitted_tokens"] += len(row["emitted_token_ids"])
            proposed_total += proposed
            accepted_total += accepted
        counters = measurement["counters"]
        if (proposed_total != counters["drafted_tokens"]
                or accepted_total != counters["accepted_draft_tokens"]):
            raise ValueError("saved round traces disagree with acceptance counters")
    result = []
    for begin in sorted(bins):
        bucket = bins[begin]
        bucket.update(
            acceptance_rate=bucket["accepted_draft_tokens"] / bucket["drafted_tokens"],
            zero_accept_rate=bucket["zero_accept_rounds"] / bucket["rounds"],
            mean_proposed_tokens=bucket["drafted_tokens"] / bucket["rounds"],
            mean_accepted_tokens=bucket["accepted_draft_tokens"] / bucket["rounds"],
            tokens_per_round=bucket["emitted_tokens"] / bucket["rounds"],
        )
        result.append(bucket)
    return result


def render_position_acceptance(rows):
    lines = ["Acceptance by generation position:", "",
             "| Prompt | Round start token index | Rounds | Mean proposed | Acceptance | Zero-accept rounds | Tokens / round |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        for bucket in row.get("acceptance_by_position", []):
            lines.append(
                f"| {row['id']} | [{bucket['generated_index_begin']}, {bucket['generated_index_end']}) | "
                f"{bucket['rounds']} | {bucket['mean_proposed_tokens']:.2f} | "
                f"{bucket['acceptance_rate']:.2%} | {bucket['zero_accept_rate']:.2%} | "
                f"{bucket['tokens_per_round']:.2f} |")
    lines += ["", "Measured repetitions only. Each whole speculative round belongs to the bin of its first emitted token; "
              "a round crossing a bin boundary is not split. Prefill/target-only rounds are excluded. "
              "No rounds in a bin means no observation, not zero acceptance."]
    return "\n".join(lines) + "\n"


def summarize_prompt(report):
    from qwen35_dflash.ascend310p.repeatability import repeatability_observation, representative_output

    ordinary, draft = report["ordinary"], report["dflash"]
    measurements = draft["measurements"]
    drafted = sum(m["counters"]["drafted_tokens"] for m in measurements)
    accepted = sum(m["counters"]["accepted_draft_tokens"] for m in measurements)
    rounds = [r for m in measurements for r in m["rounds"] if r["proposed_token_ids"]]
    emitted = sum(len(r["emitted_token_ids"]) for r in rounds)
    ordinary_ms = sum(m["latency_ms"]["model_total"] for m in ordinary["measurements"])
    dflash_ms = sum(m["latency_ms"]["model_total"] for m in measurements)
    draft_lengths = [len(m["generated_token_ids"]) for m in measurements]
    ordinary_lengths = [len(m["generated_token_ids"]) for m in ordinary["measurements"]]
    draft_tps = sum(draft_lengths) * 1000 / dflash_ms if dflash_ms > 0 else 0
    ordinary_tps = sum(ordinary_lengths) * 1000 / ordinary_ms if ordinary_ms > 0 else 0
    for benchmark in (ordinary, draft):
        latencies = [m["latency_ms"]["model_total"] for m in benchmark["measurements"]]
        latencies += [benchmark["latency_ms"]["model_total"]["median"], benchmark["generated_tokens_per_second"]]
        if any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in latencies):
            raise ValueError("invalid saved model latency or throughput")
    if not 0 <= accepted <= drafted or ordinary_ms <= 0 or dflash_ms <= 0:
        raise ValueError("invalid acceptance counters or model latency")
    return {
        "drafted_tokens": drafted, "accepted_draft_tokens": accepted,
        "stage_timings": stage_timings(report), "phase_timings": phase_timings(report),
        "acceptance_rate": accepted / drafted if drafted else None,
        "speculative_rounds": len(rounds), "tokens_emitted_in_speculative_rounds": emitted,
        "tokens_per_speculative_round": emitted / len(rounds) if rounds else None,
        "target_only_fallback_rounds": sum(m["counters"]["target_only_fallback_rounds"] for m in measurements),
        "speculation_disable_events": sum(m["counters"]["speculation_disable_events"] for m in measurements),
        "generated_tokens": statistics.mean(draft_lengths),
        "ordinary_generated_tokens": statistics.mean(ordinary_lengths),
        "generated_tokens_range": [min(draft_lengths), max(draft_lengths)],
        "ordinary_generated_tokens_range": [min(ordinary_lengths), max(ordinary_lengths)],
        "dflash_measured_tokens": sum(draft_lengths), "ordinary_measured_tokens": sum(ordinary_lengths),
        "representative_repetition": 0,
        "repeatability": {name: repeatability_observation(report[name]) for name in ("ordinary", "dflash")},
        "draft_token_share_of_output": accepted / sum(len(m["generated_token_ids"]) for m in measurements),
        "acceptance_by_position": acceptance_by_position(report),
        "stop_reason": representative_output(draft)[1],
        "ordinary_stop_reason": representative_output(ordinary)[1],
        "stop_reasons": [m["stop_reason"] for m in measurements],
        "ordinary_stop_reasons": [m["stop_reason"] for m in ordinary["measurements"]],
        "ordinary_total_measured_ms": ordinary_ms, "dflash_total_measured_ms": dflash_ms,
        "ordinary_median_ms": ordinary["latency_ms"]["model_total"]["median"],
        "dflash_median_ms": draft["latency_ms"]["model_total"]["median"],
        "speedup": ordinary["latency_ms"]["model_total"]["median"] / draft["latency_ms"]["model_total"]["median"],
        "dflash_tokens_per_second": draft_tps,
        "ordinary_tokens_per_second": ordinary_tps,
        "throughput_speedup": draft_tps / ordinary_tps,
    }


def aggregate(rows):
    good = [r for r in rows if r["status"] in MEASURED_STATUSES]
    drafted = sum(r["drafted_tokens"] for r in good)
    accepted = sum(r["accepted_draft_tokens"] for r in good)
    latency = sum(r["dflash_total_measured_ms"] for r in good)
    return {"scope": "completed measurements admitted by output-comparison policy; repeatability drift is observational",
            "passed_prompts": sum(r["status"] == "PASS" for r in rows),
            "allowed_difference_prompts": sum(r["status"] == "PASS_WITH_DIFFERENCES" or bool(r.get("output_difference")) for r in good),
            "drift_observed_prompts": sum(r["status"] == "PASS_WITH_OBSERVATIONS" for r in good),
            "measured_prompts": len(good), "failed_prompts": sum(r["status"] == "FAIL" for r in rows),
            "not_run_prompts": sum(r["status"] == "NOT_RUN" for r in rows),
            "drafted_tokens": drafted, "accepted_draft_tokens": accepted,
            "weighted_acceptance_rate": accepted / drafted if drafted else None,
            "total_model_time_speedup": sum(r["ordinary_total_measured_ms"] for r in good) / latency if latency else None}


def aggregate_metrics(rows):
    totals = aggregate(rows)
    good = [row for row in rows if row["status"] in MEASURED_STATUSES]
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
        row["generated_tokens"] == row["ordinary_generated_tokens"] == row.get("max_new_tokens") for row in good)
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
    for phase in PHASES:
        values = [row.get("phase_timings", {}).get(phase, {}) for row in good]
        if not values or any(not value.get("available") for value in values):
            totals["phase_timings"][phase] = {"available": False}
            continue
        measurements = sum(value["measurements"] for value in values)
        elapsed = math.fsum(value["total_ms"] for value in values)
        totals["phase_timings"][phase] = {"available": True, "measurements": measurements,
                                         "total_ms": elapsed, "mean_ms": elapsed / measurements}
    return totals


def dataset_results(summary):
    """One row per source file, route and output budget; never average percentages."""
    cells = summary.get("cells", [dict(summary, verify_gdr=summary.get("protocol", {}).get("verify_gdr"))])
    result = []
    for cell in cells:
        for dataset in summary.get("datasets", []):
            rows = [r for r in cell.get("cases", []) if r.get("dataset_id") == dataset["id"]]
            totals = aggregate_metrics(rows)
            totals["not_run_prompts"] += max(0, dataset["selected_samples"] - len(rows))
            complete = len(rows) == dataset["selected_samples"] and all(r["status"] in MEASURED_STATUSES for r in rows)
            status = measured_status(rows) if complete else "FAIL_OR_INCOMPLETE"
            if not rows:
                status = cell.get("status", "NOT_RUN")
                if status in MEASURED_STATUSES:
                    status = "FAIL_OR_INCOMPLETE"
            if cell.get("error"):
                status = "FAIL_OR_INCOMPLETE"
            result.append({**totals, "dataset_id": dataset["id"], "dataset_file": dataset["name"],
                           "dataset_sha256": dataset["sha256"], "total_samples": dataset["total_samples"],
                           "selected_samples": dataset["selected_samples"], "status": status,
                           "verify_gdr": cell.get("verify_gdr"), "max_new_tokens": cell.get("max_new_tokens"),
                           "error": cell.get("error")})
    return result


def render_datasets(results):
    def number(value, percent=False):
        return "N/A" if value is None else f"{value:.2%}" if percent else f"{value:.2f}"
    lines = ["Acceptance by dataset file:", "",
             "| Dataset file | GDR | Max new tokens | Status | Measured / selected | Accepted / proposed | Acceptance | Tokens / round | Ordinary tok/s | DFlash tok/s | Speedup |",
             "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    timing_rows = []
    for row in results:
        name = row["dataset_file"].replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {name} | {row['verify_gdr']} | {row['max_new_tokens']} | {row['status']} | "
                     f"{row['measured_prompts']} / {row['selected_samples']} | "
                     f"{row['accepted_draft_tokens']} / {row['drafted_tokens']} | "
                     f"{number(row['weighted_acceptance_rate'], True)} | "
                     f"{number(row['tokens_per_speculative_round'])} | "
                     f"{number(row['ordinary_tokens_per_second'])} | {number(row['dflash_tokens_per_second'])} | "
                     f"{number(row['total_model_time_speedup'])} |")
        timing_rows.append(dict(id=f"{row['verify_gdr']}/{row['max_new_tokens']}/{name}",
                                stage_timings=row["stage_ms_per_call"], phase_timings=row["phase_timings"]))
    lines += ["", "Acceptance = sum accepted / sum proposed; speedup = sum ordinary time / sum DFlash time.",
              "Only admitted measurements contribute; measured/selected exposes incomplete files. Warmups and startup excluded.",
              "Each mode generates its own output. Task accuracy is not evaluated; repeated-run drift remains observable.",
              "", render_timings(timing_rows, measured_only=False).rstrip()]
    for row in results:
        if row["drift_observed_prompts"]:
            lines += ["", f"- {row['dataset_file']} / {row['verify_gdr']} / {row['max_new_tokens']}: "
                      f"DRIFT_OBSERVED in {row['drift_observed_prompts']} questions; "
                      "per-repetition token/EOS differences retained in the per-file JSON."]
    return "\n".join(lines) + "\n"


def write_dataset_reports(root, summary):
    """Write both the cross-file table and each file's own metrics/case references."""
    if not summary.get("datasets"):
        return
    from qwen35_dflash.ascend310p.utils import atomic_write_json

    results = summary["dataset_results"] = dataset_results(summary)
    fields = ["dataset_id", "dataset_file", "dataset_sha256", "verify_gdr", "max_new_tokens", "status",
              "total_samples", "selected_samples", "measured_prompts", "failed_prompts", "not_run_prompts",
              "allowed_difference_prompts", "drift_observed_prompts",
              "accepted_draft_tokens", "drafted_tokens", "weighted_acceptance_rate", "tokens_per_speculative_round",
              "ordinary_measured_tokens", "dflash_measured_tokens", "ordinary_tokens_per_second", "dflash_tokens_per_second",
              "total_model_time_speedup", "throughput_speedup",
              *[stage + "_ms_per_call" for stage in STAGES],
              *[stage + "_ms_per_generation" for stage in STAGES],
              *[phase + "_phase_ms_per_generation" for phase in PHASES]]
    with (root / "datasets.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            row = {key: result.get(key) for key in fields}
            for stage in STAGES:
                value = result["stage_ms_per_call"][stage]
                row[stage + "_ms_per_call"] = value.get("mean_ms")
                row[stage + "_ms_per_generation"] = value.get("mean_total_ms_per_generation")
            for phase in PHASES:
                row[phase + "_phase_ms_per_generation"] = result["phase_timings"][phase].get("mean_ms")
            writer.writerow(row)
    cases = [r for c in summary.get("cells", [summary]) for r in c.get("cases", [])]
    for dataset in summary["datasets"]:
        # IDs are generated by the loader, but saved requests may have been edited.
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", dataset["id"]):
            raise ValueError("unsafe dataset ID in saved request")
        directory = root / "datasets" / dataset["id"]
        directory.mkdir(parents=True, exist_ok=True)
        selected = [r for r in results if r["dataset_id"] == dataset["id"]]
        payload = {"schema_version": 1, "dataset": dataset, "protocol": summary.get("protocol", {}),
                   "results": selected, "cases": [r for r in cases if r.get("dataset_id") == dataset["id"]],
                   "quality_evaluation": "NOT_RUN", "source_summary": str(root / "summary.json")}
        atomic_write_json(directory / "summary.json", payload)
        (directory / "summary.md").write_text(render_datasets(selected), encoding="utf-8")


def markdown(summary):
    def value(number, percent=False):
        return "N/A" if number is None else f"{number * 100:.2f}%" if percent else f"{number:.2f}"
    lines = ["| Prompt | Input tokens | Status | Acceptance | Tokens / speculative round | DFlash tok/s | Speedup | Generated |",
             "|---|---:|---|---:|---:|---:|---:|---:|"]
    for row in summary["cases"]:
        input_tokens = row.get("input_tokens", "N/A")
        if row["status"] not in MEASURED_STATUSES:
            lines.append(f"| {row['id']} | {input_tokens} | {row['status']} | — | — | — | — | — |")
        else:
            bounds = row.get("generated_tokens_range", [row["generated_tokens"]] * 2)
            generated = str(bounds[0]) if bounds[0] == bounds[1] else f"{bounds[0]}..{bounds[1]}"
            lines.append(f"| {row['id']} | {input_tokens} | {row['status']} | {value(row['acceptance_rate'], True)} | "
                         f"{value(row['tokens_per_speculative_round'])} | {value(row['dflash_tokens_per_second'])} | "
                         f"{value(row['speedup'])}x | {generated} |")
    totals = summary["aggregate"]
    protocol = summary.get("protocol", {})
    repeats = f"{protocol.get('warmup', 'N/A')}+{protocol.get('repetitions', 'N/A')}"
    lines += ["", f"Passed: {totals['passed_prompts']}; failed: {totals['failed_prompts']}; not run: {totals['not_run_prompts']}.",
              f"Warmup per mode/prompt: {protocol.get('warmup', 'N/A')}; measured repetitions: {protocol.get('repetitions', 'N/A')}.",
              f"Weighted acceptance: {value(totals['weighted_acceptance_rate'], True)}.",
              "Acceptance = accepted draft tokens / proposed draft tokens; warmups excluded.",
              "Speedup > 1 means faster than ordinary generation; acceptance alone does not establish speedup."]
    if totals.get("allowed_difference_prompts"):
        lines += [f"Allowed output differences: {totals['allowed_difference_prompts']}; "
                  f"both modes completed independent {repeats} measurements. Output parity remains FAIL; task quality was not evaluated.",
                  "Speedup compares model-loop time for each mode's own output. Different EOS lengths can change the work; "
                  "JSON also records both token counts and throughput_speedup."]
    if summary.get("ordinary_baseline"):
        lines += ["Ordinary measurements are reused from the first verification route; only DFlash ran in this cell."]
    if totals["passed_prompts"]:
        lines += [f"Each passing prompt passed token/EOS parity and the requested {repeats} repeatability checks."]
    elif not totals.get("measured_prompts"):
        lines += ["No prompt has passed the complete checks; no validated acceptance or speedup is available."]
    if totals.get("drift_observed_prompts"):
        lines += [f"Repeatability drift observed: {totals['drift_observed_prompts']} prompts; metrics retained. "
                  "Generated shows the measured token-count range when lengths differ. "
                  "Cross-mode output comparison uses measured repetition 0."]
        lines += ["", render_repeatability(summary["cases"]).rstrip()]
    if any(r.get("observed_acceptance", {}).get("available") for r in summary["cases"]):
        lines += ["", render_acceptance(summary["cases"]).rstrip()]
    timings = render_timings(summary["cases"])
    if timings:
        lines += ["", timings.rstrip()]
    if any(r.get("acceptance_by_position") for r in summary["cases"]):
        lines += ["", render_position_acceptance(summary["cases"]).rstrip()]
    for row in summary["cases"]:
        if row.get("error"):
            lines += ["", f"- {row['id']} ({row.get('failure_stage', 'report_validation')}): "
                      + row["error"].replace("\n", " ")]
        elif row.get("output_difference"):
            lines += ["", f"- {row['id']} (allowed output difference): " + row["output_difference"].replace("\n", " ")]
    if summary.get("dataset_results"):
        lines += ["", render_datasets(summary["dataset_results"]).rstrip()]
    return "\n".join(lines) + "\n"


def run_runner(command, log):
    with log.open("x") as stream:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            for line in process.stdout:
                stream.write(line)
                if line.startswith(("[prompt-batch]", "[repeatability]")):
                    print(line.rstrip(), flush=True)
            return process.wait()


def collect_results(args, prompts, raw, index, plan_hash, batch_hash, eos, exit_code, tokenizer=None):
    from qwen35_dflash.ascend310p.cpp_runtime import validate_cpp_runner_report
    from qwen35_dflash.ascend310p.repeatability import representative_output
    from qwen35_dflash.ascend310p.utils import sha256_file

    allow_differences = getattr(args, "allow_output_differences", False)
    warmup, repetitions = getattr(args, "warmup", 3), getattr(args, "repetitions", 10)
    index_error = None
    if (exit_code not in (None, 0, 1) or raw.get("fake_acl") is not False
            or raw.get("status") not in ("PASS", "PASS_WITH_OBSERVATIONS", "FAIL") or raw.get("error")
            or raw.get("prompt_batch_sha256") != batch_hash or raw.get("model_sha256") != plan_hash
            or [c["id"] for c in raw.get("cases", [])] != [p["id"] for p in prompts]):
        index_error = "missing/invalid batch index or fake ACL; inspect runner.log"
    rows = []
    for i, prompt in enumerate(prompts):
        row = {**prompt, "input_tokens": len(prompt["prompt_token_ids"]), "status": "FAIL",
               "max_new_tokens": args.max_new_tokens, "verify_gdr": getattr(args, "verify_gdr", None)}
        try:
            if index_error:
                raise RuntimeError(index_error)
            case = raw["cases"][i]
            path = Path(case["report"]).resolve()
            if path != (Path(str(index) + ".cases") / (prompt["id"] + ".json")).resolve():
                raise ValueError("unexpected case report path")
            report = json.loads(path.read_text())
            row.update(raw_report=str(path), raw_report_sha256=sha256_file(path), raw_status=report.get("status"),
                       observed_acceptance=observed_acceptance(report), ordinary_parity=report.get("ordinary_parity", {}),
                       first_difference=report.get("ordinary_parity", {}).get("first_difference"))
            if tokenizer is not None:
                row["decoded_outputs"] = decode_outputs(report, tokenizer)
            if case["status"] != report.get("status"):
                raise ValueError("batch/case report statuses differ")
            allowed = (allow_differences and case["status"] == "FAIL"
                       and report.get("failure_stage") == "ordinary_dflash_parity")
            if case["status"] not in ("PASS", "PASS_WITH_OBSERVATIONS") and not allowed:
                row.update(status="NOT_RUN" if case["status"] == "NOT_RUN" else "FAIL",
                    failure_stage=report.get("failure_stage", "unknown"), error=report.get("error", "C++ prompt failed"))
                rows.append(row)
                continue
            validate_cpp_runner_report(report, prompt_token_ids=prompt["prompt_token_ids"],
                om_sha256=plan_hash, device_id=args.device_id, max_new_tokens=args.max_new_tokens,
                max_draft_tokens=args.max_draft_tokens, chunk_abi=True, low_memory=args.low_memory,
                verify_gdr=getattr(args, "verify_gdr", None),
                warmup=warmup, repetitions=repetitions,
                allow_output_differences=allow_differences)
            if report["eos_token_ids"] != eos or report["protocol"].get("round_trace_enabled") is not True:
                raise ValueError("EOS or trace settings differ")
            row.update(summarize_prompt(report))
            drift = any(obs["differences"] for obs in row["repeatability"].values())
            row["status"] = "PASS_WITH_OBSERVATIONS" if drift else "PASS_WITH_DIFFERENCES" if allowed else "PASS"
            if allowed:
                row["output_difference"] = report.get("error", "ordinary/DFlash outputs differ")
            if tokenizer is not None:
                row["generated_text"] = tokenizer.decode(representative_output(report["dflash"])[0], skip_special_tokens=True)
        except (OSError, ValueError, KeyError, RuntimeError, TypeError) as error:
            row.update(status="FAIL", failure_stage="report_validation", error=str(error))
        rows.append(row)
    ok = bool(rows) and not index_error and all(r["status"] in MEASURED_STATUSES for r in rows)
    differences = any(r.get("output_difference") for r in rows)
    drift = any(r["status"] == "PASS_WITH_OBSERVATIONS" for r in rows)
    if not differences and (raw.get("status") not in ("PASS", "PASS_WITH_OBSERVATIONS") or exit_code not in (None, 0)):
        ok = False
    parity_failed = any(r.get("ordinary_parity", {}).get("status") == "FAIL" for r in rows)
    return {"schema_version": 1, "status": measured_status(rows) if ok else "FAIL_OR_INCOMPLETE",
        "cases": rows, "aggregate": aggregate(rows), "protocol": {
            "warmup": warmup, "repetitions": repetitions, "low_memory": args.low_memory,
            "prompt_group": getattr(args, "prompt_group", "all"),
            "output_comparison": "allow_output_differences" if allow_differences else "strict",
            "repeatability_policy": "observe",
            "dflash_speculation_policy": raw.get("dflash_speculation_policy", "not_recorded"),
            "models_reused_across_prompts": raw.get("models_reused_across_prompts"), "order": raw.get("order"),
            "verify_gdr": getattr(args, "verify_gdr", None)},
        "startup_ms": raw.get("startup_ms"), "runner_index": str(index),
        "runner_index_sha256": sha256_file(index) if index.is_file() else None, "runner_exit_code": exit_code,
        "ordinary_parity": "FAIL" if parity_failed else "PASS" if ok else "FAIL_OR_INCOMPLETE",
        "ordinary_parity_scope": "representative measurement 0 from each mode",
        "repeatability": "DRIFT_OBSERVED" if drift else "STABLE" if ok else "NOT_ESTABLISHED",
        "quality_evaluation": "NOT_RUN", "formal_latency_evidence": bool(ok and not differences and not drift
            and not raw.get("ordinary_baseline") and (warmup, repetitions) == (3, 10)),
        "scope": "This selected prompt suite. Allowed differences are experimental comparisons of each mode's own output, not quality equivalence."}


def write_summary(root, summary):
    from qwen35_dflash.ascend310p.utils import atomic_write_json

    write_dataset_reports(root, summary)
    atomic_write_json(root / "summary.json", summary)
    (root / "summary.md").write_text(markdown(summary), encoding="utf-8")
    print(render_datasets(summary["dataset_results"]) if summary.get("datasets") else markdown(summary), flush=True)
    print(f"Summary: {root / 'summary.json'}", flush=True)
    if any("decoded_outputs" in row for row in summary["cases"]):
        (root / "generations.txt").write_text(render_outputs(summary["cases"]), encoding="utf-8")
        print(f"Decoded outputs: {root / 'generations.txt'}", flush=True)
    return 0 if summary["status"] in MEASURED_STATUSES else 1


def summarize_existing(args):
    """Read one saved suite; never execute the runner or modify its evidence."""
    from qwen35_dflash.ascend310p.utils import require_run_output, sha256_file

    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir() or run_dir.is_relative_to(REPO):
        raise ValueError("run-dir must be an existing directory outside the repository")
    os.environ["AI_RUN_DIR"] = str(run_dir)
    index = args.summarize_existing.expanduser().resolve()
    request_path = index.parent / "request.json"
    request = json.loads(request_path.read_text())
    raw = json.loads(index.read_text())
    if request.get("ordinary_baseline"):
        from tools import ordinary_baseline
        ordinary_baseline.validate_saved(index, request)
    command = request["command"]

    def argument(name, default=None):
        if isinstance(command, list) and name not in command and default is not None:
            return default  # Older C++ requests omitted the then-fixed 3+10 defaults.
        if not isinstance(command, list) or command.count(name) != 1:
            raise ValueError(f"saved request needs one {name}")
        return command[command.index(name) + 1]

    prompts = request["prompts"]
    names = [p["id"] for p in prompts]
    if not names or len(set(names)) != len(names) or any(
            not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name) for name in names):
        raise ValueError("invalid saved prompt IDs")
    plan = index.parent / Path(argument("--model")).name
    batch = index.parent / Path(argument("--prompt-batch")).name
    plan_hash, batch_hash = sha256_file(plan), sha256_file(batch)
    if plan_hash != argument("--model-sha256") or batch_hash != argument("--prompt-batch-sha256"):
        raise ValueError("saved plan/batch hashes differ from the request")
    expected_batch = "QWEN35_PROMPT_BATCH_V1\n" + "".join(
        f"{p['id']} \"{','.join(map(str, p['prompt_token_ids']))}\"\n" for p in prompts)
    if batch.read_text() != expected_batch:
        raise ValueError("saved prompt tokens differ from batch inputs")
    stored = argparse.Namespace(device_id=int(argument("--device-id")),
        warmup=int(argument("--warmup", "3")), repetitions=int(argument("--repetitions", "10")),
        prompt_group=request.get("prompt_group", "all"),
        max_new_tokens=int(argument("--max-new-tokens")), max_draft_tokens=int(argument("--max-draft-tokens")),
        low_memory="--low-memory" in command, allow_output_differences=args.allow_output_differences)
    set_benchmark_counts(stored)
    if any(request.get(key, getattr(stored, key)) != getattr(stored, key) for key in ("warmup", "repetitions")):
        raise ValueError("saved request repeat counts disagree with command")
    from qwen35_dflash.ascend310p.incremental_plan import require_verify_gdr
    stored.verify_gdr = require_verify_gdr(
        {"abi": plan.read_text().splitlines()[0]}, getattr(args, "verify_gdr", None))
    if request.get("verify_gdr", stored.verify_gdr) != stored.verify_gdr:
        raise ValueError("saved verification route differs from the hashed plan")
    eos = [int(t) for t in argument("--eos-token-ids").split(",")]
    if (stored.max_new_tokens != request["max_new_tokens"] or stored.max_draft_tokens != request["max_draft_tokens"]
            or eos != request["eos_token_ids"]):
        raise ValueError("saved request limits/EOS disagree with command")
    tokenizer = None
    if args.model_dir is not None:
        from qwen35_dflash.ascend310p.workflow import load_tokenizer
        tokenizer, _ = load_tokenizer(model_dir=args.model_dir)
    summary = collect_results(stored, prompts, raw, index, plan_hash, batch_hash, eos, None, tokenizer)
    summary.update(request=str(request_path), process_wall_seconds=None,
        ordinary_baseline=request.get("ordinary_baseline"),
        datasets=request.get("datasets", []), max_new_tokens=stored.max_new_tokens,
        reanalysis={"source_index_sha256": sha256_file(index), "source_request_sha256": sha256_file(request_path),
                    "script_sha256": sha256_file(Path(__file__)), "device_execution": "NOT_RUN",
                    "source_reports_modified": False, "limits_source": "saved request, not CLI defaults"})
    root = require_run_output(Path(tempfile.mkdtemp(prefix="prompt-summary-", dir=run_dir)))
    print(f"Output: {root}", flush=True)
    return write_summary(root, summary)


def run(args):
    if getattr(args, "summarize_existing", None):
        if offline_datasets.selected(args) or getattr(args, "num_questions", None) is not None:
            raise ValueError("--summarize-existing uses saved dataset selection; do not supply dataset inputs")
        return summarize_existing(args)
    from qwen35_dflash.ascend310p.cpp_runtime import resolve_cpp_runner, validate_cpp_runner_options
    from qwen35_dflash.ascend310p.generation import tokenize_prompt
    from qwen35_dflash.ascend310p.incremental_plan import write_incremental_plan
    from qwen35_dflash.ascend310p.utils import atomic_write_json, require_run_output, sha256_file
    from qwen35_dflash.ascend310p.workflow import load_tokenizer
    from tools import ordinary_baseline

    if args.runner is None or args.model_dir is None:
        raise ValueError("running inference requires --runner and --model-dir")
    if os.environ.get("ASCEND310P_SIMULATION_ONLY") == "1" or os.environ.get("PROFILING_MODE") not in (None, "", "false"):
        raise ValueError("batch target test requires real Ascend310P outside msprof")
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir() or run_dir.is_relative_to(REPO):
        raise ValueError("run-dir must be an existing directory outside the repository")
    os.environ["AI_RUN_DIR"] = str(run_dir)
    if args.device_id < 0 or not 1 <= args.max_draft_tokens <= 15 or args.max_new_tokens <= 0:
        raise ValueError("invalid device/token limits")
    set_benchmark_counts(args)
    prompts, datasets = load_inputs(args)
    manifest = (args.deployment_manifest or run_dir / "artifacts/deployment-manifest.json").resolve()
    config = json.loads((args.runner_config or run_dir / "runner.json").read_text())
    identity = validate_cpp_runner_options(config, args.device_id)
    executable = resolve_cpp_runner(args.runner)
    help_result = subprocess.run([str(executable), "--help"], capture_output=True, text=True)
    if help_result.returncode or "--prompt-batch" not in help_result.stdout:
        raise RuntimeError("rebuild the C++ runner: --prompt-batch support is required")
    if len(prompts) > 64 and "unlimited prompt count" not in help_result.stdout:
        raise RuntimeError("rebuild the C++ runner for offline datasets with more than 64 prompts (no OM recompilation needed)")
    if (args.warmup, args.repetitions) != (3, 10) and "positive; default 10" not in help_result.stdout:
        raise RuntimeError("rebuild the C++ runner for configurable --warmup/--repetitions (no OM recompilation needed)")
    baseline = getattr(args, "_ordinary_baseline", None)
    if baseline and "dflash-only batch" not in help_result.stdout:
        raise RuntimeError("rebuild the C++ runner for ordinary baseline reuse (no OM recompilation needed)")
    root = require_run_output(Path(tempfile.mkdtemp(prefix="prompt-suite-", dir=run_dir)))
    print(f"Output: {root}", flush=True)
    plan, deployment, contract = write_incremental_plan(
        manifest, root / "chunk-plan.txt", verify_gdr=getattr(args, "verify_gdr", None))
    from qwen35_dflash.ascend310p.incremental_plan import verify_gdr_route
    args.verify_gdr = verify_gdr_route(contract)
    tokenizer, tokenizer_source = load_tokenizer(model_dir=args.model_dir)
    eos = args.eos_token_id or [248044]
    if any(token < 0 or token >= contract["vocab_size"] for token in eos):
        raise ValueError("EOS token outside model vocabulary")
    for item in prompts:
        tokens = tokenize_prompt(tokenizer, item["prompt"], chat=args.chat)
        if (not tokens or len(tokens) + args.max_new_tokens > contract["capacity"]
                or any(token < 0 or token >= contract["vocab_size"] for token in tokens)):
            raise ValueError(f"{offline_datasets.prompt_label(item)}: input {len(tokens)} + output budget {args.max_new_tokens} "
                             f"exceeds vocabulary/context limit (capacity {contract['capacity']}); no device jobs started")
        if "prompt_token_ids" in item and item["prompt_token_ids"] != tokens:
            raise ValueError(f"{offline_datasets.prompt_label(item)}: tokenization changed since matrix preflight")
        item["prompt_token_ids"] = tokens
        item["input_tokens"] = len(tokens)
    batch = root / "prompts.txt"
    batch.write_text("QWEN35_PROMPT_BATCH_V1\n" + "".join(
        f"{p['id']} \"{','.join(map(str, p['prompt_token_ids']))}\"\n" for p in prompts))
    index = root / "runner-batch.json"
    execution_index = root / "runner-dflash.json" if baseline else index
    plan_hash, batch_hash = sha256_file(plan), sha256_file(batch)
    command = [str(executable), "--model-kind", "chunk", "--mode", "dflash" if baseline else "paired", "--model", str(plan),
        "--model-sha256", plan_hash, "--prompt-batch", str(batch), "--prompt-batch-sha256", batch_hash,
        "--output", str(execution_index), "--eos-token-ids", ",".join(map(str, eos)),
        "--pad-token-id", str(identity["pad_token_id"]), "--device-id", str(args.device_id),
        "--max-new-tokens", str(args.max_new_tokens), "--max-draft-tokens", str(args.max_draft_tokens),
        "--warmup", str(args.warmup), "--repetitions", str(args.repetitions), "--trace-rounds"]
    if args.low_memory:
        command.append("--low-memory")
    request = {"schema_version": 1, "prompts": prompts, "datasets": datasets, "chat": args.chat, "eos_token_ids": eos,
        "prompt_group": getattr(args, "prompt_group", "all"),
        "warmup": args.warmup, "repetitions": args.repetitions,
        "verify_gdr": args.verify_gdr, "incremental_abi": contract["abi"],
        "allow_output_differences": getattr(args, "allow_output_differences", False),
        "max_new_tokens": args.max_new_tokens, "max_draft_tokens": args.max_draft_tokens,
        "runtime_identity": identity, "tokenizer_source": tokenizer_source,
        "ordinary_contract": ordinary_baseline.ordinary_contract(deployment, contract),
        "script_sha256": sha256_file(Path(__file__)),
        "runner": {"path": str(executable), "sha256": sha256_file(executable)},
        "deployment_manifest": {"path": str(manifest), "sha256": sha256_file(manifest)},
        "om_sha256": {g["name"]: g["om"]["sha256"] for g in deployment["graphs"]},
        "draft_atc_command": next(g["atc_command"] for g in deployment["graphs"] if g["name"] == "draft"),
        "command": command, "model_load_policy": "reuse models across all prompts"}
    if baseline:
        request["ordinary_baseline"] = ordinary_baseline.snapshot(baseline, request)
        print(f"[prompt-batch] reuse ordinary baseline: {baseline}; running DFlash only", flush=True)
    atomic_write_json(root / "request.json", request)
    start = time.monotonic()
    exit_code = run_runner(command, root / "runner.log")
    process_wall_seconds = time.monotonic() - start
    if baseline and execution_index.exists():
        ordinary_baseline.merge(index, execution_index, request)
        atomic_write_json(root / "request.json", request)
    raw = json.loads(index.read_text()) if index.exists() else {}
    summary = collect_results(args, prompts, raw, index, plan_hash, batch_hash, eos, exit_code, tokenizer)
    summary.update(request=str(root / "request.json"), process_wall_seconds=process_wall_seconds,
                   ordinary_baseline=request.get("ordinary_baseline"),
                   datasets=datasets, max_new_tokens=args.max_new_tokens)
    return write_summary(root, summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=os.environ.get("AI_RUN_DIR"), required=not os.environ.get("AI_RUN_DIR"))
    parser.add_argument("--runner", type=Path, default=os.environ.get("CPP_RUNNER"))
    parser.add_argument("--deployment-manifest", type=Path)
    parser.add_argument("--verify-gdr", choices=("chunk", "mtp"),
                        help="require the selected compiled route; omitted: read manifest")
    parser.add_argument("--runner-config", type=Path)
    parser.add_argument("--model-dir", type=Path, help="required for inference; optional for decoding saved outputs")
    offline_datasets.add_arguments(parser)
    parser.add_argument("--prompts", type=Path, help="custom JSON; default: 8 short + 12 long (~1K input) prompts")
    parser.add_argument("--prompt-group", choices=("all", "short", "long"), default="all",
                        help="select all (default), short, or long inputs; custom JSON can specify group")
    parser.add_argument("--prompt-id", action="append", help="run only selected ID(s); repeatable")
    parser.add_argument("--chat", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eos-token-id", type=int, action="append", help="repeatable; default 248044")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-draft-tokens", type=int, default=15)
    parser.add_argument("--low-memory", action="store_true")
    parser.add_argument("--warmup", type=int, default=1, help="warmup generations per mode and prompt (default 1)")
    parser.add_argument("--repetitions", type=int, default=3, help="measured generations per mode and prompt (default 3)")
    parser.add_argument("--allow-output-differences", action="store_true",
                        help="allow cross-mode differences; repeated-run drift is reported without failing")
    parser.add_argument("--summarize-existing", type=Path, metavar="BATCH_JSON",
                        help="read a saved runner batch and request; no inference or runner/model files required")
    try:
        return run(parser.parse_args())
    except (OSError, ValueError, KeyError, RuntimeError, TypeError, IndexError) as error:
        parser.exit(2, f"prompt-suite: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
