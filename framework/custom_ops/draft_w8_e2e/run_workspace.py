"""Measure full generation with a huge-first shared workspace against the saved five-OM run."""

import argparse
import math
import os
from pathlib import Path
import re
import traceback

from winners import c, require, HERE, REPO
from history import BATCH_SHA, suite
from measure import compare_generation, launch, runner
from context_measure import stats, summarize, timings
from diagnose_calls import receiver_runtime
from experiment import Stages
from run_q import run_path


def archive_inputs(folder, capture, plan):
    index = c.read(c.verify(capture["index"]))
    prompts = suite.load_prompts(None)
    ordered = [p["id"] for p in prompts]
    require(
        index["fake_acl"] is False
        and index["models_reused_across_prompts"]
        and index["low_memory"]
        and index["status"] in ("PASS", "PASS_WITH_OBSERVATIONS")
        and index["order"] == "DFlash only; ordinary baseline external"
        and index["model_sha256"] == plan["sha256"]
        and index["prompt_batch_sha256"] == BATCH_SHA
        and [r["id"] for r in index["cases"]] == ordered
        and set(capture["reports"]) == set(ordered)
        and len(prompts) == 20
        and sum(p["group"] == "short" for p in prompts) == 8,
        "baseline is not the documented 20-prompt generation",
    )
    for p in prompts:
        report = c.read(c.verify(capture["reports"][p["id"]]))
        p["tokens"] = report["prompt_token_ids"]
        timings(report, p, plan, context=True)
    # Reconstruct the identical runner input from the baseline case outputs, even if
    # its earlier tokenizer or ordinary benchmark directory has since been moved.
    batch = folder / "prompts.txt"
    batch.write_text(
        "QWEN35_PROMPT_BATCH_V1\n"
        + "".join(f'{p["id"]} "{",".join(map(str, p["tokens"]))}"\n' for p in prompts)
    )
    identity = c.identity(batch)
    require(identity["sha256"] == index["prompt_batch_sha256"], "prompt token batch differs from the control")
    return dict(batch=identity, prompts=prompts)


def memory(capture, candidate, *, model_count=5):
    file = Path(c.verify(capture["index"])).parent / "runner.log"
    log = file.read_text(errors="replace")
    match = re.findall(
        r"workspace policy=shared_serial shared_bytes=(\d+) separate_sum_bytes=(\d+) saved_work_bytes=(\d+)([^\n]*)",
        log,
    )
    require(
        len(match) == 1 and int(match[0][0]) > 0 and f"loaded_models={model_count} " in log,
        f"expected {model_count} resident OMs sharing one serial workspace",
    )
    shared, total, saved, tail = match[0]
    request = re.search(r"\bworkspace_page_request=(\w+)", tail)
    expected = "huge_first" if candidate else "normal_only"
    require(
        (request and request[1] == expected) or (not request and not candidate),
        "workspace page request does not match the experiment",
    )
    require(int(total) - int(saved) == int(shared), "inconsistent workspace size")
    return dict(
        shared_bytes=int(shared), separate_sum_bytes=int(total), page_request=expected, log=c.identity(file)
    )


def request_summary(rows):
    result = summarize(rows)
    for name, parts in (
        ("draft_per_request", ("draft", "context")),
        ("target_per_request", ("target_prefill", "verify")),
    ):
        result[name] = stats([sum(sum(r[k]) for k in parts) for r in rows], allow_zero=True)
    ordered = sorted(rows, key=lambda r: r["total"])
    at_p95 = ordered[math.ceil(0.95 * len(ordered)) - 1]
    result["p95_request"] = dict(prompt=at_p95["prompt"], repetition=at_p95["repetition"], ms=at_p95["total"])
    return result


def compare(folder, prompts, previous, candidate, plan, archived_groups, *, split_plan=None, window_plan=None):
    label = "conv_window" if window_plan else "split_head" if split_plan else "workspace"
    candidate_plan = window_plan or split_plan or plan
    samples = {s: {"short": [], "long": []} for s in ("previous", label)}
    agreement = {}
    for p in prompts:
        name = p["id"]
        a, b = [c.read(c.verify(run["reports"][name])) for run in (previous, candidate)]
        abi, protocol = dict(a["abi"]), dict(a["protocol"])
        if split_plan is not None:
            abi.update(
                split_target_prefill_head=True,
                graph_count=abi["graph_count"] + 1,
                om_count=abi["om_count"] + 1,
            )
            protocol["max_resident_models"] += 1
        require(
            abi == b["abi"] and protocol == b["protocol"],
            "generation ABI/protocol changed: " + name,
        )
        agreement[name] = compare_generation(a["dflash"], b["dflash"])
        for side, report, observed_plan in (("previous", a, plan), (label, b, candidate_plan)):
            for i, row in enumerate(timings(report, p, observed_plan, context=True)):
                samples[side][p["group"]].append(dict(row, prompt=name, repetition=i))
    groups = {}
    for group in ("all", "short", "long"):
        groups[group] = {
            side: request_summary(data[group] if group != "all" else data["short"] + data["long"])
            for side, data in samples.items()
        }
        a, b = groups[group].values()
        # Reproduce the actual previous comparison and its token/latency denominators.
        control_key = "split_head" if window_plan else "workspace" if split_plan else "context"
        control = archived_groups[group][control_key]
        required = ("total", "draft", "context", "verify", "generation_tok_s", "tokens")
        if split_plan or window_plan:
            required += ("target_prefill", "prefill", "decode", "target_per_request")
        require(
            all(a[k] == control[k] for k in required),
            "recomputed control differs from the archived result: " + group,
        )
        old_tps, new_tps = a["generation_tok_s"], b["generation_tok_s"]
        print(
            f"{group} Generation tok/s {old_tps:.3f} -> {new_tps:.3f} ({100 * (new_tps / old_tps - 1):+.2f}%), "
            f"mean request {a['total']['mean_ms']:.2f} -> {b['total']['mean_ms']:.2f} ms",
            flush=True,
        )
        for phase in ("prefill", "decode", "total"):
            x, y = a[phase], b[phase]
            print(
                f"  {group} {phase} ms/request mean/P50/P95 {x['mean_ms']:.2f}/{x['median_ms']:.2f}/{x['p95_ms']:.2f}"
                f" -> {y['mean_ms']:.2f}/{y['median_ms']:.2f}/{y['p95_ms']:.2f} "
                f"(P95 delta {y['p95_ms'] - x['p95_ms']:+.2f} ms)",
                flush=True,
            )
    a, b = groups["all"].values()
    prefill_parts = (
        (
            ("Target non-final chunks", "prefill_middle"),
            ("Target final chunk incl. Head", "prefill_last"),
            ("Target body alone", "prefill_body"),
            ("Separate Target Head", "prefill_head"),
        )
        if split_plan or window_plan
        else ()
    )
    for label, k in (
        ("Target Prefill", "target_prefill"),
        *prefill_parts,
        ("Draft context C64", "context"),
        ("Draft propose C16", "draft_c16"),
        ("Draft propose C64", "draft_c64"),
        ("Draft proposals", "draft"),
        ("Target Verify/commit", "verify"),
        ("All Draft per request", "draft_per_request"),
        ("All Target per request", "target_per_request"),
    ):
        x, y = a[k], b[k]

        def show(v):
            return "/".join(
                f"{v[t]:.3f}" if v[t] is not None else "n/a" for t in ("mean_ms", "median_ms", "p95_ms")
            )

        print(f"  {label} n={x['count']}/{y['count']} mean/P50/P95 ms {show(x)} -> {show(y)}", flush=True)
    if split_plan:
        print(
            f"Vocabulary Head calls {a['target_prefill']['count']} -> {b['prefill_head']['count']}, "
            f"skipped {b['prefill_middle']['count']} non-final blocks",
            flush=True,
        )
    n = sum(len(v) for v in samples["previous"].values())
    matches = sum(z["matching_repetitions"] for z in agreement.values())
    traces = sum(z["round_traces_matched"] for z in agreement.values())
    same_work = all(a[k] == b[k] for k in ("tokens", "accepted", "proposed")) and all(
        a[k]["count"] == b[k]["count"]
        for k in ("context", "draft_c16", "draft_c64", "target_prefill", "verify")
    )
    print(
        f"Tokens and stops {matches}/{n}; full proposal/Verify traces {traces}/{n}; "
        f"generated tokens {a['tokens']} -> {b['tokens']}; accepted/proposed {a['accepted']}/{a['proposed']} -> "
        f"{b['accepted']}/{b['proposed']}; same workload {same_work}",
        flush=True,
    )
    c.write(folder / "per-prompt-agreement.json", agreement)
    return dict(
        groups=groups,
        measurements=n,
        token_matches=matches,
        round_matches=traces,
        same_workload=same_work,
        agreement=c.identity(folder / "per-prompt-agreement.json"),
    )


def run(out, args, state):
    base = run_path(out.parent, args.from_run, "context-prefill-")
    baseline = c.identity(base / "result.json")
    previous = c.read(c.verify(baseline))
    require(
        previous["status"] == "MEASURED" and previous["token_matches"] == previous["round_matches"] == 60,
        "select the verified five-OM context-prefill baseline",
    )
    prior = Stages(out, base)
    plan = prior.completed("plan")["result"]
    c.verify(plan)
    control = prior.completed("generation")["result"]
    history = archive_inputs(out, control, plan)
    history["runner"] = previous["runners"]["context"]
    old_mem = memory(control, False)
    original = c.verify(previous["baseline"]).parent
    cann = Path(os.environ["CANN_ROOT"]).resolve(strict=True)
    env, binding = receiver_runtime(original, c.read(original / "result.json"), cann)
    state.update(
        baseline=baseline,
        plan=plan,
        prompt_batch=history["batch"],
        archived_runner=history["runner"],
        policies=dict(shared_workspace="HUGE_FIRST", io="NORMAL_ONLY"),
        baseline_executions=0,
    )
    stages = Stages(out, run_path(out.parent, args.resume, "workspace-e2e-") if args.resume else None)

    def stage(label, key, action):
        state["phase"] = label
        return stages.run(label, dict(baseline=baseline, inputs=key), action)

    cpp = REPO / "framework/runtime/cpp"
    sources = [
        *cpp.glob("src/*.cpp"),
        *cpp.glob("include/qwen35_dflash/*.hpp"),
        cpp / "CMakeLists.txt",
        HERE / "measure.py",
    ]
    exe = stage(
        "runner",
        [binding, {str(f): c.sha(f) for f in sources}],
        lambda d: runner(d, history, cann, env, rebuild=True),
    )
    state["runner"] = exe["file"]
    print(
        "Same five OMs and prompt batch; candidate-only 20 prompts, 1 warmup + 3 measured per prompt",
        flush=True,
    )
    new = stage(
        "generation",
        [plan, history["batch"]["sha256"], exe["file"], binding, c.sha(HERE / "measure.py")],
        lambda d: launch(d, history, plan, exe, env, reference_command=c.read(c.verify(control["command"]))),
    )
    new_mem = memory(new, True)
    require(
        old_mem["shared_bytes"] == new_mem["shared_bytes"]
        and old_mem["separate_sum_bytes"] == new_mem["separate_sum_bytes"],
        "model workspace sizes differ from the archived run",
    )
    state.update(workspace=dict(previous=old_mem, candidate=new_mem), capture=new["index"])
    result = stage(
        "comparison",
        [control["index"], new["index"], c.sha(Path(__file__)), c.sha(HERE / "context_measure.py")],
        lambda d: compare(d, history["prompts"], control, new, plan, previous["groups"]),
    )
    state.update(result)
    return (
        "MEASURED"
        if result["same_workload"]
        and result["token_matches"] == result["round_matches"] == result["measurements"]
        else "MEASURED_WITH_OUTPUT_OR_TRACE_DIFFERENCES"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-run", required=True)
    parser.add_argument("--resume")
    args = parser.parse_args()
    out = Path(os.environ["DFLASH_OUTPUT"])
    state = {"status": "RUNNING"}
    try:
        state["status"] = run(out, args, state)
    except Exception as error:
        state.update(status="FAILED", error=str(error))
        traceback.print_exc()
    c.write(out / "result.json", state)
    print(state["status"], "REPORT", out / "result.json", flush=True)
    raise SystemExit(state["status"] != "MEASURED")
