"""Measure only real DFlash generation and compare to the saved ordinary and native W8 outputs."""

import csv
import math
from pathlib import Path
import statistics
import subprocess

from winners import c, require, REPO
from history import suite
from package import command


def runner(d, history, cann, env, *, rebuild=False):
    def supports(p):
        r = subprocess.run([str(p), "--help"], cwd=d, env=env, capture_output=True, text=True)
        return r.returncode == 0 and "dflash-only batch" in r.stdout

    archived = None if rebuild else c.verify(history["runner"])
    if archived is not None and supports(archived):
        p = archived
    else:
        build = d / "build"
        command(
            [
                "cmake",
                "-S",
                REPO / "framework/runtime/cpp",
                "-B",
                build,
                "-DCMAKE_BUILD_TYPE=Release",
                "-DQWEN35_DFLASH_BUILD_TESTS=OFF",
                "-DASCENDCL_ROOT=" + str(cann),
            ],
            d / "configure.log",
            env,
            d,
        )
        command(
            ["cmake", "--build", build, "--target", "qwen35_dflash_acl_runner", "--parallel", "2"],
            d / "build.log",
            env,
            d,
        )
        p = build / "qwen35_dflash_acl_runner"
        require(supports(p), "real runner lacks DFlash-only batch support")
    deps = subprocess.run(["ldd", str(p)], capture_output=True, text=True, env=env, check=True).stdout
    require(
        "libascendcl.so" in deps and "not found" not in deps,
        "generation runner lacks real AscendCL dependencies",
    )
    (d / "ldd.txt").write_text(deps)
    return dict(file=c.identity(p), same_as_archive=c.sha(p) == history["runner"]["sha256"])


def launch(d, history, plan, exe, env, *, canary=False, reference_command=None):
    batch = c.verify(history["batch"])
    selected = history["prompts"]
    if canary:
        selected = selected[:1]
        require(selected[0]["id"] == "zh_explain", "unexpected canary prompt")
        lines = batch.read_text().splitlines()
        batch = d / "canary-prompt.txt"
        batch.write_text("\n".join(lines[:2]) + "\n")
    batch_hash = c.sha(batch)
    index = d / "runner-batch.json"
    cmd = list(
        map(
            str,
            (
                c.verify(exe["file"]),
                "--model-kind",
                "chunk",
                "--mode",
                "dflash",
                "--model",
                c.verify(plan),
                "--model-sha256",
                plan["sha256"],
                "--prompt-batch",
                batch,
                "--prompt-batch-sha256",
                batch_hash,
                "--output",
                index,
                "--eos-token-ids",
                "248044",
                "--pad-token-id",
                "0",
                "--device-id",
                "0",
                "--max-new-tokens",
                "128",
                "--max-draft-tokens",
                "15",
                "--warmup",
                "0" if canary else "1",
                "--repetitions",
                "1" if canary else "3",
                "--trace-rounds",
                "--low-memory",
            ),
        )
    )
    if reference_command is not None:
        expected = list(reference_command)
        expected[0] = cmd[0]
        for option in ("--prompt-batch", "--output"):
            expected[expected.index(option) + 1] = cmd[cmd.index(option) + 1]
        require(cmd == expected, "generation command differs from the archived protocol")
    c.write(d / "command.json", cmd)
    print(
        (
            "One candidate-only zh_explain generation, 0 warmup + 1 run; device check"
            if canary
            else "20 archived prompts, 8 short + 12 long; Target Prefill → optimized Draft C16/C64 → Chunk Verify/commit"
        ),
        flush=True,
    )
    with (d / "runner.log").open("x") as log, subprocess.Popen(
        cmd, cwd=d, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace"
    ) as proc:
        for line in proc.stdout:
            log.write(line)
            if not line.startswith("{"):
                print(line, end="", flush=True)
        code = proc.wait()
    if code:
        raise RuntimeError(f"generation runner exited {code}; see {d/'runner.log'}")
    raw = c.read(index)
    require(
        raw["fake_acl"] is False
        and "fake-acl" not in raw["runner_version"]
        and raw["status"] in ("PASS", "PASS_WITH_OBSERVATIONS")
        and raw["models_reused_across_prompts"]
        and raw["low_memory"]
        and raw["order"] == "DFlash only; ordinary baseline external"
        and raw["model_sha256"] == plan["sha256"]
        and raw["prompt_batch_sha256"] == batch_hash
        and [v["id"] for v in raw["cases"]] == [p["id"] for p in selected],
        "incomplete genuine candidate batch",
    )
    files = {}
    for row in raw["cases"]:
        f = Path(str(index) + ".cases") / (row["id"] + ".json")
        require(
            Path(row["report"]).resolve() == f.resolve() and c.read(f)["status"] == row["status"],
            "unexpected case report",
        )
        files[row["id"]] = c.identity(f)
    if canary:
        r = c.read(c.verify(files[selected[0]["id"]]))
        (m,) = r["dflash"]["measurements"]
        counts = {
            k: len(m["stage_ms"].get(k, []))
            for k in ("target_prefill", "draft", "target_verify", "target_decode")
        }
        require(
            "ordinary" not in r
            and r["cpu_fallback"] is False
            and r["prompt_token_ids"] == selected[0]["tokens"]
            and r["protocol"]["warmup"] == 0
            and r["protocol"]["repetitions"] == 1
            and counts["target_prefill"] == 1
            and counts["target_verify"] == counts["draft"] > 0
            and counts["target_decode"] == 0,
            "incomplete candidate device check",
        )
        print(
            "Target/Draft/Verify device check PASS",
            counts,
            "tokens",
            len(m["generated_token_ids"]),
            flush=True,
        )
    return dict(index=c.identity(index), reports=files, command=c.identity(d / "command.json"))


def compare_generation(old, new):
    changes = []
    rounds = 0
    for i, (a, b) in enumerate(zip(old["measurements"], new["measurements"], strict=True)):
        l, r = a["generated_token_ids"], b["generated_token_ids"]
        unequal = [j for j in range(max(len(l), len(r))) if l[j : j + 1] != r[j : j + 1]]
        if unequal or a["stop_reason"] != b["stop_reason"]:
            changes.append(
                dict(
                    repetition=i,
                    token_count=len(unequal),
                    first_index=unequal[0] if unequal else None,
                    old_stop=a["stop_reason"],
                    new_stop=b["stop_reason"],
                )
            )
        if a.get("rounds") != b.get("rounds"):
            rounds += 1
    return dict(matching_repetitions=3 - len(changes), round_traces_matched=3 - rounds, differences=changes)


def compare_target_rounds(old, new):
    """Verify the archived Target predictions only where the observed inputs agree."""
    anchors = same = matching = 0
    for a, b in zip(old["measurements"], new["measurements"], strict=True):
        anchors += a["generated_token_ids"][0] == b["generated_token_ids"][0]

        def calls(m):
            prefix = []
            for r in m["rounds"]:
                proposals = tuple(r["proposed_token_ids"])
                if proposals:
                    yield (tuple(prefix), proposals), tuple(r["target_token_ids"])
                prefix.extend(r["emitted_token_ids"])

        saved = dict(calls(a))
        for key, ids in calls(b):
            if key in saved:
                same += 1
                matching += ids == saved[key]
    return dict(
        prefill_anchors_matched=anchors, same_prefix_and_proposals=same, target_predictions_matched=matching
    )


def validate_dflash(z, p, plan):
    from qwen35_dflash.ascend310p.cpp_runtime import _validate_mode_report

    require(
        "ordinary" not in z
        and z["cpu_fallback"] is False
        and z["device_id"] == 0
        and z["status"] in ("PASS", "PASS_WITH_OBSERVATIONS")
        and z["prompt_token_ids"] == p["tokens"]
        and z["eos_token_ids"] == [248044]
        and z["model"]["sha256"] == plan["sha256"]
        and z["limits"] == {"max_new_tokens": 128, "max_draft_tokens": 15}
        and z["abi"]["draft_prefill_policy"] == "static_draft16_64_oms"
        and z["abi"]["sequence_length"] == 2048
        and z["abi"]["om_count"] == 4
        and z["protocol"]["low_memory"]
        and z["protocol"]["round_trace_enabled"]
        and z["protocol"]["dflash_speculation_policy"] == "always_on",
        "wrong DFlash-only execution: " + p["id"],
    )
    mode = z["dflash"]
    _validate_mode_report(
        "candidate",
        mode,
        generation_mode="dflash-strict-greedy",
        warmup=1,
        repetitions=3,
        max_new_tokens=128,
        eos_token_ids=[248044],
    )
    for run in mode["measurements"]:
        stages = run["stage_ms"]
        rounds = [r for r in run["rounds"] if r["proposed_token_ids"]]
        chunks = math.ceil(len(p["tokens"]) / 64)
        require(
            len(stages["target_prefill"]) == chunks
            and len(stages.get("target_decode", [])) == 0
            and len(stages["draft"]) == chunks - 1 + len(rounds)
            and len(stages["target_verify"]) == len(rounds)
            and run["counters"]["drafted_tokens"] == sum(len(r["proposed_token_ids"]) for r in rounds)
            and run["counters"]["accepted_draft_tokens"]
            == sum(len(r["accepted_draft_token_ids"]) for r in rounds)
            and all(0 <= v < 248320 for v in run["generated_token_ids"])
            and all(
                all(math.isfinite(x) and x > 0 for x in stages[k])
                for k in ("target_prefill", "draft", "target_verify")
            )
            and math.isclose(
                run["latency_ms"]["model_total"],
                run["latency_ms"]["prefill"] + run["latency_ms"]["decode"],
                rel_tol=1e-7,
            ),
            "incomplete executed Target/Draft trace: " + p["id"],
        )


def metric_row(p, ordinary, dflash):
    row = suite.summarize_prompt({"prompt_token_ids": p["tokens"], "ordinary": ordinary, "dflash": dflash})
    first, last = ordinary["measurements"][0], dflash["measurements"][0]
    parity = all(first[k] == last[k] for k in ("generated_token_ids", "stop_reason"))
    row.update(
        id=p["id"],
        group=p["group"],
        status="PASS" if parity else "PASS_WITH_DIFFERENCES",
        max_new_tokens=128,
        input_tokens=len(p["tokens"]),
        ordinary_measured_tokens=sum(len(x["generated_token_ids"]) for x in ordinary["measurements"]),
        dflash_measured_tokens=sum(len(x["generated_token_ids"]) for x in dflash["measurements"]),
    )
    return row


def aggregate(rows):
    q = suite.aggregate_metrics(rows)
    phase = q["phase_timings"]
    s = q["stage_ms_per_call"]
    return dict(
        prompts=len(rows),
        generations=3 * len(rows),
        accepted=q["accepted_draft_tokens"],
        proposed=q["drafted_tokens"],
        acceptance_pct=(
            100 * q["weighted_acceptance_rate"] if q["weighted_acceptance_rate"] is not None else None
        ),
        tokens_per_round=q["tokens_per_speculative_round"],
        generated_tokens=q["dflash_measured_tokens"],
        gen_tok_s=q["dflash_tokens_per_second"],
        ordinary_tok_s=q["ordinary_tokens_per_second"],
        decode_speedup_over_archived_ordinary=q["decode_time_speedup"],
        prefill_ms=phase["dflash_prefill"]["mean_ms"],
        decode_ms=phase["dflash_decode"]["mean_ms"],
        ordinary_prefill_ms=phase["ordinary_prefill"]["mean_ms"],
        ordinary_decode_ms=phase["ordinary_decode"]["mean_ms"],
        target_prefill_ms_call=s["dflash_prefill"]["mean_ms"],
        draft_ms_call=s["draft"]["mean_ms"],
        verify_ms_call=s["verify"]["mean_ms"],
        calls={k: s[k]["calls"] for k in ("dflash_prefill", "draft", "verify")},
    )


# Values printed to two decimal places in docs/RESULTS.md for the W8 offline NZ, 1+3 run.
DOCS = {
    "all": dict(
        acceptance_pct=21.03,
        tokens_per_round=3.93,
        gen_tok_s=29.16,
        decode_speedup_over_archived_ordinary=1.39,
        prefill_ms=1195.57,
        decode_ms=3193.72,
        ordinary_decode_ms=4436.34,
        target_prefill_ms_call=75.63,
        draft_ms_call=47.89,
        verify_ms_call=51.17,
    ),
    "short": dict(
        acceptance_pct=21.20,
        tokens_per_round=3.98,
        gen_tok_s=39.73,
        decode_speedup_over_archived_ordinary=1.41,
        prefill_ms=74.89,
        decode_ms=3146.91,
        ordinary_decode_ms=4431.58,
        target_prefill_ms_call=74.82,
        draft_ms_call=47.54,
        verify_ms_call=51.13,
    ),
    "long": dict(
        acceptance_pct=20.91,
        tokens_per_round=3.90,
        gen_tok_s=24.77,
        decode_speedup_over_archived_ordinary=1.38,
        prefill_ms=1942.69,
        decode_ms=3224.93,
        ordinary_decode_ms=4439.52,
        target_prefill_ms_call=75.67,
        draft_ms_call=48.04,
        verify_ms_call=51.20,
    ),
}


def analyze(d, history, plan, capture):
    from qwen35_dflash.ascend310p.cpp_runtime import _validate_mode_report

    old_rows = []
    rows = []
    consistency = {}
    gear_times = {16: [], 64: []}
    for p in history["prompts"]:
        name = p["id"]
        historical = c.read(c.verify(history["cases"][name]))
        z = c.read(c.verify(capture["reports"][name]))
        validate_dflash(z, p, plan)
        ordinary = historical["ordinary"]
        old = historical["dflash"]
        new = z["dflash"]
        for tag, mode, policy in (
            ("ordinary", ordinary, "ordinary-greedy"),
            ("archived W8", old, "dflash-strict-greedy"),
        ):
            _validate_mode_report(
                tag,
                mode,
                generation_mode=policy,
                warmup=1,
                repetitions=3,
                max_new_tokens=128,
                eos_token_ids=[248044],
            )
        old_rows.append(metric_row(p, ordinary, old))
        rows.append(metric_row(p, ordinary, new))
        consistency[name] = dict(
            ordinary=compare_generation(ordinary, new),
            previous_w8=compare_generation(old, new),
            target=compare_target_rounds(old, new),
        )
        count = len(p["tokens"])
        prefix = (count - 1) // 64
        tail = count - 64 * prefix
        for m in new["measurements"]:
            times = m["stage_ms"]["draft"]
            gear_times[64].extend(times[:prefix])
            if len(times) > prefix:
                gear_times[64 if tail > 16 else 16].append(times[prefix])
                gear_times[16].extend(times[prefix + 1 :])
    groups = {}
    for group in ("all", "short", "long"):
        subset = lambda rr: [r for r in rr if group == "all" or r["group"] == group]
        cur = aggregate(subset(rows))
        previous = aggregate(subset(old_rows))
        groups[group] = dict(
            candidate=cur,
            archived_W8=previous,
            docs=DOCS[group],
            docs_ordinary_over_candidate_decode_approx=DOCS[group]["ordinary_decode_ms"] / cur["decode_ms"],
            docs_W8_over_candidate_decode_approx=DOCS[group]["decode_ms"] / cur["decode_ms"],
            archived_W8_over_candidate_decode=previous["decode_ms"] / cur["decode_ms"],
        )
    per = []
    for old, r in zip(old_rows, rows):
        per.append(
            dict(
                prompt=r["id"],
                group=r["group"],
                input_tokens=r["input_tokens"],
                generated_3x=r["dflash_measured_tokens"],
                accepted=r["accepted_draft_tokens"],
                proposed=r["drafted_tokens"],
                gen_tok_s=r["dflash_tokens_per_second"],
                archive_ordinary_decode_ms=r["phase_timings"]["ordinary_decode"]["mean_ms"],
                archive_W8_decode_ms=old["phase_timings"]["dflash_decode"]["mean_ms"],
                candidate_decode_ms=r["phase_timings"]["dflash_decode"]["mean_ms"],
                decode_speedup=r["decode_time_speedup"],
                native_W8_matching_reps=consistency[r["id"]]["previous_w8"]["matching_repetitions"],
                ordinary_matching_reps=consistency[r["id"]]["ordinary"]["matching_repetitions"],
            )
        )
    with (d / "cases.csv").open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per[0]))
        w.writeheader()
        w.writerows(per)
    gears = {
        k: dict(
            calls=len(v),
            mean_ms=statistics.mean(v),
            median_ms=statistics.median(v),
            p95_ms=sorted(v)[math.ceil(0.95 * len(v)) - 1],
        )
        for k, v in gear_times.items()
        if v
    }
    scores = {
        k: sum(v[k]["matching_repetitions"] for v in consistency.values())
        for k in ("ordinary", "previous_w8")
    }
    target_match = {
        k: sum(v["target"][k] for v in consistency.values())
        for k in ("prefill_anchors_matched", "same_prefix_and_proposals", "target_predictions_matched")
    }
    round_matches = sum(v["previous_w8"]["round_traces_matched"] for v in consistency.values())
    lines = [
        "# W8A16 完整生成实测",
        "",
        "8 条短题和 12 条长题，thinking 开启，输出上限 128，每题预热 1 次、正式 3 次。ordinary 和离线 NZ 的原始对照来自封存的 w55diia2；docs 表为另一历史批次 mjhazgip 的舍入值。",
        "",
        "| 范围 / 来源 | 接受率 % | token/轮 | Generation tok/s | Decode 加速比（历史 ordinary） | Prefill ms/次 | Decode ms/次 | Target Prefill / Draft / Verify ms/call |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group, g in groups.items():
        for name in ("docs", "archived_W8", "candidate"):
            v = g[name]
            lines.append(
                f"| {group} / {name} | {v['acceptance_pct']:.2f} | {v['tokens_per_round']:.2f} | {v['gen_tok_s']:.2f} | {v['decode_speedup_over_archived_ordinary']:.3f}× | {v['prefill_ms']:.2f} | {v['decode_ms']:.2f} | {v['target_prefill_ms_call']:.3f} / {v['draft_ms_call']:.3f} / {v['verify_ms_call']:.3f} |"
            )
    lines += [
        "",
        f"60 次正式生成中，候选与历史 W8 token 序列和停止原因完全一致 {scores['previous_w8']}/60，与历史 ordinary 一致 {scores['ordinary']}/60。生成吞吐含 Prefill，Decode 加速比使用循环时长总和。任务质量未评估。",
        f"与历史 W8 相比，首 token 一致 {target_match['prefill_anchors_matched']}/60，整条接受轨迹一致 {round_matches}/60。在实际前缀与提议均相同的 Verify 调用中，Target 输出一致 {target_match['target_predictions_matched']}/{target_match['same_prefix_and_proposals']}。",
    ]
    (d / "comparison.md").write_text("\n".join(lines) + "\n")
    for l in lines:
        print(l)
    print("In-generation Draft static gears, calls and ms/call", gears, flush=True)
    return dict(
        groups=groups,
        gears=gears,
        cases=c.identity(d / "cases.csv"),
        comparison=c.identity(d / "comparison.md"),
        token_parity=consistency,
        target_agreement=target_match,
        matching_round_traces=round_matches,
        matching_generations=scores,
        ordinary_executions=0,
        historical_index=history["index"],
        previous_suite=history["historical"],
        candidate_capture=capture,
        quality="NOT_EVALUATED",
    )
