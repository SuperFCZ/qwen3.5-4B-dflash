"""Verify and compare the candidate against the saved optimized generation, by phase."""

import math
import statistics

from winners import c, require
from measure import compare_generation
from context_graph import STATES
from package import command


def stats(values, *, allow_zero=False):
    if not values:
        return dict(count=0, mean_ms=None, median_ms=None, p95_ms=None)
    require(
        all(math.isfinite(v) and (v >= 0 if allow_zero else v > 0) for v in values), "invalid latency sample"
    )
    return dict(
        count=len(values),
        mean_ms=statistics.mean(values),
        median_ms=statistics.median(values),
        p95_ms=sorted(values)[math.ceil(0.95 * len(values)) - 1],
    )


def boundaries(folder, runner, env, plans, tokens):
    evidence = {}
    # 65 takes the first real proposal through C16 after one swap. 192 takes it
    # through C64 after two swaps. Each side builds its state from the same real prompt.
    for length in (65, 192):
        records = []
        for label, plan in plans.items():
            f = folder / f"{length}-{label}.json"
            cmd = [
                runner,
                "--model-kind",
                "chunk",
                "--mode",
                "dflash",
                "--model",
                c.verify(plan),
                "--model-sha256",
                plan["sha256"],
                "--prompt-token-ids",
                ",".join(map(str, tokens[:length])),
                "--pad-token-id",
                "0",
                "--eos-token-ids",
                "248044",
                "--max-new-tokens",
                "16",
                "--max-draft-tokens",
                "15",
                "--debug-draft-replay",
                "1",
                "--output",
                f,
            ]
            command(cmd, f.with_suffix(".log"), env, folder)
            row = c.read(f)
            require(
                row["fake_acl"] is False
                and row["status"] == "PASS_REPLAY_CHECKS"
                and row["model_sha256"] == plan["sha256"]
                and len(row["reference_token_ids"]) == 15
                and set(row["reference_output_sha256"]) == {"draft_top1", *STATES}
                and set(STATES) <= row["snapshot_sha256"].keys(),
                "incomplete on-device cache replay",
            )
            records.append(row)
        old, new = records
        for field in ("snapshot_sha256", "reference_output_sha256", "reference_token_ids"):
            require(old[field] == new[field], f"{length}-token boundary differs in {field}; see {folder}")
        print(
            f"{length} tokens: all ten full KV inputs, next 15 proposals and ten full KV outputs match",
            flush=True,
        )
        evidence[str(length)] = {key: c.identity(folder / f"{length}-{key}.json") for key in plans}
    return evidence


def timings(report, prompt, plan, *, context):
    mode = report["dflash"]
    split = report["abi"].get("split_target_prefill_head", False)
    require(
        report["status"] in ("PASS", "PASS_WITH_OBSERVATIONS")
        and report["cpu_fallback"] is False
        and "ordinary" not in report
        and report["device_id"] == 0
        and report["model"]["sha256"] == plan["sha256"]
        and report["prompt_token_ids"] == prompt["tokens"]
        and report["eos_token_ids"] == [248044]
        and report["limits"] == dict(max_new_tokens=128, max_draft_tokens=15)
        and report["protocol"]["warmup"] == 1
        and report["protocol"]["repetitions"] == 3
        and report["protocol"]["low_memory"]
        and report["protocol"]["round_trace_enabled"]
        and bool(report["abi"].get("context_only_prefill")) == context
        and type(split) is bool
        and report["abi"]["om_count"] == 4 + int(context) + int(split)
        and len(mode["measurements"]) == 3,
        "generation protocol changed for " + prompt["id"],
    )
    chunks = math.ceil(len(prompt["tokens"]) / 64)
    rows = []
    for m in mode["measurements"]:
        t = m["stage_ms"]
        verify = t.get("target_verify", [])
        n = chunks - 1
        draft = t.get("draft", [])
        preparation = t.get("draft_context64", []) if context else draft[:n]
        proposals = draft if context else draft[n:]
        # Only the first proposal can consume more than 16 committed rows.
        first_is_c64 = len(prompt["tokens"]) - 64 * n > 16
        body = t["target_prefill"]
        head = t.get("target_prefill_head", [])
        require(
            len(body) == chunks
            and len(head) == ((1 if context else chunks) if split else 0)
            and len(preparation) == n
            and len(proposals) == len(verify) == sum(bool(r["proposed_token_ids"]) for r in m["rounds"])
            and not t.get("target_decode")
            and (context or not t.get("draft_context64")),
            "invalid context/proposal dispatch for " + prompt["id"],
        )
        effective = list(body)
        for i, elapsed in enumerate(head, start=chunks - len(head)):
            effective[i] += elapsed
        rows.append(
            dict(
                prefill=m["latency_ms"]["prefill"],
                decode=m["latency_ms"]["decode"],
                total=m["latency_ms"]["model_total"],
                context=list(preparation),
                draft=list(proposals),
                draft_c64=proposals[:1] if first_is_c64 else [],
                draft_c16=proposals[1:] if first_is_c64 else proposals,
                target_prefill=effective,
                prefill_middle=effective[:-1],
                prefill_last=effective[-1:],
                prefill_body=list(body) if split else [],
                prefill_head=list(head),
                verify=verify,
                tokens=len(m["generated_token_ids"]),
                accepted=m["counters"]["accepted_draft_tokens"],
                proposed=m["counters"]["drafted_tokens"],
            )
        )
    return rows


def summarize(rows):
    result = {
        k: stats([v[k] for v in rows], allow_zero=k == "decode") for k in ("prefill", "decode", "total")
    }
    result.update(
        {
            k: stats([x for row in rows for x in row[k]])
            for k in (
                "context",
                "draft",
                "draft_c16",
                "draft_c64",
                "target_prefill",
                "verify",
                "prefill_middle",
                "prefill_last",
                "prefill_body",
                "prefill_head",
            )
        }
    )
    result.update({k: sum(row[k] for row in rows) for k in ("tokens", "accepted", "proposed")})
    result["generation_tok_s"] = result["tokens"] * 1000 / sum(row["total"] for row in rows)
    return result


def compare(folder, prompts, old, new, plans):
    by_group = {side: {"short": [], "long": []} for side in ("previous", "context")}
    agreement = {}
    for p in prompts:
        name = p["id"]
        a, b = [c.read(c.verify(capture["reports"][name])) for capture in (old, new)]
        agreement[name] = compare_generation(a["dflash"], b["dflash"])
        for side, z in (("previous", a), ("context", b)):
            by_group[side][p["group"]].extend(timings(z, p, plans[side], context=side == "context"))
        agreement[name]["prefill_saved_ms"] = statistics.mean(
            left["latency_ms"]["prefill"] - right["latency_ms"]["prefill"]
            for left, right in zip(a["dflash"]["measurements"], b["dflash"]["measurements"], strict=True)
        )
    results = {}
    token_matches = sum(a["matching_repetitions"] for a in agreement.values())
    trace_matches = sum(a["round_traces_matched"] for a in agreement.values())
    print(
        f"Tokens and stops {token_matches}/60; full proposal and Verify traces {trace_matches}/60", flush=True
    )
    for group in ("short", "long", "all"):
        results[group] = {}
        for side, cases in by_group.items():
            rows = cases["short"] + cases["long"] if group == "all" else cases[group]
            results[group][side] = summarize(rows)
        a, b = results[group].values()
        print(
            f"{group} previous -> context: Prefill {a['prefill']['mean_ms']:.2f} -> {b['prefill']['mean_ms']:.2f} ms; "
            f"Decode {a['decode']['mean_ms']:.2f} -> {b['decode']['mean_ms']:.2f} ms; "
            f"total {a['total']['mean_ms']:.2f} -> {b['total']['mean_ms']:.2f} ms; "
            f"generation {a['generation_tok_s']:.2f} -> {b['generation_tok_s']:.2f} tok/s",
            flush=True,
        )
        if group == "all":
            print(
                f"Request Median/P95 {a['total']['median_ms']:.2f}/{a['total']['p95_ms']:.2f} -> "
                f"{b['total']['median_ms']:.2f}/{b['total']['p95_ms']:.2f} ms; "
                f"accepted/proposed {a['accepted']}/{a['proposed']} -> {b['accepted']}/{b['proposed']}, "
                f"Verify calls {a['verify']['count']} -> {b['verify']['count']}",
                flush=True,
            )
    a, b = (v["context"] for v in results["all"].values())
    require(a["count"] == b["count"] and a["count"] > 0, "no comparable context blocks")
    print(
        f"Non-final C64 blocks {b['count']}: Median/P95 {a['median_ms']:.3f}/{a['p95_ms']:.3f} -> "
        f"{b['median_ms']:.3f}/{b['p95_ms']:.3f} ms",
        flush=True,
    )
    c.write(folder / "per-prompt-agreement.json", agreement)
    return dict(
        groups=results,
        measurements=3 * len(prompts),
        token_matches=token_matches,
        round_matches=trace_matches,
        agreement=c.identity(folder / "per-prompt-agreement.json"),
    )
