"""Reuse an anchor-verified Prefill and select the same provider when compiling Verify."""

import os
from pathlib import Path
import shutil

from winners import c, require, HERE
from history import write_candidate_plan
from targets import receiver_env, compile_target
from probe_dynamic import add_provider, inventory, compare
from cache_backend import audit_compiled
from experiment import Stages


def recheck_anchors(probe, history, plan, full_runner, previous_capture):
    a = probe["anchors"]
    ref = a["runner_index"]
    index = c.read(c.verify(ref))
    require(
        index["fake_acl"] is False
        and index["status"] == "PASS"
        and index["model_sha256"] == plan["sha256"]
        and index["order"] == "DFlash only; ordinary baseline external",
        "different Prefill probe plan or runner",
    )
    d = c.verify(ref).parent
    cmd = c.read(d / "command.json")
    require(
        cmd[0] == str(c.verify(full_runner))
        and cmd[cmd.index("--model-sha256") + 1] == plan["sha256"] == c.sha(cmd[cmd.index("--model") + 1])
        and c.sha(cmd[cmd.index("--prompt-batch") + 1]) == index["prompt_batch_sha256"],
        "different measured probe inputs",
    )
    prompts = [p for p in history["prompts"] if p["group"] == "short"]
    require(
        len(prompts) == 8
        and [x["id"] for x in index["cases"]] == [p["id"] for p in prompts]
        and [x["prompt"] for x in a["rows"]] == [p["id"] for p in prompts],
        "wrong short prompt coverage",
    )
    for p, row, case in zip(prompts, a["rows"], index["cases"], strict=True):
        name = p["id"]
        f = c.verify(row["report"])
        r = c.read(f)
        require(
            f == Path(str(c.verify(ref)) + ".cases") / (name + ".json")
            and f == Path(case["report"]).resolve()
            and r["prompt_token_ids"] == p["tokens"]
            and r["status"] == case["status"] == "PASS"
            and r["cpu_fallback"] is False
            and "ordinary" not in r
            and r["device_id"] == 0
            and r["model"]["sha256"] == plan["sha256"]
            and r["limits"]["max_new_tokens"] == 1
            and r["protocol"]["warmup"] == 0
            and r["protocol"]["repetitions"] == 1,
            "invalid probe execution: " + name,
        )
        (m,) = r["dflash"]["measurements"]
        (token,) = m["generated_token_ids"]
        native = c.read(c.verify(history["cases"][name]))["dflash"]["measurements"][0]["generated_token_ids"][
            0
        ]
        previous = c.read(c.verify(previous_capture["reports"][name]))["dflash"]["measurements"][0][
            "generated_token_ids"
        ][0]
        require(
            token == native == row["candidate"] == row["historical"]
            and previous == row["previous"]
            and m["counters"]["graph_calls"] == 1
            and len(m["stage_ms"]["target_prefill"]) == 1
            and all(not m["stage_ms"].get(k) for k in ("target_decode", "target_verify", "draft")),
            "unmatched anchor: " + name,
        )


def promote(out, stage, history, prior, probe_root, drafts, cann):
    # Only the Prefill was executed by the short probe. Verify is compiled and checked here.
    result = c.read(probe_root / "result.json")
    require(
        result["status"] == "ALL_ANCHORS_MATCH",
        "short Target Prefill anchors must agree with the archived generation",
    )
    probe_ref = c.identity(probe_root / "result.json")
    if (prior / "provider-base.json").is_file():
        parent = c.read(prior / "provider-base.json")
        require(parent["probe"] == probe_ref, "resumed provider check belongs to a different probe")
        prior = c.verify(parent["previous"]).parent
        require(
            prior.parent == out.parent and prior.name.startswith("w8-e2e-"),
            "resumed measurement outside run collection",
        )
    c.write(out / "provider-base.json", dict(previous=c.identity(prior / "result.json"), probe=probe_ref))
    src = c.read(prior / "result.json")
    base_target = c.read(prior / "target.json")
    models = dict(base_target["models"])
    require(
        src["status"].startswith("E2E_MEASURED")
        and src["targets"] == models
        and base_target["historical_index"] == history["index"],
        "probe predecessor is not the measured full generation",
    )
    st = Stages(out, prior)
    sourcing = result["sourced"]
    vendor = c.verify(sourcing["script"]).parents[1]
    env, binding = add_provider(
        receiver_env(cann, os.environ, [Path(p) for p in base_target["vendors"]]), vendor
    )
    require(sourcing == binding, "sourced package changed since the Prefill test")
    atc = Path(shutil.which("atc", path=env["PATH"])).resolve(strict=True)
    sources = {name: st.completed("reconstruct-" + name)["result"] for name in models}
    for g in drafts:
        require(drafts[g]["om"] == src["drafts"][str(g)]["om"], "different Draft compared with short probe")
    name = "target_prefill"
    tested = c.verify(result["compiled"])
    cmd = c.read(tested.parent / "command.json")
    require(
        Path(cmd[0]).resolve() == atc
        and "--model=" + str(c.verify(sources[name]["air"])) in cmd
        and sources[name]["air"] == models[name]["source_air"]
        and "--output=" + str(tested.with_suffix("")) in cmd,
        "different reconstructed Prefill AIR or compiler",
    )

    def bind_prefill(d):
        proof = c.read(c.verify(result["cache_backend"]))
        old_proof = c.read(c.verify(models[name]["cache_backend"]))
        before, _ = inventory(dict(om=models[name]["om"], dump=old_proof["dump"]))
        after, _ = inventory(dict(om=result["compiled"], dump=proof["dump"]))
        compared = compare(before, after, vendor)
        q = compared["DynamicQuant"]
        require(
            compared == result["comparison"] and q["chosen"] == q["nodes"] and q["nodes"] > 0,
            "probe did not bind the requested DynamicQuant",
        )
        chosen = dict(
            models[name],
            om=result["compiled"],
            argv=cmd,
            cache_backend=result["cache_backend"],
            historical_binary_match=False,
        )
        plan = write_candidate_plan(
            d / "probe-plan.txt", history, dict(models, target_prefill=chosen), drafts
        )
        recheck_anchors(result, history, plan, src["runner"]["file"], st.completed("generation")["result"])
        return chosen

    models[name] = stage(
        "reuse-provider-prefill",
        dict(
            probe=c.identity(probe_root / "result.json"),
            archive=c.identity(prior / "result.json"),
            source=sources[name]["air"],
            runner=src["runner"],
            binding=binding,
            code=c.sha(HERE / "qualified_provider.py"),
        ),
        bind_prefill,
    )
    print("Reused Prefill with verified short anchors", models[name]["om"]["sha256"], flush=True)
    name = "target_verify"
    original = models[name]
    require(sources[name]["air"] == original["source_air"], "different Verify AIR")
    chosen = stage(
        "compile-provider-verify",
        dict(
            air=sources[name]["air"],
            provider=binding,
            cache=base_target["cache_provider"],
            tool=c.identity(atc),
            code=c.sha(HERE / "targets.py"),
        ),
        lambda d: compile_target(d, name, sources[name], history, env, atc),
    )

    def check_verify(d):
        receipt = audit_compiled(
            d, name, c.verify(chosen["om"]), atc, env, base_target["cache_provider"]["path"]
        )
        a = c.read(c.verify(original["cache_backend"]))
        b = c.read(c.verify(receipt))
        before, _ = inventory(dict(om=original["om"], dump=a["dump"]))
        after, _ = inventory(dict(om=chosen["om"], dump=b["dump"]))
        comp = compare(before, after, vendor)
        q = comp["DynamicQuant"]
        require(
            q["chosen"] == q["nodes"] and q["nodes"] > 0, "Verify did not bind the same private DynamicQuant"
        )
        print("Verify provider comparison", comp, flush=True)
        return dict(cache_backend=receipt, comparison=comp)

    checked = stage(
        "audit-provider-verify",
        dict(
            om=chosen["om"],
            prefill=models["target_prefill"]["om"],
            provider=binding,
            code=[c.sha(HERE / f) for f in ("qualified_provider.py", "cache_backend.py", "probe_dynamic.py")],
        ),
        check_verify,
    )
    c.verify(checked["cache_backend"])
    models[name] = dict(chosen, cache_backend=checked["cache_backend"])
    target = dict(
        base_target,
        models=models,
        vendors=binding["order"],
        dynamic_provider=binding,
        prefill_probe=c.identity(probe_root / "result.json"),
        verify_comparison=checked["comparison"],
    )
    c.write(out / "target.json", target)
    return target, env
