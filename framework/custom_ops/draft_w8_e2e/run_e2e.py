"""Run the optimized five-layer W8 Draft in the documented real Target generation loop."""

import argparse
from dataclasses import asdict
import os
from pathlib import Path
import shutil
import traceback

from winners import (
    c,
    require,
    selected,
    admit,
    GATE_RUN,
    GATE_RESULT,
    TRANSFER,
    TRANSFER_SHA,
    DEPLOYMENT,
    HERE,
    OPS,
)
from history import read_history, write_candidate_plan, SUITE
from features import export, expand_replay
from targets import receiver_env
from prepare_target import prepare
from measure import runner, launch
from package import command
from experiment import Stages
from fresh_run import environment
from fresh_dso import seal
from full_replay import run_om
from raw_protocol import verify_run, stats
from air import protobuf_classes
from vendor import build, local_headers
from compile_all import compile_candidate, audit
from run_q import run_path


def code(*files):
    return {str(p): c.sha(p) for p in files}


def run(out, args, state):
    runs = out.parent
    cann = Path(os.environ["CANN_ROOT"]).resolve(strict=True)
    prev = run_path(runs, args.resume, "w8-e2e-") if args.resume else None
    st = Stages(out, prev)

    def stage(name, contract, action):
        state["phase"] = name
        return st.run(name, contract, action)

    tools = {n: c.identity(shutil.which(n)) for n in ("atc", "msopgen", "cmake")}
    plain = environment(cann, tools)
    desc = c.identity(out / "cann-ge.desc")
    types = protobuf_classes(c.verify(desc).read_bytes())
    require(
        c.sha(runs / TRANSFER / "result.json") == TRANSFER_SHA
        and c.sha(runs / GATE_RUN / "result.json") == GATE_RESULT,
        "different hardware winner archive",
    )
    a = stage(
        "admit",
        dict(
            indices={k: c.identity(runs / k / "stages.json") for k in (TRANSFER, GATE_RUN, DEPLOYMENT)},
            code=c.sha(HERE / "winners.py"),
        ),
        lambda d: admit(d, runs),
    )
    arch = a["archive"]
    exe = arch["runner"]
    c.verify(exe)
    drafts = {}
    state["drafts"] = {}
    for gear in (16, 64):
        s = str(gear)
        case = arch["cases"][s]
        specs = selected(gear)
        assets = case["assets"]
        air = stage(
            "air-c" + s,
            dict(
                source=case["graph"]["optimized"],
                schema=desc["sha256"],
                receipt=st.refs["admit"],
                specs={k: asdict(v) for k, v in specs.items()},
                code=code(HERE / "features.py", OPS / "draft_all_w8/rewrite.py"),
            ),
            lambda d: export(d, types, gear, case, arch["external"]),
        )
        source = {}
        for role, asset in assets.items():
            source[role] = {
                part: dict(
                    cpp=asset[part],
                    headers={name: c.identity(p) for name, p in local_headers(c.verify(asset[part])).items()},
                )
                for part in ("host", "device")
            }
        pkg = stage(
            "opp-c" + s,
            dict(
                sources=source,
                tools=tools,
                code=code(OPS / "draft_all_w8/vendor.py", OPS / "draft_gate_tile_nd/package.py"),
            ),
            lambda d: build(d, cann, plain, gear, assets, specs=specs),
        )
        c.verify_opp(pkg["opp"])
        env = environment(cann, tools, pkg["opp"]["install"])
        mod = stage(
            "atc-c" + s,
            dict(
                air=air["air"],
                pkg=st.refs["opp-c" + s],
                tool=tools["atc"],
                code=c.sha(OPS / "draft_full_panels/compile_model.py"),
            ),
            lambda d: compile_candidate(d, air, cann, env),
        )
        proof = stage(
            "audit-c" + s,
            dict(
                model=st.refs["atc-c" + s],
                source=st.refs["admit"],
                air=st.refs["air-c" + s],
                code=c.sha(OPS / "draft_all_w8/compile_all.py"),
            ),
            lambda d: audit(d, gear, air, pkg, case, mod, specs=specs),
        )
        model = dict(proof, abi=air["abi"])
        drafts[gear] = model
        c.verify(model["om"])
        io = stage(
            "inputs-c" + s,
            dict(inputs=case["io"]["files"], code=c.sha(HERE / "features.py")),
            lambda d: expand_replay(d, case, gear),
        )
        ref = case["reference"]
        c.verify_tree(ref)
        c.verify_tree(io["files"])
        dsos = seal(out / ("dso-c" + s), pkg, exe, env)
        key = dict(
            audit=st.refs["audit-c" + s],
            runner=exe,
            inputs=io["files"],
            golden=ref,
            abi=air["abi"],
            code=code(OPS / "draft_full_panels/full_replay.py", OPS / "draft_full_panels/full_launch.py"),
        )

        def execute(d, phase):
            return run_om(d, air["abi"], io, model, exe, env, dsos, phase, ref)

        sm = stage("smoke-c" + s, key, lambda d: execute(d, "smoke"))
        require(
            sm["validation"] == verify_run(sm, {"golden": ref}, 0, False),
            "different wide-input candidate smoke",
        )
        print(f"C{gear} all 15 token IDs and ten complete KV buffers match frozen outputs", flush=True)

        def time_candidate(d):
            results = [execute(d / f"round-{i+1}", "benchmark") for i in range(3)]
            return dict(rounds=results)

        z = stage("draft-time-c" + s, dict(key, smoke=st.refs["smoke-c" + s], rounds=3), time_candidate)
        samples = []
        for row in z["rounds"]:
            checked = verify_run(row, {"golden": ref}, 0, True)
            require(
                checked == row["validation"] and row["launch_authorized"] and not row["profiled"],
                "bad standalone candidate timing",
            )
            samples.extend(checked["timing"]["samples_ms"])
        t = stats(samples, 90)
        state["drafts"][s] = dict(
            om=mod["om"],
            audit=st.refs["audit-c" + s],
            smoke=st.refs["smoke-c" + s],
            median_ms=t["median_ms"],
            p95_ms=t["p95_ms"],
            samples=90,
            selected={r: v.mode for r, v in specs.items()},
        )
        c.write(out / "drafts.json", state["drafts"])
        print(
            f"C{gear} optimized full Draft (public 28,160 features) Median/P95 {t['median_ms']:.6f}/{t['p95_ms']:.6f} ms (90 candidate only)",
            flush=True,
        )
    if args.draft_only:
        return "DRAFT_VERIFIED_AND_TIMED"
    state["phase"] = "historical-generation"
    h = read_history(SUITE)
    if args.target_run:
        tr = run_path(runs, args.target_run, "e2e-target-")
        receipt = c.read(tr / "result.json")
        require(receipt["status"] == "TARGET_READY", "Target preparation has not completed")
        target = c.read(c.verify(receipt["artifact"]))
        require(
            target["historical_index"] == h["index"], "Target preparation used a different reference workload"
        )
    else:
        target = prepare(out, st, h, state, os.getenv("DFLASH_E2E_TARGET_OM_DIR"))
    env = receiver_env(cann, os.environ, [Path(v) for v in target["vendors"]])
    settled = stage(
        "plan",
        dict(
            original=h["plan"],
            targets=target["models"],
            draft={k: v["om"] for k, v in drafts.items()},
            code=c.sha(HERE / "history.py"),
        ),
        lambda d: write_candidate_plan(d / "chunk-plan.txt", h, target["models"], drafts),
    )
    generation = stage(
        "generation-runner",
        dict(archived=h["runner"], cmake=tools["cmake"], code=c.sha(HERE / "measure.py")),
        lambda d: runner(d, h, cann, env),
    )
    cap = stage(
        "generation",
        dict(
            plan=settled,
            prompts=h["batch"],
            runner=generation,
            code=c.sha(HERE / "measure.py"),
            protocol="DFlash-only, thinking-on, max 128, device 0, 1+3, chunk, low memory, 15, trace rounds",
        ),
        lambda d: launch(d, h, settled, generation, env),
    )
    c.verify(cap["index"])
    c.verify_tree(cap["reports"])

    def compare_in_frontend(d):
        # The repository report API imports the Torch frontend on package initialization.
        # Keep the compiler/runner in its measured gate310p environment.
        front = Path(os.environ["DFLASH_RUN_ROOT"]) / "envs/full-draft-frontend/bin/python"
        request = d / "analysis-input.json"
        c.write(request, dict(history=h, plan=settled, capture=cap))
        command([front, "-B", HERE / "compare_saved.py", request], d / "analysis.log", os.environ, d)
        print((d / "analysis.log").read_text(), flush=True)
        return c.read(d / "compared.json")

    result = stage(
        "comparison",
        dict(
            capture=cap,
            archived=h["cases"],
            plan=settled,
            docs=c.sha(HERE.parents[2] / "docs/RESULTS.md"),
            code=code(
                HERE / "measure.py", HERE / "compare_saved.py", HERE.parents[2] / "tools/benchmark_prompts.py"
            ),
        ),
        compare_in_frontend,
    )
    state.update(
        targets=target["models"],
        runner=generation,
        comparison=result["comparison"],
        groups=result["groups"],
        gear_calls=result["gears"],
        matched=result["matching_generations"],
        ordinary_executions=0,
        raw_capture=cap["index"],
    )
    print("E2E COMPARISON", result["comparison"]["path"], flush=True)
    return (
        "E2E_MEASURED"
        if result["matching_generations"]["previous_w8"] == 60
        else "E2E_MEASURED_WITH_W8_OUTPUT_DIFFERENCES"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-run", help="e2e-target-... directory from prepare_target.sh")
    p.add_argument("--resume", help="w8-e2e-... directory to reuse")
    p.add_argument("--draft-only", action="store_true", help="only build, check, and time the new Draft OMs")
    args = p.parse_args()
    out = Path(os.environ["DFLASH_OUTPUT"])
    state = dict(status="RUNNING", phase="history")
    try:
        state["status"] = run(out, args, state)
    except Exception as e:
        state.update(status="FAILED", error=repr(e))
        traceback.print_exc()
        (out / "traceback.txt").write_text(traceback.format_exc())
    c.write(out / "result.json", state)
    print(state["status"], "REPORT", out / "result.json", flush=True)
    return int(state["status"] == "FAILED")


if __name__ == "__main__":
    raise SystemExit(main())
