"""Measure an additional cache-only C64 OM inside the archived full-generation workload."""

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import traceback

from winners import c, require, HERE, REPO, OPS
from history import read_history, SUITE
from targets import receiver_env
from probe_dynamic import add_provider
from measure import launch
from context_graph import context_model, STATES, ROLES
from context_measure import boundaries, compare, stats
from air import protobuf_classes
from compile_all import compile_candidate, origins, embedded, selected_image, read_om_json
from experiment import Stages
from full_replay import prepare_case
from fresh_run import environment
from package import command
from run_q import run_path


def extract(folder, record, desc):
    model_type, _ = protobuf_classes(c.verify(desc).read_bytes())
    original = model_type.FromString(c.verify(record["air"]).read_bytes())
    graph, abi = context_model(original, record["abi"], record["inventory"])
    path = folder / "draft_context64.air"
    path.write_bytes(graph.SerializeToString(deterministic=True))
    return dict(air=c.identity(path), abi=abi, source=record["air"])


def audit(compiled, previous):
    nodes = read_om_json(c.verify(compiled["dump"]))
    custom = [n for n in nodes if n["type"].startswith("DFlash")]
    require(
        len(custom) == 6
        and not any(n["type"] in ("MatMul", "MatMulV2", "BatchMatMulV2", "ArgMaxV2") for n in nodes),
        "compiled candidate still executes proposal work",
    )
    images = {}
    for role in ROLES:
        old = previous["inventory"][role]
        (node,) = (n for n in custom if old["source_node"] in origins(n))
        images[role] = selected_image(node, 8, old["op"])
    bound = embedded(c.verify(compiled["om"]), list(images.values()))
    for role, image in images.items():
        require(
            bound[image["node"]]["executable_sections"]
            == previous["inventory"][role]["selected"]["executable_sections"],
            "different C64 projection instructions: " + role,
        )
    print(
        "C64 cache graph verified: FC + five KV projections, original machine code, ten state outputs",
        flush=True,
    )
    return {role: bound[image["node"]] for role, image in images.items()}


def extend_plan(folder, previous, compiled, abi):
    path = c.verify(previous)
    lines = path.read_text().splitlines()
    require(
        lines[-1] == "done" and not any(s.startswith("graph draft_context64 ") for s in lines),
        "expected original optimized plan",
    )
    lines[-1] = (
        "graph draft_context64 " + json.dumps(str(c.verify(compiled["om"]))) + " " + compiled["om"]["sha256"]
    )
    for side, marker in (("inputs", "I"), ("outputs", "O")):
        for item in abi[side]:
            shape = item["shape"]
            lines.append(" ".join(map(str, (marker, item["name"], item["dtype"], len(shape), *shape))))
    plan = folder / "chunk-plan.txt"
    plan.write_text("\n".join([*lines, "end", "done", ""]))
    return c.identity(plan)


def build_tools(folder, cann, env):
    tools = {}
    for name, src in (
        ("ac_om_runner", OPS / "draft_ac/test"),
        ("qwen35_dflash_acl_runner", REPO / "framework/runtime/cpp"),
    ):
        dest = folder / name
        flags = (
            ["-DCANN_ROOT=" + str(cann)]
            if name == "ac_om_runner"
            else ["-DASCENDCL_ROOT=" + str(cann), "-DQWEN35_DFLASH_BUILD_TESTS=OFF"]
        )
        command(
            ["cmake", "-S", src, "-B", dest, "-DCMAKE_BUILD_TYPE=Release", *flags],
            folder / (name + "-configure.log"),
            env,
            folder,
        )
        command(
            ["cmake", "--build", dest, "--target", name, "--parallel", "2"],
            folder / (name + "-build.log"),
            env,
            folder,
        )
        tools[name] = c.identity(dest / name)
    return tools


def replay(folder, raw_runner, graph, compiled, io, ref, env, profiler=None, images=None):
    case = folder / "case"
    files = {i["name"]: io["files"][i["name"]] for i in graph["abi"]["inputs"]}
    prepare_case(case, graph["abi"], {"files": files}, None)
    args = [str(raw_runner), "0", str(c.verify(compiled["om"])), str(case), str(case / "inputs")]
    if profiler:
        args = [
            profiler["alias"],
            *profiler["flags"],
            "--output=" + str(folder / "profiler"),
            "--application=" + shlex.join(args),
        ]
    command(args, folder / "runner.log", env, folder)
    data = c.read(case / "execution.json")
    require(
        data["status"] == "PASS"
        and not data["cpu_fallback"]
        and all(data[k] for k in ("guards", "readonly", "repeat")),
        "standalone replay failed",
    )
    for name in STATES:
        for suffix in ("actual-0", "actual-1", "benchmark-last", "postcheck"):
            require(
                c.sha(case / f"{name}-{suffix}.bin") == ref[name][0]["sha256"],
                f"full KV including untouched rows differs: {name} {suffix}",
            )
    times = data["timing"]["execute_sync"]["samples_ms"]
    require(len(times) == 30, "incomplete standalone timing")
    if profiler:
        from c_profile_trace import csv_tables

        root = folder / "profiler"
        if not any(root.rglob("op_summary*.csv")):
            command(
                [profiler["alias"], "--export=on", "--output=" + str(root), "--summary-format=csv"],
                folder / "export.log",
                env,
                folder,
            )
        tables, _ = csv_tables(root)
        require(tables and any(t["rows"] for t in tables), "msprof produced no task durations")
        tasks = {}
        for role, image in images.items():
            samples = [
                r["duration_us"] / 1000 for t in tables for r in t["rows"] if r["op_name"] == image["node"]
            ]
            tasks[role] = stats(samples)
        return dict(
            op_summary=[c.identity(root / t["file"]) for t in tables],
            profiler=profiler["file"],
            scope="isolated context OM, all profiled calls including warmup and correctness checks",
            per_projection_task_ms=tasks,
        )
    result = stats(times)
    print(
        f"C64 cache-only standalone Median/P95 {result['median_ms']:.4f}/{result['p95_ms']:.4f} ms; "
        "10/10 complete KV match under both output poisons",
        flush=True,
    )
    return dict(samples_ms=times, timing=result, execution=c.identity(case / "execution.json"))


def run(out, args, state):
    base = run_path(out.parent, args.from_run, "w8-e2e-")
    prior = Stages(out, base)
    previous = c.read(base / "result.json")
    require(
        previous["status"] == "E2E_MEASURED" and previous["matched"]["previous_w8"] == 60,
        "select a completed, output-matched full generation as the baseline",
    )
    snapshot = c.identity(base / "result.json")
    state["baseline"] = snapshot
    history = read_history(SUITE)

    def saved(name):
        return prior.completed(name)["result"]

    full = saved("air-c64")
    proof = saved("audit-c64")
    pkg = saved("opp-c64")["opp"]
    io = saved("inputs-c64")
    golden = saved("smoke-c64")["outputs"]
    old_capture, old_plan = saved("generation"), saved("plan")
    require(
        proof["air"] == full["air"]
        and proof["om"] == previous["drafts"]["64"]["om"]
        and old_capture["index"] == previous["raw_capture"]
        and c.read(c.verify(old_capture["index"]))["model_sha256"] == old_plan["sha256"],
        "baseline graph or measured inputs differ",
    )
    target = c.read(base / "target.json")
    require(target["models"] == previous["targets"], "different Target models")
    cann = Path(os.environ["CANN_ROOT"]).resolve(strict=True)
    vendor = c.verify(target["dynamic_provider"]["script"]).parents[1]
    base_env = receiver_env(cann, os.environ, [Path(v) for v in target["vendors"] if Path(v) != vendor])
    runtime_env, binding = add_provider(base_env, vendor)
    require(binding == target["dynamic_provider"], "different generation environment")
    c.verify_opp(pkg)
    compile_env = environment(cann, {}, pkg["install"])
    desc = c.identity(out / "cann-ge.desc")
    checkpoint = run_path(out.parent, args.resume, "context-prefill-") if args.resume else None
    stages = Stages(out, checkpoint)
    scripts = {f: c.sha(HERE / f) for f in ("context_graph.py", "context_measure.py", "run_context.py")}

    def stage(name, key, action):
        state["phase"] = name
        return stages.run(name, dict(baseline=snapshot, code=scripts, inputs=key), action)

    reduced = stage("air", [full["air"], desc["sha256"]], lambda d: extract(d, full, desc))
    model = stage(
        "compile",
        [reduced["air"], prior.refs["opp-c64"], c.identity(cann / "bin/atc")],
        lambda d: compile_candidate(d, reduced, cann, compile_env),
    )
    state["om"] = model["om"]
    images = stage("compiled-kernels", [model, prior.refs["audit-c64"]], lambda d: audit(model, proof))
    cpp = REPO / "framework/runtime/cpp"
    sources = [
        *cpp.glob("src/*.cpp"),
        *cpp.glob("include/qwen35_dflash/*.hpp"),
        cpp / "CMakeLists.txt",
        OPS / "draft_ac/test/CMakeLists.txt",
        OPS / "draft_ac/test/runner.cpp",
        OPS / "draft_quant/test/runner_common.h",
    ]
    tools = stage("runners", {str(f): c.sha(f) for f in sources}, lambda d: build_tools(d, cann, runtime_env))
    raw = c.verify(tools["ac_om_runner"])
    exe = c.verify(tools["qwen35_dflash_acl_runner"])
    state["runners"] = dict(previous=previous["runner"]["file"], context=tools["qwen35_dflash_acl_runner"])
    run_key = [model["om"], tools, prior.refs["inputs-c64"], prior.refs["smoke-c64"]]
    state["standalone"] = stage(
        "standalone", run_key, lambda d: replay(d, raw, reduced, model, io, golden, compile_env)
    )
    plans = dict(
        previous=old_plan,
        context=stage(
            "plan", [old_plan, model["om"]], lambda d: extend_plan(d, old_plan, model, reduced["abi"])
        ),
    )
    long_prompt = next(p["tokens"] for p in history["prompts"] if len(p["tokens"]) >= 192)
    stage(
        "kv-boundaries",
        [plans, tools, long_prompt[:192]],
        lambda d: boundaries(d, exe, runtime_env, plans, long_prompt),
    )
    new = stage(
        "generation",
        [plans["context"], tools, history["batch"], stages.refs["kv-boundaries"]],
        lambda d: launch(
            d, history, plans["context"], {"file": tools["qwen35_dflash_acl_runner"]}, runtime_env
        ),
    )
    log = (c.verify(new["index"]).parent / "runner.log").read_text(errors="replace")
    memory = re.search(r"om-memory graph=draft_context64 query_status=0 weight_bytes=(\d+) work_bytes=(\d+)", log)
    if memory:
        weights, work = map(int, memory.groups())
        state["context_memory"] = dict(om_weight_bytes=weights, requested_work_bytes=work)
        print(f"C64 context OM weights {weights:,} B, workspace requested {work:,} B", flush=True)
    result = stage(
        "comparison",
        [new, old_capture, plans],
        lambda d: compare(d, history["prompts"], old_capture, new, plans),
    )
    state.update(result)
    if result["token_matches"] != result["measurements"] or result["round_matches"] != result["measurements"]:
        return "OUTPUT_OR_TRACE_DIFFERENCES"
    if args.profile:
        from panel_profiler import bind_msprof

        def profile(d):
            tool = bind_msprof(d, cann, compile_env)
            return replay(d, raw, reduced, model, io, golden, compile_env, tool, images)

        state["profile"] = stage("profile-context64", run_key, profile)
        print("msprof task CSV", state["profile"]["op_summary"][0]["path"], flush=True)
    return "MEASURED"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-run", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--profile", action="store_true")
    options = parser.parse_args()
    root = Path(os.environ["DFLASH_OUTPUT"])
    result = {"status": "RUNNING"}
    try:
        result["status"] = run(root, options, result)
    except Exception as error:
        result.update(status="FAILED", error=str(error))
        traceback.print_exc()
    c.write(root / "result.json", result)
    print(result["status"], "REPORT", root / "result.json", flush=True)
    raise SystemExit(result["status"] != "MEASURED")
