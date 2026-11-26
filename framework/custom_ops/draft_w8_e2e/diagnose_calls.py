"""Locate a full-Draft timing difference with matched actual inputs, command streams and output bytes."""

import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import subprocess
import traceback

from winners import c, require, HERE, REPO
from run_q import run_path
from experiment import Stages
from targets import receiver_env
from probe_dynamic import add_provider
from fresh_run import environment
from fresh_dso import seal
from full_replay import run_om
from package import command
from call_latency_results import compare, dist, samples, task_contrasts


def workload(capture, plan):
    index = c.read(c.verify(capture["index"]))
    require(
        not index["fake_acl"] and index["model_sha256"] == plan["sha256"] and len(index["cases"]) == 20,
        "expected the completed 20-prompt receiver run",
    )
    all_times = {"16": [], "64": []}
    chosen = None
    for entry in index["cases"]:
        name = entry["id"]
        case = c.read(c.verify(capture["reports"][name]))
        prompt = case["prompt_token_ids"]
        require(
            case["model"]["sha256"] == plan["sha256"]
            and not case["cpu_fallback"]
            and case["abi"].get("context_only_prefill") is True
            and case["protocol"]["repetitions"] == 3,
            "wrong generation archive",
        )
        for m in case["dflash"]["measurements"]:
            stages = m["stage_ms"]
            rounds = [r for r in m["rounds"] if r["proposed_token_ids"]]
            times = stages["draft"]
            nctx = (len(prompt) - 1) // 64
            require(
                len(stages.get("draft_context64", [])) == nctx
                and len(times) == len(rounds) == len(stages["target_verify"]),
                "mixed proposal/context timings",
            )
            if times:
                all_times["64" if len(prompt) - 64 * nctx > 16 else "16"].append(times[0])
                all_times["16"].extend(times[1:])
            if chosen is None and 16 < len(prompt) <= 64 and len(rounds) >= 2:
                a, b = rounds[:2]
                chosen = dict(
                    name=name,
                    case=capture["reports"][name],
                    rows=[
                        prompt,
                        m["generated_token_ids"][:1],
                        a["proposed_token_ids"],
                        a["target_token_ids"],
                        b["proposed_token_ids"],
                    ],
                )
    require(chosen is not None, "no C64 -> Verify -> C16 reference prompt")
    return {g: dist(v) for g, v in all_times.items()}, chosen


def compile_probe(d, cann, env):
    command(
        [
            "cmake",
            "-S",
            HERE / "call_latency",
            "-B",
            d / "build",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DPROBE_FAKE_ACL=OFF",
            "-DCANN_ROOT=" + str(cann),
        ],
        d / "configure.log",
        env,
        d,
    )
    command(
        ["cmake", "--build", d / "build", "--target", "draft_call_latency", "--parallel", "3"],
        d / "build.log",
        env,
        d,
    )
    exe = d / "build/draft_call_latency"
    libs = subprocess.run(["ldd", str(exe)], env=env, capture_output=True, check=True, text=True).stdout
    require(
        "libascendcl.so" in libs and "not found" not in libs and "/devlib/" not in libs,
        "expected real AscendCL",
    )
    return c.identity(exe)


def integrated(d, exe, plan, roles, case, env, mode, policy, n, *, save=False, profiler=None):
    args = list(
        map(str, [c.verify(exe), plan["path"], roles, case, d, mode, policy, n, "save" if save else "check"])
    )
    if profiler:
        args = [
            profiler["alias"],
            *profiler["flags"],
            "--output=" + str(d / "profiler"),
            "--application=" + shlex.join(args),
        ]
    c.write(d / "command.json", args)
    log = d / "runner.log"
    with log.open("xb") as f:
        rc = subprocess.run(args, stdout=f, stderr=subprocess.STDOUT, cwd=d, env=env).returncode
    msg = log.read_text(errors="replace")
    if rc and mode == "events" and "EVENT_UNAVAILABLE" in msg:
        print("Device events unavailable; uninstrumented contrasts retained", flush=True)
        return dict(unavailable=c.identity(log))
    if (
        rc
        and policy != "stock"
        and any(
            s in msg
            for s in (
                "aclmdlLoadFromFileWithMem failed",
                "aclrtMalloc failed",
                "aclrtMalloc(shared_workspace) failed",
            )
        )
    ):
        print(f"{policy} allocation unavailable; see {log}", flush=True)
        return dict(unavailable=c.identity(log))
    require(
        rc == 0 and (d / "backend.txt").read_text().strip() == "ASCENDCL_DEVICE_BOUNDARIES_VERIFIED",
        f"boundary replay failed: {msg[-2400:]}",
    )
    rows = c.identity(d / "samples.csv")
    samples(c.verify(rows), mode, n)
    result = dict(rows=rows, mode=mode, n=n, root=str(d), log=c.identity(log))
    if mode != "clean":
        result["hashes"] = {g: c.identity(d / f"c{g}-hashes.tsv") for g in ("16", "64")}
    return result


def check_same_inputs(current, original):
    for gear, record in original["hashes"].items():
        require(
            c.sha(c.verify(current["hashes"][gear])) == c.sha(c.verify(record)),
            f"C{gear} real input or complete KV/Top1 outputs changed across processes",
        )


def real_inputs(gear, pair, abi):
    base = Path(pair["root"]) / ("c" + gear)
    inputs = {t["name"]: c.identity(base / "inputs" / (t["name"] + ".bin")) for t in abi["inputs"]}
    golden = {t["name"]: c.identity(base / "golden" / (t["name"] + ".bin")) for t in abi["outputs"]}
    hashes = [line.split() for line in c.verify(pair["hashes"][gear]).read_text().splitlines()]
    require(
        len(hashes) == len(inputs) + len(golden)
        and all((inputs if side == "I" else golden)[k]["sha256"] == sha for side, k, sha in hashes),
        "snapshot payloads differ from genuine boundary",
    )
    return {"files": inputs}, golden


def standalone(d, abi, model, runner, pkg, cann, io, golden, profiler=None):
    env = environment(cann, {}, pkg["opp"]["install"])
    dsos = seal(d / "dso", pkg, runner, env)
    record = run_om(d, abi, io, model, runner, env, dsos, "benchmark", golden, profiler)
    return dict(
        times=dist(record["validation"]["timing"]["samples_ms"]), root=str(d), execution=record["execution"]
    )


def profile_info(graph, model):
    from module_profile import scopes, read_om_json

    compiled = read_om_json(c.verify(model["dump"]))
    _, mapping = scopes(graph, compiled)
    inv = model["inventory"]
    nodes = {v["node"]: role.split(".")[-1] for role, v in inv.items()}
    types = {v["node"]: v["op"] for v in inv.values()}
    heads = [
        n["name"]
        for n in compiled
        if n["type"] in ("MatMul", "MatMulV2")
        and (
            mapping.get(n["name"]) == "head.matmul"
            or any(
                [int(x) for x in desc.get("shape", {}).get("dim", [])] == [15, 248320]
                for desc in n["output_desc"]
            )
        )
    ]
    if len(heads) == 1:
        nodes[heads[0]] = "Head"
        types[heads[0]] = next(n["type"] for n in compiled if n["name"] == heads[0])
    require(len(inv) == 26, "missing verified Draft projections")
    return dict(fc=(inv["fc"]["node"], inv["fc"]["op"]), nodes=nodes, types=types)


def export_csv(root, cli, env):
    if not list((root / "profiler").rglob("op_summary*.csv")):
        command(
            [cli["alias"], "--export=on", "--output=" + str(root / "profiler"), "--summary-format=csv"],
            root / "latency-export.log",
            env,
            root,
        )


def run(out, args, state):
    base = run_path(out.parent, args.from_run, "context-prefill-")
    context = c.read(base / "result.json")
    require(
        context["status"] == "MEASURED" and context["token_matches"] == context["round_matches"] == 60,
        "expected an output-matched context-only integration",
    )
    original = c.verify(context["baseline"]).parent
    old = c.read(original / "result.json")
    require(old["status"] == "E2E_MEASURED", "baseline generation incomplete")
    archived = Stages(out, original)
    ctx = Stages(out, base)
    plan = ctx.completed("plan")["result"]
    archive, chosen = workload(ctx.completed("generation")["result"], plan)
    state["actual_generation"] = archive
    for gear, v in archive.items():
        standalone_old = old["drafts"][gear]["median_ms"]
        print(
            f"C{gear} archive full proposals n={v['n']} Median/P95 {v['p50']:.3f}/{v['p95']:.3f} ms, "
            f"isolated frozen {standalone_old:.3f}, gap {v['p50'] - standalone_old:+.3f} ms",
            flush=True,
        )
    expected = {"c" + g: v["om"] for g, v in old["drafts"].items()}
    expected.update(
        context=context["om"],
        prefill=old["targets"]["target_prefill"]["om"],
        verify=old["targets"]["target_verify"]["om"],
    )
    aliases = dict(target_prefill="prefill", target_verify="verify", draft_context64="context", draft="c16")
    bindings = {}
    for line in c.verify(plan).read_text().splitlines():
        v = shlex.split(line)
        if v and v[0] == "graph":
            bindings[aliases[v[1]]] = (v[2], v[3])
        elif v and v[0] == "static_gear64":
            bindings["c64"] = (v[1], v[2])
    require(
        bindings == {k: (v["path"], v["sha256"]) for k, v in expected.items()}, "wrong integrated model files"
    )
    roles, case = out / "roles.txt", out / "case.txt"
    roles.write_text("".join(k + " " + json.dumps(p) + "\n" for k, (p, _) in bindings.items()))
    case.write_text("".join(" ".join(map(str, row)) + "\n" for row in chosen["rows"]))
    state.update(prompt=chosen["name"], baseline=c.identity(base / "result.json"), plan=plan)
    target = c.read(original / "target.json")
    require(target["models"] == old["targets"], "different Target assets")
    cann = Path(os.environ["CANN_ROOT"]).resolve(strict=True)
    vendor = c.verify(target["dynamic_provider"]["script"]).parents[1]
    env, binding = add_provider(
        receiver_env(cann, os.environ, [Path(v) for v in target["vendors"] if Path(v) != vendor]), vendor
    )
    require(binding == target["dynamic_provider"], "different Target runtime providers")
    previous = run_path(out.parent, args.resume, "draft-call-gap-") if args.resume else None
    stages = Stages(out, previous)
    source = [
        *(HERE / "call_latency").glob("*"),
        HERE / "diagnose_calls.py",
        HERE / "call_latency_results.py",
        *(REPO / "framework/runtime/cpp/src").glob("*.cpp"),
        *(REPO / "framework/runtime/cpp/include/qwen35_dflash").glob("*.hpp"),
    ]
    code = {str(p): c.sha(p) for p in source if p.is_file()}

    def stage(name, key, action):
        state["phase"] = name
        return stages.run(name, dict(plan=plan, source=code, case=c.sha(case), key=key), action)

    exe = stage("runner", binding, lambda d: compile_probe(d, cann, env))

    def probe(name, mode, policy="stock", n=16, *, save=False, cli=None):
        return stage(
            name,
            [exe, mode, policy, n, save, cli and cli["file"]],
            lambda d: integrated(d, exe, plan, roles, case, env, mode, policy, n, save=save, profiler=cli),
        )

    def read(s):
        return samples(c.verify(s["rows"]), s["mode"], s["n"])

    clean = read(probe("normal-calls", "clean"))
    original_pair = probe("paired", "pair", save=True)
    repeated = read(original_pair)
    models = {g: archived.completed("audit-c" + g)["result"] for g in archive}
    graphs = {g: archived.completed("air-c" + g)["result"] for g in archive}
    packages = {g: archived.completed("opp-c" + g)["result"] for g in archive}
    runner = archived.completed("admit")["result"]["archive"]["runner"]

    def isolated(g, cli=None, *, frozen=False):
        require(
            models[g]["om"] == old["drafts"][g]["om"] and graphs[g]["air"] == models[g]["air"],
            "wrong isolated model",
        )
        if frozen:
            io = archived.completed("inputs-c" + g)["result"]
            original_outputs = archived.completed("smoke-c" + g)["result"]["outputs"]
            golden = {key: records[0] for key, records in original_outputs.items()}
        else:
            io, golden = real_inputs(g, original_pair, graphs[g]["abi"])
        name = "profile-isolated-c" if cli else ("frozen-c" if frozen else "isolated-c")
        return stage(
            name + g,
            [io["files"], golden, models[g]["om"], runner, cli and cli["file"]],
            lambda d: standalone(
                d,
                graphs[g]["abi"],
                models[g],
                runner,
                packages[g],
                cann,
                io,
                golden,
                cli["alias"] if cli else None,
            ),
        )

    stand, frozen = {}, {}
    for g in archive:
        for is_frozen in (False, True) if g == "16" else (True, False):
            target_stats = frozen if is_frozen else stand
            target_stats[g] = isolated(g, frozen=is_frozen)["times"]
        print(
            f"C{g} preliminary P50 ms after Target / hot + IO / hot bare / isolated same inputs: "
            + " / ".join(
                f"{dist(v['wall_ms'] for v in repeated[g][k])['p50']:.3f}"
                for k in ("after_target", "hot_full", "hot_bare")
            )
            + f" / {stand[g]['p50']:.3f}",
            flush=True,
        )
    state["isolated_frozen"] = frozen
    event = probe("events", "events", n=8)
    marked = None
    if "rows" in event:
        check_same_inputs(event, original_pair)
        marked = read(event)
    else:
        state["device_events_unavailable"] = event["unavailable"]
    memory = {}
    for name in ("huge-io-work", "draft-weights", "draft-work", "stock-end"):
        obs = probe(name, "pair", "stock" if name == "stock-end" else name, n=12)
        if "unavailable" in obs:
            state.setdefault("allocation_unavailable", {})[name] = obs["unavailable"]
            continue
        if name != "stock-end" and any(
            obs["hashes"][g]["sha256"] != original_pair["hashes"][g]["sha256"] for g in archive
        ):
            state.setdefault("allocation_output_or_input_difference", {})[name] = obs["hashes"]
            print(f"{name} real inputs or full outputs differed, timing excluded", flush=True)
            continue
        check_same_inputs(obs, original_pair)
        memory[name] = read(obs)
    state["timing"] = compare(archive, clean, repeated, stand, marked, memory)
    for g in archive:
        print(
            f"C{g} isolated Execute+Sync P50: historical {old['drafts'][g]['median_ms']:.3f}, "
            f"same frozen replay {frozen[g]['p50']:.3f}, generation inputs {stand[g]['p50']:.3f} ms",
            flush=True,
        )
    c.write(out / "timing.json", state["timing"])
    if args.profile:
        from panel_profiler import bind_msprof

        cli = bind_msprof(out, cann, env)
        captured = probe("profile-pairs", "pair", n=5, cli=cli)
        check_same_inputs(captured, original_pair)
        specs = {g: profile_info(graphs[g], models[g]) for g in archive}
        with c.verify(captured["rows"]).open() as f:
            rows = list(csv.DictReader(f))
        seq = {g: [(int(r["rep"]), r["sequence"]) for r in rows if r["gear"] == g] for g in archive}
        folder = Path(captured["root"])
        export_csv(folder, cli, env)
        result = task_contrasts(folder / "profiler", specs, seq)
        state["profile"] = {"resident": result}
        for g in archive:
            single = Path(isolated(g, cli)["root"])
            export_csv(single, cli, env)
            iso = task_contrasts(
                single / "profiler",
                {g: specs[g]},
                {g: [(i - 7 if 7 <= i < 37 else -1, "isolated") for i in range(38)]},
                single=g,
            )
            state["profile"]["isolated-c" + g] = iso
            for data in (result, iso):
                scope = data.get("gears", {}).get(g, {})
                for k, stats_row in scope.get("task_sums_ms", {}).items():
                    print(
                        f"C{g} msprof {k} device task sums P50 ms "
                        + " ".join(f"{s}={z['p50']:.3f}" for s, z in stats_row.items()),
                        flush=True,
                    )
                if scope.get("fc_to_head_span_ms"):
                    print(
                        f"C{g} msprof FC start to Head end P50 ms "
                        + " ".join(f"{s}={z['p50']:.3f}" for s, z in scope["fc_to_head_span_ms"].items()),
                        flush=True,
                    )
                if scope.get("status") != "COMPLETE":
                    print(f"C{g} msprof coverage {scope.get('status', 'UNRESOLVED')}", flush=True)
    return "MEASURED"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-run", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    out = Path(os.environ["DFLASH_OUTPUT"])
    s = {"status": "RUNNING"}
    try:
        s["status"] = run(out, args, s)
    except Exception as e:
        s.update(status="FAILED", error=str(e))
        traceback.print_exc()
    c.write(out / "result.json", s)
    print(s["status"], "REPORT", out / "result.json", flush=True)
    raise SystemExit(s["status"] != "MEASURED")
