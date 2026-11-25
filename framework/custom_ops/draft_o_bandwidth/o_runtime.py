"""Device reference replay, timed single-O processes and separate device-task profiling."""

import json
from pathlib import Path
import re
import shutil

import numpy as np

from o_panel import c, require, M, K, ABI, OPS, ROLE, CONTROL, OLD_OP, plan, BYTES
from bench import atc_args, read_om_json, selected_image, embedded, flow
from compile_model import grouped
from layout import once
from package import command
from fresh_run import run_child
from full_launch import launch
from full_replay import prepare_case, record_mismatches

REFERENCE = "supplied Layer0 O reference checked in process"


def abi_guard(spec):
    checks = []
    dtypes = {"float16": "ACL_FLOAT16", "int16": "ACL_INT16", "int64": "ACL_INT64"}
    for side in ("inputs", "outputs"):
        checks.append(f"{side}.size()=={len(spec[side])}")
        for i, t in enumerate(spec[side]):
            require(
                re.fullmatch(r"[A-Za-z0-9_]+", t["name"])
                and all(type(d) is int and d > 0 for d in t["shape"]),
                "invalid O runner ABI",
            )
            dims = ",".join(map(str, t["shape"]))
            row = f"{side}[{i}]"
            checks.extend(
                [
                    f'{row}.name=={json.dumps(t["name"])}',
                    f"{row}.dtype==" + dtypes[t["dtype"]],
                    f"{row}.shape==std::vector<int64_t>({{{dims}}})",
                ]
            )
    return " &&\n                ".join(checks)


def runner(out, cann, env, ref_abi, *, projection_abi=ABI, reference=REFERENCE):
    src = out / "runner-src"
    shutil.copytree(OPS / "draft_gate_raw_atc8/runner", src)
    p = src / "runner.cpp"
    code = p.read_text()
    code = once(
        code,
        'if(!benchmark && std::string(argv[6])!="smoke")',
        'const bool capture=std::string(argv[6])=="capture";\n        if(!capture && !benchmark && std::string(argv[6])!="smoke")',
    )
    start = code.index("        if(inputs.size()!=1 || outputs.size()!=1")
    end = code.index('throw std::runtime_error("expected frozen Gate projection ABI");', start) + len(
        'throw std::runtime_error("expected frozen Gate projection ABI");'
    )
    code = (
        code[:start]
        + f"""        if(capture) {{
            if(!({abi_guard(ref_abi)})) throw std::runtime_error("unexpected projection prefix ABI");
        }} else if(!({abi_guard(projection_abi)})) throw std::runtime_error("unexpected single projection ABI");"""
        + code[end:]
    )
    code = once(code, "golden;for(const auto&t:outputs)", "golden;if(!capture)for(const auto&t:outputs)")
    code = once(code, "i<golden[0].size()", "!capture && i<golden[0].size()")
    code = once(code, "if(value!=golden[i])", "if(!capture && value!=golden[i])")
    code = once(code, "single Gate frozen Native Golden checked in process", reference)
    code = once(
        code,
        r"\"golden_match\":true",
        r'\"reference_capture\":"<<(capture?"true":"false")<<",\"golden_match\":"<<(capture?"false":"true")<<"',
    )
    p.write_text(code)
    target = out / "runner-build"
    command(
        [
            "cmake",
            "-S",
            src,
            "-B",
            target,
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCANN_ROOT=" + str(cann),
            "-DCOMMON_INCLUDE=" + str(c.QUANT / "test"),
        ],
        out / "configure.log",
        env,
        out,
    )
    command(["cmake", "--build", target, "--parallel", "2"], out / "build.log", env, out)
    return c.identity(target / "gate_om_runner")


def compile_prefix(out, src, env, argv, *, validate_site=None):
    air = c.verify(src["reference_air"])
    pkg = src["pkg"]
    cache = out / "cache"
    cache.mkdir()
    platform = out / "platform.jsonl"
    sched = out / "scheduling.jsonl"
    for file in (platform, sched):
        file.touch()
    env = dict(
        env,
        AI_RUN_DIR=str(out),
        ASCEND_WORK_PATH=str(out),
        ASCEND_CACHE_PATH=str(cache),
        DFLASH_PLATFORM_DIAG_PATH=str(platform),
        DFLASH_PLATFORM_DIAG_ROUTE="ATC",
        DFLASH_SCHEDULER_EVIDENCE_PATH=str(sched),
    )
    args = atc_args(argv, air, out / "reference")
    c.write(out / "command.json", args)
    command(args, out / "atc.log", env, out)
    om = out / "reference.om"
    dump = out / "om-graph.json"
    command([argv[0], "--mode=1", "--om=" + str(om), "--json=" + str(dump)], out / "inspect.log", env, out)
    pp, ss = grouped(platform), grouped(sched)
    shapes = {tuple(s[k] for k in ("m", "k", "n")) for s in src["sites"].values()}
    require(set(pp) == set(ss) == shapes, "different set of ATC prefix callbacks")
    expected = {site["node"]: role for role, site in src["sites"].items()}
    selected = []
    sources = {}
    plans = {}
    for shape in sorted(shapes):
        key = "x".join(map(str, shape))
        d = out / key
        d.mkdir()
        for name, data in (("p", pp), ("s", ss)):
            (d / name).write_text("\n".join(data[shape]) + "\n")
        query = flow.read_query(d / "p", d / "s", pkg["opp"], "ATC", "A", shape)
        old = src["source_plans"][key]
        require(
            query["tiling_sha256"] == old["tiling_sha256"]
            and query["schedule"]["resources"] == old["schedule"]["resources"]
            and query["platform"]["blockDim"] == old["platform"]["blockDim"] == 8,
            "prefix tiling differs: " + key,
        )
        plans[key] = query
    if validate_site is None:
        plan(plans["16x4096x2560"], CONTROL, src["query"])
    else:
        validate_site(plans)
    for node in read_om_json(dump):
        if not node["type"].startswith("DFlash"):
            continue
        names = {node["name"]}
        for a in node["attributes"]:
            if a["key"] in ("_datadump_original_op_names", "_original_op_names"):
                names.update((a.get("value") or {}).get("list", {}).get("s", []))
        matching = names & expected.keys()
        require(
            node["type"] == OLD_OP and len(matching) == 1,
            "foreign or ambiguous projected prefix node: " + node["name"],
        )
        role = expected[matching.pop()]
        require(role not in sources.values(), "duplicate prefix projection")
        sources[node["name"]] = role
        require(
            [a["value"] for a in node["attributes"] if a["key"] == "custom_op_file_path"]
            == [{"s": str(Path(pkg["opp"]["install"]) / "vendors/customize")}],
            "different prefix private OPP",
        )
        selected.append(selected_image(node, 8, OLD_OP))
    require(set(sources.values()) == set(src["sites"]), "missing prefix projection")
    bins = embedded(om, selected)
    for n, v in bins.items():
        require(
            v["executable_sections"] == src["source_implementations"][sources[n]]["executable_sections"],
            "different prefix machine text: " + sources[n],
        )
    return dict(om=c.identity(om), dump=c.identity(dump), selected=bins, plans=plans, argv=args)


def capture(out, runner, model, env, dsos, src, label, *, input_shape=(M, K)):
    app = out / "application"
    prepare_case(app, src["reference_abi"], dict(files=src["replays"][label]), None)
    launch(out, runner, model["om"], "capture", env, dsos, None)
    e = c.read(app / "execution.json")
    require(
        e["status"] == "PASS"
        and e["reference_capture"] is True
        and e["golden_match"] is False
        and e["cpu_fallback"] is False
        and e["device"] == 0
        and e["verification_calls"] == 2
        and e["timing"]["repetitions"] == 0
        and all(e[k] is True for k in ("guards", "readonly", "repeat", "poison_verified", "sync_completed")),
        "incomplete Layer0 prefix replay",
    )
    files = dict(src["weights"])
    snapshots = {}
    for t in src["reference_abi"]["outputs"]:
        name = t["name"]
        f = c.identity(app / (name + "-actual-0.bin"))
        a = np.fromfile(f["path"], "<f2")
        require(
            f["sha256"] == c.sha(app / (name + "-actual-1.bin"))
            and a.size == np.prod(t["shape"])
            and np.isfinite(a).all(),
            "invalid prefix snapshot: " + name,
        )
        snapshots[name] = f
        if name in ("x", "y"):
            require(np.any(a), "all-zero projection prefix snapshot: " + name)
            if label == "stress" and name == "x":
                require(
                    np.unique(a.reshape(input_shape), axis=0).shape[0] == input_shape[0],
                    "projection stress does not exercise all rows distinctly",
                )
            files["x.bin" if name == "x" else "golden.bin"] = f
        elif label == "measured":
            require(
                f["sha256"] == src["expected_kv"][name]["sha256"],
                "complete Layer0 KV differs from accepted full-run replay: " + name,
            )
    return dict(
        files=files,
        outputs=snapshots,
        execution=c.identity(app / "execution.json"),
        origin=src.get("capture_origin", "LAYER0_ATTENTION_DEVICE_PREFIX") + "_" + label.upper(),
    )


def execute(folder, runner, model, env, dsos, files, timed, i, *, abi=ABI, reference=REFERENCE):
    try:
        return run_child(folder, runner, model, env, dsos, files, timed, reference, i, abi=abi)
    except Exception:
        record_mismatches(folder, abi, {"y": files["golden.bin"]})
        raise


def collect_profile(out, runner, model, env, dsos, files, cli, *, abi, reference):
    app = out / "application"
    prepare_case(app, abi, {"files": {"x": files["x.bin"]}}, {"y": files["golden.bin"]})
    require(c.identity(cli["alias"]) == cli["file"], "msprof changed")
    witness = launch(out, runner, model["om"], "benchmark", env, dsos, cli["alias"])
    e = c.read(app / "execution.json")
    require(
        e["status"] == "PASS"
        and e["scope"] == reference
        and e["device"] == 0
        and e["reference_capture"] is False
        and e["cpu_fallback"] is False
        and e["verification_calls"] == 4
        and all(
            e[k] is True
            for k in (
                "guards",
                "golden_match",
                "readonly",
                "repeat",
                "poison_verified",
                "sync_completed",
                "timed_log_stable",
            )
        )
        and (e["timing"]["warmup"], e["timing"]["repetitions"]) == (5, 30),
        "profiled projection correctness or timing structure differs",
    )
    for suffix in ("actual-0", "actual-1", "benchmark-last", "postcheck"):
        require(
            c.sha(app / ("y-" + suffix + ".bin")) == files["golden.bin"]["sha256"],
            "profiled projection output drift",
        )
    return dict(witness, profiler=str(out / "profiler"), execution=c.identity(app / "execution.json"))


def analyze_profile(out, model, capture, *, role, logical_bytes):
    from module_profile import analyze

    node = model["selected"]["node"]
    report = analyze(
        out,
        capture,
        model,
        dict(modules={node: role}, node_types={node: model["op"]}),
    )
    (row,) = (r for r in report["modules"] if r["module"] == role)
    require(
        row["calls_observed"] == row["tasks"] == 30 and row["p50_task_sum_ms"] > 0,
        f"need one {role} device task in every timed window",
    )
    ms = row["p50_task_sum_ms"]
    return dict(
        attribution=report["scope"],
        kernel=row,
        logical_gbps=logical_bytes / ms / 1e6,
        floor_ratio=ms / (logical_bytes / 204.8e6),
        timeline=report["timeline"],
        op_summary=report["op_summary"],
        execution=capture["execution"],
    )


def profile(
    out, runner, model, env, dsos, files, cli, *, abi=ABI, role=ROLE, reference=REFERENCE, logical_bytes=BYTES
):
    capture = collect_profile(out, runner, model, env, dsos, files, cli, abi=abi, reference=reference)
    return analyze_profile(out, model, capture, role=role, logical_bytes=logical_bytes)
