"""One untimed Q reference capture, guarded A/B replay and independent msprof device time."""

import shutil

import numpy as np

from q_panel import (
    c,
    OPS,
    M,
    K,
    N,
    ABI,
    REF_ABI,
    CONTROL,
    OLD_OP,
    require,
    embedded,
    SHAPE,
    plan,
    BYTES,
    FLOOR_MS,
)
from bench import atc_args, selected_image, read_om_json, flow
from package import command
from layout import once
from full_launch import launch
from full_replay import prepare_case, record_mismatches
from fresh_run import run_child

REFERENCE = "supplied Layer0 Q reference checked in process"


def runner(out, cann, env):
    src = out / "runner-src"
    shutil.copytree(OPS / "draft_gate_raw_atc8/runner", src)
    file = src / "runner.cpp"
    code = file.read_text()
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
        code[:start] + """        auto fp=[](const TensorSpec& t,const char* name,std::vector<int64_t> shape){
            return t.name==name && t.dtype==ACL_FLOAT16 && t.shape==shape;
        };
        if(capture) {
            if(inputs.size()!=1 || outputs.size()!=2 || !fp(inputs[0],"embedding",{1,16,2560}) ||
               !fp(outputs[0],"x",{16,2560}) || !fp(outputs[1],"y",{16,4096}))
                throw std::runtime_error("Q reference prefix ABI");
        } else if(inputs.size()!=1 || outputs.size()!=1 || !fp(inputs[0],"x",{16,2560}) ||
                  !fp(outputs[0],"y",{16,4096})) throw std::runtime_error("Q projection ABI");""" + code[end:]
    )
    code = once(code, "golden;for(const auto&t:outputs)", "golden;if(!capture)for(const auto&t:outputs)")
    code = once(code, "i<golden[0].size()", "!capture && i<golden[0].size()")
    code = once(code, "if(value!=golden[i])", "if(!capture && value!=golden[i])")
    code = once(code, "single Gate frozen Native Golden checked in process", REFERENCE)
    code = once(
        code,
        r"\"golden_match\":true",
        r'\"reference_capture\":"<<(capture?"true":"false")<<",\"golden_match\":"<<(capture?"false":"true")<<"',
    )
    file.write_text(code)
    dest = out / "runner-build"
    command(
        [
            "cmake",
            "-S",
            src,
            "-B",
            dest,
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCANN_ROOT=" + str(cann),
            "-DCOMMON_INCLUDE=" + str(c.QUANT / "test"),
        ],
        out / "configure.log",
        env,
        out,
    )
    command(["cmake", "--build", dest, "--parallel", "2"], out / "build.log", env, out)
    return c.identity(dest / "gate_om_runner")


def compile_reference(out, src, env, argv):
    air = c.verify(src["reference_air"])
    pkg = src["control_package"]
    cache = out / "cache"
    cache.mkdir()
    platform = out / "platform.jsonl"
    sched = out / "schedule.jsonl"
    for file in (platform, sched):
        file.touch()
    local = dict(
        env,
        ASCEND_CACHE_PATH=str(cache),
        ASCEND_WORK_PATH=str(out),
        AI_RUN_DIR=str(out),
        DFLASH_PLATFORM_DIAG_PATH=str(platform),
        DFLASH_PLATFORM_DIAG_ROUTE="ATC",
        DFLASH_SCHEDULER_EVIDENCE_PATH=str(sched),
    )
    args = atc_args(argv, air, out / "reference")
    command(args, out / "atc.log", local, out)
    om = out / "reference.om"
    dump = out / "om-graph.json"
    command([argv[0], "--mode=1", "--om=" + str(om), "--json=" + str(dump)], out / "inspect.log", env, out)
    q = flow.read_query(platform, sched, pkg["opp"], "ATC", "A", SHAPE)
    proof = plan(q, CONTROL, src["query"])
    (node,) = (n for n in read_om_json(dump) if n["type"] == OLD_OP)
    image = selected_image(node, proof["block_dim"], OLD_OP)
    selected = embedded(om, [image])[node["name"]]
    require(
        selected["executable_sections"] == src["compiled_q"]["executable_sections"],
        "reference Q does not run the sealed full-Draft machine text",
    )
    return dict(om=c.identity(om), dump=c.identity(dump), argv=args, selected=selected, query=q)


def capture(out, runner, model, env, dsos, weights, embedding, origin):
    folder = out / "application"
    prepare_case(folder, REF_ABI, {"files": {"embedding": embedding}}, None)
    launch(out, runner, model["om"], "capture", env, dsos, None)
    e = c.read(folder / "execution.json")
    require(
        e["status"] == "PASS"
        and e["device"] == 0
        and e["reference_capture"] is True
        and e["golden_match"] is False
        and e["cpu_fallback"] is False
        and e["verification_calls"] == 2
        and e["timing"]["repetitions"] == 0
        and all(e[k] is True for k in ("guards", "readonly", "repeat", "poison_verified", "sync_completed")),
        "Q prefix capture checks failed",
    )
    files = dict(weights)
    for name, key, count in (("x", "x.bin", M * K), ("y", "golden.bin", M * N)):
        row = c.identity(folder / (name + "-actual-0.bin"))
        actual = np.fromfile(row["path"], dtype="<f2")
        require(
            row["sha256"] == c.sha(folder / (name + "-actual-1.bin"))
            and actual.size == count
            and np.isfinite(actual).all()
            and np.any(actual),
            "invalid Q reference capture: " + name,
        )
        files[key] = row
    return dict(files=files, execution=c.identity(folder / "execution.json"), origin=origin)


def execute(folder, runner, model, env, dsos, files, timed, i):
    try:
        return run_child(folder, runner, model, env, dsos, files, timed, REFERENCE, i, abi=ABI)
    except Exception:
        record_mismatches(folder, ABI, {"y": files["golden.bin"]})
        raise


def profile_kernel(out, runner, model, env, dsos, files, cli):
    from module_profile import analyze

    folder = out / "application"
    prepare_case(folder, ABI, {"files": {"x": files["x.bin"]}}, {"y": files["golden.bin"]})
    require(c.identity(cli["alias"]) == cli["file"], "msprof changed before Q capture")
    witness = launch(out, runner, model["om"], "benchmark", env, dsos, cli["alias"])
    e = c.read(folder / "execution.json")
    require(
        e["scope"] == REFERENCE
        and e["status"] == "PASS"
        and e["cpu_fallback"] is False
        and e["reference_capture"] is False
        and e["device"] == 0
        and all(
            e[k] is True
            for k in (
                "golden_match",
                "sync_completed",
                "poison_verified",
                "guards",
                "readonly",
                "repeat",
                "timed_log_stable",
            )
        )
        and e["verification_calls"] == 4
        and e["timing"]["repetitions"] == 30
        and e["timing"]["warmup"] == 5,
        "profiled Q replay failed",
    )
    for tag in ("actual-0", "actual-1", "benchmark-last", "postcheck"):
        require(
            c.sha(folder / ("y-" + tag + ".bin")) == files["golden.bin"]["sha256"], "profiled Q output drift"
        )
    node = model["selected"]["node"]
    stats = analyze(
        out,
        dict(witness, profiler=str(out / "profiler")),
        model,
        dict(modules={node: "layer0.q"}, node_types={node: model["op"]}),
    )
    (row,) = (r for r in stats["modules"] if r["module"] == "layer0.q")
    require(
        row["calls_observed"] == 30 and row["tasks"] == 30 and row["p50_task_sum_ms"] > 0,
        "no unique Q kernel per measured Execute",
    )
    ms = row["p50_task_sum_ms"]
    return dict(
        attribution=stats["scope"],
        kernel=row,
        timeline=stats["timeline"],
        op_summary=stats["op_summary"],
        kernel_floor_ratio=ms / FLOOR_MS,
        logical_gbps=BYTES / ms / 1e6,
        execution=c.identity(folder / "execution.json"),
    )
