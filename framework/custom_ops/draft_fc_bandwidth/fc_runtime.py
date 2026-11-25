"""Guarded raw FC device references, projection timings and separately collected task times."""

import csv
import shutil
from statistics import median
import numpy as np

from fc_panel import c, require, OPS, K, N, abi, logical_bytes
from layout import once
from package import command
from fresh_run import run_child
from full_replay import prepare_case, record_mismatches
from full_launch import launch

SCOPE = "supplied FC reference checked in process"


def runner(out, cann, env):
    src = out / "runner-src"
    shutil.copytree(OPS / "draft_gate_raw_atc8/runner", src)
    f = src / "runner.cpp"
    code = f.read_text()
    code = once(
        code,
        'if(!benchmark && std::string(argv[6])!="smoke")',
        'const bool capture=std::string(argv[6])=="capture";\n        if(!capture && !benchmark && std::string(argv[6])!="smoke")',
    )
    code = once(
        code,
        "inputs[0].shape!=std::vector<int64_t>({16,2560}) || outputs[0].shape!=std::vector<int64_t>({16,19456})",
        """!((inputs[0].shape==std::vector<int64_t>({16,12800}) && outputs[0].shape==std::vector<int64_t>({16,2560})) ||
             (inputs[0].shape==std::vector<int64_t>({64,12800}) && outputs[0].shape==std::vector<int64_t>({64,2560})))""",
    )
    code = code.replace("expected frozen Gate projection ABI", "unexpected C16/C64 FC ABI")
    code = once(code, "golden;for(const auto&t:outputs)", "golden;if(!capture)for(const auto&t:outputs)")
    code = once(code, "i<golden[0].size()", "!capture && i<golden[0].size()")
    code = once(code, "if(value!=golden[i])", "if(!capture && value!=golden[i])")
    code = once(code, "single Gate frozen Native Golden checked in process", SCOPE)
    code = once(
        code,
        r"\"golden_match\":true",
        r'\"reference_capture\":"<<(capture?"true":"false")<<",\"golden_match\":"<<(capture?"false":"true")<<"',
    )
    f.write_text(code)
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


def capture(d, m, runner, raw, env, dsos, weights, x, expected):
    require(
        raw["mode"] == "raw-nz-brcb"
        and raw["selected"]["executable_sections"] == expected["executable_sections"],
        "FC reference must use original machine text",
    )
    a = abi(m)
    app = d / "application"
    prepare_case(app, a, {"files": {"x": x}}, None)
    witness = launch(d, runner, raw["om"], "capture", env, dsos, None)
    e = c.read(app / "execution.json")
    snapshot = c.identity(app / "y-actual-0.bin")
    y = np.fromfile(snapshot["path"], "<f2")
    require(
        e["status"] == "PASS"
        and e["reference_capture"] is True
        and e["golden_match"] is False
        and e["device"] == 0
        and e["cpu_fallback"] is False
        and e["verification_calls"] == 2
        and e["timing"]["repetitions"] == 0
        and all(e[k] is True for k in ("guards", "readonly", "repeat", "poison_verified", "sync_completed"))
        and snapshot["sha256"] == c.sha(app / "y-actual-1.bin")
        and y.size == m * N
        and np.isfinite(y).all()
        and np.any(y),
        "invalid captured FC result",
    )
    return dict(
        files={**weights, "x.bin": x, "golden.bin": snapshot},
        execution=c.identity(app / "execution.json"),
        **witness,
    )


def execute(folder, m, runner, model, env, dsos, files, timed, i):
    try:
        return run_child(folder, runner, model, env, dsos, files, timed, SCOPE, i, abi=abi(m))
    except Exception:
        record_mismatches(folder, abi(m), {"y": files["golden.bin"]})
        raise


def pipeline_times(folder, summary, *, module="fc", profiler_folder=None):
    from module_profile import trace
    from hotspots import counters

    with (folder / "timed-tasks.csv").open(newline="") as handle:
        tasks = [
            dict(
                idx=int(r["idx"]),
                call=int(r["call"]),
                task_id=r["task_id"],
                stream=r["stream"],
                op=r["op"],
                kind=r["kind"],
                start=r["start"],
                model=None,
                label=module.upper(),
            )
            for r in csv.DictReader(handle)
            if r["module"] == module
        ]
    raw = folder / "profiler" if profiler_folder is None else profiler_folder
    tables, _ = trace.csv_tables(raw)
    require(
        len(tables) == 1 and c.identity(raw / tables[0]["file"]) == summary["op_summary"],
        "different FC counter source",
    )
    data = counters(tasks, tables[0])
    c.write(folder / "counters.json", data)
    values = {}
    if data["joined"] == 30:
        for field in data["columns"]:
            numbers = [x["metrics"][field] for x in data["unique_csv_links"]]
            if all(v is not None for v in numbers):
                values[field] = median(numbers)
    return dict(joined=data["joined"], medians=values, file=c.identity(folder / "counters.json"))


def profile(d, m, runner, model, env, dsos, files, cli):
    from module_profile import analyze

    app = d / "application"
    prepare_case(app, abi(m), {"files": {"x": files["x.bin"]}}, {"y": files["golden.bin"]})
    require(c.identity(cli["alias"]) == cli["file"], "changed bound msprof")
    witness = launch(d, runner, model["om"], "benchmark", env, dsos, cli["alias"])
    e = c.read(app / "execution.json")
    require(
        e["scope"] == SCOPE
        and e["status"] == "PASS"
        and e["device"] == 0
        and e["reference_capture"] is False
        and e["cpu_fallback"] is False
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
        and e["verification_calls"] == 4
        and (e["timing"]["warmup"], e["timing"]["repetitions"]) == (5, 30),
        "profiled FC runner failed",
    )
    for suf in ("actual-0", "actual-1", "benchmark-last", "postcheck"):
        require(
            c.sha(app / ("y-" + suf + ".bin")) == files["golden.bin"]["sha256"], "profiled FC output drift"
        )
    node = model["selected"]["node"]
    res = analyze(
        d,
        dict(witness, profiler=str(d / "profiler")),
        model,
        dict(modules={node: "fc"}, node_types={node: model["op"]}),
    )
    (row,) = (r for r in res["modules"] if r["module"] == "fc")
    require(
        row["calls_observed"] == row["tasks"] == 30 and row["p50_task_sum_ms"] > 0,
        "need one measured FC task in each of 30 calls",
    )
    ms = row["p50_task_sum_ms"]
    nbytes = logical_bytes(m)
    return dict(
        pipeline=pipeline_times(d, res),
        attribution=res["scope"],
        kernel=row,
        timeline=res["timeline"],
        op_summary=res["op_summary"],
        execution=c.identity(app / "execution.json"),
        logical_gbps=nbytes / ms / 1e6,
        matrix_tflops=2 * m * K * N / ms / 1e9,
        floor_ratio=ms / (nbytes / 204.8e6),
    )
