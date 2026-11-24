"""One manual 310P3 run for the actual first C16 attention output projection."""

import argparse
import os
from pathlib import Path
import shutil
import sys
import traceback

import o_panel as o
from o_runtime import runner, compile_prefix, capture, execute, profile
from o_source import read_source
from experiment import Stages
from air import protobuf_classes
from fresh_dso import seal
from fresh_run import environment
from run_q import run_path
from run_gate import pair_statistics, boundary_case

MODES = (o.CONTROL, o.PANEL)


def benchmark(folder, call, files):
    rows = []
    for i in range(3):
        order = MODES[::-1] if i == 1 else MODES
        print(f"Round {i+1}: " + " → ".join(order), flush=True)
        for mode in order:
            rows.append(call(folder / str(i) / mode, mode, files, True, i))
    stats = pair_statistics(rows, o.PANEL, o.CONTROL, flops=o.FLOPS, logical_bytes=o.BYTES)
    return dict(runs=rows, statistics=stats, selected=o.PANEL if stats["decision"] == "GO" else o.CONTROL)


def run(out, args, state):
    runs = Path(os.environ["DFLASH_RUN_ROOT"]) / "runs"
    prior = run_path(runs, args.resume, "o-bandwidth-") if args.resume else None
    old = run_path(runs, args.full_run, "full-draft-panels-")
    stages = Stages(out, prior)

    def stage(name, key, action):
        state["phase"] = name
        return stages.run(name, key, action)

    def code(*files):
        return {str(p): o.c.sha(p) for p in files}

    cann = Path(os.environ["CANN_ROOT"]).resolve(strict=True)
    tools = {k: o.c.identity(shutil.which(k)) for k in ("atc", "cmake", "msopgen")}
    base = environment(cann, tools)
    types = protobuf_classes((out / "cann-ge.desc").read_bytes())
    src = stage(
        "source",
        dict(
            archive=o.c.identity(old / "stages.json"),
            code=code(o.HERE / "o_source.py"),
            schema=o.c.sha(out / "cann-ge.desc"),
        ),
        lambda d: read_source(d, old, types),
    )
    o.c.verify(src["reference_air"])
    o.c.verify_tree(src["weights"])
    o.c.verify_tree(src["archive"])
    print(
        f"C16 Layer0 O | {o.SHAPE} | {o.M*o.N:,} FP16 values | native Attention + O reference | {src['o_node']}",
        flush=True,
    )
    state.update(
        source=stages.refs["source"],
        assumed_ddr_gbps=204.8,
        one_pass_ms=o.FLOOR_MS,
        logical_bytes=o.BYTES,
        matrix_flops=o.FLOPS,
    )
    exe = stage(
        "runner",
        dict(
            abi=src["reference_abi"],
            code=code(
                o.HERE / "o_runtime.py",
                o.OPS / "draft_gate_raw_atc8/runner/runner.cpp",
                o.c.QUANT / "test/runner_common.h",
            ),
            cmake=tools["cmake"],
        ),
        lambda d: runner(d, cann, base, src["reference_abi"]),
    )
    o.c.verify(exe)
    packages = {o.CONTROL: src["pkg"]}
    packages[o.PANEL] = stage(
        "build-panel",
        dict(
            tools=tools,
            cann=str(cann),
            device_sha256=o.q.sha(o.device_source().encode()),
            host_sha256=o.q.sha(o.host_source().encode()),
            code=code(o.HERE / "o_panel.py", o.HERE / "o_spec.h", o.q.gate.HERE / "panel_vector.h"),
        ),
        lambda d: o.build(d, cann, base),
    )
    envs = {k: environment(cann, tools, pkg["opp"]["install"]) for k, pkg in packages.items()}
    atc = [
        str(cann / "bin/atc"),
        "--mode=0",
        "--framework=1",
        "--model=INPUT",
        "--output=OUTPUT",
        "--soc_version=Ascend310P3",
        "--precision_mode=must_keep_origin_dtype",
        "--deterministic=0",
    ]
    models = {}
    for mode, pkg in packages.items():
        o.c.verify_opp(pkg["opp"])
        models[mode] = stage(
            "atc-" + mode,
            dict(
                source=stages.refs["source"],
                atc=tools["atc"],
                pkg=pkg,
                mode=mode,
                code=code(o.HERE / "o_panel.py"),
            ),
            lambda d: o.compile_o(d / "compiled", types, pkg, envs[mode], atc, mode, src),
        )
        o.c.verify(models[mode]["om"])
        proof = models[mode]["contract"]
        cube = proof["cube"]
        print(
            f"{mode}: ATC blocks={proof['block_dim']}  baseK={cube['baseK']}  steps={cube['stepKa']}/{cube['stepKb']} "
            f"Iterate/N64={proof['iterates_per_n64']}  user UB/scratch={proof['resources']['user_ub']}/{proof['resources']['scratch']}",
            flush=True,
        )
    refmodel = stage(
        "reference-om",
        dict(source=stages.refs["source"], code=code(o.HERE / "o_runtime.py"), atc=tools["atc"]),
        lambda d: compile_prefix(d, src, envs[o.CONTROL], atc),
    )
    if args.compile_only:
        return "COMPILED_NOT_EXECUTED"
    state["phase"] = "DSO binding"
    dsos = {mode: seal(out / ("dso-" + mode), pkg, exe, envs[mode]) for mode, pkg in packages.items()}
    cases = {}
    for label in ("measured", "stress"):
        o.c.verify_tree(src["replays"][label])
        ref = stage(
            "capture-" + label,
            dict(
                om=refmodel["om"],
                source=stages.refs["source"],
                runner=exe,
                inputs=src["replays"][label],
                label=label,
            ),
            lambda d: capture(d, exe, refmodel, envs[o.CONTROL], dsos[o.CONTROL], src, label),
        )
        cases[label] = ref["files"]
        o.c.verify_tree(ref["files"])
    cases["boundaries"] = stage(
        "boundary-input",
        dict(
            weights=src["weights"],
            ref=stages.refs["capture-measured"],
            code=code(o.OPS / "draft_gate_bandwidth/run_gate.py"),
        ),
        lambda d: boundary_case(
            d / "data",
            cases["measured"],
            shape=o.SHAPE,
            pairs=(
                (0, 4095),
                (15, 16),
                (31, 32),
                (127, 128),
                (255, 256),
                (1023, 1024),
                (2047, 2048),
                (3967, 3968),
            ),
        ),
    )

    def call(d, mode, data, timed, i):
        o.c.verify_opp(packages[mode]["opp"])
        return execute(d, exe, models[mode], envs[mode], dsos[mode], data, timed, i)

    common = dict(
        cases=cases,
        runner=exe,
        models=[stages.refs["atc-" + m] for m in MODES],
        protocol=code(o.HERE / "o_runtime.py", o.OPS / "draft_gate_tile_nd/fresh_run.py"),
    )

    def smoke(d):
        rows = []
        for mode in MODES:
            for label, files in cases.items():
                rows.append(dict(call(d / mode / label, mode, files, False, -1), case=label))
            print(
                "Bitwise + guards PASS",
                mode,
                "(40,960 FP16 outputs, three inputs, two poisons each)",
                flush=True,
            )
        return rows

    stage("smoke", common, smoke)
    measured = stage(
        "benchmark",
        dict(
            common, smoke=stages.refs["smoke"], statistics=o.c.sha(o.OPS / "draft_gate_bandwidth/run_gate.py")
        ),
        lambda d: benchmark(d, call, cases["measured"]),
    )
    stat = measured["statistics"]
    a, b = (stat["variants"][mode] for mode in MODES)
    state.update(statistics=stat, selected=measured["selected"], benchmark_report=stages.refs["benchmark"])
    print(
        f"O Execute+Sync Median/P95 (ms, n=90/90): {a['median_ms']:.6f}/{a['p95_ms']:.6f} → "
        f"{b['median_ms']:.6f}/{b['p95_ms']:.6f}, {stat['speedup']:.4f}x, {stat['decision']}",
        flush=True,
    )
    if args.no_profile:
        return "MEASURED"
    from panel_profiler import bind_msprof

    state["phase"] = "msprof"
    cli = bind_msprof(out, cann, envs[o.CONTROL])
    state["profiles"] = {}
    for mode in MODES:
        row = stage(
            "profile-" + mode,
            dict(
                benchmark=stages.refs["benchmark"],
                mode=mode,
                cli=cli["file"],
                flags=cli["flags"],
                code=code(o.HERE / "o_runtime.py", o.OPS / "draft_full_panels/module_profile.py"),
            ),
            lambda d: profile(d, exe, models[mode], envs[mode], dsos[mode], cases["measured"], cli),
        )
        state["profiles"][mode] = row
        print(
            f"{mode} device O task Median/P95 {1000*row['kernel']['p50_task_sum_ms']:.2f}/{1000*row['kernel']['p95_task_sum_ms']:.2f} us, "
            f"{row['logical_gbps']:.2f} logical GB/s, {row['floor_ratio']:.3f}x one-pass 53.04 us, {row['attribution']}",
            flush=True,
        )
    return "MEASURED_AND_PROFILED"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--full-run", default="full-draft-panels-UFqlys")
    p.add_argument("--resume", help="reuse successful stages from a previous o-bandwidth run")
    flags = p.add_mutually_exclusive_group()
    for flag in ("compile-only", "no-profile"):
        flags.add_argument("--" + flag, action="store_true")
    args = p.parse_args()
    out = Path(os.environ["DFLASH_OUTPUT"])
    state = dict(status="FAILED", phase="setup")
    try:
        state["status"] = run(out, args, state)
    except Exception as e:
        state["error"] = repr(e)
        traceback.print_exc()
    o.c.write(out / "result.json", state)
    print(state["status"], "REPORT", out / "result.json", flush=True)
    return int(state["status"] == "FAILED")


if __name__ == "__main__":
    sys.exit(main())
