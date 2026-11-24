"""Single-FC experiments on the two actual full-Draft gears; one private OPP per dataflow."""

import argparse
import os
from pathlib import Path
import shutil
import sys
import traceback

import fc_panel as fc
from fc_runtime import runner, capture, execute, profile
from fc_source import read
from air import protobuf_classes
from experiment import Stages
from fresh_run import environment
from fresh_dso import seal
from run_q import run_path
from run_gate import pair_statistics, boundary_case

PAIRS = ((fc.CONTROL, fc.LOCAL), (fc.CONTROL, fc.GLOBAL), (fc.LOCAL, fc.GLOBAL))


def benchmark(out, m, call, files):
    rows = []
    for i in range(3):
        for pair in PAIRS[i:] + PAIRS[:i]:
            seq = pair[::-1] if (i + PAIRS.index(pair) + int(m == 64)) % 2 else pair
            tag = "__".join(pair)
            print(f"C{m} round {i+1}: " + " → ".join(seq), flush=True)
            for mode in seq:
                rows.append(dict(call(out / str(i) / tag / mode, mode, files, True, i), pair=tag))
    all_stats = {}
    for a, b in PAIRS:
        tag = a + "__" + b
        all_stats[tag] = pair_statistics(
            [r for r in rows if r["pair"] == tag],
            b,
            a,
            flops=2 * m * fc.K * fc.N,
            logical_bytes=fc.logical_bytes(m),
        )

    def passed(a, b):
        return all_stats[a + "__" + b]["decision"] == "GO"

    chosen = (
        fc.GLOBAL
        if passed(fc.CONTROL, fc.GLOBAL) and (not passed(fc.CONTROL, fc.LOCAL) or passed(fc.LOCAL, fc.GLOBAL))
        else fc.LOCAL if passed(fc.CONTROL, fc.LOCAL) else fc.CONTROL
    )
    return dict(runs=rows, statistics=all_stats, selected=chosen)


def run(out, args, state):
    base = Path(os.environ["DFLASH_RUN_ROOT"]) / "runs"
    prior = run_path(base, args.resume, "fc-bandwidth-") if args.resume else None
    old = run_path(base, args.full_run, "full-draft-panels-")
    stages = Stages(out, prior)

    def stage(name, key, action):
        state["phase"] = name
        return stages.run(name, key, action)

    def code(*paths):
        return {str(p): fc.c.sha(p) for p in paths}

    cann = Path(os.environ["CANN_ROOT"]).resolve(strict=True)
    tools = {k: fc.c.identity(shutil.which(k)) for k in ("atc", "msopgen", "cmake")}
    env = environment(cann, tools)
    descriptor = fc.c.identity(out / "cann-ge.desc")
    types = protobuf_classes(fc.c.verify(descriptor).read_bytes())
    gears = list(fc.GEARS) if args.gear == "both" else [int(args.gear)]
    src = stage(
        "source",
        dict(
            archive=fc.c.identity(old / "stages.json"),
            gears=gears,
            schema=descriptor["sha256"],
            source=fc.c.sha(fc.HERE / "fc_source.py"),
        ),
        lambda d: read(d, old, types, gears),
    )
    fc.c.verify_tree(src["archive"])
    fc.c.verify_tree(src["weights"])
    print("UFqlys checkpoint FC | shared INT8 NZ and GN scales | gears", gears, flush=True)
    exe = stage(
        "runner",
        dict(
            cmake=tools["cmake"],
            code=code(
                fc.HERE / "fc_runtime.py",
                fc.OPS / "draft_gate_raw_atc8/runner/runner.cpp",
                fc.c.QUANT / "test/runner_common.h",
            ),
        ),
        lambda d: runner(d, cann, env),
    )
    fc.c.verify(exe)
    pkgs = {fc.CONTROL: src["control_package"]}
    for mode in (fc.LOCAL, fc.GLOBAL):
        pkgs[mode] = stage(
            "build-" + mode,
            dict(
                tools=tools,
                cann=str(cann),
                mode=mode,
                host=fc.q.sha(fc.host_source().encode()),
                code=code(
                    fc.HERE / "fc_panel.py",
                    fc.HERE / "fc_spec.h",
                    fc.HERE / "op_kernel" / (fc.STEM + ".cpp"),
                    fc.q.gate.HERE / "panel_vector.h",
                ),
            ),
            lambda d: fc.build(d, cann, env, mode),
        )
    envs = {mode: environment(cann, tools, pkg["opp"]["install"]) for mode, pkg in pkgs.items()}
    argv = [
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
    for m in gears:
        models[m] = {}
        print(
            f"C{m} FC [{m},{fc.K},{fc.N}] single-pass reference {fc.logical_bytes(m)/204800:.2f} us at 204.8 GB/s",
            flush=True,
        )
        for mode, pkg in pkgs.items():
            fc.c.verify_opp(pkg["opp"])
            result = stage(
                f"atc-c{m}-" + mode,
                dict(
                    atc=tools["atc"],
                    pkg=pkg,
                    source=stages.refs["source"],
                    m=m,
                    mode=mode,
                    code=fc.c.sha(fc.HERE / "fc_panel.py"),
                ),
                lambda d: fc.compile_fc(d / "compiled", types, pkg, envs[mode], argv, m, mode, src),
            )
            fc.c.verify(result["om"])
            models[m][mode] = result
            proof = result["contract"]
            cube = proof["cube"]
            resources = proof["resources"]
            print(
                f"C{m} {mode}: blocks={proof['block_dim']} base={cube['baseM']}/{cube['baseN']}/{cube['baseK']} "
                f"steps={cube['stepKa']}/{cube['stepKb']} Iterate/N64={proof['iterates_per_n64']} UB/scratch={resources['user_ub']}/{resources['scratch']}",
                flush=True,
            )
        fc.require(
            len(
                {tuple(s["sha256"] for s in a["selected"]["executable_sections"]) for a in models[m].values()}
            )
            == len(fc.MODES),
            "coincident FC executables",
        )
    if len(gears) == 2:
        for mode in fc.MODES:
            fc.require(
                [i["sha256"] for i in models[16][mode]["packing"]["constants"]]
                == [i["sha256"] for i in models[64][mode]["packing"]["constants"]],
                "cross-gear weight/scale mismatch",
            )
    state.update(
        source=stages.refs["source"],
        roofline={
            str(m): dict(
                bytes=fc.logical_bytes(m),
                flops=2 * m * fc.K * fc.N,
                assumed_gbps=204.8,
                one_pass_us=fc.logical_bytes(m) / 204800,
                ddr_traffic="unmeasured",
            )
            for m in gears
        },
    )
    if args.compile_only:
        return "COMPILED_NOT_EXECUTED"
    state["phase"] = "DSO admission"
    dsos = {mode: seal(out / ("dso-" + mode), pkg, exe, envs[mode]) for mode, pkg in pkgs.items()}
    state["measurements"] = {}
    inputs = {}
    for m in gears:
        case = src["gears"][str(m)]
        cases = {}
        for label, key in (("features", "x"), ("stress", "stress")):
            fc.c.verify(case[key])
            ref = stage(
                f"reference-c{m}-" + label,
                dict(om=models[m][fc.CONTROL]["om"], x=case[key], runner=exe, archive=stages.refs["source"]),
                lambda d: capture(
                    d,
                    m,
                    exe,
                    models[m][fc.CONTROL],
                    envs[fc.CONTROL],
                    dsos[fc.CONTROL],
                    src["weights"],
                    case[key],
                    case["kernel"],
                ),
            )
            cases[label] = ref["files"]
            fc.c.verify_tree(ref["files"])
        cases["boundaries"] = stage(
            f"boundaries-c{m}",
            dict(
                source=stages.refs["source"], m=m, code=fc.c.sha(fc.OPS / "draft_gate_bandwidth/run_gate.py")
            ),
            lambda d: boundary_case(
                d / "data",
                cases["features"],
                shape=(m, fc.K, fc.N),
                pairs=(
                    (0, 12799),
                    (15, 16),
                    (127, 128),
                    (255, 256),
                    (1279, 1280),
                    (2559, 2560),
                    (6399, 6400),
                    (10239, 10240),
                ),
            ),
        )
        inputs[m] = cases["features"]

        def call(d, mode, files, timed, i):
            fc.c.verify_opp(pkgs[mode]["opp"])
            return execute(d, m, exe, models[m][mode], envs[mode], dsos[mode], files, timed, i)

        key = dict(
            cases=cases,
            runner=exe,
            models=[stages.refs[f"atc-c{m}-" + mode] for mode in fc.MODES],
            code=code(fc.HERE / "fc_runtime.py", fc.OPS / "draft_gate_tile_nd/fresh_run.py"),
        )

        def smoke(d):
            rows = []
            for mode in fc.MODES:
                for label, files in cases.items():
                    rows.append(dict(call(d / mode / label, mode, files, False, -1), case=label))
                print(
                    f"Bitwise + guards PASS C{m} {mode}: {m*fc.N:,} outputs, three inputs, two poisons each",
                    flush=True,
                )
            return rows

        stage(f"smoke-c{m}", key, smoke)
        result = stage(
            f"benchmark-c{m}",
            dict(
                key,
                smoke=stages.refs[f"smoke-c{m}"],
                protocol="3 interleaved pairs x 3 rounds; 5+30",
                stats=fc.c.sha(fc.OPS / "draft_gate_bandwidth/run_gate.py"),
            ),
            lambda d: benchmark(d, m, call, cases["features"]),
        )
        selection = result["selected"]
        state["measurements"][str(m)] = dict(
            selected=selection, stats=result["statistics"], report=stages.refs[f"benchmark-c{m}"]
        )
        for a, b in PAIRS:
            t = result["statistics"][a + "__" + b]
            x, y = (t["variants"][v] for v in (a, b))
            print(
                f"C{m} FC Execute+Sync {a} → {b}, Median/P95 (ms; n=90/90): {x['median_ms']:.6f}/{x['p95_ms']:.6f} → "
                f"{y['median_ms']:.6f}/{y['p95_ms']:.6f}, {t['speedup']:.4f}x {t['decision']}",
                flush=True,
            )
        print(f"Selected C{m}: {selection}", flush=True)
    if args.no_profile:
        return "MEASURED"
    from panel_profiler import bind_msprof

    state["phase"] = "msprof discovery"
    cli = bind_msprof(out, cann, envs[fc.CONTROL])
    state["profiles"] = {}
    for m in gears:
        state["profiles"][str(m)] = {}
        for mode in fc.MODES:
            row = stage(
                f"profile-c{m}-" + mode,
                dict(
                    benchmark=stages.refs[f"benchmark-c{m}"],
                    cli=cli["file"],
                    flags=cli["flags"],
                    mode=mode,
                    analyzer=code(fc.HERE / "fc_runtime.py", fc.OPS / "draft_full_panels/module_profile.py"),
                ),
                lambda d: profile(d, m, exe, models[m][mode], envs[mode], dsos[mode], inputs[m], cli),
            )
            state["profiles"][str(m)][mode] = row
            print(
                f"C{m} {mode} device FC task Median/P95 {1000*row['kernel']['p50_task_sum_ms']:.2f}/{1000*row['kernel']['p95_task_sum_ms']:.2f} us, "
                f"{row['logical_gbps']:.2f} logical GB/s, {row['matrix_tflops']:.3f} GEMM TF/s, {row['floor_ratio']:.3f}x reference, {row['attribution']}",
                flush=True,
            )
            pipe = row["pipeline"]
            times = {
                k: round(v, 2)
                for k, v in pipe["medians"].items()
                if "time" in k.lower() and "us" in k.lower()
            }
            print(f"  matched counters {pipe['joined']}/30", times, flush=True)
    return "MEASURED_AND_PROFILED"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gear", choices=("16", "64", "both"), default="both")
    p.add_argument("--resume", help="reuse successful stages from an fc-bandwidth run")
    p.add_argument("--full-run", default="full-draft-panels-UFqlys")
    group = p.add_mutually_exclusive_group()
    for f in ("no-profile", "compile-only"):
        group.add_argument("--" + f, action="store_true")
    args = p.parse_args()
    out = Path(os.environ["DFLASH_OUTPUT"])
    state = dict(phase="setup", status="FAILED")
    try:
        state["status"] = run(out, args, state)
    except Exception as err:
        state["error"] = repr(err)
        traceback.print_exc()
    fc.c.write(out / "result.json", state)
    print(state["status"], "REPORT", out / "result.json", flush=True)
    return int(state["status"] == "FAILED")


if __name__ == "__main__":
    sys.exit(main())
