"""Time and separately profile the current C16 Layer0 Gate/Up and two cross-N64 schedules."""

import argparse
import os
from pathlib import Path
import shutil
import sys
import traceback

import gate_next as g
from gate_input import DEPLOYMENT, FULL, O, source
from experiment import Stages
from air import protobuf_classes
from fresh_run import environment
from fresh_dso import seal
from run_gate import boundary_case, pair_statistics, FLOPS, LOGICAL_BYTES
from run_q import run_path
from run_kv import select, numeric_rejection
import o_runtime
from fc_runtime import pipeline_times
from panel_profiler import bind_msprof


def benchmark(out, left, right, call, measured, reverse):
    rows = []
    for i in range(3):
        order = (right, left) if (i + int(reverse)) % 2 else (left, right)
        print(f"Round {i+1}: " + " → ".join(order), flush=True)
        for mode in order:
            rows.append(call(out / str(i) / mode, mode, measured, True, i))
    return dict(
        runs=rows, statistics=pair_statistics(rows, right, left, flops=FLOPS, logical_bytes=LOGICAL_BYTES)
    )


def run(out, args, state):
    runs = Path(os.environ["DFLASH_RUN_ROOT"]) / "runs"
    prior = run_path(runs, args.resume, "gate-next-n-") if args.resume else None
    phases = Stages(out, prior)

    def stage(name, contract, action):
        state["phase"] = name
        return phases.run(name, contract, action)

    cann = Path(os.environ["CANN_ROOT"]).resolve(strict=True)
    tools = {name: g.c.identity(shutil.which(name)) for name in ("atc", "msopgen", "cmake")}
    plain = environment(cann, tools)
    descriptor = out / "cann-ge.desc"
    types = protobuf_classes(descriptor.read_bytes())
    src = stage(
        "source",
        dict(
            indices={n: g.c.identity(runs / n / "stages.json") for n in (DEPLOYMENT, FULL, O.root)},
            schema=g.c.sha(descriptor),
            code=g.c.sha(g.HERE / "gate_input.py"),
        ),
        lambda d: source(d, runs, types),
    )
    for key in (
        "single_air",
        "reference_air",
        "weights",
        "replays",
        "packing",
        "full_receipts",
        "o_receipts",
    ):
        g.c.verify_tree(src[key])
    print(
        "C16 Layer0 Gate/Up | M16 K2560 N19456 | full-Draft Gate ELF", src["deployed"]["sha256"], flush=True
    )
    print("Timed graph: one Gate task; 304 N64 tiles, 38 per core. One-pass reference 250.44 us.", flush=True)
    state.update(
        source=phases.refs["source"],
        scope=g.ROLE,
        deployed_om=src["full_om"],
        shape=g.SHAPE,
        single_pass_logical_bytes=LOGICAL_BYTES,
        conditional_one_pass_us=LOGICAL_BYTES / 204800,
    )
    exe = stage(
        "runner",
        dict(
            abi=[src["reference_abi"], g.ABI],
            scope=g.SCOPE,
            cmake=tools["cmake"],
            code=[
                g.c.sha(g.OPS / "draft_o_bandwidth/o_runtime.py"),
                g.c.sha(g.OPS / "draft_gate_raw_atc8/runner/runner.cpp"),
                g.c.sha(g.c.QUANT / "test/runner_common.h"),
            ],
        ),
        lambda d: o_runtime.runner(
            d, cann, plain, src["reference_abi"], projection_abi=g.ABI, reference=g.SCOPE
        ),
    )
    g.c.verify(exe)
    pkgs = {g.CONTROL: src["control_package"]}
    for mode in g.CANDIDATES:
        dev = g.device_source(mode)
        pkgs[mode] = stage(
            "build-" + mode,
            dict(
                device=g.q.sha(dev.encode()),
                header=g.q.sha(g.gate.panel_header(2560, 2)),
                host=g.c.sha(g.gate.HERE / "op_host" / (g.gate.STEM + ".cpp")),
                helper=g.c.sha(g.gate.HERE / "panel_vector.h"),
                builder=g.c.sha(g.gate.HERE / "gate_panel.py"),
                tools=tools,
            ),
            lambda d: g.gate.build(d / "package", cann, plain, 2560, w16_banks=2, device_source=dev),
        )
    atc = [
        tools["atc"]["path"],
        "--mode=0",
        "--framework=1",
        "--model=INPUT",
        "--output=OUTPUT",
        "--soc_version=Ascend310P3",
        "--precision_mode=must_keep_origin_dtype",
        "--deterministic=0",
    ]
    models = {}
    envs = {}
    for mode, pkg in pkgs.items():
        g.c.verify_opp(pkg["opp"])
        envs[mode] = environment(cann, tools, pkg["opp"]["install"])
        models[mode] = stage(
            "atc-" + mode,
            dict(
                air=src["single_air"],
                source=phases.refs["source"],
                pkg=pkg,
                atc=tools["atc"],
                control=phases.refs.get("atc-" + g.CONTROL),
                code=g.c.sha(g.HERE / "gate_next.py"),
            ),
            lambda d: g.compile_single(d, pkg, envs[mode], atc, src, mode, models.get(g.CONTROL)),
        )
        model = models[mode]
        g.c.verify(model["om"])
        print(
            mode,
            "ATC blocks",
            model["contract"]["block_dim"],
            "scratch",
            model["contract"]["resources"]["scratch"],
            "Gate ELF",
            model["selected"]["sha256"],
            flush=True,
        )
    g.require(
        len(
            {
                tuple(s["sha256"] for s in model["selected"]["executable_sections"])
                for model in models.values()
            }
        )
        == len(models),
        "duplicate implementation in measured set",
    )
    raw_pkg = src["pkg"]
    g.c.verify_opp(raw_pkg["opp"])
    raw_env = environment(cann, tools, raw_pkg["opp"]["install"])

    def check_ref(plans):
        g.require(
            plans[g.SHAPE_KEY]["tiling_sha256"] == src["source_plans"][g.SHAPE_KEY]["tiling_sha256"],
            "reference must use original Gate tiling",
        )

    ref = stage(
        "reference-om",
        dict(
            source=phases.refs["source"],
            atc=tools["atc"],
            code=g.c.sha(g.OPS / "draft_o_bandwidth/o_runtime.py"),
        ),
        lambda d: o_runtime.compile_prefix(d, src, raw_env, atc, validate_site=check_ref),
    )
    if args.compile_only:
        return "COMPILED_NOT_EXECUTED"
    dsos = {mode: seal(out / ("dso-" + mode), p, exe, envs[mode]) for mode, p in pkgs.items()}
    raw_maps = seal(out / "dso-reference", raw_pkg, exe, raw_env)
    cases = {}
    for label in ("measured", "stress"):
        capture = stage(
            "capture-" + label,
            dict(source=phases.refs["source"], runner=exe, reference=ref["om"], inputs=src["replays"][label]),
            lambda d: o_runtime.capture(d, exe, ref, raw_env, raw_maps, src, label, input_shape=g.SHAPE[:2]),
        )
        cases[label] = capture["files"]
    cases["boundaries"] = stage(
        "boundaries",
        dict(weights=src["weights"], shape=g.SHAPE, code=g.c.sha(g.gate.HERE / "run_gate.py")),
        lambda d: boundary_case(d / "inputs", cases["measured"], shape=g.SHAPE),
    )
    g.c.verify_tree(cases)

    def call(d, mode, files, timed, i):
        return o_runtime.execute(
            d, exe, models[mode], envs[mode], dsos[mode], files, timed, i, abi=g.ABI, reference=g.SCOPE
        )

    def smoke(d, mode):
        result = {label: call(d / label, mode, files, False, -1) for label, files in cases.items()}
        print("Gate bitwise + guards PASS", mode, "(311,296 FP16 outputs; 3 inputs, 2 poisons)", flush=True)
        return result

    accepted = []
    state["rejected"] = {}
    for mode in models:
        name = "smoke-" + mode
        try:
            stage(
                name,
                dict(
                    runner=exe,
                    cases=cases,
                    atc=phases.refs["atc-" + mode],
                    protocol=g.c.sha(g.OPS / "draft_gate_tile_nd/fresh_run.py"),
                ),
                lambda d: smoke(d, mode),
            )
        except ValueError:
            failure = numeric_rejection(out / name) if mode != g.CONTROL else None
            if not failure:
                raise
            print("REJECTED", mode, failure["reason"], flush=True)
            state["rejected"][mode] = failure
            continue
        if mode != g.CONTROL:
            accepted.append(mode)
    g.require(accepted, "no candidate passed Gate correctness")
    state["pairs"] = {}

    def pair(a, b):
        label = a + "__" + b
        name = "benchmark-" + label
        data = stage(
            name,
            dict(
                smokes=[phases.refs["smoke-" + mode] for mode in (a, b)],
                inputs=cases["measured"],
                runner=exe,
                protocol=g.c.sha(Path(__file__)),
                stats=g.c.sha(g.gate.HERE / "run_gate.py"),
            ),
            lambda d: benchmark(d, a, b, call, cases["measured"], reverse=(b == g.EARLY and a == g.CONTROL)),
        )
        stat = pair_statistics(data["runs"], b, a, flops=FLOPS, logical_bytes=LOGICAL_BYTES)
        g.require(data["statistics"] == stat, "paired statistics differ")
        state["pairs"][label] = dict(report=phases.refs[name], statistics=stat)
        left, right = (stat["variants"][v] for v in (a, b))
        print(
            f"Gate Execute+Sync {a} → {b}: {left['median_ms']:.6f}/{left['p95_ms']:.6f} → "
            f"{right['median_ms']:.6f}/{right['p95_ms']:.6f} ms (Median/P95, 90/90), "
            f"{stat['speedup']:.4f}x {stat['decision']} | rounds "
            + str([round(r, 4) for r in stat["round_ratios"]]),
            flush=True,
        )
        return stat

    best = select(accepted, pair, g.CONTROL)
    state["selected"] = best
    print("Selected Gate/Up:", best, flush=True)
    g.c.write(out / "latency.json", {"selected": best, "pairs": state["pairs"]})
    if args.no_profile:
        return "MEASURED"
    state["phase"] = "msprof"
    cli = bind_msprof(out, cann, envs[g.CONTROL])
    state["profiles"] = {}

    def prof(d, mode):
        row = o_runtime.profile(
            d,
            exe,
            models[mode],
            envs[mode],
            dsos[mode],
            cases["measured"],
            cli,
            abi=g.ABI,
            role=g.ROLE,
            reference=g.SCOPE,
            logical_bytes=LOGICAL_BYTES,
        )
        row["pipeline"] = pipeline_times(d, row, module=g.ROLE)
        return row

    for mode in (g.CONTROL, *accepted):
        name = "profile-" + mode
        row = stage(
            name,
            dict(
                om=models[mode]["om"],
                runner=exe,
                inputs=cases["measured"],
                smoke=phases.refs["smoke-" + mode],
                cli=cli["file"],
                flags=cli["flags"],
                code=[
                    g.c.sha(g.OPS / "draft_o_bandwidth/o_runtime.py"),
                    g.c.sha(g.OPS / "draft_fc_bandwidth/fc_runtime.py"),
                    g.c.sha(g.OPS / "draft_full_panels/module_profile.py"),
                ],
            ),
            lambda d: prof(d, mode),
        )
        state["profiles"][mode] = dict(row, report=phases.refs[name])
        ker = row["kernel"]
        print(
            f"{mode} device Gate Median/P95 {ker['p50_task_sum_ms']*1000:.2f}/{ker['p95_task_sum_ms']*1000:.2f} us, "
            f"{row['logical_gbps']:.2f} logical GB/s, {row['floor_ratio']:.3f}x one-pass, {row['attribution']}",
            flush=True,
        )
        print(
            f"  matched counters {row['pipeline']['joined']}/30",
            {
                k: round(v, 2)
                for k, v in row["pipeline"]["medians"].items()
                if "time" in k.lower() or "utilization" in k.lower()
            },
            flush=True,
        )
    return "MEASURED_AND_PROFILED"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", metavar="PREVIOUS_GATE_NEXT_N_RUN")
    flags = parser.add_mutually_exclusive_group()
    flags.add_argument("--compile-only", action="store_true")
    flags.add_argument("--no-profile", action="store_true")
    out = Path(os.environ["DFLASH_OUTPUT"])
    state = dict(status="FAILED", phase="setup")
    try:
        state["status"] = run(out, parser.parse_args(), state)
    except Exception as e:
        state["error"] = repr(e)
        (out / "traceback.txt").write_text(traceback.format_exc())
        traceback.print_exc()
    g.c.write(out / "result.json", state)
    print(state["status"], "REPORT", out / "result.json", flush=True)
    return int(state["status"] == "FAILED")


if __name__ == "__main__":
    sys.exit(main())
