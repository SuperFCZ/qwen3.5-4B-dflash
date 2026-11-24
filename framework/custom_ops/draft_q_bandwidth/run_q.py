"""Build, validate and time only the measured checkpoint's first C16 Q projection."""

import argparse
import os
from pathlib import Path
import shutil
import sys
import traceback

import q_panel as q
from q_panel import c, require
from q_source import read_source
from q_runtime import runner, compile_reference, capture, execute, profile_kernel
from q_timing import smoke, timing, display
from experiment import Stages
from fresh_dso import seal
from fresh_run import environment
from air import protobuf_classes


def run_path(base, name, prefix):
    path = (base / name).resolve(strict=True)
    require(
        path.parent == base.resolve() and path.name.startswith(prefix), "invalid run directory: " + str(path)
    )
    return path


def run(out, args, state):
    runs = Path(os.environ["DFLASH_RUN_ROOT"]) / "runs"
    prior = run_path(runs, args.resume, "q-bandwidth-") if args.resume else None
    if args.next_n:
        import q_next_pair

        return q_next_pair.run(out, args, state, run_path(runs, args.next_n, "q-bandwidth-"), prior)
    old = run_path(runs, args.full_run, "full-draft-panels-")
    stages = Stages(out, prior)

    def stage(name, contract, action):
        state["phase"] = name
        return stages.run(name, contract, action)

    def code(*paths):
        return {str(p): c.sha(p) for p in paths}

    cann = Path(os.environ["CANN_ROOT"]).resolve(strict=True)
    tools = {name: c.identity(shutil.which(name)) for name in ("atc", "msopgen", "cmake")}
    env = environment(cann, tools)
    desc = c.identity(out / "cann-ge.desc")
    types = protobuf_classes(c.verify(desc).read_bytes())
    src = stage(
        "source",
        dict(
            archive=c.identity(old / "stages.json"),
            schema_sha256=desc["sha256"],
            code=code(q.HERE / "q_source.py"),
        ),
        lambda d: read_source(d, old, types),
    )
    c.verify_tree(src["weights"])
    c.verify(src["embedding"])
    c.verify(src["reference_air"])
    c.verify_tree(src["source_stages"])
    print(
        f"Real C16 Layer0 Q {q.SHAPE}, {q.M*q.N:,} outputs | selected node {src['q_node']} | {q.BYTES:,} logical bytes",
        flush=True,
    )
    state.update(
        source=stages.refs["source"],
        weight=src["weights"]["w_nz.bin"],
        scale=src["weights"]["s_gn.bin"],
        roofline=dict(
            bytes=q.BYTES,
            flops=q.FLOPS,
            assumed_gbps=204.8,
            single_pass_ms=q.FLOOR_MS,
            w8_and_scale_bytes=sum(src["weights"][name]["bytes"] for name in ("w_nz.bin", "s_gn.bin")),
            sdk_l2_bytes=16 * 1024 * 1024,
            cache_state="repeat of one OM; actual DDR traffic unmeasured",
        ),
    )
    exe = stage(
        "runner",
        dict(
            cmake=tools["cmake"],
            code=code(
                q.HERE / "q_runtime.py",
                q.OPS / "draft_gate_raw_atc8/runner/runner.cpp",
                c.QUANT / "test/runner_common.h",
            ),
        ),
        lambda d: runner(d, cann, env),
    )
    c.verify(exe)
    packages = {q.CONTROL: src["control_package"]}
    envs = {}
    models = {}
    args_atc = [
        str(cann / "bin/atc"),
        "--mode=0",
        "--framework=1",
        "--model=INPUT",
        "--output=OUTPUT",
        "--soc_version=Ascend310P3",
        "--precision_mode=must_keep_origin_dtype",
        "--deterministic=0",
    ]
    for mode in q.MODES:
        if mode != q.CONTROL:
            packages[mode] = stage(
                "build-" + mode,
                dict(
                    tools=tools,
                    mode=mode,
                    code=code(
                        q.HERE / "q_panel.py",
                        q.HERE / "q_spec.h",
                        q.HERE / "op_kernel" / (q.STEM + ".cpp"),
                        q.gate.HERE / "op_host/d_flash_gate_panel.cpp",
                        q.gate.HERE / "panel_vector.h",
                    ),
                ),
                lambda d: q.build(d / "package", cann, env, mode),
            )
        pkg = packages[mode]
        c.verify_opp(pkg["opp"])
        envs[mode] = environment(cann, tools, pkg["opp"]["install"])
        models[mode] = stage(
            "atc-" + mode,
            dict(
                source=stages.refs["source"],
                pkg=pkg,
                atc=tools["atc"],
                mode=mode,
                packed_control_air=models[q.PANEL]["air"] if mode == q.PREFETCH else None,
                code=code(q.HERE / "q_panel.py"),
            ),
            lambda d: q.compile_q(
                d / "compiled",
                types,
                pkg,
                envs[mode],
                args_atc,
                mode,
                src,
                packed_control=models[q.PANEL] if mode == q.PREFETCH else None,
            ),
        )
        c.verify(models[mode]["om"])
        proof = models[mode]["contract"]
        print(
            f"{mode}: ATC blocks={proof['block_dim']} baseK={proof['cube']['baseK']} "
            f"Iterate/N64={proof['iterates_per_n64']} scratch={proof['resources']['scratch']}",
            flush=True,
        )
    first, second = (models[name] for name in (q.PANEL, q.PREFETCH))
    require(
        first["air"]["sha256"] == second["air"]["sha256"]
        and first["contract"]["cube_sha256"] == second["contract"]["cube_sha256"],
        "Q panel pair changed input bytes/graph or serialized Cube plan",
    )
    require(
        len({tuple(y["sha256"] for y in m["selected"]["executable_sections"]) for m in models.values()}) == 3,
        "duplicate Q machine text",
    )
    ref_model = stage(
        "reference-om",
        dict(source=stages.refs["source"], atc=tools["atc"], code=code(q.HERE / "q_runtime.py")),
        lambda d: compile_reference(d, src, envs[q.CONTROL], args_atc),
    )
    if args.compile_only:
        return "COMPILED_NOT_EXECUTED"
    state["phase"] = "DSO admission"
    dsos = {mode: seal(out / ("dso-" + mode), pkg, exe, envs[mode]) for mode, pkg in packages.items()}
    ref = stage(
        "device-reference",
        dict(om=ref_model["om"], embedding=src["embedding"], runner=exe, source=stages.refs["source"]),
        lambda d: capture(
            d,
            exe,
            ref_model,
            envs[q.CONTROL],
            dsos[q.CONTROL],
            src["weights"],
            src["embedding"],
            "SEALED_C16_TOKEN_DEVICE_PREFIX",
        ),
    )
    stress = stage(
        "dense-reference",
        dict(om=ref_model["om"], embedding=src["stress_embedding"], runner=exe, source=stages.refs["source"]),
        lambda d: capture(
            d,
            exe,
            ref_model,
            envs[q.CONTROL],
            dsos[q.CONTROL],
            src["weights"],
            src["stress_embedding"],
            "DISTINCT_CHECKPOINT_TOKEN_STRESS",
        ),
    )
    files = ref["files"]
    c.verify_tree(files)
    c.verify_tree(stress["files"])

    def call(d, mode, data, timed, i):
        c.verify_opp(packages[mode]["opp"])
        return execute(d, exe, models[mode], envs[mode], dsos[mode], data, timed, i)

    common = dict(
        reference=stages.refs["device-reference"],
        runner=exe,
        models=[stages.refs["atc-" + m] for m in q.MODES],
        code=code(
            q.HERE / "q_timing.py",
            q.HERE / "q_runtime.py",
            q.OPS / "draft_gate_bandwidth/run_gate.py",
            q.OPS / "draft_gate_tile_nd/fresh_run.py",
        ),
    )
    stage(
        "smoke",
        dict(common, dense=stages.refs["dense-reference"]),
        lambda d: smoke(d, call, files, stress["files"]),
    )
    measured = stage("benchmark", dict(common, smoke=stages.refs["smoke"]), lambda d: timing(d, call, files))
    display(measured)
    state.update(
        benchmark_report=stages.refs["benchmark"],
        selected=measured["selected"],
        statistics=measured["statistics"],
    )
    print(
        f"One-pass hardware reference {q.FLOOR_MS*1000:.2f} us at 204.8 GB/s (conditional reference).",
        flush=True,
    )
    if args.no_profile:
        return "MEASURED"
    from panel_profiler import bind_msprof

    state["phase"] = "msprof discovery"
    # Use the archived control environment for the CLI help. Actual profiled executions use the selected private OPP.
    cli = bind_msprof(out, cann, envs[q.CONTROL])
    profiles = {}
    for mode in dict.fromkeys((q.CONTROL, measured["selected"])):
        row = stage(
            "profile-" + mode,
            dict(
                benchmark=stages.refs["benchmark"],
                cli=cli["file"],
                flags=cli["flags"],
                mode=mode,
                code=code(
                    q.HERE / "q_runtime.py",
                    q.OPS / "draft_full_panels/module_profile.py",
                    q.OPS / "draft_full_panels/full_launch.py",
                ),
            ),
            lambda d: profile_kernel(d, exe, models[mode], envs[mode], dsos[mode], files, cli),
        )
        profiles[mode] = row
        print(
            f"{mode} device Q task: Median={1000*row['kernel']['p50_task_sum_ms']:.2f} us "
            f"P95={1000*row['kernel']['p95_task_sum_ms']:.2f} us  {row['logical_gbps']:.2f} logical GB/s  "
            f"{row['kernel_floor_ratio']:.3f}x floor  {row['attribution']}",
            flush=True,
        )
    state["profiles"] = profiles
    return "MEASURED_AND_PROFILED"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--full-run", default="full-draft-panels-UFqlys")
    p.add_argument(
        "--next-n",
        metavar="MEASURED_Q_RUN",
        help="compare cross-N preparation against the measured single-raw Q",
    )
    p.add_argument("--resume", help="reuse passed stages from a q-bandwidth run")
    p.add_argument("--compile-only", action="store_true")
    p.add_argument("--no-profile", action="store_true")
    args = p.parse_args()
    out = Path(os.environ["DFLASH_OUTPUT"])
    state = dict(status="FAILED", phase="setup")
    try:
        state["status"] = run(out, args, state)
    except Exception as e:
        state["error"] = repr(e)
        traceback.print_exc()
    c.write(out / "result.json", state)
    print(state["status"], "REPORT", out / "result.json", flush=True)
    return int(state["status"] == "FAILED")


if __name__ == "__main__":
    sys.exit(main())
