"""Direct, uninstrumented N64 versus wide-output KV measurements using the first sweep's assets."""

import os
from pathlib import Path
import shutil

import kv_panel as kv
from kv_archive import winner
from experiment import Stages
from run_kv import measurements, select
from run_q import run_path
from fresh_run import environment
from fresh_dso import seal
from air import protobuf_classes


def run(out, args, state, old):
    runs = Path(os.environ["DFLASH_RUN_ROOT"]) / "runs"
    prior = run_path(runs, args.resume, "kv-bandwidth-") if args.resume else None
    stages = Stages(out, prior)

    def stage(name, key, action):
        state["phase"] = name
        return stages.run(name, key, action)

    gears = (16, 64) if args.gear == "both" else (int(args.gear),)
    cann = Path(os.environ["CANN_ROOT"]).resolve(strict=True)
    tools = {n: kv.c.identity(shutil.which(n)) for n in ("atc", "cmake", "msopgen")}
    env = environment(cann, tools)
    types = protobuf_classes((out / "cann-ge.desc").read_bytes())
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
    record = kv.c.identity(old / "result.json")
    state.update(
        parent_run=str(old),
        parent_result=record,
        control=kv.DIRECT,
        gears={},
        bandwidth_assumption_GB_s=204.8,
    )
    archives = {}
    for gear in gears:
        archives[gear] = stage(
            f"admit-c{gear}",
            dict(
                parent=record,
                index=kv.c.identity(old / "stages.json"),
                gear=gear,
                code=kv.c.sha(kv.HERE / "kv_archive.py"),
            ),
            lambda d: winner(d, old, gear, select),
        )
    pkgs = {}
    runtime_id = {
        p.name: kv.c.sha(p)
        for p in (
            kv.HERE / "run_kv.py",
            Path(__file__),
            kv.OPS / "draft_o_bandwidth/o_runtime.py",
            kv.OPS / "draft_gate_tile_nd/fresh_run.py",
        )
    }
    for mode in kv.WIDE:
        key = dict(
            mode=mode,
            tools=tools,
            header=kv.sha(kv.header(mode)),
            device=kv.sha(kv.device_source(mode).encode()),
            code=[
                kv.c.sha(kv.HERE / "kv_panel.py"),
                kv.c.sha(kv.HERE / "d_flash_kv_panel_host.cpp"),
                kv.c.sha(kv.gate.HERE / "panel_vector.h"),
                kv.c.sha(kv.OPS / "draft_a_core_scheduler_flow/op_host/scheduler_evidence.h"),
            ],
        )
        pkgs[mode] = stage("build-" + mode, key, lambda d: kv.build(d, cann, env, mode))
    cli = None
    for gear in gears:
        a = archives[gear]
        exe = a["runner"]
        kv.c.verify(exe)
        pkgs[kv.DIRECT] = a["package"]
        pkgs[kv.RAW] = a["source"]["pkg"]
        models = {kv.DIRECT: a["control"], kv.RAW: a["raw"]}
        cases = a["cases"]
        kv.c.verify_tree(cases)
        envs = {}
        for mode, pkg in pkgs.items():
            kv.c.verify_opp(pkg["opp"])
            envs[mode] = environment(cann, tools, pkg["opp"]["install"])

        def get_profiler():
            nonlocal cli
            if cli is None:
                from panel_profiler import bind_msprof

                cli = bind_msprof(out, cann, envs[kv.DIRECT])
            return cli

        for mode, width in kv.WIDE.items():
            models[mode] = stage(
                f"atc-c{gear}-" + mode,
                dict(
                    admit=stages.refs[f"admit-c{gear}"],
                    build=stages.refs["build-" + mode],
                    atc=tools["atc"],
                    width=width,
                ),
                lambda d: kv.compile_model(
                    d / "compiled", types, pkgs[mode], envs[mode], argv, gear, mode, a["source"]
                ),
            )
        texts = []
        for mode, model in models.items():
            kv.c.verify(model["om"])
            pf = model["contract"]
            width = kv.WIDE.get(mode, 64)
            txt = pf["cube"]
            kv.require(
                model["selected"]["executable_sections"] not in texts and not model["prep_nodes"],
                "unchanged KV kernel or unexpected external activation packing",
            )
            texts.append(model["selected"]["executable_sections"])
            print(
                f"C{gear} {mode}: ATC {pf['block_dim']} blocks, SDK base "
                f"{txt['baseM']}/{txt['baseN']}/{txt['baseK']}, K{pf['resources']['tile_k']}, "
                f"output writebacks/block={kv.N//(8*width)}",
                flush=True,
            )
        m, k, n = kv.shape(gear)
        logical = kv.logical_bytes(gear)
        result = state["gears"][str(gear)] = dict(
            shape=[m, k, n], logical_bytes=logical, one_pass_us=logical / 204800
        )
        if not args.compile_only:
            dsos = {
                mode: seal(out / f"dso-c{gear}-{mode}", pkg, exe, envs[mode]) for mode, pkg in pkgs.items()
            }
            measurements(
                gear,
                result,
                stages,
                stage,
                runtime_id,
                cases,
                (exe, models, envs, dsos),
                None if args.no_profile else get_profiler,
                control=kv.DIRECT,
                candidates=tuple(kv.WIDE),
            )
    return (
        "COMPILED_NOT_EXECUTED"
        if args.compile_only
        else "MEASURED" if args.no_profile else "MEASURED_AND_PROFILED"
    )
