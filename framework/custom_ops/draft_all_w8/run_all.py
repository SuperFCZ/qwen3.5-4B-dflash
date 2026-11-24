"""Run the accepted MLP-only full Draft and its all-W8 counterpart on identical public inputs."""

import argparse
from dataclasses import asdict
import os
from pathlib import Path
import shutil
import sys
import traceback

from selection import HERE, OPS, FULL, LATENCY, SINGLE, c, require, selections
from archive import admit
from vendor import build, local_headers
from rewrite import export
from compile_all import audit, compile_candidate
from breakdown import summarize
from air import protobuf_classes
from experiment import Stages
from fresh_dso import seal
from fresh_run import environment
from full_replay import run_om, compare_rounds
from raw_protocol import verify_run
from run_q import run_path
from panel_profiler import profile_modules


def hashes(*paths):
    return {str(p): c.sha(p) for p in paths}


def run(out, args, result):
    runs = Path(os.environ["DFLASH_RUN_ROOT"]) / "runs"
    previous = run_path(runs, args.resume, "full-all-w8-") if args.resume else None
    st = Stages(out, previous)

    def stage(name, key, action):
        result["phase"] = name
        return st.run(name, key, action)

    gears = [16, 64] if args.gear == "both" else [int(args.gear)]
    cann = Path(os.environ["CANN_ROOT"]).resolve(strict=True)
    toolset = {n: c.identity(shutil.which(n)) for n in ("atc", "msopgen", "cmake")}
    base = environment(cann, toolset)
    descriptor = c.identity(out / "cann-ge.desc")
    types = protobuf_classes(c.verify(descriptor).read_bytes())
    pinned = {FULL: LATENCY, **{v.root: v.result for g in gears for v in SINGLE[g].values()}}
    input_index = {}
    for name, digest in pinned.items():
        file = runs / name / ("latency.json" if name == FULL else "result.json")
        require(c.sha(file) == digest, "changed source report: " + str(file))
        input_index[name] = c.identity(runs / name / "stages.json")
    arch = stage(
        "admit",
        dict(
            gears=gears,
            toolkit=str(cann),
            archives=input_index,
            code=hashes(HERE / "selection.py", HERE / "archive.py"),
        ),
        lambda d: admit(d, runs, cann, gears),
    )
    ctl = arch["baseline_package"]
    exe = arch["runner"]
    c.verify(exe)
    c.verify_opp(ctl["opp"])
    ctl_env = environment(cann, toolset, ctl["opp"]["install"])
    ctl_dsos = None
    measurements = {}
    profiles = {}
    plan_for_report = {}
    result.update(
        control=FULL,
        gears={},
        control_description="accepted Gate/Down full Draft",
        candidate_description="all 26 optimized W8 projections",
        source=st.refs["admit"],
    )
    for gear in gears:
        case = arch["cases"][str(gear)]
        graph = case["graph"]
        assets = case["assets"]
        specs = selections(gear)
        key = dict(
            control=graph["optimized"],
            schema=descriptor["sha256"],
            selected={name: asdict(s) for name, s in specs.items()},
            bindings={k: v.get("layer0_constants") for k, v in assets.items()},
            code=hashes(HERE / "rewrite.py", OPS / "draft_fc_bandwidth/packed_panel.py"),
        )
        air = stage(f"air-c{gear}", key, lambda d: export(d, types, gear, case, arch["external"]))
        c.verify(air["air"])
        sources = {}
        for name, data in assets.items():
            group = {}
            for role in ("device", "host"):
                file = c.verify(data[role])
                group[role] = dict(
                    cpp=data[role], headers={n: c.identity(f) for n, f in local_headers(file).items()}
                )
            sources[name] = group
        pkg = stage(
            f"opp-c{gear}",
            dict(
                sources=sources,
                tools=toolset,
                code=hashes(HERE / "vendor.py", OPS / "draft_gate_tile_nd/package.py"),
            ),
            lambda d: build(d, cann, base, gear, assets),
        )
        c.verify_opp(pkg["opp"])
        env = environment(cann, toolset, pkg["opp"]["install"])
        compiled = stage(
            f"atc-c{gear}",
            dict(
                air=air["air"],
                build=st.refs[f"opp-c{gear}"],
                atc=toolset["atc"],
                compiler=hashes(
                    OPS / "draft_full_panels/compile_model.py", OPS / "draft_full_panels/ipc_check.py"
                ),
            ),
            lambda d: compile_candidate(d, air, cann, env),
        )
        model = stage(
            f"audit-c{gear}",
            dict(
                compiled=st.refs[f"atc-c{gear}"],
                archive=st.refs["admit"],
                graph=st.refs[f"air-c{gear}"],
                code=hashes(HERE / "compile_all.py"),
            ),
            lambda d: audit(d, gear, air, pkg, case, compiled),
        )
        old = case["model"]
        require(
            model["air"] == air["air"] and model["om"] == compiled["om"] and model["om"] != old["om"],
            "wrong measured models",
        )
        for item in (old["om"], model["om"], model["dump"]):
            c.verify(item)
        plan_for_report[str(gear)] = st.refs[f"audit-c{gear}"]
        result["gears"][str(gear)] = dict(
            control=old["om"], candidate=model["om"], inventory=st.refs[f"audit-c{gear}"]
        )
        if args.compile_only:
            continue
        if ctl_dsos is None:
            ctl_dsos = seal(out / "dso-control", ctl, exe, ctl_env)
        dsos = seal(out / f"dso-candidate-c{gear}", pkg, exe, env)
        io = case["io"]
        gold = case["reference"]
        c.verify_tree(io["files"])
        c.verify_tree(gold)
        require(
            set(gold) == {o["name"] for o in graph["abi"]["outputs"]} and len(gold) == 11,
            "incomplete Draft Golden",
        )

        def execute(d, mode, phase):
            mod, private, local, maps = (
                (old, ctl, ctl_env, ctl_dsos) if mode == "baseline" else (model, pkg, env, dsos)
            )
            c.verify_opp(private["opp"])
            return run_om(d, graph["abi"], io, mod, exe, local, maps, phase, gold)

        runtime_key = dict(
            models=[old["om"], model["om"]],
            audit=st.refs[f"audit-c{gear}"],
            runner=exe,
            inputs=io["files"],
            reference=gold,
            code=hashes(OPS / "draft_full_panels/full_replay.py", OPS / "draft_full_panels/full_launch.py"),
        )
        sm = stage(
            f"smoke-c{gear}",
            runtime_key,
            lambda d: {m: execute(d / m, m, "smoke") for m in ("baseline", "optimized")},
        )
        for row in sm.values():
            require(
                not row["profiled"]
                and row["launch_authorized"]
                and verify_run(row, {"golden": gold}, 0, False) == row["validation"],
                "different smoke",
            )
        print(
            f"C{gear} full smoke PASS: 15 token IDs and ten complete KV buffers matched in both OMs",
            flush=True,
        )

        def measure(d):
            rounds = []
            for i in range(3):
                pair = {}
                for mode in (
                    ("optimized", "baseline") if (i + int(gear == 64)) % 2 else ("baseline", "optimized")
                ):
                    print(
                        f"C{gear} round {i+1}: "
                        + {"optimized": "all 26 W8", "baseline": "Gate/Down control"}[mode],
                        flush=True,
                    )
                    pair[mode] = execute(d / f"round-{i+1}" / mode, mode, "benchmark")
                rounds.append(pair)
            return dict(rounds=rounds, statistics=compare_rounds(rounds))

        measured = stage(f"benchmark-c{gear}", dict(runtime_key, smoke=st.refs[f"smoke-c{gear}"]), measure)
        for pair in measured["rounds"]:
            for row in pair.values():
                require(
                    row["launch_authorized"]
                    and row["validation"] == verify_run(row, {"golden": gold}, 0, True),
                    "different complete-model timing/output",
                )
        formal = compare_rounds(measured["rounds"])
        require(formal == measured["statistics"], "archived formal samples changed")
        measurements[str(gear)] = formal
        result["gears"][str(gear)]["benchmark"] = st.refs[f"benchmark-c{gear}"]
        b, o = (formal["variants"][m] for m in ("baseline", "optimized"))
        print(
            f"C{gear} Full Draft Execute+Sync: Gate/Down {b['median_ms']:.6f}/{b['p95_ms']:.6f} → "
            f"all W8 {o['median_ms']:.6f}/{o['p95_ms']:.6f} ms, Median/P95 n=90/90, {formal['ratio']:.4f}x",
            flush=True,
        )
        profiles[gear] = dict(
            graph=dict(air, verified_projections=model["inventory"]),
            io=io,
            reference=gold,
            model=model,
            runtime=(env, pkg, dsos),
        )
        c.write(out / "latency.json", measurements)
    if args.compile_only:
        return "COMPILED_AND_AUDITED"
    if args.no_profile:
        return "MEASURED"
    result["phase"] = "profile"
    profile_modules(st, cann, ctl_env, ctl, exe, ctl_dsos, profiles, concise=True)
    for gear, case in profiles.items():
        name = f"components-c{gear}"
        mod = st.refs[f"modules-c{gear}"]
        summary = stage(
            name,
            dict(
                profile=mod,
                formal=st.refs[f"benchmark-c{gear}"],
                inventory=plan_for_report[str(gear)],
                code=c.sha(HERE / "breakdown.py"),
            ),
            lambda d: summarize(d, mod, gear, case["model"]["inventory"], measurements[str(gear)]),
        )
        c.verify_tree(summary["files"])
        result["gears"][str(gear)].update(
            breakdown=st.refs[name],
            attribution=summary["attribution"],
            components=summary["components"],
            layers=summary["layers"],
        )
        print("Summary", summary["files"]["summary.md"]["path"], flush=True)
    return "MEASURED_AND_PROFILED"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gear", choices=("16", "64", "both"), default="both")
    parser.add_argument("--resume", metavar="PRIOR_FULL_ALL_W8_RUN")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--compile-only", action="store_true")
    group.add_argument("--no-profile", action="store_true")
    args = parser.parse_args()
    out = Path(os.environ["DFLASH_OUTPUT"])
    result = dict(status="FAILED", phase="setup")
    try:
        result["status"] = run(out, args, result)
    except Exception as error:
        result["error"] = repr(error)
        (out / "traceback.txt").write_text(traceback.format_exc())
        traceback.print_exc()
    c.write(out / "result.json", result)
    print(result["status"], "REPORT", out / "result.json", flush=True)
    return int(result["status"] == "FAILED")


if __name__ == "__main__":
    sys.exit(main())
