"""One server command, separate uninstrumented timing and profiling, immutable resumable outputs."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

from joint import HERE, OPS, build, c, require
from compile_model import compiled
from full_replay import inputs, run_om, build_runner, compare_rounds
from fresh_run import environment
from fresh_dso import seal


class Stages:
    def __init__(self, folder, prior):
        self.folder = folder
        self.refs = {}
        self.previous = {}
        if prior:
            self.previous = c.read(prior / "stages.json")

    def _read_receipt(self, name):
        ref = self.previous[name]
        row = c.read(c.verify(ref))
        require(row["name"] == name and row["status"] == "PASS", "invalid prior receipt: " + name)
        digest = hashlib.sha256(json.dumps(row["contract"], sort_keys=True).encode()).hexdigest()
        require(row["contract_sha256"] == digest, "invalid prior contract digest: " + name)
        return row

    def completed(self, name):
        require(name in self.previous, "profiling requires a completed stage: " + name)
        row = self._read_receipt(name)
        self.refs[name] = self.previous[name]
        return row

    def run(self, name, contract, action):
        key = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
        ref = self.previous.get(name)
        row = self._read_receipt(name) if ref else None
        if row is not None and row["contract_sha256"] == key:
            print("Reusing", name, flush=True)
        else:
            if row is not None:
                print("Rebuilding changed stage in new directory:", name, flush=True)
            print("Starting", name, flush=True)
            folder = self.folder / name
            folder.mkdir()
            value = action(folder)
            row = dict(name=name, status="PASS", contract_sha256=key, contract=contract, result=value)
            c.write(folder / "report.json", row)
            ref = c.identity(folder / "report.json")
        self.refs[name] = ref
        c.write(self.folder / "stages.json", self.refs)
        return row["result"]


def profile_completed(stages, prior, cann, gears):
    from panel_profiler import profile_modules
    from profile_archive import measured_cases

    pkg, runner, cases = measured_cases(stages, prior, cann, gears)
    env = environment(cann, {}, pkg["opp"]["install"])
    dsos = seal(stages.folder / "dso-binding", pkg, runner, env)
    profile_modules(stages, cann, env, pkg, runner, dsos, cases)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gear", choices=("16", "64", "both"), default="both")
    p.add_argument("--resume", help="name of a previous experiment run under DFLASH_RUN_ROOT/runs")
    p.add_argument(
        "--inputs-dir", type=Path, help="existing c16/ and c64/ directories of public ABI binaries"
    )
    group = p.add_mutually_exclusive_group()
    for flag in ("prepare-only", "compile-only", "no-profile"):
        group.add_argument("--" + flag, action="store_true")
    group.add_argument(
        "--profile-only",
        metavar="COMPLETED_RUN",
        help="profile the measured optimized OMs using sealed reports",
    )
    a = p.parse_args()
    if a.profile_only and (a.resume or a.inputs_dir):
        p.error(
            "--profile-only takes the measured run and its original inputs; omit --resume and --inputs-dir"
        )
    out = Path(os.environ["DFLASH_OUTPUT"])
    runs = Path(os.environ["DFLASH_RUN_ROOT"]) / "runs"
    prior = None
    if a.resume or a.profile_only:
        value = Path(a.profile_only or a.resume)
        prior = (runs / value).resolve(strict=True)
        require(
            prior.parent == runs.resolve() and prior.name.startswith("full-draft-panels-") and prior != out,
            "bad resume directory",
        )
    s = Stages(out, prior)
    if a.inputs_dir:
        a.inputs_dir = a.inputs_dir.resolve(strict=True)
    cann = Path(os.environ["CANN_ROOT"]).resolve(strict=True)
    gears = [16, 64] if a.gear == "both" else [int(a.gear)]
    if a.profile_only:
        profile_completed(s, prior, cann, gears)
        return
    tools = {n: c.identity(shutil.which(n)) for n in ("msopgen", "atc", "cmake")}
    base = environment(cann, tools)
    descriptor = out / "cann-ge.desc"
    draft = Path(os.environ["DFLASH_DRAFT_CHECKPOINT"]).resolve(strict=True)
    target = Path(os.environ["DFLASH_TARGET_CHECKPOINT"]).resolve(strict=True)
    print("Draft", draft, "Target", target, "gears", gears, flush=True)
    recipes = lambda *names: {str(n): c.sha(HERE / n) for n in names}

    def frontend(folder):
        argv = [
            os.environ["DFLASH_FRONTEND"],
            "-B",
            "-u",
            str(HERE / "model_frontend.py"),
            str(folder),
            str(descriptor),
            str(draft),
            str(target),
            "--gears",
            *map(str, gears),
        ]
        subprocess.run(argv, check=True, cwd=folder)
        return c.identity(folder / "graphs.json")

    index = target / "model.safetensors.index.json"
    ckpt = {
        str(f): c.sha(f)
        for f in [draft / "model.safetensors", draft / "config.json", index, target / "config.json"]
    }
    shards = c.read(index)["weight_map"]
    for f in {v for k, v in shards.items() if k.endswith(("embed_tokens.weight", "lm_head.weight"))}:
        ckpt[str(target / f)] = c.sha(target / f)
    graphs = c.read(
        c.verify(
            s.run(
                "checkpoint-graphs",
                dict(
                    checkpoint=ckpt,
                    gears=gears,
                    descriptor=c.sha(descriptor),
                    code=recipes("ge_frontend.py", "model_frontend.py"),
                    model_sources={
                        str(p): c.sha(p)
                        for p in [
                            *sorted((OPS.parents[1] / "models/dflash_v1").glob("*.py")),
                            *(
                                OPS.parents[1] / "framework/python/qwen35_dflash/ascend310p" / f
                                for f in ("incremental.py", "quant_factory.py", "draft_ac_export.py")
                            ),
                        ]
                    },
                ),
                frontend,
            )
        )
    )
    for path in {
        r["fp16"]["path"]: r["fp16"] for k, r in graphs["target"].items() if k in ("head", "embedding")
    }.values():
        c.verify(path)
    if a.prepare_only:
        return
    kernel_sources = [
        p
        for folder in (
            "draft_gate_bandwidth",
            "draft_down_bandwidth",
            "draft_gate_tile_nd",
            "draft_gate_scale_broadcast",
            "draft_quant/op_kernel",
            "draft_quant/op_host",
        )
        for p in (OPS / folder).rglob("*")
        if p.is_file() and p.suffix in (".py", ".h", ".cpp", ".json") and "test" not in p.parts
    ]
    pkg = s.run(
        "private-opp",
        dict(
            tools=tools,
            cann=str(cann),
            code=recipes("joint.py"),
            kernel_sources={str(p): c.sha(p) for p in sorted(kernel_sources)},
        ),
        lambda d: build(d / "package", cann, base),
    )
    c.verify_opp(pkg["opp"])
    c.verify_tree(pkg["sources"])
    env = environment(cann, tools, pkg["opp"]["install"])
    runner = s.run(
        "runner",
        dict(
            code=recipes("full_replay.py"),
            source=c.sha(OPS / "draft_a_core_scheduler_flow/om/runner.cpp"),
            header=c.sha(c.QUANT / "test/runner_common.h"),
            cmake=tools["cmake"],
        ),
        lambda d: build_runner(d, cann, base),
    )
    c.verify(runner)
    models = {}
    cases = {}
    for gear in gears:
        g = graphs["gears"][str(gear)]
        models[gear] = {}
        for mode in ("baseline", "optimized"):
            case = models[gear]
            case[mode] = s.run(
                f"atc-c{gear}-{mode}",
                dict(air=g[mode], pkg=s.refs["private-opp"], code=recipes("compile_model.py")),
                lambda d: compiled(d, gear, mode, g, pkg, cann, env, case.get("baseline")),
            )
            c.verify(case[mode]["om"])
    if a.compile_only:
        return
    # Fresh pre-launch library binding for this invocation (including restart); never admit an unknown mapping.
    dsos = seal(out / "dso-binding", pkg, runner, env)

    def per_run(d, gear, mode, phase, ref=None, profiler=None):
        c.verify_opp(pkg["opp"])
        return run_om(
            d,
            graphs["gears"][str(gear)]["abi"],
            cases[gear],
            models[gear][mode],
            runner,
            env,
            dsos,
            phase,
            ref,
            profiler,
        )

    def runkey(gear):
        return dict(
            io=s.refs[f"inputs-c{gear}"],
            runner=runner,
            models=[s.refs[f"atc-c{gear}-{m}"] for m in ("baseline", "optimized")],
            code=recipes("full_replay.py", "experiment.py"),
        )

    references = {}
    measurements = {}
    for gear in gears:
        g = graphs["gears"][str(gear)]
        replay = (
            {
                o["name"]: c.identity(a.inputs_dir / f"c{gear}" / (o["name"] + ".bin"))
                for o in g["abi"]["inputs"]
            }
            if a.inputs_dir
            else None
        )
        cases[gear] = s.run(
            f"inputs-c{gear}",
            dict(abi=g["abi"], seed=310300 + gear, replay=replay, code=recipes("full_replay.py")),
            lambda d: inputs(d, gear, g["abi"], a.inputs_dir),
        )
        for item in cases[gear]["files"].values():
            c.verify(item)
        refs = s.run(f"reference-c{gear}", runkey(gear), lambda d: per_run(d, gear, "baseline", "capture"))[
            "reference"
        ]
        references[gear] = refs

        def smoke(d):
            return {m: per_run(d / m, gear, m, "smoke", refs) for m in ("baseline", "optimized")}

        key = dict(runkey(gear), reference=refs)
        s.run(f"smoke-c{gear}", key, smoke)

        def benchmark(d):
            rounds = []
            for i in range(3):
                round_ = {}
                for mode in (
                    ("optimized", "baseline") if (i + (gear == 64)) % 2 else ("baseline", "optimized")
                ):
                    print(f"C{gear} round {i+1}: {mode}", flush=True)
                    round_[mode] = per_run(d / f"round-{i+1}" / mode, gear, mode, "benchmark", refs)
                rounds.append(round_)
            return dict(statistics=compare_rounds(rounds), rounds=rounds)

        t = s.run(f"benchmark-c{gear}", dict(key, smoke=s.refs[f"smoke-c{gear}"]), benchmark)["statistics"]
        measurements[str(gear)] = t
        b, o = (t["variants"][m] for m in ("baseline", "optimized"))
        print(
            f"C{gear} Full Draft Execute+Sync  baseline {b['median_ms']:.4f}/{b['p95_ms']:.4f} ms  optimized {o['median_ms']:.4f}/{o['p95_ms']:.4f} ms  (Median/P95, 90 each, {t['ratio']:.4f}x)",
            flush=True,
        )
    c.write(out / "latency.json", measurements)
    if a.no_profile:
        return
    from panel_profiler import profile_modules

    profile_cases = {
        gear: dict(
            graph=graphs["gears"][str(gear)],
            io=cases[gear],
            reference=references[gear],
            model=models[gear]["optimized"],
        )
        for gear in gears
    }
    profile_modules(s, cann, env, pkg, runner, dsos, profile_cases)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        print("FAILED:", e, "| logs:", os.environ.get("DFLASH_OUTPUT"), file=sys.stderr)
        sys.exit(1)
