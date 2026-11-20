#!/usr/bin/env python3
"""Run exactly one stage of the frozen Gate scale-transpose expansion."""

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import sys
import traceback

from evidence import (
    CANDIDATE,
    CONTROL,
    FIRST,
    GEARS,
    GATE8_ROOT,
    LAYERS,
    OLD,
    PILOT,
    bind_gear,
    c,
    context,
    historical,
    require,
    source_files,
)
from cpu_review import review_scales
from expansion_models import check_model, compile_model
from pairs import five_decision, get_assets, groups, layer_statistics, runtime
from routing import prepare, check_package
from scale_build import validate_package
from scale_contracts import resume, tree

STEPS = ("source", "build", "atc", "review", "smoke", "benchmark", "compare")
STAGES = (
    "pilot-lock",
    *(f"c{g}-{s}" for g in GEARS for s in STEPS),
    "full-draft-export",
    "full-draft-smoke",
    "full-draft-benchmark",
    "final-comparison",
)
DEVICE = {s for s in STAGES if s.endswith(("-smoke", "-benchmark"))}


def location(stage):
    if stage.startswith(("c16-", "c64-")):
        gear, step = stage.split("-")
        return Path("five-gate-" + gear) / step
    return Path(stage)


def status(stage, result=None):
    if stage.endswith("-compare"):
        return stage[:3].upper() + "_FIVE_GATE_" + result["decision"]
    if stage == "final-comparison":
        return "FULL_DRAFT_SCALE_" + result["decision"]
    return "GATE_SCALE_" + stage.replace("-", "_").upper() + "_PASS"


def green(stage, actual):
    return actual == status(stage, {"decision": "GO"})


@contextmanager
def stage_directory(root, stage):
    require(root.is_absolute() and root == root.resolve(), "noncanonical experiment root")
    for p in (c.REPO, GATE8_ROOT, OLD, PILOT, Path("/usr/local/Ascend")):
        require(
            p != root and p not in root.parents and root not in p.parents,
            "protected evidence path: " + str(p),
        )
    require(
        not any(
            (p / "source/report.json").is_file() or (p / "pilot-lock/report.json").exists()
            for p in root.parents
        ),
        "experiment nested inside historical evidence",
    )
    if stage == "pilot-lock":
        root.mkdir(parents=True, exist_ok=True)
    fd = os.open(root, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if stage == "pilot-lock":
            require(not any(root.iterdir()), "pilot-lock needs a new empty root")
        else:
            require((root / "pilot-lock/report.json").is_file(), "missing pilot-lock")
        out = root / location(stage)
        require(out == out.resolve() and root in out.parents, "stage output redirects outside root")
        out.mkdir(parents=True, exist_ok=False)
        yield out
    finally:
        os.close(fd)


def build_witness(ctx):
    raw, candidate = ctx["builds"][CONTROL], ctx["builds"][CANDIDATE]
    validate_package(candidate, raw, ctx["src"])
    for b in (raw, candidate):
        c.verify_opp(b["opp"])
        require(tree(Path(b["opp"]["install"])) == b["package_files"], "sealed package drift")
        resume.combined(b, ctx["src"]["cann"], ctx["src"]["extras"], c.read(resume.POLICY))
    return {
        k: dict(
            opp=b["opp"],
            kernel=b["kernel"],
            wrapper_sources=b["dynamic"],
            projection_runner=b["runner"],
            full_runner=b["om"],
            api=b["api"],
            dso_manifest=b["manifest"],
            environment=b["env"],
        )
        for k, b in ctx["builds"].items()
    }


def projection_atc(out, ctx, chain, r, gear):
    models = r["models"] = {}
    proof = r["evidence"] = {}
    for layer in LAYERS:
        r["active_case"] = [gear, layer]
        if gear == "64":
            # Own C64 case must have bound the same node constants and the same compiled ABI.
            predecessor = chain["c16-atc"]["models"][layer]
            a = chain["c16-source"]["binding"]["cases"][layer]
            b = chain["c64-source"]["binding"]["cases"][layer]
            require(
                a["single_node"] == b["single_node"]
                and a["raw_single_om"] == b["raw_single_om"]
                and a["case"]["golden"]["sha256"] != b["case"]["golden"]["sha256"]
                and a["case"]["inputs"]["x.bin"]["sha256"] != b["case"]["inputs"]["x.bin"]["sha256"],
                "single-OM reuse does not bind C64's own case",
            )
            models[layer] = predecessor
        elif layer == FIRST:
            models[layer] = ctx["pilot"]["atc"]["candidate"]
        else:
            r["atc"] = "ATTEMPTED"
            models[layer] = compile_model(out / layer, ctx, ctx["builds"][CANDIDATE], layer, full=False)
        proof[layer] = check_model(ctx, ctx["builds"][CANDIDATE], layer, models[layer], full=False)
    r["atc"] = "COMPILED_FOUR_VERIFIED_FIVE" if gear == "16" else "VERIFIED_FIVE_REUSED"
    r["active_case"] = None


def gear_result(chain, gear):
    bench = chain[f"c{gear}-benchmark"]
    stats = {}
    for layer in LAYERS:
        if (gear, layer) == ("16", FIRST):
            value = dict(
                chain["pilot-lock"]["lock"]["comparison"],
                scope=f"C16/{FIRST}",
                sample_origin="FROZEN_PILOT",
                historical_samples_used=True,
            )
        else:
            value = layer_statistics(
                [r for r in bench["runs"] if r["layer"] == layer], f"C{gear}/{layer}", "FRESH"
            )
        stats[layer] = value
    require(stats == bench["statistics"], "per-layer timing sample replay differs")
    return five_decision(gear, stats)


def full_result(chain):
    bench = chain["full-draft-benchmark"]
    stats = {
        g: layer_statistics([r for r in bench["runs"] if r["gear"] == g], f"C{g}/full", "FRESH")
        for g in GEARS
    }
    require(stats == bench["statistics"], "fresh full-OM statistics drift")
    passed = all(s["decision"] == "GO" for s in stats.values())
    return dict(
        scope="FULL_DRAFT_EXECUTE_SYNC",
        gears=stats,
        decision="GO" if passed else "NO_GO",
        default_change=False,
        projections={g: chain[f"c{g}-compare"]["result"]["decision"] for g in GEARS},
        pilot_benchmark=chain["pilot-lock"]["lock"]["reports"]["benchmark"],
        full_control_root=str(GATE8_ROOT),
        comparison="single-W Raw Brcb versus single-W Raw Transpose B16",
    )


def verify_header(root, stage, r, refs, sources):
    require(
        r["schema"] == 1
        and r["phase"] == stage
        and r["root"] == str(root)
        and r["folder"] == str(root / location(stage))
        and r["sources"] == sources
        and r["predecessors"] == refs
        and r["error"] is None
        and r["integrity_error"] is None
        and green(stage, r["status"]),
        "missing, blocked or misbound predecessor: " + stage,
    )
    require(
        r["device_execution"] == ("COMPLETED" if stage in DEVICE else "NOT_RUN")
        and r["benchmark"] == ("MEASURED" if stage.endswith("-benchmark") else "NOT_RUN"),
        "invalid predecessor execution status: " + stage,
    )


def load_chain(root, stage, ctx, sources):
    chain, refs, seen = {}, {}, {}
    for previous in STAGES[: STAGES.index(stage)]:
        f = root / location(previous) / "report.json"
        require(f == f.resolve(strict=True), "redirected predecessor: " + str(f))
        item = c.identity(f)
        r = c.read(c.verify(item))
        verify_header(root, previous, r, refs, sources)
        historical.verify_tree(r, seen)
        if previous == "pilot-lock":
            require(r["lock"] == ctx["lock"], "frozen Pilot lock drift")
        elif previous.endswith("-compare"):
            require(r["result"] == gear_result(chain, previous[1:3]), "decision differs from paired samples")
        c.verify(item)
        chain[previous], refs[previous] = r, item
    return chain, refs


def admit_runtime(ctx, chain, gear, timed):
    # Verify actual package, compiled graph and same-gear layout before releasing a fresh Runner process.
    build_witness(ctx)
    if gear:
        require(bind_gear(ctx, gear) == chain[f"c{gear}-source"]["binding"], "source contract drift")
        review = chain[f"c{gear}-review"]
        folder = Path(review["folder"])
        require(
            review["scale_words"] == review_scales(folder, ctx, chain, gear, execute=False),
            "per-layer layout proof changed",
        )
    for g, k in groups(gear):
        builds, models = get_assets(ctx, chain, g, k)
        if k == "full":
            export = chain["full-draft-export"]
            check_package(ctx, g, builds[CANDIDATE])
            stored = export["evidence"][g]
        else:
            stored = chain[f"c{g}-atc"]["evidence"][k]
        require(
            stored
            == check_model(
                ctx, builds[CANDIDATE], g if k == "full" else k, models[CANDIDATE], full=k == "full"
            ),
            "model, resource or constant evidence drift before launch",
        )
    if timed:
        s = chain[f"c{gear}-smoke" if gear else "full-draft-smoke"]
        runtime(Path(s["folder"]), ctx, chain, s, gear, False, launch=False)


def perform(out, stage, r, ctx, chain):
    if stage == "pilot-lock":
        r["lock"] = ctx["lock"]
        return
    if stage == "final-comparison":
        bench = chain["full-draft-benchmark"]
        require(
            runtime(Path(bench["folder"]), ctx, chain, bench, None, True, launch=False)
            == bench["statistics"],
            "full benchmark replay differs",
        )
        r["result"] = full_result(chain)
        return
    full = stage.startswith("full-draft-")
    gear = None if full else stage[1:3]
    step = stage.rsplit("-", 1)[-1]
    if step == "source":
        r["binding"] = bind_gear(ctx, gear)
    elif step == "build":
        r["build_action"] = "REVALIDATED_SEALED_PRIVATE_PACKAGES"
        r["builds"] = build_witness(ctx)
    elif step == "atc":
        projection_atc(out, ctx, chain, r, gear)
    elif step == "review":
        r["scale_words"] = review_scales(out, ctx, chain, gear, execute=True)
        r["models"] = {k: chain[f"c{gear}-atc"]["models"][k]["om"] for k in LAYERS}
    elif step == "export":
        r.update(packages={}, models={}, evidence={})
        for g in GEARS:
            r["active_case"] = [g, "full"]
            require(
                chain[f"c{g}-compare"]["result"] == gear_result(chain, g)
                and chain[f"c{g}-compare"]["result"]["decision"] == "GO",
                "five-Gate NO_GO: " + g,
            )
            require(bind_gear(ctx, g) == chain[f"c{g}-source"]["binding"], "full source admission drift")
            b = r["packages"][g] = prepare(out / g / "private", ctx, g)
            r["atc"] = "ATTEMPTED"
            m = r["models"][g] = compile_model(out / g / "atc", ctx, b, g, full=True)
            r["evidence"][g] = check_model(ctx, b, g, m, full=True)
            constants = r["evidence"][g]["constants"]
            original = chain[f"c{g}-source"]["binding"]
            require(
                constants["raw_partition"] == original["raw_weight_partition"]
                and constants["const_nodes"] == original["raw_const_nodes"],
                "compiled constants lost Source binding",
            )
            check_package(ctx, g, b)
        r.update(active_case=None, atc="COMPILED_TWO_RAW_CONTROL_OMS_REUSED")
    elif step in ("smoke", "benchmark"):
        timed = step == "benchmark"
        admit_runtime(ctx, chain, gear, timed)
        r["statistics"] = runtime(out, ctx, chain, r, gear, timed, launch=True)
        if gear == "16":
            r["layer0_pilot"] = ctx["lock"]["reports"][step]
    else:
        bench = chain[f"c{gear}-benchmark"]
        require(
            runtime(Path(bench["folder"]), ctx, chain, bench, gear, True, launch=False)
            == bench["statistics"],
            "projection benchmark replay differs",
        )
        r["result"] = gear_result(chain, gear)


def run(args):
    stage = args.stage
    require(
        (stage in DEVICE) == args.allow_device,
        "only a manual Smoke/Benchmark accepts and requires --allow-device",
    )
    root = args.root.absolute()
    with stage_directory(root, stage) as out:
        r = dict(
            schema=1,
            phase=stage,
            folder=str(out),
            root=str(root),
            predecessors={},
            status="BLOCKED",
            sources=source_files(),
            error=None,
            integrity_error=None,
            device_execution="NOT_RUN",
            atc="NOT_RUN",
            benchmark="NOT_RUN",
            active_case=None,
        )
        try:
            c.clean()
            ctx = context(replay=stage == "pilot-lock")
            chain, r["predecessors"] = load_chain(root, stage, ctx, r["sources"])
            perform(out, stage, r, ctx, chain)
            r["status"] = status(stage, r.get("result"))
        except (Exception, KeyboardInterrupt) as exc:
            r["error"] = repr(exc)
            (out / "traceback.txt").write_text(traceback.format_exc())
        try:
            historical.verify_tree(r, {})
            require(source_files() == r["sources"], "source set changed during stage")
            c.clean()
        except Exception as exc:
            r.update(status="BLOCKED", integrity_error=repr(exc))
        tmp = out / "report.tmp"
        c.write(tmp, r)
        tmp.replace(out / "report.json")
        print(
            json.dumps(
                {
                    k: r[k]
                    for k in (
                        "phase",
                        "status",
                        "active_case",
                        "atc",
                        "device_execution",
                        "benchmark",
                        "error",
                        "integrity_error",
                    )
                },
                indent=2,
            )
        )
        if "result" in r and r["status"] != "BLOCKED":
            projections = stage.endswith("-compare")
            for row in r["result"]["layers" if projections else "gears"].values():
                a, b = row["variants"][CONTROL], row["variants"][CANDIDATE]
                print(
                    f"{row['scope']} {row['sample_origin']}  median {a['median_ms']:.6f} -> {b['median_ms']:.6f} ms  "
                    f"P95 {a['p95_ms']:.6f} -> {b['p95_ms']:.6f} ms  "
                    f"n={a['count']}/{b['count']}  round ratios {row['round_ratios']}"
                )
            if projections:
                print("Sum of independent layer medians, ms", r["result"]["sum_of_layer_medians_ms"])
        print("REPORT", out / "report.json")
        return 0 if green(stage, r["status"]) else (3 if r["status"].endswith("_NO_GO") else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--allow-device", action="store_true")
    try:
        sys.exit(run(parser.parse_args()))
    except (OSError, ValueError) as error:
        parser.exit(1, "BLOCKED: " + str(error) + "\n")
