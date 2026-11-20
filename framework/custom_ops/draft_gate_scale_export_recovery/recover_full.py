#!/usr/bin/env python3
"""Continue the sealed five-Gate experiment, one explicitly selected full-OM stage at a time."""

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import sys
import traceback

from archive import ARCHIVE, HERE, archived, c, require
from evidence import GEARS, GATE8_ROOT, OLD, PILOT, bind_gear, context, historical, source_files
from expansion_models import compile_model
from full_proof import check_full
from pairs import runtime
from routing import prepare
import run_expansion as prior

STAGES = ("admit-c16", "full-draft-export", "full-draft-smoke", "full-draft-benchmark", "final-comparison")
DEVICE = {"full-draft-smoke", "full-draft-benchmark"}
PASS = {
    "admit-c16": "FULL_SCALE_ARCHIVED_C16_ADMITTED",
    "full-draft-export": "FULL_SCALE_EXPORT_RECOVERED",
    "full-draft-smoke": "FULL_SCALE_TOKEN_KV_MATCH",
    "full-draft-benchmark": "FULL_SCALE_PAIRED_MEASURED",
    "final-comparison": "FULL_SCALE_GO",
}


def sources():
    ours = [c.identity(p) for p in sorted(HERE.rglob("*")) if p.is_file() and "__pycache__" not in p.parts]
    return ours + source_files()


@contextmanager
def new_stage(root, phase):
    require(root == root.resolve() and root.is_absolute(), "use a canonical absolute experiment root")
    for p in (ARCHIVE, GATE8_ROOT, OLD, PILOT, c.REPO, Path("/usr/local/Ascend")):
        require(
            root != p and p not in root.parents and root not in p.parents,
            "output overlaps protected evidence",
        )
    require(
        not any(
            (p / d / "report.json").is_file()
            for p in root.parents
            for d in ("source", "pilot-lock", "admit-c16")
        ),
        "output nested in an existing experiment",
    )
    if phase == "admit-c16":
        root.mkdir(parents=True, exist_ok=True)
    fd = os.open(root, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if phase == "admit-c16":
            require(not any(root.iterdir()), "admission requires a fresh empty root")
        else:
            require((root / "admit-c16/report.json").is_file(), "missing C16 admission")
        out = root / phase
        require(out.resolve() == out, "redirected stage folder")
        out.mkdir()
        yield out
    finally:
        os.close(fd)


def history():
    ctx = context()
    ref, failed, _, _ = archived()
    historical.verify_tree(failed, {})
    chain, refs = prior.load_chain(ARCHIVE, "full-draft-export", ctx, source_files())
    require(failed["predecessors"] == refs, "different predecessors in failed export")
    require(
        all(
            chain[f"c{g}-compare"]["result"] == prior.gear_result(chain, g)
            and chain[f"c{g}-compare"]["result"]["decision"] == "GO"
            for g in GEARS
        ),
        "both five-Gate paired decisions must be GO",
    )
    return ctx, chain, dict(failed_export=ref, five_gate=refs), failed


def check_export(ctx, chain, report):
    require(
        set(report["models"]) == set(report["packages"]) == set(report["evidence"]) == set(GEARS),
        "incomplete controlled full export",
    )
    for g in GEARS:
        if g == "16":
            admit = chain["admit-c16"]
            require(
                report["models"][g] == admit["model"]
                and report["packages"][g] == admit["package"]
                and report["evidence"][g] == admit["evidence"],
                "substituted C16 after offline admission",
            )
        require(
            check_full(ctx, chain, g, report["packages"][g], report["models"][g]) == report["evidence"][g],
            "full model admission drift: C" + g,
        )


def load_prior(root, phase, chain, base):
    refs, seen = {}, {}
    for s in STAGES[: STAGES.index(phase)]:
        p = root / s / "report.json"
        require(p.resolve(strict=True) == p, "redirected predecessor " + str(p))
        item = c.identity(p)
        r = c.read(c.verify(item))
        require(
            all(r.get(k) == v for k, v in base.items())
            and r["phase"] == s
            and r["folder"] == str(p.parent)
            and r["predecessors"] == refs
            and r["status"] == PASS[s]
            and r["error"] is None
            and r["integrity_error"] is None
            and r["device_execution"] == ("COMPLETED" if s in DEVICE else "NOT_RUN")
            and r["benchmark"] == ("MEASURED" if s == "full-draft-benchmark" else "NOT_RUN")
            and r["atc"] == ("C64_COMPILED_C16_REUSED" if s == "full-draft-export" else "NOT_RUN"),
            "blocked, missing or misbound predecessor " + s,
        )
        historical.verify_tree(r, seen)
        c.verify(item)
        chain[s] = r
        refs[s] = item
    return refs


def perform(phase, out, ctx, chain, r, failed):
    if phase == "admit-c16":
        prior.build_witness(ctx)
        for g in GEARS:
            bench = chain[f"c{g}-benchmark"]
            require(
                runtime(Path(bench["folder"]), ctx, chain, bench, g, True, launch=False)
                == bench["statistics"],
                "archived NPU evidence failed replay: C" + g,
            )
        r["model"] = failed["models"]["16"]
        r["package"] = failed["packages"]["16"]
        r["active_case"] = ["16", "full"]
        r["evidence"] = check_full(ctx, chain, "16", r["package"], r["model"])
        r["active_case"] = None
    elif phase == "full-draft-export":
        admit = chain["admit-c16"]
        r.update(
            packages={"16": admit["package"]},
            models={"16": admit["model"]},
            evidence={"16": admit["evidence"]},
            active_case=["16", "full"],
        )
        require(
            check_full(ctx, chain, "16", admit["package"], admit["model"]) == admit["evidence"],
            "C16 no longer matches admission",
        )
        g = "64"
        r["active_case"] = [g, "full"]
        require(bind_gear(ctx, g) == chain["c64-source"]["binding"], "C64 source drift")
        r["packages"][g] = b = prepare(out / g / "private", ctx, g)
        r["atc"] = "ATTEMPTED"
        r["models"][g] = m = compile_model(out / g / "atc", ctx, b, g, full=True)
        r["evidence"][g] = check_full(ctx, chain, g, b, m)
        r.update(atc="C64_COMPILED_C16_REUSED", active_case=None)
    elif phase in DEVICE:
        timed = phase == "full-draft-benchmark"
        prior.build_witness(ctx)
        check_export(ctx, chain, chain["full-draft-export"])
        if timed:
            smoke = chain["full-draft-smoke"]
            runtime(Path(smoke["folder"]), ctx, chain, smoke, None, False, launch=False)
        r["statistics"] = runtime(out, ctx, chain, r, None, timed, launch=True)
    else:
        benchmark = chain["full-draft-benchmark"]
        require(
            runtime(Path(benchmark["folder"]), ctx, chain, benchmark, None, True, launch=False)
            == benchmark["statistics"],
            "full benchmark replay failed",
        )
        r["result"] = prior.full_result(chain)


def run(args):
    phase = args.stage
    require(
        (phase in DEVICE) == args.allow_device,
        "only manual smoke/benchmark require and accept --allow-device",
    )
    root = args.root.absolute()
    with new_stage(root, phase) as out:
        r = dict(
            schema=1,
            root=str(root),
            folder=str(out),
            phase=phase,
            status="BLOCKED",
            sources=sources(),
            predecessors={},
            error=None,
            integrity_error=None,
            atc="NOT_RUN",
            device_execution="NOT_RUN",
            benchmark="NOT_RUN",
            active_case=None,
        )
        try:
            c.clean()
            ctx, chain, r["history"], failed = history()
            base = {k: r[k] for k in ("schema", "root", "sources", "history")}
            r["predecessors"] = load_prior(root, phase, chain, base)
            perform(phase, out, ctx, chain, r, failed)
            r["status"] = (
                "FULL_SCALE_NO_GO"
                if phase == "final-comparison" and r["result"]["decision"] == "NO_GO"
                else PASS[phase]
            )
        except (Exception, KeyboardInterrupt) as exc:
            r["error"] = repr(exc)
            (out / "traceback.txt").write_text(traceback.format_exc())
        try:
            historical.verify_tree(r, {})
            require(r["sources"] == sources(), "source inventory changed during stage")
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
        if r["status"] == PASS["admit-c16"]:
            print("C16", r["model"]["om"]["sha256"], r["evidence"]["images"]["counts"])
            print(
                "Bound constants",
                len(r["evidence"]["graph"]["const_records"]),
                "generated",
                len(r["evidence"]["graph"]["generated_constants"]),
            )
        if "result" in r and r["status"] != "BLOCKED":
            for g, v in r["result"]["gears"].items():
                print(
                    "C" + g,
                    {k: (x["median_ms"], x["p95_ms"], x["count"]) for k, x in v["variants"].items()},
                    "round ratios",
                    v["round_ratios"],
                )
        print("REPORT", out / "report.json")
        return 0 if r["status"] == PASS[phase] else 3 if r["status"] == "FULL_SCALE_NO_GO" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--allow-device", action="store_true")
    try:
        sys.exit(run(parser.parse_args()))
    except (OSError, ValueError) as exc:
        parser.exit(1, "BLOCKED: " + str(exc) + "\n")
