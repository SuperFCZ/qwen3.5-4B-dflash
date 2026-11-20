#!/usr/bin/env python3
"""Run one C16 Layer0 scale-expansion stage. Only manual smoke and benchmark use the device."""

import argparse
from contextlib import contextmanager
from decimal import Decimal
import fcntl
import json
import os
from pathlib import Path
import sys
import traceback

from scale_contracts import (
    CANDIDATE,
    CONTROL,
    FIRST,
    GATE8_ROOT,
    HERE,
    MODES,
    c,
    command,
    historical,
    host,
    layout_contract,
    read_baseline,
    require,
    sdk_files,
    sources,
)
from scale_build import validate_package
from scale_review import check_model, compile_candidate, run_cpu, validate_cpu
from w16_runtime import case_run, verify_runtime_row
from raw_protocol import stats

STAGES = ("source", "build", "atc", "review", "smoke", "benchmark", "compare")
DEVICE = ("smoke", "benchmark")
STATUS = {s: "SCALE_" + s.upper() + "_PASS" for s in STAGES}
STATUS.update(benchmark="SCALE_PAIRED_MEASURED", compare="SCALE_GATE_GO")
NO_GO = "SCALE_GATE_NO_GO"


def schedule(timed):
    return [(r, m) for r in range(3 if timed else 2) for m in (MODES[::-1] if r % 2 else MODES)]


def statistics(rows):
    require(
        [(x["round"], x["mode"]) for x in rows] == schedule(True) and len({x["folder"] for x in rows}) == 6,
        "invalid fresh A/B round inventory",
    )
    ordered = {}
    overall = {}
    for m in MODES:
        picked = [x for x in rows if x["mode"] == m]
        require(
            all(
                x["validation"]["timing"] is not None
                and x["profiled"] is False
                and x["launch_authorized"] is True
                and x["bit_mismatches"] == 0
                and x["atc_plan"]["block_dim"] == 8
                for x in picked
            ),
            "timing without 8-block numerical/safety evidence",
        )
        ordered[m] = [stats(x["validation"]["timing"]["samples_ms"], 30) for x in picked]
        overall[m] = stats([s for round_ in ordered[m] for s in round_["samples_ms"]], 90)
    base, new = overall[CONTROL], overall[CANDIDATE]
    ratios = [a["median_ms"] / b["median_ms"] for a, b in zip(ordered[CONTROL], ordered[CANDIDATE])]
    faster = Decimal(str(new["median_ms"])) <= Decimal("0.97") * Decimal(str(base["median_ms"]))
    tail = new["p95_ms"] > base["p95_ms"]
    return dict(
        rounds=ordered,
        variants=overall,
        round_ratios=ratios,
        median_ratio=base["median_ms"] / new["median_ms"],
        median_saving_ms=base["median_ms"] - new["median_ms"],
        p95_regression=tail,
        latency_reduction_at_least_3pct=faster,
        decision="GO" if faster and min(ratios) > 1 and not tail else "NO_GO",
        scope="ONE_C16_LAYER0_GATE_ONLY",
        historical_samples_used=False,
    )


def variants(src, raw, control, chain):
    return {CONTROL: raw, CANDIDATE: chain["build"]["candidate"]}, {
        CONTROL: control,
        CANDIDATE: chain["atc"]["candidate"],
    }


def activity(out, phase, src, raw, control, chain, r, *, launch=False):
    timed = phase == "benchmark"
    builds, models = variants(src, raw, control, chain)
    prior = {x["mode"]: x for x in chain.get("smoke", {}).get("runs", [])}
    if launch:
        require(
            chain["review"]["status"] == STATUS["review"] and (not timed or set(prior) == set(MODES)),
            "device launch requires reviewed resources/layout and timing requires both accepted Smoke variants",
        )
        r["runs"] = []
    else:
        require([(x["round"], x["mode"]) for x in r["runs"]] == schedule(timed), "missing/reordered run")
    allocations = set()
    for idx, (i, m) in enumerate(schedule(timed)):
        folder = out / (str(i) + "-" + m)
        if launch:
            r["active_case"] = [i, m]
            r["device_execution"] = "ATTEMPTED"
            if timed:
                r["benchmark"] = "ATTEMPTED"
            case_run(
                folder,
                src,
                builds[m],
                models[m],
                "16",
                FIRST,
                m,
                i,
                "scale-pair" if timed else "smoke",
                timed,
                r,
                prior.get(m),
            )
        row = r["runs"][idx]
        require(
            (row["gear"], row["layer"], row["round"], row["mode"], row["pair"])
            == ("16", FIRST, i, m, "scale-pair" if timed else "smoke"),
            "wrong run/case/mode",
        )
        verify_runtime_row(folder, row, src, builds[m], models[m], timed, prior.get(m))
        require(
            all(s["sha256"] == row["golden"]["y"]["sha256"] for s in row["outputs"]["y"]),
            "cross-variant/repeat output differs",
        )
        allocations.add((row["validation"]["workspace_bytes"], row["validation"]["model_weight_bytes"]))
        require(len(allocations) == 1, "OM allocations drift between variants/rounds")
    if launch:
        r["device_execution"] = "COMPLETED"
        r["benchmark"] = "MEASURED" if timed else "NOT_RUN"
        r.pop("active_case", None)
    return statistics(r["runs"]) if timed else None


def verify(phase, out, r, chain, ctx):
    binding, src, raw, control = ctx
    require(
        r["device_execution"] == ("COMPLETED" if phase in DEVICE else "NOT_RUN")
        and r["atc"] == ("COMPILED_NOT_EXECUTED" if phase == "atc" else "NOT_RUN")
        and r["benchmark"] == ("MEASURED" if phase == "benchmark" else "NOT_RUN"),
        "incorrect stage state",
    )
    if phase == "source":
        require(
            r["binding"] == binding
            and r["layout"] == layout_contract()
            and r["sdk"] == sdk_files(src["cann"]),
            "source admission drift",
        )
    elif phase == "build":
        worker = c.read(c.verify(r["worker"]))
        require(
            r["worker"]["path"] == str(out / "worker.json")
            and worker["status"] == "SCALE_PRIVATE_BUILD_RECORDED"
            and worker["candidate"] == r["candidate"]
            and not worker.get("error"),
            "mixed build worker",
        )
        require(
            r["candidate"]["opp"]["install"] == str(out / "package/opp"), "private build outside experiment"
        )
        validate_package(r["candidate"], raw, src)
    elif phase == "atc":
        require(
            c.verify(r["candidate"]["dump"]).parent == out / CANDIDATE
            and r["safety"] == check_model(r["candidate"], src, chain["build"]["candidate"], control),
            "ATC replay failed",
        )
    elif phase == "review":
        require(
            r["resource_evidence"] == chain["atc"]["safety"] and r["layout"] == layout_contract(),
            "review for another OM or algorithm",
        )
        validate_cpu(r["sdk_cpu"], out / "sdk-cpu", src, binding["frozen"]["s_gn.bin"])
    elif phase in DEVICE:
        require(
            activity(out, phase, src, raw, control, chain, r) == r["statistics"],
            "stored samples or checks drift",
        )
    else:
        actual = statistics(chain["benchmark"]["runs"])
        require(r["comparison"] == actual == chain["benchmark"]["statistics"], "paired decision drift")
        if actual["decision"] == "NO_GO":
            return NO_GO
    return STATUS[phase]


@contextmanager
def stage_dir(root, phase, archive):
    for p in (c.REPO, archive, Path("/usr/local/Ascend")):
        require(
            root != p and root not in p.parents and p not in root.parents,
            "output overlaps read-only evidence/source",
        )
    require(
        not any((p / "source/report.json").is_file() for p in root.parents),
        "output nested inside an existing experiment",
    )
    if phase == "source":
        root.mkdir(exist_ok=True, parents=True)
    fd = os.open(root, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if phase == "source":
            require(not any(root.iterdir()), "Source needs a fresh empty directory")
        for s in STAGES[: STAGES.index(phase)]:
            f = root / s / "report.json"
            require(
                f.is_file() and f.resolve() == f and c.read(f).get("status") == STATUS[s],
                "missing/blocked predecessor " + s,
            )
        out = root / phase
        out.mkdir()
        yield out
    finally:
        os.close(fd)


def run(args):
    root = args.root.resolve()
    archive = args.archive.resolve(strict=True)
    phase = args.phase
    require((phase in DEVICE) == args.allow_device, "only smoke/benchmark require and accept --allow-device")
    with stage_dir(root, phase, archive) as out:
        base = dict(
            schema=1,
            root=str(root),
            archive=str(archive),
            sources=sources(),
            scope="C16_LAYER0_GATE_SCALE_EXPERIMENT",
        )
        r = dict(
            **base,
            phase=phase,
            folder=str(out),
            status="BLOCKED",
            predecessors={},
            error=None,
            integrity_error=None,
            device_execution="NOT_RUN",
            benchmark="NOT_RUN",
            atc="NOT_RUN",
        )
        try:
            c.clean()
            ctx = read_baseline(archive)
            binding, src, raw, control = ctx
            chain = {}
            for s in STAGES[: STAGES.index(phase)]:
                path = root / s / "report.json"
                item = c.identity(path)
                p = c.read(c.verify(item))
                require(
                    all(p[k] == v for k, v in base.items())
                    and p["phase"] == s
                    and p["folder"] == str(path.parent)
                    and p["predecessors"] == r["predecessors"]
                    and not p["error"]
                    and not p["integrity_error"],
                    "mixed predecessor " + s,
                )
                historical.verify_tree(p, {})
                require(
                    verify(s, path.parent, p, chain, ctx) == p["status"] == STATUS[s],
                    "predecessor evidence changed: " + s,
                )
                c.verify(item)
                chain[s] = p
                r["predecessors"][s] = item
            if phase == "source":
                r.update(binding=binding, layout=layout_contract(), sdk=sdk_files(src["cann"]))
            elif phase == "build":
                dest = out / "package"
                prior = c.read(c.verify(src["host_source"]))["binding"]["prior"]
                env, _ = host.envs.for_prior(prior, dest / "opp", ready=False)
                req = out / "request.json"
                c.write(req, dict(source=r["predecessors"]["source"], sources=r["sources"], dest=str(dest)))
                try:
                    command(
                        [sys.executable, "-B", HERE / "scale_build.py", "worker", req],
                        out / "builder.log",
                        env,
                        cwd=out,
                    )
                except RuntimeError:
                    if (out / "worker.json").is_file():
                        raise RuntimeError(c.read(out / "worker.json").get("error"))
                    raise
                r["worker"] = c.identity(out / "worker.json")
                r["candidate"] = c.read(c.verify(r["worker"]))["candidate"]
            elif phase == "atc":
                r["atc"] = "ATTEMPTED"
                r["candidate"] = compile_candidate(out / CANDIDATE, src, chain["build"]["candidate"], control)
                r["safety"] = check_model(r["candidate"], src, chain["build"]["candidate"], control)
                r["atc"] = "COMPILED_NOT_EXECUTED"
            elif phase == "review":
                r.update(layout=layout_contract(), resource_evidence=chain["atc"]["safety"])
                r["sdk_cpu"] = run_cpu(out / "sdk-cpu", src, raw)
            elif phase in DEVICE:
                r["statistics"] = activity(out, phase, src, raw, control, chain, r, launch=True)
            else:
                r["comparison"] = statistics(chain["benchmark"]["runs"])
            r["status"] = verify(phase, out, r, chain, ctx)
        except (Exception, KeyboardInterrupt) as exc:
            r.update(status="BLOCKED", error=repr(exc))
            (out / "traceback.txt").write_text(traceback.format_exc())
            if phase in DEVICE:
                active = r.get("active_case")
                row = next(
                    (v for v in reversed(r.get("runs", [])) if [v.get("round"), v.get("mode")] == active), {}
                )
                r["failure_scope"] = dict(
                    case=active,
                    runner_gate_released=row.get("launch_authorized"),
                    sync="CHECK_RAW_EXECUTION_OR_UNCONFIRMED",
                )
        try:
            historical.verify_tree(r, {})
            require(base["sources"] == sources(), "sources changed during stage")
            c.clean()
        except Exception as exc:
            r.update(status="BLOCKED", integrity_error=repr(exc))
        tmp = out / "report.tmp"
        c.write(tmp, r)
        tmp.replace(out / "report.json")
        print(
            json.dumps(
                {
                    k: r.get(k)
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
        if phase == "compare" and "comparison" in r:
            x = r["comparison"]
            for m, res in x["variants"].items():
                print(m, "median_ms", res["median_ms"], "P95_ms", res["p95_ms"], "samples", res["count"])
            print(
                "round_ratios",
                x["round_ratios"],
                "median_ratio",
                x["median_ratio"],
                "decision",
                x["decision"],
            )
        print("REPORT", out / "report.json")
        return 0 if r["status"] == STATUS[phase] else 3 if r["status"] == NO_GO else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("phase", choices=STAGES)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--archive", type=Path, default=GATE8_ROOT)
    p.add_argument("--allow-device", action="store_true")
    try:
        sys.exit(run(p.parse_args()))
    except (OSError, ValueError) as exc:
        p.exit(1, "BLOCKED: " + str(exc) + "\n")
