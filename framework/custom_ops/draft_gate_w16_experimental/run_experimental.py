#!/usr/bin/env python3
"""One manually selected, independently evidenced C16 Layer0 stage per process."""

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from contracts import (
    DOUBLE,
    FIRST,
    ISA_ROOT,
    LABELS,
    OVERLAP,
    SERIAL,
    SINGLE,
    archive,
    c,
    require,
    same,
    source_manifest,
)
from admission import cpu_witness, load_context, safety_case, verify_cpu
from compiler_audit import compiler_evidence
from w16_runtime import case_run, decision, order, paired, row_key, verify_runtime_row

STAGES = (
    "compiler-audit",
    "admit",
    "smoke-serial",
    "smoke-overlap",
    "stress",
    "schedule-benchmark",
    "net-benchmark",
    "compare",
)
DEVICE = set(STAGES[2:7])
BENCH = {"schedule-benchmark": "schedule", "net-benchmark": "net"}
PASS = {s: "W16_EXPERIMENTAL_" + s.upper().replace("-", "_") + "_PASS" for s in STAGES}
PASS["admit"] = "W16_EXPERIMENTAL_ADMISSION_PASS"
BLOCKED = {s: PASS[s][:-4] + "BLOCKED" for s in STAGES}
NO_GO = "W16_EXPERIMENTAL_NO_GO"
GEAR = "16"
STRESS_ROUNDS = 3


def runs_for(phase):
    if phase in BENCH:
        return order(GEAR, [FIRST], [BENCH[phase]])

    def key(tag, i, mode):
        return (GEAR, FIRST, tag, i, mode)

    if phase in ("smoke-serial", "smoke-overlap"):
        return [key("smoke", 0, SERIAL if phase == "smoke-serial" else OVERLAP)]
    if phase == "stress":
        # Refresh the single-W timing control; B and C already have their own untimed Smoke.
        return [key("control-smoke", 0, SINGLE)] + [
            key("stress", i, m) for i in range(STRESS_ROUNDS) for m in (DOUBLE[::-1] if i % 2 else DOUBLE)
        ]
    raise ValueError("no runner for phase " + phase)


def prior_run(stages, mode):
    if mode in DOUBLE:
        stage = "smoke-serial" if mode == SERIAL else "smoke-overlap"
        return stages[stage]["runs"][0] if stage in stages else None
    return stages["stress"]["runs"][0] if "stress" in stages else None


def placement(out, key):
    gear, layer, pair, i, mode = key
    return out / ("c" + gear) / layer / pair / str(i) / mode


def validate_runs(out, r, ctx, stages):
    phase = r["phase"]
    keys = runs_for(phase)
    timed = phase in BENCH
    require([row_key(row) for row in r["runs"]] == keys, "missing, reused, or misordered runner invocation")
    golden = ctx.src["cases"][GEAR][FIRST]["golden"]
    sizes = set()
    folders = set()
    for key, row in zip(keys, r["runs"]):
        mode = key[-1]
        folder = placement(out, key)
        require(
            folder.resolve(strict=True) == folder and folder not in folders and row["labels"] == LABELS,
            "redirected/reused/unlabelled experimental run",
        )
        folders.add(folder)
        reference = prior_run(stages, mode)
        verify_runtime_row(folder, row, ctx.src, ctx.built[mode], ctx.models[mode], timed, reference)
        require(
            all(
                x["sha256"] == golden["sha256"] and x["bytes"] == golden["bytes"] for x in row["outputs"]["y"]
            ),
            "Golden, cross-process repeat, or cross-variant bits differ",
        )
        sizes.add((row["validation"]["workspace_bytes"], row["validation"]["model_weight_bytes"]))
    for name in ("smoke-serial", "smoke-overlap", "stress"):
        for row in stages.get(name, {}).get("runs", []):
            sizes.add((row["validation"]["workspace_bytes"], row["validation"]["model_weight_bytes"]))
    require(len(sizes) == 1, "OM allocations drift across variants or repetitions")
    return paired(r["runs"], GEAR, FIRST, BENCH[phase]) if timed else None


def validate(root, r, ctx, stages, compilers):
    phase = r["phase"]
    out = root / phase
    if phase == "compiler-audit":
        same(r["compilation"], compilers, "compiler journal/inputs/options/intermediates drift")
    elif phase == "admit":
        verify_cpu(r["cpu_witness"], out, ctx)
        same(r["safety"], safety_case(ctx, compilers), "source-level admission no longer holds")
    elif phase in DEVICE:
        same(r["statistics"], validate_runs(out, r, ctx, stages), "timing/safety evidence changed")
    else:
        result = decision(stages["schedule-benchmark"]["statistics"], stages["net-benchmark"]["statistics"])
        same(r["comparison"], result, "preregistered comparison drift")
    same(r["device_execution"], "COMPLETED" if phase in DEVICE else "NOT_RUN", "invalid device scope")
    same(r["benchmark"], "MEASURED" if phase in BENCH else "NOT_RUN", "invalid timing scope")
    return NO_GO if phase == "compare" and r["comparison"]["decision"] != "GO" else PASS[phase]


def predecessors(root, phase, ctx, compilers, base):
    records = {}
    refs = {}
    for name in STAGES[: STAGES.index(phase)]:
        path = root / name / "report.json"
        require(path.resolve(strict=True) == path, "redirected predecessor report")
        item = c.identity(path)
        row = c.read(c.verify(item))
        require(
            row["phase"] == name
            and row["folder"] == str(path.parent)
            and not row["error"]
            and not row["integrity_error"]
            and row["status"] == PASS[name],
            "incomplete/blocked predecessor " + name,
        )
        for field, v in base.items():
            same(row[field], v, "mixed predecessor " + name + " " + field)
        same(row["predecessors"], refs, "mixed predecessor chain " + name)
        archive.historical.verify_tree(row, {})
        require(
            validate(root, row, ctx, records, compilers) == row["status"], "status without evidence: " + name
        )
        c.verify(item)
        refs[name] = item
        records[name] = row
    return records, refs


def execute(out, r, ctx, previous, allow_device):
    phase = r["phase"]
    timed = phase in BENCH
    require(
        phase in DEVICE and allow_device and previous["admit"]["status"] == PASS["admit"],
        "experimental device admission required",
    )
    r["runs"] = []
    for key in runs_for(phase):
        gear, layer, pair, i, mode = key
        r["active_case"] = list(key)
        folder = placement(out, key)
        r["device_execution"] = "PREPARING_RUN"
        if timed:
            r["benchmark"] = "ATTEMPTED"
        try:
            case_run(
                folder,
                ctx.src,
                ctx.built[mode],
                ctx.models[mode],
                gear,
                layer,
                mode,
                i,
                pair,
                timed,
                r,
                prior_run(previous, mode),
            )
        finally:
            if r["runs"] and r["runs"][-1]["folder"] == str(folder):
                row = r["runs"][-1]
                row["labels"] = list(LABELS)
                if row["launch_authorized"]:
                    r["device_execution"] = "LAUNCH_AUTHORIZED"
        # The child has exited. Replay that very run before launching another process.
        verify_runtime_row(
            folder,
            r["runs"][-1],
            ctx.src,
            ctx.built[mode],
            ctx.models[mode],
            timed,
            prior_run(previous, mode),
        )
        sizes = (
            r["runs"][-1]["validation"]["workspace_bytes"],
            r["runs"][-1]["validation"]["model_weight_bytes"],
        )
        for row in [*r["runs"][:-1], *previous.get("smoke-serial", {}).get("runs", [])]:
            require(
                sizes == (row["validation"]["workspace_bytes"], row["validation"]["model_weight_bytes"]),
                "OM allocation drift",
            )
    r["statistics"] = validate_runs(out, r, ctx, previous)
    r["device_execution"] = "COMPLETED"
    r["benchmark"] = "MEASURED" if timed else "NOT_RUN"
    r.pop("active_case", None)


def failure(out, r, exc):
    row = r.get("runs", [None])[-1] if r.get("runs") else None
    key = r.get("active_case")
    folder = placement(out, key) if key else None
    log = folder / "runner.log" if folder else None
    text = log.read_text(errors="replace").splitlines() if log and log.is_file() else []
    return dict(
        exception=type(exc).__name__,
        detail=str(exc),
        active_case=key,
        authorized=bool(row and folder and row.get("folder") == str(folder) and row.get("launch_authorized")),
        runner_messages=[line for line in text if line.startswith("FAIL:")],
        timeout=isinstance(exc, (TimeoutError, subprocess.TimeoutExpired)),
        sync="NOT_CONFIRMED"
        if not row or not row.get("execution") or row.get("folder") != str(folder)
        else c.read(c.verify(row["execution"])).get("sync_completed"),
        artifacts=[c.identity(p) for p in sorted(folder.iterdir()) if p.is_file() and not p.is_symlink()]
        if folder and folder.is_dir()
        else [],
    )


@contextmanager
def phase_directory(root, phase, protected):
    require(
        all(root != p and p not in root.parents and root not in p.parents for p in protected),
        "experimental root overlaps protected evidence",
    )
    if phase == "compiler-audit":
        root.mkdir(parents=True, exist_ok=True)
    fd = os.open(root, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if phase == "compiler-audit":
            require(not any(root.iterdir()), "compiler-audit needs an empty new root")
        else:
            for name in STAGES[: STAGES.index(phase)]:
                p = root / name / "report.json"
                require(
                    p.is_file() and c.read(p).get("status") == PASS[name],
                    "missing or blocked experimental stage: " + name,
                )
        dest = root / phase
        dest.mkdir(exist_ok=False)
        yield dest
    finally:
        os.close(fd)


def main(args):
    phase = args.phase
    root = args.root.resolve()
    if (phase in DEVICE) != args.allow_device:
        raise ValueError("only a separately requested device stage accepts and requires --allow-device")
    old = args.archive.resolve(strict=True)
    isa = args.isa_audit.resolve(strict=True)
    protected = {
        c.REPO,
        Path("/usr/local/Ascend"),
        old,
        isa,
        *map(Path, c.read(old / "static/report.json")["history_roots"]),
    }
    with phase_directory(root, phase, protected) as out:
        base = dict(
            schema=1,
            root=str(root),
            archive=str(old),
            isa_audit=str(isa),
            labels=LABELS,
            scope="C16_LAYER0_GATE_EXPERIMENTAL_ONLY",
            sources=source_manifest(),
        )
        r = dict(
            **base,
            phase=phase,
            folder=str(out),
            predecessors={},
            device_execution="NOT_RUN",
            benchmark="NOT_RUN",
            status=BLOCKED[phase],
            error=None,
            integrity_error=None,
        )
        try:
            c.clean()
            ctx = load_context(old, isa)
            base["archived_evidence"] = dict(**ctx.refs, isa_audit=ctx.negative["archive"])
            r["archived_evidence"] = base["archived_evidence"]
            compilers = compiler_evidence(ctx)
            previous, refs = predecessors(root, phase, ctx, compilers, base)
            r["predecessors"] = refs
            if phase == "compiler-audit":
                r["compilation"] = compilers
            elif phase == "admit":
                r["safety"] = safety_case(ctx, compilers)
                r["cpu_witness"] = cpu_witness(out, ctx)
            elif phase in DEVICE:
                execute(out, r, ctx, previous, args.allow_device)
            else:
                r["comparison"] = decision(
                    previous["schedule-benchmark"]["statistics"], previous["net-benchmark"]["statistics"]
                )
            r["status"] = validate(root, r, ctx, previous, compilers)
        except (Exception, KeyboardInterrupt) as exc:
            r["status"] = BLOCKED[phase]
            r["error"] = repr(exc)
            (out / "traceback.txt").write_text(traceback.format_exc())
            if phase in DEVICE:
                try:
                    r["failure"] = failure(out, r, exc)
                except Exception as evidence_error:
                    r["failure_evidence_error"] = repr(evidence_error)
        try:
            archive.historical.verify_tree(r, {})
            require(r["sources"] == source_manifest(), "experiment verifier changed during stage")
            c.clean()
        except Exception as exc:
            r.update(status=BLOCKED[phase], integrity_error=repr(exc))
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
                        "labels",
                        "device_execution",
                        "benchmark",
                        "active_case",
                        "error",
                        "integrity_error",
                    )
                },
                indent=2,
            )
        )
        if phase == "compare" and "comparison" in r:
            print("Decision", r["comparison"]["decision"])
        print("REPORT", out / "report.json")
        return 0 if r["status"] == PASS[phase] else 3 if r["status"] == NO_GO else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("phase", choices=STAGES)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--archive", type=Path, default=archive.DEFAULT_ROOT)
    parser.add_argument("--isa-audit", type=Path, default=ISA_ROOT)
    parser.add_argument("--allow-device", action="store_true")
    try:
        sys.exit(main(parser.parse_args()))
    except (OSError, ValueError) as exc:
        parser.exit(1, "BLOCKED: " + str(exc) + "\n")
