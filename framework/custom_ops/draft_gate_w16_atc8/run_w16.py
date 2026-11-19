#!/usr/bin/env python3
"""Run one requested W16 stage. Static evidence, device permission and performance gates are independent."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback

from w16_common import (
    API,
    DOUBLE,
    FIRST,
    GATE8_ROOT,
    HERE,
    LAYERS,
    OVERLAP,
    PROFILE_ROOT,
    SINGLE,
    c,
    command,
    gate_kernel,
    historical,
    host,
    require,
    resume,
    sources,
    tree,
)
from w16_export import check_model, compile_model
from w16_runtime import case_run, decision, five_layer_decision, order, paired, row_key
from w16_source import admit, recovered, verify_source
from w16_static import NoOverlap, StaticBlocked, admit_static, collect, proof_index, review_template

STAGES = (
    "source",
    "build",
    "atc",
    "static",
    "smoke",
    "schedule-benchmark",
    "net-benchmark",
    "compare",
    "five-c16-atc",
    "five-c16-static",
    "five-c16-smoke",
    "five-c16-benchmark",
    "five-c64-smoke",
    "five-c64-benchmark",
    "full-atc",
    "full-static",
    "full-smoke",
    "full-benchmark",
)
EXTRA = "resource-benchmark"
DEVICE = {s for s in (*STAGES, EXTRA) if s.endswith(("smoke", "benchmark"))}
ATC = {"atc", "five-c16-atc", "full-atc"}
STATIC = {"static": "atc", "five-c16-static": "five-c16-atc", "full-static": "full-atc"}
NO_GO = "W16_NO_GO"


def requirements(phase):
    return STAGES[: STAGES.index(phase)] if phase != EXTRA else STAGES[: STAGES.index("compare") + 1]


def passed(phase):
    return "W16_" + phase.replace("-", "_").upper() + "_PASS"


def previous(root, phase):
    attempts = [root / phase / "report.json", *sorted((root / "retries" / phase).glob("*/report.json"))]
    ok = []
    for file in attempts:
        if file.is_file():
            r = c.read(file)
            if r.get("status") in (passed(phase), NO_GO):
                ok.append((file, r))
    require(len(ok) == 1, "need exactly one completed " + phase + " report, got " + str(len(ok)))
    return ok[0]


def fresh_stage(root, phase, protected):
    require(
        all(root != p and root not in p.parents and p not in root.parents for p in protected),
        "root overlaps protected evidence",
    )
    if phase == "source":
        require(
            not root.exists() or (root.is_dir() and not any(root.iterdir())), "source needs an empty new root"
        )
    else:
        require(
            (root / "source/report.json").is_file()
            and c.read(root / "source/report.json").get("root") == str(root),
            "not an independent W16 root",
        )
    dest = root / phase
    if not dest.exists():
        return c.fresh(dest, protected)
    prior = c.read(dest / "report.json")
    require(phase != "source" and prior["status"].startswith("BLOCKED"), "stage exists; refusing overwrite")
    retry = root / "retries" / phase
    retry.mkdir(parents=True, exist_ok=True)
    require(
        retry.resolve() == retry
        and all(c.read(p)["status"].startswith("BLOCKED") for p in retry.glob("*/report.json")),
        "already completed retry or redirected directory",
    )
    return Path(tempfile.mkdtemp(prefix="attempt-", dir=retry))


def verify_build(data, src, raw):
    require(set(data["variants"]) == set(DOUBLE), "incomplete private builds")
    installs = [raw["opp"]["install"]]
    for variant, b in data["variants"].items():
        worker = c.read(c.verify(data["workers"][variant]))
        require(
            worker["status"] == "PRIVATE_W16_PACKAGE_BUILT"
            and worker["result"] == b
            and worker["variant"] == variant,
            "wrong private build worker",
        )
        c.verify_opp(b["opp"])
        root = Path(b["opp"]["install"])
        installs.append(str(root))
        require(
            tree(root) == b["package_files"]
            and gate_kernel(b["opp"]) == b["kernel"]
            and b["kernel"]["elf"]["sha256"] == b["compiler"]["binary"]["sha256"]
            and b["api"]["sha256"] == API
            and b["runner"] == raw["runner"]
            and b["om"] == raw["om"],
            "installed OPP/Runner drift",
        )
        require(host.envs.standard_identity(b["env"]) == src["standard_env"], "environment drift")
        resume.combined(b, src["cann"], src["extras"], c.read(resume.POLICY))
    require(len(set(installs)) == 3, "overlapping private packages")
    a, b = (data["variants"][v] for v in DOUBLE)
    host_file = "op_host/d_flash_group_quant_linear.cpp"
    require(
        a["generated_sources"][host_file]["file"]["sha256"]
        == b["generated_sources"][host_file]["file"]["sha256"],
        "different B/C Host",
    )
    from w16_common import STEM, render_kernel

    name = "op_kernel/" + STEM + ".cpp"
    base = c.verify(raw["generated_sources"][name]["file"]).read_bytes()
    for mode, package in data["variants"].items():
        require(
            c.verify(package["generated_sources"][name]["file"]).read_bytes() == render_kernel(base, mode),
            "unreviewed device source in build",
        )


def run_build(out, r, src, source_path):
    prior = c.read(c.verify(src["host_source"]))["binding"]["prior"]
    r.update(variants={}, workers={})
    for mode in DOUBLE:
        r["active_case"] = mode
        folder = out / mode
        folder.mkdir()
        dest = folder / "package"
        env, _ = host.envs.for_prior(prior, dest / "opp", ready=False)
        req = folder / "request.json"
        c.write(req, dict(dest=str(dest), source=c.identity(source_path), variant=mode, sources=r["sources"]))
        file = folder / "worker.json"
        try:
            command([sys.executable, "-B", HERE / "private_worker.py", req], folder / "builder.log", env)
        except RuntimeError as exc:
            if file.is_file():
                r["workers"][mode] = c.identity(file)
                raise RuntimeError(f"{mode} package: {c.read(file).get('error')} ({file})") from exc
            raise
        worker = c.read(file)
        require(worker["status"] == "PRIVATE_W16_PACKAGE_BUILT", "private build failed")
        r["variants"][mode] = worker["result"]
        r["workers"][mode] = c.identity(file)


def atc_keys(phase):
    return {"atc": (FIRST,), "five-c16-atc": LAYERS[1:], "full-atc": ("16", "64")}[phase]


def run_atc(out, r, src, selected, baseline, binding):
    tool = binding["tool"]
    phase = r["phase"]
    full = phase == "full-atc"
    r.update(models={}, captures={}, baseline_images={}, atc="ATTEMPTED")
    for key in atc_keys(phase):
        r["captures"][key] = {}
        control = baseline[key]
        r["baseline_images"][key] = {}
        for index, name in enumerate(sorted(control["gate_nodes"].values())):
            r["baseline_images"][key][name] = collect(
                out / SINGLE / key / "machine" / str(index), control["om"], control["atc_kernels"][name], tool
            )
        for mode in (OVERLAP,) if full else DOUBLE:
            r["active_case"] = [key, mode]
            folder = out / mode / key
            model = compile_model(folder, src, selected[mode], mode, key, baseline[key], full)
            r["models"].setdefault(mode, {})[key] = model
            evidence = {}
            for index, name in enumerate(sorted(model["gate_nodes"].values())):
                node = model["atc_kernels"][name]
                dest = folder / "machine" / str(index)
                cap = collect(dest, model["om"], node, tool)
                source = selected[mode]["generated_sources"]["op_kernel/d_flash_group_quant_linear.cpp"]["file"]
                template = dest / "review-input.json"
                c.write(
                    template,
                    review_template(
                        cap, mode, source, model["plans"]["16x2560x19456"]["contract"], binding["sdk"]
                    ),
                )
                evidence[name] = dict(cap, review_input=c.identity(template))
            r["captures"][key][mode] = evidence
    r["atc"] = "FROZEN_AIR_COMPILED_AND_EMBEDDED_ELF_CAPTURED"


def verify_atc(r, src, built, base, tool):
    full = r["phase"] == "full-atc"
    keys = set(atc_keys(r["phase"]))
    modes = (OVERLAP,) if full else DOUBLE
    require(
        set(r["models"]) == set(modes) and set(r["captures"]) == set(r["baseline_images"]) == keys, "ATC scope"
    )
    for key, images in r["baseline_images"].items():
        expect = base[key]
        require(
            set(images) == set(expect["gate_nodes"].values())
            and all(
                cap["om"] == expect["om"]
                and cap["node"] == expect["atc_kernels"][n]
                and cap["elf"]["sha256"] == cap["node"]["sha256"]
                and cap["tool"] == tool
                for n, cap in images.items()
            ),
            "cross-wired single-W control ELF",
        )
    for mode in modes:
        require(set(r["models"][mode]) == keys, "ATC omitted/extra model")
        for key in keys:
            model = r["models"][mode][key]
            require(
                model["mode"] == mode
                and model["key"] == key
                and model["scope"] == ("full" if full else "projection"),
                "ATC model mislabeled",
            )
            native = check_model(model, src, built[mode], base[key], full)
            if full:
                require(native == model["protected_native_images"], "native embedded-code evidence changed")
            cap = r["captures"][key][mode]
            require(set(cap) == set(model["gate_nodes"].values()), "missing disassembly")
            for name, rec in cap.items():
                require(
                    rec["node"] == model["atc_kernels"][name]
                    and rec["om"] == model["om"]
                    and rec["tool"] == tool
                    and rec["elf"]["sha256"] == rec["node"]["sha256"]
                    and rec["elf"]["bytes"] == rec["node"]["bytes"],
                    "machine evidence from different model",
                )
            if not full:
                left, right = [r["models"][v][key]["plans"] for v in DOUBLE]
                require(
                    {k: p["contract"] for k, p in left.items()} == {k: p["contract"] for k, p in right.items()},
                    "B/C complete parent Tiling differs",
                )


def schedule(phase):
    if phase.startswith("full-"):
        return [
            x
            for gear in ("16", "64")
            for x in order(gear, ["full"], ["net"] if phase.endswith("benchmark") else (), full=True)
        ]
    gear = "64" if phase.startswith("five-c64") else "16"
    layers = LAYERS if phase.startswith("five-") else (FIRST,)
    pairs = (
        ()
        if phase.endswith("smoke")
        else (
            ("schedule", "net")
            if phase.startswith("five-")
            else ({"schedule-benchmark": "schedule", "net-benchmark": "net", EXTRA: "resource-control"}[phase],)
        )
    )
    return order(gear, layers, pairs)


def model_map(chain, old_models, full):
    models = {SINGLE: old_models["full-export" if full else "export"]}
    for phase in ("full-atc",) if full else ("atc", "five-c16-atc"):
        for mode, rows in chain.get(phase, {}).get("models", {}).items():
            models.setdefault(mode, {}).update(rows)
    return models


def smoke_index(chain, phase):
    name = (
        "full-smoke"
        if phase.startswith("full-")
        else phase[:8] + "-smoke"
        if phase.startswith("five-")
        else "smoke"
    )
    return {(r["gear"], r["layer"], r["mode"]): r for r in chain.get(name, {}).get("runs", [])}


def location(out, key):
    gear, layer, pair, round_id, mode = key
    return out / ("c" + gear) / layer / pair / str(round_id) / mode


def execute(out, r, src, built, models, chain):
    r["runs"] = []
    timed = r["phase"].endswith("benchmark")
    previous_rows = smoke_index(chain, r["phase"])
    r["device_execution"] = "ATTEMPTED"
    if timed:
        r["benchmark"] = "ATTEMPTED"
    for key in schedule(r["phase"]):
        gear, layer, pair, round_id, mode = key
        r["active_case"] = list(key)
        model = models[mode][gear if layer == "full" else layer]
        case_run(
            location(out, key),
            src,
            built[mode],
            model,
            gear,
            layer,
            mode,
            round_id,
            pair,
            timed,
            r,
            previous_rows.get((gear, layer, mode)),
        )
    r["device_execution"] = "COMPLETED"
    if timed:
        r["benchmark"] = "FRESH_RESIDENT_EXECUTE_SYNC_MEASURED"


def replay_runs(out, r, src, built, models, chain):
    from w16_runtime import verify_runtime_row

    timed = r["phase"].endswith("benchmark")
    keys = schedule(r["phase"])
    require([row_key(row) for row in r["runs"]] == keys, "run inventory/order mismatch")
    smoke = smoke_index(chain, r["phase"])
    common = {}
    memory = {}
    for row, key in zip(r["runs"], keys):
        gear, layer, pair, round_id, mode = key
        model = models[mode][gear if layer == "full" else layer]
        verify_runtime_row(
            location(out, key), row, src, built[mode], model, timed, smoke.get((gear, layer, mode))
        )
        output = {name: snaps[0]["sha256"] for name, snaps in row["outputs"].items()}
        require(common.setdefault((gear, layer), output) == output, "cross-variant output bits differ")
        used = (row["validation"]["workspace_bytes"], row["validation"]["model_weight_bytes"])
        require(memory.setdefault((gear, layer), used) == used, "cross-variant OM allocation drift")
    return common


def results(r):
    groups = {}
    for row in r["runs"]:
        key = (row["gear"], row["layer"], row["pair"])
        groups.setdefault(key, []).append(row)
    return {"/".join(key): paired(rows, *key) for key, rows in groups.items()}


def compare_single(chain):
    k = "16/" + FIRST + "/"
    return decision(
        chain["schedule-benchmark"]["statistics"][k + "schedule"],
        chain["net-benchmark"]["statistics"][k + "net"],
    )


def validate(out, r, chain, assets):
    phase = r["phase"]
    require(
        r["device_execution"] == ("COMPLETED" if phase in DEVICE else "NOT_RUN")
        and r["atc"] == ("FROZEN_AIR_COMPILED_AND_EMBEDDED_ELF_CAPTURED" if phase in ATC else "NOT_RUN")
        and r["benchmark"]
        == ("FRESH_RESIDENT_EXECUTE_SYNC_MEASURED" if phase.endswith("benchmark") else "NOT_RUN"),
        "stage scope/execution state mismatch",
    )
    if phase == "source":
        verify_source(r["binding"], *map(Path, r["history_roots"]))
        return
    src, raw, old_models = assets
    if phase == "build":
        verify_build(r, src, raw)
        return
    built = {SINGLE: raw, **chain["build"]["variants"]}
    if phase in ATC:
        verify_atc(
            r,
            src,
            built,
            old_models["full-export" if phase == "full-atc" else "export"],
            chain["source"]["binding"]["tool"],
        )
    elif phase in STATIC:
        exp = chain[STATIC[phase]]
        require(
            r["proofs"]
            == admit_static(
                exp["captures"],
                exp["models"],
                built,
                chain["source"]["binding"]["sdk"],
                r["review_commit"],
                r.get("external_proof"),
            ),
            "machine proof no longer verifies",
        )
    elif phase == "compare":
        require(r["comparison"] == compare_single(chain), "decision drift")
    else:
        models = model_map(chain, old_models, phase.startswith("full-"))
        require(
            replay_runs(out, r, src, built, models, chain) == r["cross_variant_output_sha256"],
            "output hash mismatch",
        )
        if phase.endswith("benchmark"):
            require(results(r) == r["statistics"], "fresh sample/statistics mismatch")
            if phase.startswith("five-"):
                require(
                    five_layer_decision(r["statistics"], phase[6:8]) == r["five_layer_decision"],
                    "five-layer decision drift",
                )


def status(r):
    if r["phase"] == "compare" and r["comparison"]["decision"] != "GO":
        return NO_GO
    if (
        r["phase"].startswith("five-")
        and r["phase"].endswith("benchmark")
        and r["five_layer_decision"]["decision"] != "GO"
    ):
        return NO_GO
    if r["phase"] == "full-benchmark":
        if any(row["decision"] != "GO" for row in r["statistics"].values()):
            return NO_GO
    return passed(r["phase"])


def main(args):
    root = args.root.resolve()
    history = [p.resolve() for p in (args.gate8_root, args.profile_root)]
    out = fresh_stage(root, args.phase, [*history, c.REPO, Path("/usr/local/Ascend")])
    r = dict(
        schema=1,
        phase=args.phase,
        root=str(root),
        folder=str(out),
        history_roots=list(map(str, history)),
        status="BLOCKED",
        error=None,
        integrity_error=None,
        predecessors={},
        sources=sources(),
        device_execution="NOT_RUN",
        atc="NOT_RUN",
        benchmark="NOT_RUN",
        production="NOT_MODIFIED",
    )
    chain = {}
    assets = None
    try:
        c.clean()
        require(args.phase == "source" or not args.objdump, "select the SDK decoder at Source")
        require(
            args.phase in STATIC or not (args.review_commit or args.proof_index or args.proof_index_sha256),
            "PC proof inputs belong to the static stage",
        )
        seen = {}
        for phase in requirements(args.phase):
            path, prior = previous(root, phase)
            require(
                prior["phase"] == phase
                and prior["root"] == str(root)
                and prior["folder"] == str(path.parent)
                and prior["predecessors"] == r["predecessors"]
                and prior["sources"] == r["sources"]
                and prior["schema"] == 1
                and prior["history_roots"] == r["history_roots"]
                and not prior["error"]
                and not prior["integrity_error"],
                "mixed stage/source history: " + phase,
            )
            require(
                prior["status"] == passed(phase)
                or (args.phase == EXTRA and phase == "compare" and prior["status"] == NO_GO),
                "NO_GO blocks expansion",
            )
            historical.verify_tree(prior, seen)
            validate(path.parent, prior, chain, assets)
            require(status(prior) == prior["status"], "predecessor status differs from evidence")
            chain[phase] = prior
            r["predecessors"][phase] = c.identity(path)
            if phase == "source":
                assets = recovered(prior["binding"])
        phase = args.phase
        if phase == "source":
            r["binding"] = admit(*history, args.objdump)
        elif phase == "build":
            run_build(out, r, assets[0], Path(r["predecessors"]["source"]["path"]))
        else:
            src, raw, old_models = assets
            built = {SINGLE: raw, **chain["build"]["variants"]}
            if phase in ATC:
                run_atc(
                    out,
                    r,
                    src,
                    built,
                    old_models["full-export" if phase == "full-atc" else "export"],
                    chain["source"]["binding"],
                )
            elif phase in STATIC:
                exp = chain[STATIC[phase]]
                r["review_commit"] = (
                    args.review_commit
                    or subprocess.check_output(
                        ["git", "-C", str(c.REPO), "rev-parse", "HEAD"], text=True
                    ).strip()
                )
                require(
                    bool(args.proof_index) == bool(args.proof_index_sha256)
                    and not (args.proof_index and args.review_commit),
                    "provide one exact external proof index/digest or one Git revision",
                )
                r["external_proof"] = (
                    proof_index(args.proof_index, args.proof_index_sha256) if args.proof_index else None
                )
                r["proofs"] = admit_static(
                    exp["captures"],
                    exp["models"],
                    built,
                    chain["source"]["binding"]["sdk"],
                    r["review_commit"],
                    r["external_proof"],
                )
            elif phase == "compare":
                r["comparison"] = compare_single(chain)
            else:
                require(phase in DEVICE and args.allow_device, "explicit --allow-device required")
                models = model_map(chain, old_models, phase.startswith("full-"))
                execute(out, r, src, built, models, chain)
                r["cross_variant_output_sha256"] = replay_runs(out, r, src, built, models, chain)
                if phase.endswith("benchmark"):
                    r["statistics"] = results(r)
                    if phase.startswith("five-"):
                        r["five_layer_decision"] = five_layer_decision(r["statistics"], phase[6:8])
                    if phase == EXTRA:
                        r["serial_minus_single_fresh_ms"] = -r["statistics"][
                            "16/" + FIRST + "/resource-control"
                        ]["median_saving_ms"]
        validate(out, r, chain, assets)
        r["status"] = status(r)
        r.pop("active_case", None)
    except Exception as exc:
        r["error"] = f"{type(exc).__name__}: {exc}"
        r["status"] = (
            "NO_TESTABLE_OVERLAP"
            if isinstance(exc, NoOverlap)
            else "BLOCKED_STATIC_ADMISSION"
            if isinstance(exc, StaticBlocked) or args.phase in STATIC
            else "BLOCKED"
        )
        (out / "traceback.txt").write_text(traceback.format_exc())
    finally:
        try:
            if r["error"]:
                r["failure_files"] = [
                    c.identity(p)
                    for p in sorted(out.rglob("*"))
                    if p.is_file()
                    and not p.is_symlink()
                    and (
                        p.suffix in (".log", ".jsonl", ".json", ".bin", ".elf", ".om", ".disassembly")
                        or p.name == "traceback.txt"
                    )
                ]
            historical.verify_tree(r, {})
            require(r["sources"] == sources(), "local source drift during stage")
            c.clean()
        except Exception as exc:
            r.update(status="BLOCKED", integrity_error=repr(exc))
        c.write(out / "report.json", r)
    print(
        json.dumps(
            {
                k: r.get(k)
                for k in (
                    "phase",
                    "status",
                    "device_execution",
                    "atc",
                    "benchmark",
                    "active_case",
                    "error",
                    "integrity_error",
                )
            },
            indent=2,
        )
    )
    print("REPORT", out / "report.json")
    return 0 if r["status"] == passed(args.phase) else 3 if r["status"] in (NO_GO, "NO_TESTABLE_OVERLAP") else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("phase", choices=(*STAGES, EXTRA))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--gate8-root", type=Path, default=GATE8_ROOT)
    p.add_argument("--profile-root", type=Path, default=PROFILE_ROOT)
    p.add_argument("--objdump", type=Path, help="Source only: exact decoder inside selected CANN")
    p.add_argument("--review-commit", help="static only: immutable revision of per-ELF PC evidence")
    p.add_argument(
        "--proof-index", type=Path, help="static only: external, independently reviewed machine evidence index"
    )
    p.add_argument("--proof-index-sha256", help="expected SHA256 of the external index")
    p.add_argument("--allow-device", action="store_true")
    raise SystemExit(main(p.parse_args()))
