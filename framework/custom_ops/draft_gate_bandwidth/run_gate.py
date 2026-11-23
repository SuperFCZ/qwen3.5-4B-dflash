"""Benchmark complete W8A16 Gate/Up OMs on the caller's 310P3."""

import argparse
import os
from pathlib import Path
import shutil
import sys
import traceback

from gate_panel import PANELS, compile_panel, make_air, build
from fresh_run import data_files, runner_binary, environment, run_child
from bench import M, K, N, SHAPE_KEY, TILE_MODE, OP, c, compile_gate, require, stats, prepare_air
from fresh_dso import seal
from package import make_package
from air import classes
from tile_gate_graph import model_proto
from measurement import mismatch_evidence
import numpy as np

FLOPS = 2 * M * K * N
LOGICAL_BYTES = N * K + 2 * (K // 128) * N + 2 * M * K + 2 * M * N


def boundary_case(folder, files, *, shape=(M, K, N), pairs=None):
    """Two signed terms/row across packing and reduction boundaries, same compiled weights/scale."""
    m_rows, k_dim, n_dim = shape
    folder.mkdir()
    codes = np.frombuffer(c.verify(files["w_nz.bin"]).read_bytes(), dtype=np.int8).reshape(
        k_dim // 32, n_dim, 32
    )
    gn = np.frombuffer(c.verify(files["s_gn.bin"]).read_bytes(), dtype="<f2").reshape(k_dim // 128, n_dim)
    if pairs is None:
        pairs = (
            (0, 2559),
            (15, 16),
            (31, 32),
            (127, 128),
            (255, 256),
            (511, 512),
            (1279, 1280),
            (2431, 2432),
        )
    x = np.zeros((m_rows, k_dim), dtype="<f2")
    y = np.empty((m_rows, n_dim), dtype="<f2")
    for m in range(m_rows):
        left, right = pairs[m % len(pairs)]
        x[m, left] = np.float16(1 + m / 64)
        x[m, right] = np.float16(-1 + (m - 8) / 128)

        def term(kk):
            w = codes[kk // 32, :, kk % 32].astype(np.float16) * gn[kk // 128]
            return w.astype(np.float32) * np.float32(x[m, kk])

        accum = term(left) + term(right)
        y[m] = np.where(accum == 0, np.float32(0), accum).astype("<f2")
    require(np.isfinite(y).all(), "nonfinite boundary-case reference")
    (folder / "x.bin").write_bytes(x.tobytes())
    (folder / "golden.bin").write_bytes(y.tobytes())
    return dict(files, **{key: c.identity(folder / key) for key in ("x.bin", "golden.bin")})


def pair_statistics(rows, mode, control_mode=TILE_MODE, *, flops=FLOPS, logical_bytes=LOGICAL_BYTES):
    require(mode != control_mode, "identical modes in pair")
    order = [r["mode"] for r in rows]
    require(
        len(rows) == 6
        and len({r["folder"] for r in rows}) == 6
        and all(
            set(order[i : i + 2]) == {control_mode, mode}
            and [r["round"] for r in rows[i : i + 2]] == [i // 2] * 2
            for i in (0, 2, 4)
        )
        and order[:2] == order[4:6] == order[2:4][::-1],
        "invalid paired schedule",
    )
    require(
        all(r["bit_mismatches"] == 0 and r["launch_authorized"] and not r["profiled"] for r in rows),
        "unvalidated timing",
    )
    per = {}
    summary = {}
    for key in (control_mode, mode):
        per[key] = [stats(r["validation"]["timing"]["samples_ms"], 30) for r in rows if r["mode"] == key]
        summary[key] = stats([v for rd in per[key] for v in rd["samples_ms"]], 90)
    ctrl, cand = summary[control_mode], summary[mode]
    ratios = [a["median_ms"] / b["median_ms"] for a, b in zip(per[control_mode], per[mode])]
    ms = cand["median_ms"]
    return dict(
        variants=summary,
        round_pairs=per,
        round_ratios=ratios,
        speedup=ctrl["median_ms"] / ms,
        reduction_pct=100 * (1 - ms / ctrl["median_ms"]),
        equivalent_matrix_tflops=flops / ms / 1e9,
        logical_gbps=logical_bytes / ms / 1e6,
        decision=(
            "GO"
            if ms <= 0.97 * ctrl["median_ms"] and min(ratios) > 1 and cand["p95_ms"] <= ctrl["p95_ms"]
            else "NO_GO"
        ),
    )


def run(root, panels, real, reuse, compile_only, r, w16_pair=None):
    cann = Path(os.environ["CANN_ROOT"]).resolve(strict=True)
    tools = {name: c.identity(shutil.which(name)) for name in ("atc", "msopgen", "cmake")}
    env = environment(cann, tools)
    files, label = data_files(root, real)
    reference = "supplied Gate FP16 reference checked in process"
    atc = [
        str(cann / "bin/atc"),
        "--mode=0",
        "--framework=1",
        "--model=INPUT",
        "--output=OUTPUT",
        "--soc_version=Ascend310P3",
        "--precision_mode=must_keep_origin_dtype",
        "--deterministic=0",
    ]
    r.update(
        data_source=label,
        inputs=files,
        tools=tools,
        models={},
        packages={},
        rejected={},
        roofline=dict(
            min_logical_bytes=LOGICAL_BYTES,
            gemm_flops=FLOPS,
            bandwidth_assumption_gbps=204.8,
            single_pass_ms=LOGICAL_BYTES / 204.8e6,
            same_card_copy_only_ms=0.55971,
        ),
    )
    models, pkgs = r["models"], r["packages"]
    print(f"{label}  M{M} K{K} N{N}  single Gate/Up OM", flush=True)
    control_mode = TILE_MODE
    if w16_pair:
        from w16_control import load_control, CONTROL

        control_mode = CONTROL
        model, pkg, runner, original_packing, ref = load_control(
            w16_pair, tools, files, label, pair_statistics
        )
        models[control_mode], pkgs[control_mode] = model, pkg
        r["control_receipt"] = ref
        print("Verified single-W K2560 OM", model["om"]["sha256"], flush=True)
    else:
        runner = runner_binary(root, cann, env, reference)
        types = classes(cann, root)
        if reuse:
            from reuse_build import load_build

            old, ref = load_build(reuse, tools, files, label)
            r["control_receipt"] = ref
            models[TILE_MODE] = old["models"][TILE_MODE]
            pkgs[TILE_MODE] = old["packages"][TILE_MODE]
            print("Verified tile-ND/B16 control OM", models[TILE_MODE]["om"]["sha256"], flush=True)
        else:
            r["phase"] = "build tile-ND control"
            source = root / "source.air"
            proto = model_proto(
                types, c.verify(files["w_nz.bin"]).read_bytes(), c.verify(files["s_gn.bin"]).read_bytes()
            )
            source.write_bytes(proto.SerializeToString())
            del proto
            d = root / "control-atc"
            prepare_air(d, {"cann": str(cann)}, env, source, files)
            work = root / "control-build"
            work.mkdir()
            pkg = make_package(work, None, cann, env)
            models[TILE_MODE] = compile_gate(
                d,
                pkg,
                environment(cann, tools, pkg["opp"]["install"]),
                atc,
                OP,
                c.identity(d / "weights.tile-nd.bin"),
            )
            pkgs[TILE_MODE] = pkg
    r["runner"] = runner
    r["control_mode"] = control_mode
    control = models[control_mode]
    executable_images = {
        tuple(i["sha256"] for i in next(iter(control["atc_kernels"].values()))["executable_sections"])
    }
    for panel in panels:
        mode = f"panel-k{panel}" + ("-w2" if w16_pair else "")
        r["phase"] = "compile " + mode
        folder = root / mode
        folder.mkdir()
        print("Packing, building and compiling", mode, flush=True)
        try:
            if w16_pair:
                packing = original_packing
                (folder / "atc").mkdir()
                shutil.copyfile(c.verify(control["air"]), folder / "atc/gate.air")
            else:
                packing = make_air(folder / "atc", types, files, panel)
            pkg = build(folder / "build", cann, env, panel, w16_banks=2 if w16_pair else 1)
            model = compile_panel(
                folder / "atc",
                pkg,
                environment(cann, tools, pkg["opp"]["install"]),
                atc,
                control,
                packing,
                w16_banks=2 if w16_pair else 1,
            )
            signature = tuple(
                i["sha256"] for i in next(iter(model["atc_kernels"].values()))["executable_sections"]
            )
            require(signature not in executable_images, "duplicated ATC executable image")
            executable_images.add(signature)
        except (OSError, ValueError, RuntimeError) as error:
            r["rejected"][mode] = repr(error)
            (folder / "traceback.txt").write_text(traceback.format_exc())
            print("REJECTED BEFORE LAUNCH", mode, repr(error), flush=True)
            continue
        models[mode] = model
        pkgs[mode] = pkg
        proof = model["plans"][SHAPE_KEY]["contract"]
        print(
            f"{mode}  ATC blocks={proof['block_dim']}  baseK={proof['cube']['baseK']}  "
            f"Iterate/N64={proof['outer_iterates_per_n64']}  scratch={proof['resources']['scratch']}",
            flush=True,
        )
    modes = [m for m in models if m != control_mode]
    require(modes, "no candidate passed Build and actual ATC resource admission")
    if compile_only:
        return "COMPILED_NOT_EXECUTED"
    stress = boundary_case(root / "boundary-inputs", files)
    r.update(phase="prelaunch DSO binding", boundary_inputs=stress, smoke=[], runs=[], statistics={})
    envs = {m: environment(cann, tools, pkg["opp"]["install"]) for m, pkg in pkgs.items()}
    dsos = {m: seal(root / ("dso-" + m), pkg, runner, envs[m]) for m, pkg in pkgs.items()}

    def execute(folder, mode, data, iteration, timed):
        r["phase"] = str(folder.relative_to(root))
        c.verify_opp(pkgs[mode]["opp"])
        try:
            return run_child(
                folder, runner, models[mode], envs[mode], dsos[mode], data, timed, reference, iteration
            )
        except Exception:
            differences = mismatch_evidence(folder, {"golden": {"y": data["golden.bin"]}})
            r["output_differences"] = differences
            for item in differences:
                print(
                    "Output mismatch",
                    mode,
                    item.get("bit_mismatches"),
                    "first",
                    item.get("first"),
                    "max finite ULP",
                    item.get("max_finite_ulp"),
                    flush=True,
                )
            raise

    for m in models:
        for case, data in (("dense", files), ("boundaries", stress)):
            r["smoke"].append(dict(execute(root / "check" / m / case, m, data, -1, False), case=case))
        print("Bitwise + guards PASS", m, "(two inputs, two poisons each)", flush=True)
    for i in range(3):
        for m in modes[i % len(modes) :] + modes[: i % len(modes)]:
            order = [control_mode, m] if (i + modes.index(m)) % 2 == 0 else [m, control_mode]
            print("Round", i + 1, " → ".join(order), flush=True)
            for kind in order:
                r["runs"].append(
                    dict(execute(root / "timing" / str(i) / m / kind, kind, files, i, True), pair=m)
                )
    c.verify_tree(files)
    print("Full Gate/Up Execute+Sync, milliseconds; 90 fresh samples per side and per pair")
    print("candidate       control_p50  cand_p50  control_p95  cand_p95  speedup  logical_GB/s  GEMM_TF/s")
    for m in modes:
        t = pair_statistics([row for row in r["runs"] if row["pair"] == m], m, control_mode)
        r["statistics"][m] = t
        a, b = t["variants"][control_mode], t["variants"][m]
        print(
            f"{m:15} {a['median_ms']:11.6f} {b['median_ms']:9.6f} {a['p95_ms']:11.6f} "
            f"{b['p95_ms']:9.6f} {t['speedup']:7.4f}x {t['logical_gbps']:12.2f} "
            f"{t['equivalent_matrix_tflops']:10.3f} {t['decision']}",
            flush=True,
        )
    good = [m for m, t in r["statistics"].items() if t["decision"] == "GO"]
    r["fastest_passing"] = (
        min(good, key=lambda m: r["statistics"][m]["variants"][m]["median_ms"]) if good else None
    )
    print("Fastest passing candidate:", r["fastest_passing"] or "none")
    print("Roofline reference only: 0.25044 ms at 204.8 GB/s, one external pass; no traffic counters.")
    return "MEASURED_WITH_REJECTIONS" if r["rejected"] else "MEASURED"


def main():
    p = argparse.ArgumentParser()
    selection = p.add_mutually_exclusive_group()
    selection.add_argument("--panels", help="comma-separated K panels")
    selection.add_argument(
        "--w16-pair", metavar="RUN", help="compare two W16 banks directly to a measured single-W K2560 run"
    )
    p.add_argument("--compile-only", action="store_true")
    p.add_argument("--layer0", action="store_true")
    p.add_argument(
        "--reuse-build", help="prior two-OM Gate build under the run root or an absolute directory"
    )
    args = p.parse_args()
    try:
        require(not (args.reuse_build and args.w16_pair), "--w16-pair already supplies its control")
        require(args.w16_pair is None or args.w16_pair.strip(), "empty single-W run path")
        selection = "256,512,1280,2560" if args.panels is None else args.panels
        panels = [K] if args.w16_pair else [int(k) for k in selection.split(",")]
        require(len(panels) == len(set(panels)) and set(panels) <= set(PANELS) and panels, "invalid K panels")
    except ValueError as exc:
        p.error(str(exc))
    root = Path(os.environ["DFLASH_OUTPUT"]) / "experiment"
    root.mkdir()
    run_root = Path(os.environ["DFLASH_RUN_ROOT"])
    reuse = run_root / "runs" / args.reuse_build if args.reuse_build else None
    r = {"status": "FAILED", "phase": "setup"}
    try:
        r["status"] = run(
            root,
            panels,
            run_root / "inputs/c16-layer0-gate" if args.layer0 else None,
            reuse,
            args.compile_only,
            r,
            run_root / "runs" / args.w16_pair if args.w16_pair else None,
        )
    except Exception as error:
        r["error"] = repr(error)
        traceback.print_exc()
    c.write(root / "result.json", r)
    print(r["status"], "REPORT", root / "result.json", flush=True)
    return int(r["status"] == "FAILED" or bool(r.get("rejected")))


if __name__ == "__main__":
    sys.exit(main())
