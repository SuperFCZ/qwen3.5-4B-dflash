"""One manual 310P3 Down benchmark. Checks both formulations before collecting any timings."""

import argparse
import os
from pathlib import Path
import shutil
import sys
import traceback

from down_panel import (
    M,
    K,
    N,
    TAG,
    SHAPE,
    ABI,
    CONTROL,
    SINGLE,
    DOUBLE,
    FLOPS,
    LOGICAL_BYTES,
    c,
    require,
    inputs,
    prepare_air,
    build,
    baseline_package,
    compile_model,
)
from fresh_run import runner_binary, environment, run_child
from fresh_dso import seal
from air import classes
from run_gate import boundary_case, pair_statistics
from down_fallback import differences

PAIRS = ((CONTROL, SINGLE), (CONTROL, DOUBLE), (SINGLE, DOUBLE))
BOUNDARIES = (
    (0, K - 1),
    (15, 16),
    (31, 32),
    (127, 128),
    (255, 256),
    (2431, 2432),
    (4863, 4864),
    (7295, 7296),
)


def check_w_pair(models):
    a, b = models[SINGLE], models[DOUBLE]
    require(a["air"]["sha256"] == b["air"]["sha256"], "W16 pair used different AIRs")
    a, b = (m["plans"][TAG] for m in (a, b))
    require(
        a["contract"]["cube_sha256"] == b["contract"]["cube_sha256"]
        and a["query"]["schedule"]["tiling_hex"][48:] == b["query"]["schedule"]["tiling_hex"][48:],
        "W16 pair changed serialized Cube tiling",
    )
    ra, rb = (x["contract"]["resources"] for x in (a, b))
    fixed = set(ra) - {"weight_bytes", "scratch", "user_ub"}
    require(
        all(ra[k] == rb[k] for k in fixed)
        and rb["weight_bytes"] - ra["weight_bytes"] == 77824
        and ra["scratch"] - rb["scratch"] == rb["user_ub"] - ra["user_ub"] == 77824,
        "unexpected W16 resource delta",
    )


def run(out, args, r):
    cann = Path(os.environ["CANN_ROOT"]).resolve(strict=True)
    tools = {}
    for name in ("atc", "msopgen", "cmake"):
        path = shutil.which(name)
        require(path is not None, "missing tool: " + name)
        tools[name] = c.identity(path)
    base_env = environment(cann, tools)
    files, label, origin = inputs(out, args.gate8_source)
    reference = "supplied Down FP16 reference checked in process"
    runner = runner_binary(out, cann, base_env, reference, abi=ABI)
    types = classes(cann, out)
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
        shape=SHAPE,
        inputs=files,
        input_provenance=origin,
        runner=runner,
        tools=tools,
        models={},
        packages={},
        logical_bytes=LOGICAL_BYTES,
        matrix_flops=FLOPS,
    )
    print(f"{label}  M{M} K{K} N{N}  {M*N} FP16 output values", flush=True)
    modes = [CONTROL, SINGLE, DOUBLE]
    envs = {}
    for mode in modes:
        folder = out / mode
        folder.mkdir()
        r["phase"] = "build " + mode
        print("Building private OPP:", mode, flush=True)
        if mode == CONTROL:
            pkg = baseline_package(folder / "build", cann, base_env)
        else:
            pkg = build(folder / "build", cann, base_env, 1 if mode == SINGLE else 2)
        r["packages"][mode] = pkg
        envs[mode] = environment(cann, tools, pkg["opp"]["install"])
        r["phase"] = "ATC " + mode
        dst = folder / "atc"
        if mode == DOUBLE:
            dst.mkdir()
            shutil.copyfile(c.verify(r["models"][SINGLE]["air"]), dst / "down.air")
            packing = single_packing
        else:
            packing = prepare_air(dst, types, files, mode == CONTROL)
            if mode == SINGLE:
                single_packing = packing
        print("Compiling single Down OM:", mode, flush=True)
        model = compile_model(dst, pkg, envs[mode], atc, mode, packing, r["models"].get(CONTROL))
        r["models"][mode] = model
        proof = model["plans"][TAG]["contract"]
        print(
            f"{mode}  ATC blocks={proof['block_dim']}  baseK={proof['cube']['baseK']}  "
            f"Iterate/N64={proof['outer_iterates_per_n64']}  scratch={proof['resources']['scratch']}",
            flush=True,
        )
    check_w_pair(r["models"])
    images = [
        tuple(x["sha256"] for x in next(iter(m["atc_kernels"].values()))["executable_sections"])
        for m in r["models"].values()
    ]
    require(len(set(images)) == len(modes), "duplicated specialized Down executable")
    if args.compile_only:
        return "COMPILED_NOT_EXECUTED"

    r["phase"] = "DSO admission"
    records = {m: seal(out / m / "dso", r["packages"][m], runner, envs[m]) for m in modes}
    sparse = boundary_case(out / "boundary-inputs", files, shape=SHAPE, pairs=BOUNDARIES)
    r.update(boundary_inputs=sparse, smoke=[], runs=[], statistics={})

    def execute(folder, mode, data, timed, i):
        r["phase"] = str(folder.relative_to(out))
        c.verify_opp(r["packages"][mode]["opp"])
        try:
            return run_child(
                folder,
                runner,
                r["models"][mode],
                envs[mode],
                records[mode],
                data,
                timed,
                reference,
                i,
                abi=ABI,
            )
        except Exception:
            rows = differences(folder, {"golden": {"y": data["golden.bin"]}})
            r["output_differences"] = rows
            for row in rows:
                first = row.get("first_byte", 0) // 2
                print(
                    "FP16 mismatch:",
                    mode,
                    row.get("bit_mismatches"),
                    "first row/column",
                    divmod(first, N),
                    "max finite ULP",
                    row.get("max_finite_ulp"),
                    flush=True,
                )
            raise

    for mode in modes:
        for name, data in (("dense", files), ("boundaries", sparse)):
            row = execute(out / "smoke" / mode / name, mode, data, False, -1)
            r["smoke"].append(dict(row, case=name))
        print("Bitwise + guards PASS", mode, "(40,960 outputs; two inputs, two poisons each)", flush=True)
    for i in range(3):
        for control, variant in PAIRS[i:] + PAIRS[:i]:
            pair = control + "__" + variant
            order = (
                (control, variant) if (i + PAIRS.index((control, variant))) % 2 == 0 else (variant, control)
            )
            print("Round", i + 1, " → ".join(order), flush=True)
            for mode in order:
                row = execute(out / "timing" / str(i) / pair / mode, mode, files, True, i)
                r["runs"].append(dict(row, pair=pair))
    c.verify_tree(files)
    print("C16 single Down Execute+Sync, ms; 90 fresh samples per side of each pair")
    print(
        "control → candidate          control_p50  cand_p50  control_p95  cand_p95  speedup  logical_GB/s  GEMM_TF/s"
    )
    for control, variant in PAIRS:
        pair = control + "__" + variant
        t = pair_statistics(
            [row for row in r["runs"] if row["pair"] == pair],
            variant,
            control,
            flops=FLOPS,
            logical_bytes=LOGICAL_BYTES,
        )
        a, b = t["variants"][control], t["variants"][variant]
        r["statistics"][pair] = t
        print(
            f"{control} → {variant}  {a['median_ms']:.6f}  {b['median_ms']:.6f}  "
            f"{a['p95_ms']:.6f}  {b['p95_ms']:.6f}  {t['speedup']:.4f}x  "
            f"{t['logical_gbps']:.2f}  {t['equivalent_matrix_tflops']:.3f}  {t['decision']}",
            flush=True,
        )

    def go(pair):
        return r["statistics"]["__".join(pair)]["decision"] == "GO"

    if go((CONTROL, DOUBLE)) and (not go((CONTROL, SINGLE)) or go((SINGLE, DOUBLE))):
        selected = DOUBLE
    elif go((CONTROL, SINGLE)):
        selected = SINGLE
    else:
        selected = CONTROL
    r["selected"] = selected
    print("Selected:", selected, flush=True)
    return "MEASURED"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--compile-only", action="store_true", help="stop after both optimized and control OMs")
    p.add_argument(
        "--gate8-source", type=Path, help="optional frozen Gate8 Source report referencing original C16 Down"
    )
    args = p.parse_args()
    out = Path(os.environ["DFLASH_OUTPUT"]) / "experiment"
    out.mkdir()
    r = dict(status="FAILED", phase="setup")
    try:
        r["status"] = run(out, args, r)
    except Exception as error:
        r["error"] = repr(error)
        traceback.print_exc()
    c.write(out / "result.json", r)
    print(r["status"], "REPORT", out / "result.json", flush=True)
    return int(r["status"] == "FAILED")


if __name__ == "__main__":
    sys.exit(main())
