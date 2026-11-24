"""Single Layer0 KV experiment. The public input is always contiguous FP16 ND."""

import argparse
from itertools import combinations
import os
from pathlib import Path
import shutil
import sys
import traceback

import kv_panel as kv
from kv_source import read
import o_runtime
from fc_runtime import pipeline_times
from packed_run import charged_span
from experiment import Stages
from air import protobuf_classes
from fresh_run import environment
from fresh_dso import seal
from run_gate import boundary_case, pair_statistics
from run_q import run_path


def benchmark(out, call, files, gear, left, right):
    runs = []
    for i in range(3):
        order = (right, left) if (i + int(gear == 64)) % 2 else (left, right)
        print(f"C{gear} round {i+1}: " + " → ".join(order), flush=True)
        for mode in order:
            runs.append(call(out / str(i) / mode, mode, files, True, i))
    m, k, n = kv.shape(gear)
    return dict(
        runs=runs,
        statistics=pair_statistics(
            runs, right, left, flops=2 * m * k * n, logical_bytes=kv.logical_bytes(gear)
        ),
    )


def select(modes, pair):
    """All challengers need a fresh win over raw; promotions also need a direct incumbent win."""
    accepted = [mode for mode in modes if pair(kv.RAW, mode)["decision"] == "GO"]
    if not accepted:
        return kv.RAW
    best = accepted[0]
    for mode in accepted[1:]:
        if pair(best, mode)["decision"] == "GO":
            best = mode
    return best


def numeric_rejection(folder):
    """Exclude only an isolated device Golden mismatch after the guarded runner wrote a snapshot."""
    for case in ("measured", "stress", "boundaries"):
        d = folder / case
        log, details = d / "runner.log", d / "mismatches.json"
        if not (log.is_file() and details.is_file()):
            continue
        failures = [line for line in log.read_text(errors="replace").splitlines() if line.startswith("FAIL:")]
        mismatches = kv.c.read(details)
        if failures == ["FAIL: frozen Golden mismatch: y"] and any(
            row.get("count", 0) > 0 for row in mismatches.values()
        ):
            return dict(
                case=case, reason=failures[0], evidence=kv.c.identity(details), log=kv.c.identity(log)
            )
    return None


def run(out, args, state):
    runs = Path(os.environ["DFLASH_RUN_ROOT"]) / "runs"
    old = run_path(runs, args.full_run, "full-draft-panels-")
    prior = run_path(runs, args.resume, "kv-bandwidth-") if args.resume else None
    stages = Stages(out, prior)

    def stage(name, key, action):
        state["phase"] = name
        return stages.run(name, key, action)

    gears = (16, 64) if args.gear == "both" else (int(args.gear),)
    modes = (kv.RAW, *kv.CANDIDATES)
    cann = Path(os.environ["CANN_ROOT"]).resolve(strict=True)
    tools = {k: kv.c.identity(shutil.which(k)) for k in ("atc", "cmake", "msopgen")}
    env = environment(cann, tools)
    descriptor = out / "cann-ge.desc"
    types = protobuf_classes(descriptor.read_bytes())
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
    pkg, src = {}, {}
    old_index = kv.c.identity(old / "stages.json")
    runtime_id = {
        p.name: kv.c.sha(p)
        for p in (
            kv.HERE / "run_kv.py",
            kv.OPS / "draft_o_bandwidth/o_runtime.py",
            kv.OPS / "draft_gate_tile_nd/fresh_run.py",
        )
    }
    for gear in gears:
        src[gear] = stage(
            f"source-c{gear}",
            dict(
                archive=old_index,
                gear=gear,
                schema=kv.c.sha(descriptor),
                code=[kv.c.sha(kv.HERE / "kv_source.py"), kv.c.sha(kv.OPS / "draft_o_bandwidth/o_source.py")],
            ),
            lambda d: read(d, old, types, gear),
        )
        s = src[gear]
        for key in ("reference_air", "weights", "expected_kv", "archive"):
            kv.c.verify_tree(s[key])
        kv.require(kv.RAW not in pkg or pkg[kv.RAW] == s["pkg"], "different raw OPP across gears")
        pkg[kv.RAW] = s["pkg"]
        nb = kv.logical_bytes(gear)
        print(
            f"C{gear} Layer0 KV {kv.shape(gear)}, {nb:,} logical bytes; "
            f"one-pass reference {nb/204800:.2f} us @204.8 GB/s",
            flush=True,
        )
    code = {
        p.name: kv.c.sha(p)
        for p in (
            kv.HERE / "kv_panel.py",
            kv.HERE / "d_flash_kv_panel_host.cpp",
            kv.HERE / "kv_activation.h",
            kv.HERE / "kv_spec.h",
            kv.gate.HERE / "panel_vector.h",
        )
    }
    for mode in kv.CANDIDATES:
        pkg[mode] = stage(
            "build-" + mode,
            dict(
                mode=mode,
                tools=tools,
                headers=code,
                device=kv.sha(kv.device_source().encode()),
                config=kv.sha(kv.header(mode)),
            ),
            lambda d: kv.build(d, cann, env, mode),
        )
    envs = {}
    for mode in modes:
        kv.c.verify_opp(pkg[mode]["opp"])
        envs[mode] = environment(cann, tools, pkg[mode]["opp"]["install"])
    cli = None
    state.update(gears={}, archive=old_index, bandwidth_assumption_GB_s=204.8)
    for gear in gears:
        s = src[gear]
        m, k, n = kv.shape(gear)
        logical = kv.logical_bytes(gear)
        gear_result = state["gears"][str(gear)] = {
            "shape": [m, k, n],
            "one_pass_us": logical / 204800,
            "logical_bytes": logical,
            "gemm_flops": 2 * m * k * n,
            "tflops_at_one_pass": 2 * m * k * n / (logical / 204800) / 1e6,
        }
        bound = stages.refs[f"source-c{gear}"]
        abi = kv.abi(gear)
        exe = stage(
            f"runner-c{gear}",
            dict(
                prefix=s["reference_abi"],
                abi=abi,
                protocol=runtime_id,
                cpp=kv.c.sha(kv.OPS / "draft_gate_raw_atc8/runner/runner.cpp"),
                compiler=tools["cmake"],
            ),
            lambda d: o_runtime.runner(
                d, cann, env, s["reference_abi"], projection_abi=abi, reference=kv.SCOPE
            ),
        )
        kv.c.verify(exe)
        ms = {}
        for mode in modes:
            ms[mode] = stage(
                f"atc-c{gear}-" + mode,
                dict(
                    mode=mode,
                    source=bound,
                    pkg=pkg[mode],
                    atc=tools["atc"],
                    compiler=code["kv_panel.py"],
                    graph_builder=kv.c.sha(kv.OPS / "draft_fc_bandwidth/packed_panel.py"),
                ),
                lambda d: kv.compile_model(d / "compiled", types, pkg[mode], envs[mode], argv, gear, mode, s),
            )
            model = ms[mode]
            kv.c.verify(model["om"])
            pf = model["contract"]
            print(
                f"C{gear} {mode}: ATC blocks={pf['block_dim']} baseK={pf['cube']['baseK']} "
                f"Iterate/N64={pf['iterates_per_n64']} prep={model['prep_nodes']}",
                flush=True,
            )
        opt = [ms[mode] for mode in kv.CANDIDATES]
        kv.require(
            all(
                a["query"]["schedule"]["tiling_hex"] == b["query"]["schedule"]["tiling_hex"]
                and [x["sha256"] for x in a["constants"]] == [x["sha256"] for x in b["constants"]]
                and a["selected"]["executable_sections"] != b["selected"]["executable_sections"]
                for a, b in combinations(opt, 2)
            ),
            "activation-only candidates crossed weight/tiling or code identity",
        )

        def validate(plans):
            kv.plan(plans[f"{m}x{k}x{n}"], s["query"], gear, kv.RAW)

        ref = stage(
            f"reference-om-c{gear}",
            dict(source=bound, atc=tools["atc"], protocol=runtime_id),
            lambda d: o_runtime.compile_prefix(d, s, envs[kv.RAW], argv, validate_site=validate),
        )
        if args.compile_only:
            continue
        dsos = {mode: seal(out / f"dso-c{gear}-{mode}", pkg[mode], exe, envs[mode]) for mode in modes}
        cases = {}
        for label in ("measured", "stress"):
            kv.c.verify_tree(s["replays"][label])
            result = stage(
                f"capture-c{gear}-" + label,
                dict(source=bound, om=ref["om"], runner=exe, replay=s["replays"][label], protocol=runtime_id),
                lambda d: o_runtime.capture(
                    d, exe, ref, envs[kv.RAW], dsos[kv.RAW], s, label, input_shape=(m, k)
                ),
            )
            cases[label] = result["files"]
        cases["boundaries"] = stage(
            f"boundaries-c{gear}",
            dict(
                weights=s["weights"],
                shape=[m, k, n],
                code=kv.c.sha(kv.OPS / "draft_gate_bandwidth/run_gate.py"),
            ),
            lambda d: boundary_case(
                d / "inputs",
                cases["measured"],
                shape=(m, k, n),
                pairs=(
                    (0, 2559),
                    (15, 16),
                    (31, 32),
                    (127, 128),
                    (159, 160),
                    (255, 256),
                    (1279, 1280),
                    (2431, 2432),
                ),
            ),
        )
        kv.c.verify_tree(cases)

        def call(path, mode, files, timed, i):
            return o_runtime.execute(
                path, exe, ms[mode], envs[mode], dsos[mode], files, timed, i, abi=abi, reference=kv.SCOPE
            )

        def smoke(d, mode):
            rows = {key: call(d / key, mode, files, False, -1) for key, files in cases.items()}
            print(
                f"C{gear} bitwise + guards PASS {mode}: {m*n:,} outputs, three inputs, two poisons",
                flush=True,
            )
            return rows

        admitted = []
        gear_result["rejected"] = {}
        for mode in modes:
            smoke_name = f"smoke-c{gear}-" + mode
            try:
                stage(
                    smoke_name,
                    dict(cases=cases, model=ms[mode]["om"], runner=exe, protocol=runtime_id),
                    lambda d: smoke(d, mode),
                )
            except ValueError:
                rejected = numeric_rejection(out / smoke_name) if mode != kv.RAW else None
                if not rejected:
                    raise
                gear_result["rejected"][mode] = rejected
                print(f"C{gear} {mode} REJECTED: {rejected['reason']} ({rejected['case']})", flush=True)
                continue
            if mode != kv.RAW:
                admitted.append(mode)
        kv.require(admitted, f"C{gear}: all KV candidates failed correctness")
        gear_result["pairs"] = {}

        def measure(left, right):
            pair_id = left + "__" + right
            name = f"benchmark-c{gear}-" + pair_id
            result = stage(
                name,
                dict(
                    runner=exe,
                    smoke=[stages.refs[f"smoke-c{gear}-" + v] for v in (left, right)],
                    inputs=cases["measured"],
                    protocol=runtime_id,
                    stats=kv.c.sha(kv.OPS / "draft_gate_bandwidth/run_gate.py"),
                ),
                lambda d: benchmark(d, call, cases["measured"], gear, left, right),
            )
            stat = result["statistics"]
            a, b = (stat["variants"][v] for v in (left, right))
            gear_result["pairs"][pair_id] = dict(
                report=stages.refs[name],
                speedup=stat["speedup"],
                decision=stat["decision"],
                reduction_pct=stat["reduction_pct"],
                round_ratios=stat["round_ratios"],
                median_ms=[a["median_ms"], b["median_ms"]],
                p95_ms=[a["p95_ms"], b["p95_ms"]],
            )
            print(
                f"C{gear} KV Execute+Sync incl. input prep {left} → {right}: "
                f"{a['median_ms']:.6f}/{a['p95_ms']:.6f} → {b['median_ms']:.6f}/{b['p95_ms']:.6f} ms "
                f"(Median/P95, 90/90) {stat['speedup']:.4f}x {stat['decision']}",
                flush=True,
            )
            return stat

        best = select(admitted, measure)
        gear_result["selected"] = best
        print(f"Selected C{gear}: {best}", flush=True)
        if args.no_profile:
            continue
        from panel_profiler import bind_msprof

        if cli is None:
            cli = bind_msprof(out, cann, envs[kv.RAW])

        def profile(d, mode):
            row = o_runtime.profile(
                d,
                exe,
                ms[mode],
                envs[mode],
                dsos[mode],
                cases["measured"],
                cli,
                abi=abi,
                role=kv.ROLE,
                reference=kv.SCOPE,
                logical_bytes=logical,
            )
            row["pipeline"] = pipeline_times(d, row, module=kv.ROLE)
            row["span"] = charged_span(d, ms[mode], row, module=kv.ROLE, logical_bytes=logical)
            if row["span"]["status"] != "UNRESOLVED":
                row["span"]["equivalent_gemm_tflops"] = 2 * m * k * n / row["span"]["median_us"] / 1e6
            return row

        gear_result["profiles"] = {}
        for mode in (kv.RAW, *admitted):
            name = f"profile-c{gear}-" + mode
            row = stage(
                name,
                dict(
                    smoke=stages.refs[f"smoke-c{gear}-" + mode],
                    cli=cli["file"],
                    runner=exe,
                    inputs=cases["measured"],
                    protocol=runtime_id,
                ),
                lambda d: profile(d, mode),
            )
            gear_result["profiles"][mode] = stages.refs[name]
            span = row["span"]
            print(f"C{gear} {mode} device KV {row['kernel']['p50_task_sum_ms']*1000:.2f} us", end="; ")
            if span["status"] == "UNRESOLVED":
                print("prep→KV interval UNRESOLVED", span["reason"], flush=True)
            else:
                print(
                    f"prep→KV Median/P95 {span['median_us']:.2f}/{span['p95_us']:.2f} us, "
                    f"{span['logical_gbps']:.2f} logical GB/s, {span['equivalent_gemm_tflops']:.3f} GEMM TF/s, "
                    f"{span['floor_ratio']:.3f}x one-pass, "
                    f"{span['status']}",
                    flush=True,
                )
            print("  matched counters", row["pipeline"]["joined"], row["pipeline"]["medians"], flush=True)
    return (
        "COMPILED_NOT_EXECUTED"
        if args.compile_only
        else "MEASURED" if args.no_profile else "MEASURED_AND_PROFILED"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gear", choices=("16", "64", "both"), default="both")
    p.add_argument("--full-run", default="full-draft-panels-UFqlys")
    p.add_argument("--resume")
    flags = p.add_mutually_exclusive_group()
    flags.add_argument("--no-profile", action="store_true")
    flags.add_argument("--compile-only", action="store_true")
    args = p.parse_args()
    out = Path(os.environ["DFLASH_OUTPUT"])
    state = dict(status="FAILED", phase="setup")
    try:
        state["status"] = run(out, args, state)
    except Exception as error:
        state["error"] = repr(error)
        traceback.print_exc()
    kv.c.write(out / "result.json", state)
    print(state["status"], "REPORT", out / "result.json", flush=True)
    return int(state["status"] == "FAILED")


if __name__ == "__main__":
    sys.exit(main())
