#!/usr/bin/env python3
"""Independent C16 Layer0 Down 2-D DMA fallback. Every command runs one stage only."""

import argparse
from pathlib import Path
import shutil
import struct
import sys
import traceback

from w16_common import (
    GATE,
    GATE8,
    HERE,
    GATE8_ROOT,
    PROFILE_ROOT,
    c,
    command,
    flow,
    historical,
    host,
    one,
    replace_once,
    require,
    sources,
    stats,
    tree,
)
from w16_source import admit, recovered, verify_source
from w16_export import compile_air, protected_graph
from w16_runtime import case_run, mismatch_evidence, order, paired, verify_runtime_row
from gate_common import option, payloads
from gate_plan import embedded, node_binding, non_gate, recheck_embedded
from c_profile_graph import read_om_json
from flow_query import read_query
from reference import ordered_half

KEY = "layer-0-down"
SHAPE = [16, 9728, 2560]
TAG = "16x9728x2560"
LOOP, TWO = "down-loop", "down-2d"
ABI = {
    "inputs": [dict(name="x", dtype="float16", shape=[16, 9728])],
    "outputs": [dict(name="y", dtype="float16", shape=[16, 2560])],
}
STAGES = ("source", "build", "atc", "smoke", "benchmark")


def dma_extents():
    result = {}
    for name, repeat, length, gap in (("codes", 8, 64, 2496), ("scales", 2, 4, 156)):
        starts = []
        for n in range(0, 2560, 64):
            for k in range(0, 9728, 256):
                if name == "codes":
                    original = [((k // 32 + i) * 160 + n // 16) * 512 for i in range(repeat)]
                    total = 24903680
                else:
                    original = [((k // 128 + i) * 2560 + n) * 2 for i in range(repeat)]
                    total = 389120
                moved = [original[0] + 32 * (length + gap) * i for i in range(repeat)]
                require(moved == original and moved[-1] + 32 * length <= total, "DMA crosses frozen tensor end")
                starts.extend(moved)
        require(sorted(starts) == list(range(0, total, 32 * length)), "Down source overlap or hole")
        result[name] = dict(
            descriptor=[repeat, length, gap, 0],
            unit_bytes=32,
            tensor_bytes=total,
            last_exclusive=max(starts) + 32 * length,
            spans_checked=len(starts),
        )
    return result


def dma_sdk(cann):
    root = Path(cann).resolve(strict=True)
    header = c.identity(root / "aarch64-linux/asc/include/basic_api/kernel_struct_data_copy.h")
    impl = c.identity(root / "aarch64-linux/asc/impl/basic_api/dav_m200/kernel_operator_data_copy_impl.h")
    require(
        all(root in Path(x["path"]).parents for x in (header, impl))
        and header["sha256"] == "8402ed99bc1be3c05f688fb92e9262531ebce6199ba953c827dcef2a47bf5343"
        and impl["sha256"] == "8a890aa80c39380519bcba461bfd19a67060e650253622d8b49b4b2c01ea3c8f",
        "unreviewed 310P basic DataCopy descriptor or GM2UB",
    )
    return dict(parameters=header, m200_gm_to_ub=impl, extents=dma_extents())


def down_node(graph, case):
    draft = graph["runtime_input_abi"]["draft_ac"]
    nodes = [n for n in draft["nodes"] if n["ge_type"] == GATE]
    w, s = case["inputs"]["w_nz.bin"], case["inputs"]["s_gn.bin"]
    n = one(
        (n for n in nodes if n["nz_sha256"] == w["sha256"] and n["scale_gn_sha256"] == s["sha256"]),
        "Down weight/Scale binding",
    )
    require(
        n["roundtrip"] == "BIT_EXACT"
        and n["storage_shape"] == [304, 160, 16, 32]
        and n["scale_shape"] == [76, 2560]
        and (n["k"], n["n"], n["nz_bytes"]) == (9728, 2560, 24903680),
        "Down NZ/GN contract",
    )
    return n


def historical_down(src, old):
    orig = c.read(c.verify(src["history"]["source"]))["binding"]["cases"]["16/" + KEY]
    c.verify_tree(orig)
    require(
        orig["name"] == KEY
        and orig["shape"] == SHAPE
        and orig["golden"]["bytes"] == 81920
        and {k: v["bytes"] for k, v in orig["inputs"].items()}
        == {"x.bin": 311296, "s_gn.bin": 389120, "w_nz.bin": 24903680},
        "Down input extents",
    )
    agg = c.verify(src["single_suite"])
    row = one((r for r in c.read(agg)["cases"] if r["name"] == "c16-" + KEY), "missing frozen Down AIR")
    require(row["status"] == "PASS", "Down probe not accepted")
    folder = agg.parent / ("a-" + row["name"])
    paths = {name: c.linked(folder, row["artifacts"][name]) for name in ("air", "deployment", "om")}
    require(all(folder in p.parents for p in paths.values()), "Down historical path escape")
    air = c.read(paths["air"])
    dep = c.read(paths["deployment"])
    require(
        dep["status"] == "PASS"
        and len(dep["graphs"]) == len(air["graphs"]) == 1
        and c.linked(paths["deployment"].parent, dep["air_manifest"]) == paths["air"],
        "Down AIR deployment",
    )
    graph = air["graphs"][0]
    custom = graph["runtime_input_abi"]["draft_ac"]
    require(
        graph["metadata"]["tensor_abi"] == ABI
        and graph["role"] == "diagnostic"
        and graph["input_names"] == ["x"]
        and graph["output_names"] == ["y"]
        and custom["status"] == "PASS"
        and custom["counts"] == {GATE: 1, "DFlashDraftLmHeadTop1": 0, "DFlashDraftLmHeadTop1Audit": 0},
        "not single Down A-only graph",
    )
    single = down_node(graph, orig)
    full = down_node(c.read(c.verify(old["deployment"]))["graphs"][0], orig)
    require(
        all(
            single[k] == full[k]
            for k in ("weight_sha256", "scale_gn_sha256", "nz_sha256", "k", "n", "nz_bytes")
        ),
        "Down full/single constants differ",
    )
    compiled = dep["graphs"][0]
    require(
        all(graph[k] == compiled[k] for k in ("metadata", "runtime_input_abi", "custom_op_audit", "air"))
        and c.linked(paths["deployment"].parent, compiled["om"]) == paths["om"],
        "Down historical model binding",
    )
    cmd = compiled["atc_command"]
    ref = src["singles"]["layer-0-gate_up"]
    require(
        c.identity(cmd[0]) == ref["atc_binary"]
        and all(
            option(cmd, k) == option(ref["command"], k)
            for k in ("--mode", "--framework", "--soc_version", "--precision_mode", "--deterministic")
        )
        and option(cmd, "--model") == str(c.linked(paths["air"].parent, graph["air"]))
        and option(cmd, "--output") + ".om" == str(paths["om"]),
        "Down compiler identity",
    )
    frozen = dict(
        air=c.identity(paths["air"]),
        deployment=c.identity(paths["deployment"]),
        om=c.identity(paths["om"]),
        payloads=payloads(paths["air"], air),
        command=cmd,
        atc_binary=ref["atc_binary"],
        compiler_identity_text=dep["compiler"]["identity"],
    )
    return dict(
        case=orig,
        frozen=frozen,
        anchor=old["atc_kernels"][full["name"]],
        plan=old["plans"][TAG],
        node=single["name"],
    )


def build_runner(out, cann, env):
    src = out / "runner-src"
    src.mkdir()
    shutil.copyfile(GATE8 / "runner/CMakeLists.txt", src / "CMakeLists.txt")
    text = (GATE8 / "runner/runner.cpp").read_text()
    text = replace_once(
        text,
        "inputs[0].shape!=std::vector<int64_t>({16,2560})",
        "inputs[0].shape!=std::vector<int64_t>({16,9728})",
    )
    text = replace_once(
        text,
        "outputs[0].shape!=std::vector<int64_t>({16,19456})",
        "outputs[0].shape!=std::vector<int64_t>({16,2560})",
    )
    text = replace_once(
        text,
        "single Gate frozen Native Golden checked in process",
        "single Down frozen Native Golden checked in process",
    )
    (src / "runner.cpp").write_text(text)
    build = out / "runner"
    command(
        [
            "cmake",
            "-S",
            src,
            "-B",
            build,
            "-DCANN_ROOT=" + str(cann),
            "-DCOMMON_INCLUDE=" + str(c.QUANT / "test"),
        ],
        out / "runner-configure.log",
        env,
    )
    command(["cmake", "--build", build, "--parallel", "2"], out / "runner-compile.log", env)
    binary = c.identity(build / "gate_om_runner")
    return dict(runner=binary, deps=c.dependencies(binary, out / "runner-ldd.log", env))


def compile_down(folder, src, selected, d, mode):
    child, dump = compile_air(folder, src, selected, d["frozen"], "projection")
    q = read_query(
        c.verify(child["platform"]), c.verify(child["scheduling"]), selected["opp"], "ATC", "A", SHAPE
    )
    non_gate(q, d["plan"]["query"])
    nodes = [node_binding(n) for n in read_om_json(dump) if n["type"] == GATE]
    code = embedded(c.verify(child["om"]), nodes)
    require(
        set(code) == {d["node"]}
        and down_node(c.read(c.verify(child["deployment"]))["graphs"][0], d["case"])["name"] == d["node"],
        "unbound Down implementation",
    )
    equal = code[d["node"]]["executable_sections"] == d["anchor"]["executable_sections"]
    require(equal == (mode == LOOP), "Down control does not match Raw/full or 2-D code unchanged")
    return dict(
        om=child["om"],
        abi=ABI,
        mode=mode,
        key=KEY,
        air=d["frozen"]["air"],
        dump=c.identity(dump),
        deployment=child["deployment"],
        worker=c.identity(folder / "compiled/report.json"),
        private_kernel=selected["kernel"],
        opp_root=selected["opp"]["install"],
        plans={TAG: dict(query=q, contract=d["plan"]["contract"])},
        atc_kernels=code,
    )


def verify_down(row, spec, timed):
    e = c.read(c.verify(row["execution"]))
    contract = dict(
        scope="single Down frozen Native Golden checked in process",
        status="PASS",
        device=0,
        guards=True,
        readonly=True,
        repeat=True,
        cpu_fallback=False,
        golden_match=True,
        poison_verified=True,
        sync_completed=True,
        timed_log_stable=True,
        verification_calls=4 if timed else 2,
        row_permutation_checked=False,
    )
    t = e["timing"]
    protocol = dict(
        protocol="continuous-v1" if timed else "checked-v1",
        status="MEASURED" if timed else "NOT_RUN",
        warmup=5 if timed else 0,
        repetitions=30 if timed else 0,
        correctness_before_calls=2,
        correctness_after_calls=int(timed),
        timed_tail_checked=timed,
        per_timed_call_readback=not timed,
        per_timed_call_poison=not timed,
    )
    for actual, want in ((e, contract), (t, protocol)):
        require(
            all(type(actual.get(k)) is type(v) and actual[k] == v for k, v in want.items()),
            "Down safety/timing protocol",
        )
    require(
        t["prepare"]["samples_ms"] == []
        and all(type(e[k]) is int and e[k] > 0 for k in ("workspace_bytes", "model_weight_bytes")),
        "Down allocation/prepare",
    )
    samples = t["execute_sync"]["samples_ms"]
    require(timed or samples == [], "timing in Down Smoke")
    golden = c.verify(spec["golden"]["y"]).read_bytes()
    require(
        len(golden) == 81920 and all((u & 0x7C00) != 0x7C00 for (u,) in struct.iter_unpack("<H", golden)),
        "invalid Down Golden",
    )
    require(
        set(row["outputs"]) == {"y"}
        and len(row["outputs"]["y"]) == (4 if timed else 2)
        and all(c.verify(f).read_bytes() == golden for f in row["outputs"]["y"]),
        "40960 Down outputs differ",
    )
    return dict(
        status="DOWN_40960_FP16_BITWISE_MATCH",
        workspace_bytes=e["workspace_bytes"],
        model_weight_bytes=e["model_weight_bytes"],
        timing=stats(samples, 30) if timed else None,
        profiled=False,
    )


def differences(folder, spec):
    rows = mismatch_evidence(folder, spec)
    for row in rows:
        data = c.verify(row["output"]).read_bytes()
        gold = c.verify(row["golden"]).read_bytes()
        if len(data) == len(gold) == 81920:
            pairs = [
                (a, b)
                for (a,), (b,) in zip(struct.iter_unpack("<H", data), struct.iter_unpack("<H", gold))
                if a != b
            ]
            ulps = [
                abs(ordered_half(a) - ordered_half(b))
                for a, b in pairs
                if (a & 0x7C00) != 0x7C00 and (b & 0x7C00) != 0x7C00
            ]
            row.update(bit_mismatches=len(pairs), max_finite_ulp=max(ulps, default=None))
    return rows


def run(args):
    root = args.root.resolve()
    protected = [args.gate8_root.resolve(), args.profile_root.resolve(), c.REPO, Path("/usr/local/Ascend")]
    require(
        all(root != p and root not in p.parents and p not in root.parents for p in protected),
        "overlapping roots",
    )
    phase = args.phase
    if phase == "source":
        require(not root.exists() or not any(root.iterdir()), "new independent Down root required")
    else:
        require(
            (root / "source/report.json").is_file()
            and c.read(root / "source/report.json").get("track") == "down-copy",
            "not a Down root",
        )
    out = c.fresh(root / phase, protected)
    r = dict(
        schema=1,
        track="down-copy",
        phase=phase,
        root=str(root),
        sources=sources(),
        predecessors={},
        status="BLOCKED",
        device_execution="NOT_RUN",
        atc="NOT_RUN",
        benchmark="NOT_RUN",
        error=None,
        integrity_error=None,
    )
    chain = {}
    try:
        c.clean()
        for stage in STAGES[: STAGES.index(phase)]:
            file = root / stage / "report.json"
            prev = c.read(file)
            require(
                prev["phase"] == stage
                and prev["track"] == "down-copy"
                and prev["root"] == str(root)
                and prev["status"] == "DOWN_" + stage.upper() + "_PASS"
                and not prev["error"]
                and not prev["integrity_error"]
                and prev["predecessors"] == r["predecessors"]
                and prev["sources"] == r["sources"],
                "Down predecessor " + stage,
            )
            historical.verify_tree(prev, {})
            chain[stage] = prev
            r["predecessors"][stage] = c.identity(file)
        if phase == "source":
            r["binding"] = admit(protected[0], protected[1], needs_machine_decoder=False)
            src, raw, old = recovered(r["binding"])
            r["down"] = historical_down(src, old["full-export"]["16"])
            r["copy_contract"] = dma_sdk(src["cann"])
        else:
            verify_source(chain["source"]["binding"], protected[0], protected[1])
            src, raw, old = recovered(chain["source"]["binding"])
            d = chain["source"]["down"]
            require(
                d == historical_down(src, old["full-export"]["16"])
                and chain["source"]["copy_contract"] == dma_sdk(src["cann"]),
                "changed frozen Down evidence",
            )
            prior = c.read(c.verify(src["host_source"]))["binding"]["prior"]
            if phase == "build":
                worker = out / TWO
                worker.mkdir()
                dest = worker / "package"
                build_env, _ = host.envs.for_prior(prior, dest / "opp", ready=False)
                req = worker / "request.json"
                c.write(
                    req,
                    dict(variant=TWO, dest=str(dest), source=r["predecessors"]["source"], sources=r["sources"]),
                )
                command(
                    [sys.executable, "-B", HERE / "private_worker.py", req], worker / "builder.log", build_env
                )
                r["worker"] = c.identity(worker / "worker.json")
                sealed = c.read(c.verify(r["worker"]))
                require(
                    sealed["status"] == "PRIVATE_W16_PACKAGE_BUILT" and sealed["variant"] == TWO,
                    "Down package failed",
                )
                r["candidate"] = sealed["result"]
                env = flow.build_env(prior, raw)
                r["runner"] = build_runner(out, src["cann"], env)
                require(r["runner"]["deps"] == raw["runner"]["deps"], "Down runner dependency drift")
            else:
                b = chain["build"]
                selected = {LOOP: dict(raw, runner=b["runner"]), TWO: dict(b["candidate"], runner=b["runner"])}
                for mode, s in selected.items():
                    c.verify_opp(s["opp"])
                    require(tree(Path(s["opp"]["install"])) == s["package_files"], "Down private OPP drift")
                if phase == "atc":
                    r["atc"] = "ATTEMPTED"
                    r["models"] = {m: compile_down(out / m, src, s, d, m) for m, s in selected.items()}
                    protected_graph(
                        c.verify(r["models"][LOOP]["dump"]),
                        c.verify(r["models"][TWO]["dump"]),
                        {d["node"]},
                        raw_opp_root=selected[LOOP]["opp"]["install"],
                        new_opp_root=selected[TWO]["opp"]["install"],
                    )
                    r["atc"] = "TWO_FROZEN_AIR_OMS_COMPILED"
                else:
                    require(args.allow_device, "explicit --allow-device required")
                    models = chain["atc"]["models"]
                    spec = dict(
                        abi=ABI, inputs={"x": d["case"]["inputs"]["x.bin"]}, golden={"y": d["case"]["golden"]}
                    )
                    timed = phase == "benchmark"
                    previous = {}
                    for m, model in models.items():
                        require(
                            model["air"] == d["frozen"]["air"]
                            and model["opp_root"] == selected[m]["opp"]["install"],
                            "mixed Down ATC assets",
                        )
                        node = one(
                            (
                                node_binding(x)
                                for x in read_om_json(c.verify(model["dump"]))
                                if x["type"] == GATE
                            ),
                            "one Down node required",
                        )
                        recheck_embedded(c.verify(model["om"]), [node], model["atc_kernels"])
                        q = model["plans"][TAG]["query"]
                        require(
                            q
                            == read_query(
                                c.verify(q["platform_file"]),
                                c.verify(q["scheduling_file"]),
                                selected[m]["opp"],
                                "ATC",
                                "A",
                                SHAPE,
                            ),
                            "Down callback drift",
                        )
                        non_gate(q, d["plan"]["query"])
                    options = dict(
                        spec=spec,
                        plan_key=TAG,
                        check_outputs=verify_down,
                        elements=40960,
                        constants={k: v for k, v in d["case"]["inputs"].items() if k != "x.bin"},
                    )
                    if timed:
                        for row in chain["smoke"]["runs"]:
                            mode = row["mode"]
                            verify_runtime_row(
                                root / "smoke" / mode / "0",
                                row,
                                src,
                                selected[mode],
                                models[mode],
                                False,
                                None,
                                **options,
                            )
                            previous[mode] = row
                        require(set(previous) == {LOOP, TWO}, "missing Down Smoke")
                    keys = (
                        order("16", [KEY], ["down-copy"])
                        if timed
                        else [("16", KEY, "smoke", 0, m) for m in (LOOP, TWO)]
                    )
                    r.update(
                        runs=[], device_execution="ATTEMPTED", benchmark="ATTEMPTED" if timed else "NOT_RUN"
                    )
                    for g, l, p, i, m in keys:
                        folder = out / m / str(i)
                        case_run(
                            folder,
                            src,
                            selected[m],
                            models[m],
                            g,
                            l,
                            m,
                            i,
                            p,
                            timed,
                            r,
                            previous.get(m),
                            diagnose=differences,
                            **options,
                        )
                        verify_runtime_row(
                            folder,
                            r["runs"][-1],
                            src,
                            selected[m],
                            models[m],
                            timed,
                            previous.get(m),
                            **options,
                        )
                    require(
                        len({row["outputs"]["y"][0]["sha256"] for row in r["runs"]}) == 1,
                        "Down cross-variant mismatch",
                    )
                    r["device_execution"] = "COMPLETED"
                    if timed:
                        r["statistics"] = paired(r["runs"], "16", KEY, "down-copy")
                        r["benchmark"] = "MEASURED"
        r["status"] = "DOWN_" + phase.upper() + "_PASS"
        if phase == "benchmark" and r["statistics"]["decision"] != "GO":
            r["status"] = "DOWN_NO_GO"
    except Exception as exc:
        r["error"] = repr(exc)
        (out / "traceback.txt").write_text(traceback.format_exc())
    finally:
        try:
            require(r["sources"] == sources(), "source drift")
            historical.verify_tree(r, {})
            c.clean()
        except Exception as exc:
            r.update(status="BLOCKED", integrity_error=repr(exc))
        c.write(out / "report.json", r)
    print(r["status"], r["error"], r["integrity_error"], out / "report.json")
    return 1 if r["status"] == "BLOCKED" else 3 if r["status"] == "DOWN_NO_GO" else 0


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=STAGES)
    cli.add_argument("--root", type=Path, required=True)
    cli.add_argument("--gate8-root", type=Path, default=GATE8_ROOT)
    cli.add_argument("--profile-root", type=Path, default=PROFILE_ROOT)
    cli.add_argument("--allow-device", action="store_true")
    sys.exit(run(cli.parse_args()))
