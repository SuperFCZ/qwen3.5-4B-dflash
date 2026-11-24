"""Direct comparison against an already measured single-raw K2560 Q. No prior device stage is replayed."""

from pathlib import Path
import os
import shutil

import q_panel as q
from experiment import Stages
from fresh_run import environment
from fresh_dso import seal
from run_gate import pair_statistics
from q_runtime import execute, profile_kernel
from q_timing import select, PAIRS

NEXT = "q-k2560-w2-next-n"


def measured(out, root):
    old = Stages(out, root)
    src = old.completed("source")["result"]
    base = old.completed("atc-" + q.PANEL)["result"]
    pkg = old.completed("build-" + q.PANEL)["result"]
    runner = old.completed("runner")["result"]
    original = old.completed("device-reference")["result"]["files"]
    dense = old.completed("dense-reference")["result"]["files"]
    sm = old.completed("smoke")
    bench = old.completed("benchmark")
    br = bench["result"]
    previous = q.c.read(root / "result.json")
    q.require(
        previous["status"] in ("MEASURED", "MEASURED_AND_PROFILED")
        and previous["benchmark_report"] == old.refs["benchmark"]
        and sm["contract"]["reference"] == old.refs["device-reference"]
        and sm["contract"]["dense"] == old.refs["dense-reference"]
        and sm["contract"]["runner"] == runner
        and old.refs["atc-" + q.PANEL] in sm["contract"]["models"]
        and bench["contract"]["smoke"] == old.refs["smoke"],
        "incomplete or crossed original Q measurement",
    )
    stats = {}
    for a, b in PAIRS:
        key = a + "__" + b
        rows = [r for r in br["runs"] if r["pair"] == key]
        for r in rows:
            e = q.c.read(q.c.verify(r["execution"]))
            q.require(
                e["status"] == "PASS"
                and e["golden_match"]
                and e["sync_completed"]
                and r["validation"]["timing"]["samples_ms"] == e["timing"]["execute_sync"]["samples_ms"],
                "archived sample drift",
            )
        stats[key] = pair_statistics(rows, b, a, flops=q.FLOPS, logical_bytes=q.BYTES)
    q.require(
        stats == br["statistics"] == previous["statistics"]
        and select(stats) == br["selected"] == previous["selected"] == q.PANEL,
        "requires measured q-k2560-w2 winner",
    )
    q.c.verify(runner)
    q.c.verify_opp(pkg["opp"])
    q.c.verify(pkg["source"])
    q.c.verify(pkg["host_source"])
    q.c.verify(base["om"])
    q.c.verify(base["air"])
    q.c.verify_tree(src["weights"])
    cases = {
        "measured-tokens": original,
        "dense-token-stress": dense,
        "boundaries": sm["result"]["boundaries"],
    }
    for label, data in cases.items():
        q.c.verify_tree(data)
        q.require(all(data[k] == v for k, v in src["weights"].items()), "different Q weight/scale: " + label)
    return dict(source=src, model=base, runner=runner, package=pkg, cases=cases, receipts=old.refs)


def build(folder, cann, env, baseline):
    # Fresh SDK scaffold; bind the exact host, tiling and buffer definitions of the selected control.
    schema = q.c.read(q.OPS / "draft_gate_tile_nd/DFlashGateTileNd.json")
    schema[0]["op"] = q.OP
    definition = folder / (q.OP + ".json")
    q.c.write(definition, schema)
    project = q.generate(folder, definition, env)
    original = q.c.verify(baseline["host_source"]).parent
    q.require((original / (q.STEM + ".cpp")).read_text() == q.host_source(), "unexpected archived Q Host")
    expected = {
        q.STEM
        + "_tiling.h": (q.c.QUANT / "op_host" / (q.OLD_STEM + "_tiling.h"))
        .read_text()
        .replace(q.OLD_OP, q.OP)
        .encode(),
        "q_spec.h": q.header(q.PANEL),
        q.OLD_STEM + "_contract.h": (q.c.QUANT / "op_host" / (q.OLD_STEM + "_contract.h")).read_bytes(),
    }
    for name in ("platform_diag.h", "scheduler_evidence.h"):
        expected[name] = (q.OPS / "draft_a_core_scheduler_flow/op_host" / name).read_bytes()
    host, dev = project / "op_host", project / "op_kernel"
    shutil.copyfile(original / (q.STEM + ".cpp"), host / (q.STEM + ".cpp"))
    for name, contents in expected.items():
        q.require((original / name).read_bytes() == contents, "archived host header differs: " + name)
        (host / name).write_bytes(contents)
    dyn = q.c.verify(baseline["source"])
    q.require(
        q.c.sha(dyn) == q.c.sha(q.HERE / "op_kernel" / (q.STEM + ".cpp")),
        "archived control implementation changed",
    )
    helper = q.gate.HERE / "panel_vector.h"
    for name, contents in (("panel_vector.h", helper.read_bytes()), ("q_spec.h", expected["q_spec.h"])):
        q.require((dyn.parent / name).read_bytes() == contents, "different control helper: " + name)
        (dev / name).write_bytes(contents)
    shutil.copyfile(q.HERE / "op_kernel/next_n.cpp", dev / (q.STEM + ".cpp"))
    preset = project / "CMakePresets.json"
    q.write_preset(preset, preset, cann)
    return q.install_project(
        folder,
        project,
        cann,
        env,
        q.OP,
        q.STEM,
        {"q_spec.h": q.sha(expected["q_spec.h"]), "panel_vector.h": q.c.sha(helper)},
        "PRIVATE_Q_NEXT_N",
    )


def compile_next(out, types, pkg, env, atc, archive):
    base = archive["model"]
    m = q.compile_q(out / "compiled", types, pkg, env, atc, q.PANEL, archive["source"], packed_control=base)
    q.require(
        m["air"]["sha256"] == base["air"]["sha256"]
        and m["query"]["schedule"]["tiling_hex"] == base["query"]["schedule"]["tiling_hex"]
        and m["contract"] == base["contract"]
        and m["selected"]["executable_sections"] != base["selected"]["executable_sections"],
        "next-N requires identical AIR, full SDK tiling/resources and different machine code",
    )
    return dict(m, mode=NEXT)


def run(out, args, state, root, prior):
    from air import protobuf_classes

    stages = Stages(out, prior)

    def stage(name, contract, action):
        state["phase"] = name
        return stages.run(name, contract, action)

    cann = Path(os.environ["CANN_ROOT"]).resolve(strict=True)
    tools = {n: q.c.identity(shutil.which(n)) for n in ("atc", "msopgen", "cmake")}
    shared = environment(cann, tools)
    state["phase"] = "read measured control"
    archive = measured(out, root)
    runner = archive["runner"]
    cases = archive["cases"]
    print("Reusing measured single-raw Q OM", archive["model"]["om"]["sha256"], flush=True)
    source = q.c.identity(q.HERE / "op_kernel/next_n.cpp")
    pkg = stage(
        "build-next-n",
        dict(
            control=archive["receipts"]["build-" + q.PANEL],
            tools=tools,
            device=source,
            builder=q.c.sha(Path(__file__)),
        ),
        lambda d: build(d, cann, shared, archive["package"]),
    )
    q.c.verify_opp(pkg["opp"])
    envs = {
        m: environment(cann, tools, p["opp"]["install"])
        for m, p in ((q.PANEL, archive["package"]), (NEXT, pkg))
    }
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
    types = protobuf_classes((out / "cann-ge.desc").read_bytes())
    model = stage(
        "atc-next-n",
        dict(
            build=stages.refs["build-next-n"],
            control=archive["receipts"]["atc-" + q.PANEL],
            source=archive["receipts"]["source"],
            compiler=q.c.sha(q.HERE / "q_panel.py"),
        ),
        lambda d: compile_next(d, types, pkg, envs[NEXT], atc, archive),
    )
    q.c.verify(model["om"])
    state.update(
        parent_run=str(root),
        runner=runner,
        control_om=archive["model"]["om"],
        candidate_om=model["om"],
        conditional_floor_ms=q.FLOOR_MS,
    )
    print(
        "Q next-N: same 8-block SDK plan, 46,592 B scratch, single raw, two W16, one Iterate/N64", flush=True
    )
    if args.compile_only:
        return "COMPILED_NOT_EXECUTED"
    models = {q.PANEL: archive["model"], NEXT: model}
    packages = {q.PANEL: archive["package"], NEXT: pkg}
    state["phase"] = "DSO binding"
    dsos = {m: seal(out / ("dso-" + m), p, runner, envs[m]) for m, p in packages.items()}

    def call(d, m, files, timed, rd):
        q.c.verify_opp(packages[m]["opp"])
        return execute(d, runner, models[m], envs[m], dsos[m], files, timed, rd)

    def smoke(d):
        rows = []
        for m in models:
            for label, files in cases.items():
                rows.append(dict(call(d / m / label, m, files, False, -1), case=label))
            print("Bitwise + guards PASS", m, "(three cases, two poisons each)", flush=True)
        return rows

    common = dict(
        atc=stages.refs["atc-next-n"],
        prior=archive["receipts"]["benchmark"],
        runner=runner,
        cases=cases,
        protocol=q.c.sha(q.HERE / "q_runtime.py"),
    )
    stage("smoke", common, smoke)

    def measure(d):
        rows = []
        for i in range(3):
            order = (NEXT, q.PANEL) if i == 1 else (q.PANEL, NEXT)
            print(f"Round {i+1}: " + " → ".join(order), flush=True)
            for m in order:
                rows.append(call(d / str(i) / m, m, cases["measured-tokens"], True, i))
        return dict(
            runs=rows, statistics=pair_statistics(rows, NEXT, q.PANEL, flops=q.FLOPS, logical_bytes=q.BYTES)
        )

    stat = stage("benchmark", dict(common, smoke=stages.refs["smoke"]), measure)["statistics"]
    ctrl, cand = (stat["variants"][m] for m in (q.PANEL, NEXT))
    state.update(
        statistics=stat,
        benchmark_report=stages.refs["benchmark"],
        selected=NEXT if stat["decision"] == "GO" else q.PANEL,
    )
    print(
        f"Q direct Execute+Sync Median/P95, ms: {ctrl['median_ms']:.6f}/{ctrl['p95_ms']:.6f} → "
        f"{cand['median_ms']:.6f}/{cand['p95_ms']:.6f}, {stat['speedup']:.4f}x, {stat['decision']} | Selected: {state['selected']}",
        flush=True,
    )
    if args.no_profile:
        return "MEASURED"
    from panel_profiler import bind_msprof

    state["phase"] = "msprof"
    cli = bind_msprof(out, cann, envs[q.PANEL])
    state["profiles"] = {}
    for m in models:
        row = stage(
            "profile-" + m,
            dict(benchmark=stages.refs["benchmark"], cli=cli["file"], flags=cli["flags"], mode=m),
            lambda d: profile_kernel(d, runner, models[m], envs[m], dsos[m], cases["measured-tokens"], cli),
        )
        state["profiles"][m] = row
        print(
            f"{m} device task Median/P95 {1000*row['kernel']['p50_task_sum_ms']:.2f}/{1000*row['kernel']['p95_task_sum_ms']:.2f} us, "
            f"{row['logical_gbps']:.2f} logical GB/s, {row['kernel_floor_ratio']:.3f}x reference, {row['attribution']}",
            flush=True,
        )
    return "MEASURED_AND_PROFILED"
