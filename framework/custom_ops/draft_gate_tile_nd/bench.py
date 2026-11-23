#!/usr/bin/env python3
"""One real C16 Layer0 Gate/Up OM A/B. Build, check both outputs, then time Execute+Sync."""

import mmap
from pathlib import Path
import re
import sys

from layout import HERE, K, M, N, OP, OLD_OP, NZ, TILE, require
from package import make_package, command
from scale_contracts import c, flow, host, one, FIRST, FIRST_HASHES, SHAPE, SHAPE_KEY, read_om_json
from gate_common import ABI
from gate_plan import embedded, recheck_embedded, node_binding
from atc_evidence import option
from raw_protocol import stats
from w16_runtime import case_run
from scale_review import allocated_sections

NZ_MODE = "raw-transpose-b16"
TILE_MODE = "tile-nd-b16"
MODES = (NZ_MODE, TILE_MODE)
PILOT_COMPARE = "3b495f2fc68270b97ab7121166b0b39f7227167f39c287de6c3924facc62e297"
PILOT_ELF = "d822a9f91266ebea904bb5022f61d35ec543b20fa503941190ffeb814da14ac6"
PARENT = "0bbc0ee73d673d9141065d6bb73bee22b9714cc96bfe2e03c0477dfa68eae1a3"


def baseline(root):
    path = root / "gate-scale-reviewfix-bXR45o/compare/report.json"
    require(c.sha(path) == PILOT_COMPARE, "unexpected Gate B16 pilot: " + str(path))
    report = c.read(path)
    require(
        report["status"] == "SCALE_GATE_GO" and not report["error"] and not report["integrity_error"],
        "unaccepted pilot",
    )
    prev = report["predecessors"]
    source = c.read(c.verify(prev["source"]))["binding"]
    pkg = c.read(c.verify(prev["build"]))["candidate"]
    model = c.read(c.verify(prev["atc"]))["candidate"]
    src = c.read(c.verify(source["reports"]["source"]))["binding"]
    case = src["cases"]["16"][FIRST]
    files = dict(case["inputs"], golden=case["golden"])
    require(
        case["shape"] == [M, K, N]
        and files == source["frozen"]
        and {k: c.sha(c.verify(v)) for k, v in files.items()} == FIRST_HASHES,
        "wrong real Layer0 inputs/Golden",
    )
    require(
        model["mode"] == NZ_MODE
        and model["private_kernel"] == pkg["kernel"]
        and model["opp_root"] == pkg["opp"]["install"]
        and model["abi"] == ABI
        and model["key"] == FIRST,
        "mixed B16 control",
    )
    c.verify_opp(pkg["opp"])
    bindings = [
        node_binding(n)
        for n in read_om_json(c.verify(model["dump"]))
        if n["type"] == "DFlashGroupQuantLinear"
    ]
    require(len(bindings) == 1 and bindings[0]["sha256"] == PILOT_ELF, "wrong specialized Gate B16 control")
    recheck_embedded(c.verify(model["om"]), bindings, model["atc_kernels"])
    anchor = model["plans"][SHAPE_KEY]
    q = anchor["query"]
    require(
        q
        == flow.read_query(
            c.verify(q["platform_file"]), c.verify(q["scheduling_file"]), pkg["opp"], "ATC", "A", SHAPE
        )
        and q["tiling_sha256"] == PARENT,
        "archived Gate launch drift",
    )
    prior = c.read(c.verify(src["host_source"]))["binding"]["prior"]
    worker = c.read(c.verify(model["worker_report"]))
    require(
        worker["status"] == "GATE8_PROJECTION_AIR_COMPILED_NOT_EXECUTED"
        and worker["om"] == model["om"]
        and worker["deployment"] == model["deployment"],
        "unbound baseline worker",
    )
    argv = worker["atc_command"]
    dep = c.verify(model["deployment"])
    manifest = c.linked(dep.parent, c.read(dep)["air_manifest"])
    graph = c.read(manifest)["graphs"][0]
    air = c.linked(manifest.parent, graph["air"])
    require(
        option(argv, "--model") == str(air)
        and option(argv, "--output") + ".om" == model["om"]["path"]
        and option(argv, "--soc_version") == "Ascend310P3"
        and option(argv, "--precision_mode") == "must_keep_origin_dtype"
        and c.identity(argv[0]) == src["singles"][FIRST]["atc_binary"],
        "unbound control compiler/AIR",
    )
    return src, pkg, model, files, prior, air, argv


def atc_args(argv, air, prefix):
    new = list(argv)
    for flag, path in (("--model", air), ("--output", prefix)):
        old = option(argv, flag)
        new[new.index(flag + "=" + old)] = flag + "=" + str(path)
    return new


def selected_image(node, block_dim, op=OP):
    def field(key, kind):
        return one(
            (x["value"][kind] for x in node["attributes"] if x.get("key") == key),
            "missing or duplicate " + key,
        )

    key = field("_kernel_bin_id", "s")
    symbol = field("_kernelname", "s")
    size = int(field("_tbeKernelSize", "i"))
    require(
        node["type"] == op
        and re.fullmatch(r".+[0-9a-f]{64}", key)
        and re.fullmatch(re.escape(key[:-64]) + r"__kernel\d+", symbol)
        and 64 < size < 16 << 20
        and int(field("tvm_blockdim", "i")) == block_dim
        and field("_ge_attr_op_kernel_lib_name", "s") == "AIcoreEngine",
        "unbound candidate AI Core kernel or blockDim",
    )
    return dict(
        node=node["name"],
        id=key,
        symbol=symbol,
        bytes=size,
        sha256=key[-64:],
        block_dim=block_dim,
        engine="AIcoreEngine",
    )


def prepare_air(out, src, env, original, files):
    out.mkdir()
    air = out / "gate.air"
    command(
        [
            sys.executable,
            "-B",
            HERE / "air.py",
            original,
            air,
            c.verify(files["w_nz.bin"]),
            c.verify(files["s_gn.bin"]),
            src["cann"],
        ],
        out / "pack.log",
        env,
        out,
    )
    packing = c.read(out / "pack.json")
    require(
        packing["weight_node"]
        and packing["nz_sha256"] == files["w_nz.bin"]["sha256"]
        and packing["scale_sha256"] == files["s_gn.bin"]["sha256"]
        and packing["shape"] == list(TILE),
        "pack evidence differs",
    )
    return packing


def compile_gate(out, pkg, env, argv, op, weight):
    air = out / "gate.air"
    platform = out / "platform.jsonl"
    schedule = out / "scheduling.jsonl"
    platform.touch(exist_ok=False)
    schedule.touch(exist_ok=False)
    cache = out / "cache"
    cache.mkdir()
    atc_env = dict(
        env,
        AI_RUN_DIR=str(out),
        ASCEND_WORK_PATH=str(out),
        ASCEND_CACHE_PATH=str(cache),
        DFLASH_PLATFORM_DIAG_PATH=str(platform),
        DFLASH_PLATFORM_DIAG_ROUTE="ATC",
        DFLASH_SCHEDULER_EVIDENCE_PATH=str(schedule),
    )
    args = atc_args(argv, air, out / "gate")
    command(args, out / "atc.log", atc_env, out)
    om = out / "gate.om"
    dump = out / "om-graph.json"
    command([argv[0], "--mode=1", "--om=" + str(om), "--json=" + str(dump)], out / "inspect.log", env, out)
    q = flow.read_query(platform, schedule, pkg["opp"], "ATC", "A", SHAPE)
    nodes = read_om_json(dump)
    operators = [n for n in nodes if n["type"] not in ("Data", "Const", "FileConstant", "NetOutput")]
    node = one(operators, "candidate OM requires exactly one compute node")
    image = selected_image(node, q["platform"]["blockDim"], op)
    require(len(node["input_desc"]) == len(node["inputs"]) == 3, "wrong compiled graph/entry")
    vendor = one(
        (item["value"] for item in node["attributes"] if item["key"] == "custom_op_file_path"),
        "missing selected private vendor",
    )
    require(
        vendor == {"s": str(Path(pkg["opp"]["install"]) / "vendors/customize")}, "ATC selected another OPP"
    )
    w = node["input_desc"][1]
    layout, shape = ("ND", TILE) if op == OP else ("FRACTAL_NZ", NZ)
    require(
        w["layout"] == layout and list(map(int, w["shape"]["dim"])) == list(shape),
        f"ATC changed the {op} weight descriptor: {w}",
    )
    require(
        q["platform"]["blockDim"] == 8 and q["tiling_sha256"] == PARENT,
        "ATC did not select the C16 Gate 8-block K256 plan",
    )
    bins = embedded(om, [image])
    storage = allocated_sections({"om": c.identity(om), "atc_kernels": bins}, node["name"])
    data = c.verify(weight).read_bytes()
    require(len(data) == N * K, "wrong compiled weight extent")
    with om.open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as view:
        offset = view.find(data)
        require(offset >= 0, "OM does not contain the supplied weight bytes")
    receipt = out / "compile.json"
    c.write(
        receipt,
        dict(
            argv=args,
            air=c.identity(air),
            om=c.identity(om),
            dump=c.identity(dump),
            weights=weight,
            packed_constant_offset=offset,
            declared_storage=storage,
            selected=image,
            query=q,
        ),
    )
    return dict(
        mode=TILE_MODE if op == OP else NZ_MODE,
        key=FIRST,
        scope="projection",
        abi=ABI,
        air=c.identity(air),
        om=c.identity(om),
        deployment=c.identity(receipt),
        dump=c.identity(dump),
        atc_kernels=bins,
        plans={SHAPE_KEY: dict(query=q, contract={"block_dim": q["platform"]["blockDim"]})},
        opp_root=pkg["opp"]["install"],
        private_kernel=pkg["kernel"],
    )


def compile_candidate(out, control, pkg, env, argv, packing):
    weight = c.identity(out / "weights.tile-nd.bin")
    require(weight["sha256"] == packing["tile_sha256"], "packed Gate weights changed")
    model = compile_gate(out, pkg, env, argv, OP, weight)
    q = model["plans"][SHAPE_KEY]["query"]
    old = control["plans"][SHAPE_KEY]
    ref = old["query"]
    require(
        q["tiling_sha256"] == ref["tiling_sha256"] == PARENT
        and q["schedule"] == ref["schedule"]
        and {k: v for k, v in q["platform"].items() if k != "host_dso"}
        == {k: v for k, v in ref["platform"].items() if k != "host_dso"},
        "real ATC Cube, workspace or scheduling differs",
    )
    (old_image,) = control["atc_kernels"].values()
    (new_image,) = model["atc_kernels"].values()
    require(
        allocated_sections(model, new_image["node"])["writable"]
        == allocated_sections(control, old_image["node"])["writable"],
        "selected Gate writable storage changed",
    )
    require(
        new_image["executable_sections"] != old_image["executable_sections"],
        "unchanged executable Gate kernel",
    )
    model["plans"][SHAPE_KEY]["contract"] = old["contract"]
    return model, packing


def paired(rows):
    per = {}
    overall = {}
    require(
        [(r["round"], r["mode"]) for r in rows]
        == [(i, m) for i in range(3) for m in (MODES if i % 2 == 0 else MODES[::-1])],
        "timing pair/order incomplete",
    )
    require(len({r["folder"] for r in rows}) == len(rows), "duplicated benchmark execution")
    for mode in MODES:
        picked = [r for r in rows if r["mode"] == mode]
        require(
            all(r["bit_mismatches"] == 0 and r["launch_authorized"] and not r["profiled"] for r in picked),
            "unverified timing",
        )
        per[mode] = [stats(r["validation"]["timing"]["samples_ms"], 30) for r in picked]
        overall[mode] = stats([t for each in per[mode] for t in each["samples_ms"]], 90)
    a, b = (overall[m] for m in MODES)
    return dict(
        variants=overall,
        rounds=per,
        speedup=a["median_ms"] / b["median_ms"],
        reduction_pct=100 * (1 - b["median_ms"] / a["median_ms"]),
        p95_regression=b["p95_ms"] > a["p95_ms"],
        round_ratios=[l["median_ms"] / r["median_ms"] for l, r in zip(per[NZ_MODE], per[TILE_MODE])],
    )


def run(root, out, result):
    src, base, ctl, files, prior, original, argv = baseline(root)
    cann = Path(src["cann"])
    env, _ = host.envs.for_prior(prior, None)
    print("Packing the real C16 Layer0 Gate/Up weights...", flush=True)
    packing = prepare_air(out / "atc", src, env, original, files)
    print("Building tile-ND Gate/Up and compiling its single-op OM...", flush=True)
    build = out / "build"
    build.mkdir()
    candidate = make_package(build, base["generated_sources"], cann, env)
    isolated, candidate["env"] = host.envs.for_prior(prior, candidate["opp"]["install"])
    manifest = build / "dso.json"
    c.write(
        manifest,
        host.dsos.with_private(
            c.read(c.verify(src["standard_manifest"])), candidate["opp"], src["standard_manifest"]
        ),
    )
    candidate.update(manifest=c.identity(manifest), runner=base["runner"])
    model, packing = compile_candidate(out / "atc", ctl, candidate, isolated, argv, packing)
    variants = {NZ_MODE: (base, ctl), TILE_MODE: (candidate, model)}
    result.update(
        frozen=files,
        packing=packing,
        models={m: mdl["om"] for m, (_, mdl) in variants.items()},
        plans={m: mdl["plans"][SHAPE_KEY]["contract"] for m, (_, mdl) in variants.items()},
        kernels={m: mdl["atc_kernels"] for m, (_, mdl) in variants.items()},
    )
    smoke = {}
    spec = dict(abi=ABI, inputs={"x": files["x.bin"]}, golden={"y": files["golden"]})

    def execute(folder, mode, i, timed, batch):
        package, md = variants[mode]
        result["active_run"] = str(folder)
        constants = (
            {"scale": files["s_gn.bin"], "weight": files["w_nz.bin"]}
            if mode == NZ_MODE
            else {"scale": files["s_gn.bin"], "tile_nd_sha256": packing["tile_sha256"], "compiled": md["om"]}
        )
        try:
            case_run(
                folder,
                src,
                package,
                md,
                "16",
                FIRST,
                mode,
                i,
                "layout",
                timed,
                batch,
                smoke.get(mode),
                spec=spec,
                constants=constants,
            )
        except Exception:
            log = folder / "runner.log"
            if log.is_file():
                sys.stderr.write(log.read_text(errors="replace"))
            raise

    result["smoke"] = {"runs": []}
    for m in MODES:
        execute(out / "check" / m, m, 0, False, result["smoke"])
        smoke[m] = result["smoke"]["runs"][-1]
    print("Gate/Up: 311296/311296 FP16 outputs matched Native Golden for both layouts.", flush=True)
    result["benchmark"] = {"runs": []}
    for i in range(3):
        for m in MODES if i % 2 == 0 else MODES[::-1]:
            execute(out / "timing" / str(i) / m, m, i, True, result["benchmark"])
    result.pop("active_run", None)
    return paired(result["benchmark"]["runs"])


if __name__ == "__main__":
    import traceback

    root, out = map(Path, sys.argv[1:])
    if out.resolve() != out or not out.is_dir() or any(out.iterdir()):
        sys.exit("ERROR: expected a fresh empty experiment output directory")
    r = {"status": "FAILED"}
    try:
        r["timing"] = run(root, out, r)
        r["status"] = "MEASURED"
    except (Exception, KeyboardInterrupt) as exc:
        r["error"] = repr(exc)
        traceback.print_exc()
    finally:
        c.write(out / "result.json", r)
    if r["status"] == "MEASURED":
        t = r["timing"]
        print(
            "C16 layer0 Gate/Up, M16 K2560 N19456, blocks=%d, Cube + Group128 dequant, Execute+Sync"
            % r["plans"][NZ_MODE]["block_dim"]
        )
        for m, x in t["variants"].items():
            print(f"{m:20s} Median={x['median_ms']:.6f} ms  P95={x['p95_ms']:.6f} ms  n={x['count']}")
        print(
            "speedup=%.5fx  latency reduction=%.2f%%  round ratios=%s  P95 regression=%s"
            % (
                t["speedup"],
                t["reduction_pct"],
                [round(x, 5) for x in t["round_ratios"]],
                t["p95_regression"],
            )
        )
        print("offline prepack=%.3f ms (one weight, excluded from Execute+Sync)" % r["packing"]["pack_ms"])
    print("REPORT", out / "result.json")
    sys.exit(0 if r["status"] == "MEASURED" else 1)
