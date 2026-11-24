"""C64 A1 panels with a charged device ND -> block-major activation conversion."""

import mmap
from pathlib import Path
import shutil
import struct

import numpy as np
import fc_panel as f
from fc_panel import c, require
from bench import atc_args, selected_image, embedded, read_om_json
from layout import once
from package import command
from platform_contracts import unique_object
import json

M, K, N = 64, f.K, f.N
OP = "DFlashFcPackedPanel"
STEM = "d_flash_fc_packed_panel"
MODES = {64: "fc-pack-a-n64", 160: "fc-pack-a-n160", 320: "fc-pack-a-n320"}
BANKED = "fc-n160-l1-pair"
QUERY = {
    64: (2560, 256, 1, 10, 2, 10),
    160: (1280, 80, 2, 16, 4, 16),
    320: (1280, 32, 2, 1, 4, 2),
}  # PK, baseK, steps and depths


def recipe(width, banked=False):
    require(width in MODES and (not banked or width == 160), "unsupported FC geometry")
    return (640, 80, 8, 8, 8, 8) if banked else QUERY[width]


def header(width, banked=False):
    require(width in MODES, "unknown FC width")
    recipe(width, banked)
    if width == 160:
        return (
            f"#define DFLASH_FC_BANKED_L1 {int(banked)}U\n".encode() + (f.HERE / "n160_spec.h").read_bytes()
        )
    return f"#define DFLASH_FC_COLUMNS {width}U\n".encode() + (f.HERE / "packed_spec.h").read_bytes()


def device_source(banked=False):
    if banked:
        return (f.HERE / "op_kernel/banked.cpp").read_text()
    s = (
        (f.HERE / "op_kernel" / (f.STEM + ".cpp"))
        .read_text()
        .replace('"fc_spec.h"', '"packed_spec.h"')
        .replace("namespace fc_panel", "namespace packed_fc")
        .replace(f.STEM + "(", STEM + "(")
    )
    body = """    const DataCopyParams scatter{LocalK / 16, 1, 0, M - 1};
    for (uint32_t row = 0; row < M; ++row) {
        DataCopy(a[row * 16], x[row * K + k0], scatter);
    }"""
    s = once(s, body, "    DataCopy(a, x[k0 * M], M * LocalK);")
    s = once(
        s,
        "    if (td.globalM == 16) Project<16>(x, w_tiles, s, y, td);\n    else if (td.globalM == 64)",
        "    if (td.globalM == 64)",
    )
    return s.replace(
        "// One contiguous GM segment per row -> local A1 [LocalK/16, M, 16].",
        "// The charged graph transpose has already produced GM [K/16,M,16].",
    )


def host_source(width=64):
    s = (
        f.host_source()
        .replace('"fc_spec.h"', '"packed_spec.h"')
        .replace("namespace fc_panel", "namespace packed_fc")
        .replace(f.STEM, STEM)
        .replace(f.OP, OP)
    )
    for a, b in (
        ("x->GetStorageShape().GetDimNum()!=2", "x->GetStorageShape().GetDimNum()!=3"),
        ("x->GetStorageShape().GetDim(0)", "x->GetStorageShape().GetDim(1)"),
        ("if (rows!=16 && rows!=64)", "if (rows!=64)"),
        ("0, ge::DT_FLOAT16, {M,K}", "0, ge::DT_FLOAT16, {K/16,M,16}"),
        ("x->GetDimNum()!=2", "x->GetDimNum()!=3"),
        ("const int64_t M=x->GetDim(0);", "const int64_t M=x->GetDim(1);"),
        ("if (M!=16 && M!=64)", "if (M!=64)"),
        ("IsShape(ctx->GetInputShape(0),{M,K})", "IsShape(ctx->GetInputShape(0),{K/16,M,16})"),
        ("draft_quant_contract::LaunchBlockCount(N, cores, 0)", "std::min(cores,Tiles)"),
        ("mm.SetFixSplit(M, Columns, 256)", "mm.SetFixSplit(M, Columns, BaseK)"),
        ("cube.get_baseK() != 256", "cube.get_baseK() != BaseK"),
        (
            "cube.get_stepKa() != (M==64 ? 1 : (StreamA ? 5 : 10)) || cube.get_stepKb() != ((!StreamA && M==64) ? 10 : 1)",
            "cube.get_stepKa() != (Wide ? 2 : 1) || cube.get_stepKb() != (Wide ? 1 : 10)",
        ),
        ("a = M*256*2, b = Columns*256*2", "a = M*BaseK*2, b = Columns*BaseK*2"),
    ):
        s = once(s, a, b)
    if width == 160:
        return once(
            s,
            "cube.get_stepKa() != (Wide ? 2 : 1) || cube.get_stepKb() != (Wide ? 1 : 10)",
            "cube.get_stepKa() != StepKa || cube.get_stepKb() != StepKb",
        )
    return s


def scheduler_source():
    p = f.OPS / "draft_a_core_scheduler_flow/op_host/scheduler_evidence.h"
    return once(
        p.read_text(), r",\"tile_n\":64,\"n_tiles\":", r',\"tile_n\":"<<packed_fc::Columns<<",\"n_tiles\":'
    )


def build(out, cann, env, width, *, banked=False):
    spec = out / (OP + ".json")
    entries = c.read(f.OPS / "draft_gate_tile_nd/DFlashGateTileNd.json")
    entries[0]["op"] = OP
    c.write(spec, entries)
    proj = f.generate(out, spec, env)
    host = proj / "op_host"
    dev = proj / "op_kernel"
    config = header(width, banked)
    for folder in (host, dev):
        (folder / "packed_spec.h").write_bytes(config)
    (host / (STEM + ".cpp")).write_text(host_source(width))
    (dev / (STEM + ".cpp")).write_text(device_source(banked))
    helper = f.q.gate.HERE / "panel_vector.h"
    shutil.copyfile(helper, dev / helper.name)
    (host / (STEM + "_tiling.h")).write_text(
        (c.QUANT / "op_host" / (f.OLD_STEM + "_tiling.h")).read_text().replace(f.OLD_OP, OP)
    )
    shutil.copyfile(c.QUANT / "op_host" / (f.OLD_STEM + "_contract.h"), host / (f.OLD_STEM + "_contract.h"))
    shutil.copyfile(f.OPS / "draft_a_core_scheduler_flow/op_host/platform_diag.h", host / "platform_diag.h")
    (host / "scheduler_evidence.h").write_text(scheduler_source())
    preset = proj / "CMakePresets.json"
    f.write_preset(preset, preset, cann)
    return f.install_project(
        out,
        proj,
        cann,
        env,
        OP,
        STEM,
        {"packed_spec.h": f.q.sha(config), helper.name: c.sha(helper)},
        "PRIVATE_PACKED_C64_FC",
    )


def packed_weights(nz, gn, n, k, width, pk):
    require(
        n % width == 0
        and width % 16 == 0
        and k % pk == 0
        and pk % 128 == 0
        and len(nz) == n * k
        and len(gn) == k // 128 * n * 2,
        "incomplete FC pack",
    )
    nk = np.frombuffer(nz, np.int8).reshape(k // 32, n // 16, 16, 32).transpose(1, 2, 0, 3).reshape(n, k)
    scales = np.frombuffer(gn, "<u2").reshape(k // 128, n)
    data = nk.reshape(n // width, width // 16, 16, k // pk, pk).transpose(0, 3, 1, 4, 2).tobytes()
    bits = scales.reshape(k // 128, n // width, width).transpose(1, 0, 2).tobytes()
    decoded = (
        np.frombuffer(data, np.int8)
        .reshape(n // width, k // pk, width // 16, pk, 16)
        .transpose(0, 2, 4, 1, 3)
        .reshape(n, k)
    )
    recovered = (
        np.frombuffer(bits, "<u2")
        .reshape(n // width, k // 128, width)
        .transpose(1, 0, 2)
        .reshape(k // 128, n)
    )
    require(np.array_equal(decoded, nk) and np.array_equal(recovered, scales), "C64 packed constants differ")
    return data, bits


def add_pack_graph(model, width, pk, weights, scales, *, m=M, k=K, n=N):
    g = model.graph[0]
    original = list(g.op)
    del g.op[:]

    def desc(d, name, dt, shape):
        d.name = name
        d.dtype = dt
        d.layout = "ND"
        d.shape.dim[:] = shape
        d.attr["format_for_int"].i = 2
        d.attr["origin_format_for_int"].i = 2
        d.attr["origin_shape"].list.val_type = 2
        d.attr["origin_shape"].list.i[:] = shape
        d.attr["origin_format_is_set"].b = True
        d.attr["origin_shape_initialized"].b = True

    table = {}
    for node in original:
        if node.type == f.OLD_OP:
            for name, data in (("fc_a_dims", (m, k // 16, 16)), ("fc_a_order", (1, 0, 2))):
                p = g.op.add(name=name, type="Const", id=len(g.op))
                d = p.output_desc.add()
                desc(d, "y", 7, [3])
                p.attr["value"].t.desc.CopyFrom(d)
                p.attr["value"].t.data = np.array(data, "<i4").tobytes()
                table[name] = p
            for name, t, edges, ports, shape in (
                ("fc_a_rows", "Reshape", ("x", "fc_a_dims"), ("x", "shape"), (m, k // 16, 16)),
                ("fc_a_pack", "Transpose", ("fc_a_rows", "fc_a_order"), ("x", "perm"), (k // 16, m, 16)),
            ):
                p = g.op.add(name=name, type=t, id=len(g.op))
                for edge, port in zip(edges, ports):
                    p.input.append(edge + ":0")
                    dd = p.input_desc.add()
                    dd.CopyFrom(table[edge].output_desc[0])
                    dd.name = port
                desc(p.output_desc.add(), "y", 2, shape)
                table[name] = p
        p = g.op.add()
        p.CopyFrom(node)
        p.id = len(g.op) - 1
        table[p.name] = p
        if p.name in ("weight", "scale"):
            is_w = p.name == "weight"
            d = p.output_desc[0]
            d.Clear()
            desc(
                d,
                "w_tiles" if is_w else "s",
                3 if is_w else 2,
                (n // width, k // pk, width * pk) if is_w else (n // width, k // 128, width),
            )
            p.attr["value"].t.desc.CopyFrom(d)
            p.attr["value"].t.data = weights if is_w else scales
            p.attr["_out_shape_locked"].b = True
        if p.type == f.OLD_OP:
            p.type = OP
            p.input[0] = "fc_a_pack:0"
            for idx, name in enumerate(("fc_a_pack", "weight", "scale")):
                p.input_desc[idx].CopyFrom(table[name].output_desc[0])
            p.input_desc[0].name = "x"
    return model


def actual_plan(out, pkg, platform, schedule, original, width, *, banked=False):
    def records(file):
        return [json.loads(line, object_pairs_hook=unique_object) for line in file.read_text().splitlines()]

    pp, ss = records(platform), records(schedule)
    require(
        pp and len(pp) == len(ss) and all(x == pp[0] for x in pp) and all(x == ss[0] for x in ss),
        "packed FC callback drift",
    )
    p, s = pp[0], ss[0]
    oldp = original["platform"]
    require(
        {k: v for k, v in p.items() if k != "host_dso"} == {k: v for k, v in oldp.items() if k != "host_dso"}
        and p["blockDim"] == p["availableCores"] == 8
        and p["coreLimit"] == 0,
        "different packed FC platform",
    )
    lib = Path(p["host_dso"]).resolve(strict=True)
    require(
        Path(pkg["opp"]["install"]).resolve() in lib.parents
        and any(
            Path(x["path"]).resolve() == lib and c.sha(lib) == x["sha256"]
            for x in pkg["opp"]["preflight"]["files"]
        ),
        "different packed FC tiler",
    )
    pk, bk, sa, sb, da, db = recipe(width, banked)
    codes = (width if banked else 16) * pk
    scales = pk // 128 * width * 2
    weight_bytes = 4 * 16 * pk
    user = codes + weight_bytes + scales + 8192
    scratch = 262144 - user
    raw = bytes.fromhex(s["tiling_hex"])
    require(
        len(raw) == 224 and struct.unpack_from("<4IQ", raw) == (M, K, N, pk, scratch),
        "packed FC parent differs",
    )
    cube = {v["name"]: struct.unpack_from("<I", raw, 24 + v["offset"])[0] for v in f.schema()}
    fixed = dict(
        singleCoreM=M,
        singleCoreN=width,
        singleCoreK=pk,
        baseM=M,
        baseN=width,
        baseK=bk,
        stepKa=sa,
        stepKb=sb,
        depthA1=da,
        depthB1=db,
        dbL0A=2,
        dbL0B=2,
        dbL0C=1,
        isBias=0,
    )
    require(
        all(cube[k] == v for k, v in fixed.items()) and 0 < cube["transLength"] <= scratch,
        "unexpected packed FC Cube plan",
    )
    r = dict(
        tile_k=pk,
        raw_banks=1,
        user_ub=user,
        scratch=scratch,
        codes_bytes=codes,
        scale_bytes=scales,
        weight_bytes=weight_bytes,
        workspace=p["workspace"],
        l1_used=(2 if banked else 1) * 2 * (M + width) * pk + 2 * bk * (M * da + width * db),
        l0a_used=M * bk * 4,
        l0b_used=width * bk * 4,
        l0c_used=4 * M * width,
    )
    tiles = N // width
    ass = [list(range(i, tiles, p["blockDim"])) for i in range(p["blockDim"])]
    wanted = dict(
        original["schedule"],
        tile_n=width,
        n_tiles=tiles,
        baseline_assignments=ass,
        candidate_assignments=ass,
        resources=r,
        tiling_bytes=len(raw),
        tiling_hex=s["tiling_hex"],
    )
    require(
        s == wanted
        and sorted(i for row in ass for i in row) == list(range(tiles))
        and all(
            r[k] <= p["capacity"][cap]
            for k, cap in (("l1_used", "L1"), ("l0a_used", "L0A"), ("l0b_used", "L0B"), ("l0c_used", "L0C"))
        ),
        "packed FC scheduler/resource contract differs",
    )
    if width == 64:
        require(
            raw.hex() == original["schedule"]["tiling_hex"],
            "activation-only pair changed Cube tiling or buffers",
        )
    return dict(
        platform=p,
        schedule=s,
        tiling_sha256=f.q.sha(raw),
        cube=cube,
        host_dso=c.identity(lib),
        callbacks=len(pp),
    )


def compile_model(out, types, pkg, env, argv, width, archive, *, banked=False):
    pk = recipe(width, banked)[0]
    src = archive["source"]
    nz, gn = (c.verify(src["weights"][name]).read_bytes() for name in ("w_nz.bin", "s_gn.bin"))
    w, s = packed_weights(nz, gn, N, K, width, pk)
    if width == 64:
        require(
            [f.q.sha(b) for b in (w, s)] == [v["sha256"] for v in archive["control"]["packing"]["constants"]],
            "N64 layout changed W/scale bytes",
        )
    proto = f.model_proto(types, nz, gn, projection_shape=(M, K, N), name=f"c64_fc_pack_n{width}")
    add_pack_graph(proto, width, pk, w, s)
    air = out / "fc.air"
    air.write_bytes(proto.SerializeToString(deterministic=True))
    del proto
    pfile = out / "platform.jsonl"
    sfile = out / "scheduling.jsonl"
    pfile.touch()
    sfile.touch()
    cache = out / "cache"
    cache.mkdir()
    local = dict(
        env,
        AI_RUN_DIR=str(out),
        ASCEND_CACHE_PATH=str(cache),
        ASCEND_WORK_PATH=str(out),
        DFLASH_PLATFORM_DIAG_PATH=str(pfile),
        DFLASH_PLATFORM_DIAG_ROUTE="ATC",
        DFLASH_SCHEDULER_EVIDENCE_PATH=str(sfile),
    )
    args = atc_args(argv, air, out / "fc")
    om = out / "fc.om"
    dump = out / "om-graph.json"
    command(args, out / "atc.log", local, out)
    command([argv[0], "--mode=1", "--om=" + str(om), "--json=" + str(dump)], out / "inspect.log", env, out)
    query = actual_plan(out, pkg, pfile, sfile, archive["control"]["query"], width, banked=banked)
    nodes = read_om_json(dump)
    table = {n["name"]: n for n in nodes}
    require(len(nodes) == len(table), "ambiguous compiled FC graph")
    (fc,) = (n for n in nodes if n["type"] == OP)

    def parents(n):
        return [t.rsplit(":", 1)[0] for t in n["inputs"] if t and not t.endswith(":-1")]

    def upstream(name, seen):
        if name not in seen:
            seen.add(name)
            for n in parents(table[name]):
                upstream(n, seen)

    inputs = parents(fc)
    require(len(inputs) == 3, "packed FC edge inventory")
    ancestors = set()
    upstream(inputs[0], ancestors)
    public = [table[k] for k in ancestors if table[k]["type"] == "Data"]
    require(
        len(public) == 1
        and public[0]["output_desc"][0]["layout"] == "ND"
        and fc["input_desc"][0]["layout"] == "ND"
        and list(map(int, public[0]["output_desc"][0]["shape"]["dim"])) == [M, K]
        and list(map(int, fc["input_desc"][0]["shape"]["dim"])) == [K // 16, M, 16],
        "FC public ND input or internal activation descriptor changed",
    )
    skip = {"Const", "FileConstant", "Data", "NetOutput"}
    prep = {n for n in ancestors if table[n]["type"] not in skip}
    require(
        prep and all(n["name"] in prep | {fc["name"]} or n["type"] in skip for n in nodes),
        "activation preparation not bound to runtime input or unexpected compiled work",
    )
    require(
        [a["value"] for a in fc["attributes"] if a["key"] == "custom_op_file_path"]
        == [{"s": str(Path(pkg["opp"]["install"]) / "vendors/customize")}],
        "different packed FC OPP",
    )
    image = selected_image(fc, query["platform"]["blockDim"], OP)
    kernel = embedded(om, [image])[fc["name"]]
    require(
        kernel["executable_sections"] != archive["control"]["selected"]["executable_sections"],
        "packed FC kernel unchanged",
    )
    offsets = []
    with om.open("rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mem:
        for d, shape, data in zip(
            fc["input_desc"][1:], ((N // width, K // pk, width * pk), (N // width, K // 128, width)), (w, s)
        ):
            require(
                d["layout"] == "ND" and tuple(map(int, d["shape"]["dim"])) == shape,
                "packed FC constant descriptor drift",
            )
            off = mem.find(data)
            require(off >= 0, "FC packed constant missing in OM")
            offsets.append(off)
    return dict(
        mode=BANKED if banked else MODES[width],
        op=OP,
        abi=f.abi(M),
        om=c.identity(om),
        dump=c.identity(dump),
        air=c.identity(air),
        selected=kernel,
        query=query,
        prep_nodes=sorted(prep),
        compute_names={n["name"]: n["type"] for n in nodes if n["name"] in prep | {fc["name"]}},
        offsets=offsets,
        weight_sha256=f.q.sha(w),
        scale_sha256=f.q.sha(s),
        argv=args,
    )
