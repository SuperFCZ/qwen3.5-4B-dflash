"""Private Cube-ready panels for one 310P3 C16 Gate/Up projection."""

import mmap
from pathlib import Path
import shutil
import struct
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
OPS = HERE.parent
sys.path.insert(0, str(OPS / "draft_gate_tile_nd"))
from package import c, generate, install_project, write_preset
from bench import ABI, FIRST, M, K, N, SHAPE_KEY, compile_image, embedded, require
from layout import OLD_OP, OLD_STEM, sha
from gate_plan import schema
from tile_gate_graph import model_proto

OP = "DFlashGatePanel"
STEM = "d_flash_gate_panel"
PANELS = (256, 512, 1280, 2560)


def pack_inputs(nz, gn, panel, *, k=K, n=N):
    require(
        panel in PANELS
        and k % panel == 0
        and n % 64 == 0
        and len(nz) == n * k
        and len(gn) == 2 * k // 128 * n,
        "wrong panel input extents",
    )
    start = time.perf_counter_ns()
    nk = np.frombuffer(nz, np.int8).reshape(k // 32, n // 16, 16, 32).transpose(1, 2, 0, 3).reshape(n, k)
    shape = (n // 64, k // panel, 4, panel, 16)
    codes = nk.reshape(n // 64, 4, 16, k // panel, panel).transpose(0, 3, 1, 4, 2).tobytes()
    original_scale = np.frombuffer(gn, dtype="<u2").reshape(k // 128, n)
    scales = original_scale.reshape(k // 128, n // 64, 64).transpose(1, 0, 2).tobytes()
    elapsed = (time.perf_counter_ns() - start) / 1e6
    decoded = np.frombuffer(codes, np.int8).reshape(shape).transpose(0, 2, 4, 1, 3).reshape(n, k)
    inverse = (
        np.frombuffer(scales, dtype="<u2")
        .reshape(n // 64, k // 128, 64)
        .transpose(1, 0, 2)
        .reshape(k // 128, n)
    )
    require(
        np.array_equal(nk, decoded) and np.array_equal(original_scale, inverse),
        "panel packing inverse differs",
    )
    return codes, scales, elapsed


def make_air(folder, types, files, panel):
    folder.mkdir()
    nz = c.verify(files["w_nz.bin"]).read_bytes()
    gn = c.verify(files["s_gn.bin"]).read_bytes()
    codes, scales, ms = pack_inputs(nz, gn, panel)
    model = model_proto(types, nz, gn)
    graph = model.graph[0]
    nodes = {node.name: node for node in graph.op}
    op = nodes[OLD_OP]
    op.type = OP
    shapes = ((N // 64, K // panel, panel * 64), (N // 64, K // 128, 64))
    constants = []
    for i, (name, port, shape, data) in enumerate(
        zip(("weight", "scale"), ("w_tiles", "s"), shapes, (codes, scales)), 1
    ):
        node = nodes[name]
        desc = node.output_desc[0]
        desc.name = port
        desc.layout = "ND"
        desc.shape.dim[:] = shape
        desc.attr["format_for_int"].i = 2
        desc.attr["origin_shape"].list.i[:] = shape
        node.attr["value"].t.desc.CopyFrom(desc)
        op.input_desc[i].CopyFrom(desc)
        node.attr["value"].t.data = data
        node.attr["_out_shape_locked"].b = True
        file = folder / (name + ".bin")
        file.write_bytes(data)
        constants.append(c.identity(file))
    (folder / "gate.air").write_bytes(model.SerializeToString())
    return dict(
        panel_k=panel,
        pack_ms=ms,
        physical_shapes=shapes,
        constants=constants,
        original_nz=files["w_nz.bin"],
        original_gn=files["s_gn.bin"],
    )


def panel_header(panel, w16_banks=1):
    require(panel in PANELS, "unsupported Gate panel")
    require(w16_banks == 1 or (panel == K and w16_banks == 2), "invalid W16 buffer count")
    config = f"#define DFLASH_PANEL_K {panel}U\n#define DFLASH_W16_BANKS {w16_banks}U\n"
    return config.encode() + (HERE / "panel_spec.h").read_bytes()


def build(folder, cann, env, panel, *, w16_banks=1):
    folder.mkdir()
    rows = c.read(OPS / "draft_gate_tile_nd/DFlashGateTileNd.json")
    rows[0]["op"] = OP
    op_schema = folder / (OP + ".json")
    c.write(op_schema, rows)
    project = generate(folder, op_schema, env)
    hdr = panel_header(panel, w16_banks)
    for sub in ("op_kernel", "op_host"):
        name = "full_l1.cpp" if panel == K and sub == "op_kernel" else STEM + ".cpp"
        shutil.copyfile(HERE / sub / name, project / sub / (STEM + ".cpp"))
        (project / sub / "panel_spec.h").write_bytes(hdr)
    helpers = HERE / "panel_vector.h"
    shutil.copyfile(helpers, project / "op_kernel" / helpers.name)
    tiling = (c.QUANT / "op_host" / (OLD_STEM + "_tiling.h")).read_text().replace(OLD_OP, OP)
    (project / "op_host" / (STEM + "_tiling.h")).write_text(tiling)
    for name in ("platform_diag.h", "scheduler_evidence.h"):
        shutil.copyfile(OPS / "draft_a_core_scheduler_flow/op_host" / name, project / "op_host" / name)
    base = OLD_STEM + "_contract.h"
    shutil.copyfile(c.QUANT / "op_host" / base, project / "op_host" / base)
    preset = project / "CMakePresets.json"
    write_preset(preset, preset, cann)
    return install_project(
        folder,
        project,
        cann,
        env,
        OP,
        STEM,
        {"panel_spec.h": sha(hdr), helpers.name: c.sha(helpers)},
        "PRIVATE_GATE_CUBE_PANEL",
    )


def validate_plan(query, base, panel, *, w16_banks=1):
    require(w16_banks == 1 or (panel == K and w16_banks == 2), "invalid W16 plan")
    platform = query["platform"]
    baseline = base["plans"][SHAPE_KEY]["query"]
    require(
        {k: v for k, v in platform.items() if k != "host_dso"}
        == {k: v for k, v in baseline["platform"].items() if k != "host_dso"}
        and platform["blockDim"] == 8,
        "Gate platform, blockDim or workspace changed",
    )
    sched = query["schedule"]
    require(
        sched["baseline_assignments"] == baseline["schedule"]["baseline_assignments"], "N64 ownership differs"
    )
    banks = 1 if panel in (1280, K) else 2
    raw_columns = {256: 64, 512: 64, 1280: 32, 2560: 16}[panel]
    codes = banks * raw_columns * panel
    scales = banks * 2 * (panel // 128) * 64
    w = w16_banks * 2 * (16 if panel == K else 64) * panel
    l1_inputs = 2 * K * (M + 64) if panel == K else 0
    user = codes + scales + w + 8192
    scratch = platform["capacity"]["UB"] - user
    data = bytes.fromhex(sched["tiling_hex"])
    require(
        len(data) == 224 and struct.unpack_from("<4IQ", data) == (M, K, N, panel, scratch),
        "wrong parent tiling",
    )
    cube = {field["name"]: struct.unpack_from("<I", data, 24 + field["offset"])[0] for field in schema()}
    fixed = dict(
        singleCoreM=M,
        singleCoreN=64,
        singleCoreK=panel,
        baseM=M,
        baseN=64,
        baseK=256,
        stepKa=panel // 256,
        stepKb=panel // 256,
        dbL0C=1,
    )
    require(
        all(cube[k] == v for k, v in fixed.items()) and 0 < cube["transLength"] <= scratch,
        "actual Cube plan differs from panel recipe",
    )
    resources = dict(
        raw_banks=banks,
        tile_k=panel,
        codes_bytes=codes,
        scale_bytes=scales,
        weight_bytes=w,
        user_ub=user,
        scratch=scratch,
        workspace=platform["workspace"],
        l1_used=l1_inputs + 8192 * cube["depthA1"] + 32768 * cube["depthB1"],
        l0a_used=8192 * cube["dbL0A"],
        l0b_used=32768 * cube["dbL0B"],
        l0c_used=4096 * cube["dbL0C"],
    )
    require(sched["resources"] == resources, "actual per-core memory differs from declared allocations")
    return dict(
        block_dim=8,
        cube=cube,
        resources=resources,
        outer_iterates_per_n64=K // panel,
        resident_l1_input_bytes=l1_inputs,
        w16_banks=w16_banks,
        fp32_base_k=256,
        tiling_sha256=query["tiling_sha256"],
    )


def validate_w16_pair(query, base, air):
    ref = base["plans"][SHAPE_KEY]["query"]
    require(
        base["mode"] == f"panel-k{K}" and c.sha(air) == base["air"]["sha256"],
        "W16 pair requires the measured full-K control and identical AIR",
    )
    a, b = ref["schedule"], query["schedule"]
    require(
        bytes.fromhex(a["tiling_hex"])[24:] == bytes.fromhex(b["tiling_hex"])[24:],
        "W16 buffering changed the serialized Cube tiling",
    )
    fixed = set(a["resources"]) - {"user_ub", "weight_bytes", "scratch"}
    require(
        all(a["resources"][k] == b["resources"][k] for k in fixed)
        and a["resources"]["weight_bytes"] == 81920
        and b["resources"]["weight_bytes"] == 163840
        and b["resources"]["user_ub"] - a["resources"]["user_ub"] == 81920
        and a["resources"]["scratch"] - b["resources"]["scratch"] == 81920,
        "unexpected W16 UB, L1 or workspace delta",
    )


def compile_panel(folder, pkg, env, argv, base, packing, *, w16_banks=1):
    panel = packing["panel_k"]
    args, om, dump, q, node, image = compile_image(folder, pkg, env, argv, OP)
    contract = validate_plan(q, base, panel, w16_banks=w16_banks)
    if w16_banks == 2:
        validate_w16_pair(q, base, folder / "gate.air")
    offsets = []
    with om.open("rb") as fp, mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ) as view:
        for desc, shape, item in zip(
            node["input_desc"][1:], packing["physical_shapes"], packing["constants"]
        ):
            require(
                desc["layout"] == "ND" and tuple(map(int, desc["shape"]["dim"])) == tuple(shape),
                "ATC changed panel constant descriptor",
            )
            offset = view.find(c.verify(item).read_bytes())
            require(offset >= 0, "compiled OM lacks the packed constant")
            offsets.append(offset)
    bins = embedded(om, [image])
    (old,) = base["atc_kernels"].values()
    require(
        bins[image["node"]]["executable_sections"] != old["executable_sections"], "unchanged control kernel"
    )
    receipt = folder / "compile.json"
    c.write(receipt, dict(argv=args, packing=packing, constant_offsets=offsets, selected=bins, query=q))
    return dict(
        mode=f"panel-k{panel}" + ("-w2" if w16_banks == 2 else ""),
        key=FIRST,
        scope="projection",
        abi=ABI,
        air=c.identity(folder / "gate.air"),
        om=c.identity(om),
        dump=c.identity(dump),
        deployment=c.identity(receipt),
        atc_kernels=bins,
        plans={SHAPE_KEY: dict(query=q, contract=contract)},
        opp_root=pkg["opp"]["install"],
        private_kernel=pkg["kernel"],
    )
