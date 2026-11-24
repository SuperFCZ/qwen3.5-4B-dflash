"""Specialize the measured Q next-N kernel for C16 O using K2048 UB chunks in a full K4096 Cube panel."""

from pathlib import Path
import mmap
import shutil
import struct
import sys

HERE = Path(__file__).resolve().parent
OPS = HERE.parent
sys.path.insert(0, str(OPS / "draft_q_bandwidth"))
import q_panel as q
from q_panel import c, require, OLD_OP, OLD_STEM
from bench import compile_image, embedded
from gate_plan import schema
from layout import once
from package import generate, install_project, write_preset
from tile_gate_graph import model_proto

M, K, N = 16, 4096, 2560
SHAPE = (M, K, N)
TAG = "16x4096x2560"
ROLE = "layer0.o"
OP, STEM = "DFlashOPanel", "d_flash_o_panel"
CONTROL = "raw-nz-brcb"
PANEL = "o-k4096-v2048-next-n"
ABI = dict(
    inputs=[dict(name="x", dtype="float16", shape=[M, K])],
    outputs=[dict(name="y", dtype="float16", shape=[M, N])],
)
BYTES = K * N + 2 * (K // 128) * N + 2 * M * (K + N)
FLOPS = 2 * M * K * N
FLOOR_MS = BYTES / 204.8e6


def device_source():
    # Preserve the proven Q outer schedule, event order and failure handling verbatim. Piece i
    # is N16 stripe i//Chunks, with the selected K chunk i%Chunks inside that original stripe.
    src = (q.HERE / "op_kernel/next_n.cpp").read_text()
    for old, new in [
        ('"q_spec.h"', '"o_spec.h"'),
        ("using namespace q_panel;", "using namespace o_panel;"),
        (q.STEM + "(", STEM + "("),
        ("g < Groups; ++g", "g < GroupsPerChunk; ++g"),
        (
            "scales[g * Columns + stripe * 16]",
            "scales[((stripe % Chunks) * GroupsPerChunk + g) * Columns + (stripe / Chunks) * 16]",
        ),
        ("stripe + 1 < 4", "stripe + 1 < Pieces"),
        ("stripe < 4; ++stripe", "stripe < Pieces; ++stripe"),
        ("stripe < 3) Pair", "stripe + 1 < Pieces) Pair"),
    ]:
        src = once(src, old, new)
    return src


def host_source():
    src = (
        q.host_source()
        .replace(q.OP, OP)
        .replace(q.STEM, STEM)
        .replace("namespace q_panel", "namespace o_panel")
    )
    src = once(src, '"q_spec.h"', '"o_spec.h"')
    return once(src, "cube.get_stepKb() != PanelK/256", "cube.get_stepKb() != 1")


def build(out, cann, env):
    definition = c.read(OPS / "draft_gate_tile_nd/DFlashGateTileNd.json")
    definition[0]["op"] = OP
    spec = out / (OP + ".json")
    c.write(spec, definition)
    project = generate(out, spec, env)
    host, dev = project / "op_host", project / "op_kernel"
    header = HERE / "o_spec.h"
    for folder in (host, dev):
        shutil.copyfile(header, folder / header.name)
    (host / (STEM + ".cpp")).write_text(host_source())
    (dev / (STEM + ".cpp")).write_text(device_source())
    helper = q.gate.HERE / "panel_vector.h"
    shutil.copyfile(helper, dev / helper.name)
    (host / (STEM + "_tiling.h")).write_text(
        (c.QUANT / "op_host" / (OLD_STEM + "_tiling.h")).read_text().replace(OLD_OP, OP)
    )
    shutil.copyfile(c.QUANT / "op_host" / (OLD_STEM + "_contract.h"), host / (OLD_STEM + "_contract.h"))
    for name in ("platform_diag.h", "scheduler_evidence.h"):
        shutil.copyfile(OPS / "draft_a_core_scheduler_flow/op_host" / name, host / name)
    preset = project / "CMakePresets.json"
    write_preset(preset, preset, cann)
    return install_project(
        out,
        project,
        cann,
        env,
        OP,
        STEM,
        {header.name: c.sha(header), helper.name: c.sha(helper)},
        "PRIVATE_C16_O_PANEL",
    )


def plan(query, mode, original):
    p, s = query["platform"], query["schedule"]
    require(
        mode in (CONTROL, PANEL)
        and p["availableCores"] == p["blockDim"] == 8
        and p["coreLimit"] == 0
        and p["workspace"] == 2097152
        and p["capacity"]["UB"] == 262144,
        "wrong O platform or workspace",
    )
    require(
        {k: v for k, v in p.items() if k != "host_dso"}
        == {k: v for k, v in original["platform"].items() if k != "host_dso"}
        and s["baseline_assignments"]
        == [list(range(i, N // 64, p["blockDim"])) for i in range(p["blockDim"])],
        "different O platform or N64 ownership",
    )
    raw = bytes.fromhex(s["tiling_hex"])
    require(len(raw) == 224 and q.sha(raw) == query["tiling_sha256"], "invalid O parent tiling")
    cube = {f["name"]: struct.unpack_from("<I", raw, 24 + f["offset"])[0] for f in schema()}
    if mode == CONTROL:
        require(
            query["tiling_sha256"] == original["tiling_sha256"]
            and s["resources"] == original["schedule"]["resources"],
            "O raw plan differs from full Draft",
        )
    else:
        fixed = dict(
            singleCoreM=M,
            singleCoreN=64,
            singleCoreK=K,
            baseM=M,
            baseN=64,
            baseK=256,
            stepKa=16,
            stepKb=1,
            depthA1=16,
            depthB1=2,
            dbL0A=2,
            dbL0B=2,
            dbL0C=1,
            isBias=0,
        )
        expected = dict(
            raw_banks=1,
            tile_k=4096,
            codes_bytes=32768,
            scale_bytes=4096,
            weight_bytes=131072,
            user_ub=176128,
            scratch=86016,
            workspace=p["workspace"],
            l1_used=851968,
            l0a_used=16384,
            l0b_used=65536,
            l0c_used=4096,
        )
        require(
            struct.unpack_from("<4IQ", raw) == (M, K, N, K, expected["scratch"])
            and all(cube[k] == v for k, v in fixed.items())
            and 0 < cube["transLength"] <= expected["scratch"]
            and s["resources"] == expected,
            "O Cube/UB resource drift",
        )
        require(
            all(
                expected[k] <= p["capacity"][cap]
                for k, cap in (
                    ("l1_used", "L1"),
                    ("l0a_used", "L0A"),
                    ("l0b_used", "L0B"),
                    ("l0c_used", "L0C"),
                )
            ),
            "O on-chip capacity exceeded",
        )
    return dict(
        cube=cube,
        resources=s["resources"],
        block_dim=p["blockDim"],
        iterates_per_n64=K // s["resources"]["tile_k"],
    )


def compile_o(folder, types, pkg, env, argv, mode, src):
    files = src["weights"]
    if mode == CONTROL:
        folder.mkdir()
        consts = [files["w_nz.bin"], files["s_gn.bin"]]
        proto = model_proto(
            types, *(c.verify(f).read_bytes() for f in consts), projection_shape=SHAPE, name="c16_o"
        )
        g = proto.graph[0]
        name = src["o_node"]
        for node in g.op:
            if node.name == OLD_OP:
                node.name = name
            for i, v in enumerate(node.input):
                if v == OLD_OP + ":0":
                    node.input[i] = name + ":0"
        g.output[:] = [name + ":0"]
        (folder / "o.air").write_bytes(proto.SerializeToString(deterministic=True))
        packing = dict(constants=consts, physical_shapes=((K // 32, N // 16, 16, 32), (K // 128, N)))
    else:
        packing = q.gate.make_projection_air(
            folder, types, files, K, shape=SHAPE, op_type=OP, model_name="o", graph_name="c16_o"
        )
    typ = OLD_OP if mode == CONTROL else OP
    args, om, dump, query, node, image = compile_image(
        folder, pkg, env, argv, typ, shape=SHAPE, model_name="o"
    )
    proof = plan(query, mode, src["query"])
    selected = embedded(om, [image])[node["name"]]
    same = selected["executable_sections"] == src["compiled_o"]["executable_sections"]
    require(same == (mode == CONTROL), "wrong O machine text relative to the full Draft control")
    with om.open("rb") as fd, mmap.mmap(fd.fileno(), 0, access=mmap.ACCESS_READ) as blob:
        offsets = []
        for desc, fmt, shape, item in zip(
            node["input_desc"][1:],
            ("FRACTAL_NZ" if mode == CONTROL else "ND", "ND"),
            packing["physical_shapes"],
            packing["constants"],
        ):
            require(
                desc["layout"] == fmt and tuple(map(int, desc["shape"]["dim"])) == tuple(shape),
                "wrong O OM constant descriptor",
            )
            offset = blob.find(c.verify(item).read_bytes())
            require(offset >= 0, "O compiled constant payload mismatch")
            offsets.append(offset)
    return dict(
        mode=mode,
        op=typ,
        abi=ABI,
        argv=args,
        om=c.identity(om),
        air=c.identity(folder / "o.air"),
        dump=c.identity(dump),
        selected=selected,
        query=query,
        contract=proof,
        packing=packing,
        offsets=offsets,
    )
