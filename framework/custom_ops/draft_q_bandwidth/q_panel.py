"""Private M16/K2560/N4096 Q panel operators and their real ATC contracts."""

import mmap
from pathlib import Path
import shutil
import struct
import sys

HERE = Path(__file__).resolve().parent
OPS = HERE.parent
sys.path[:0] = [str(OPS / "draft_gate_bandwidth"), str(OPS / "draft_full_panels")]
import gate_panel as gate
from bench import compile_image, embedded
from package import c, generate, install_project, write_preset
from layout import OLD_OP, OLD_STEM, sha, require
from gate_plan import schema
from tile_gate_graph import model_proto

M, K, N = 16, 2560, 4096
SHAPE = (M, K, N)
TAG = "16x2560x4096"
ROLE = "layer0.q"
OP = "DFlashQPanel"
STEM = "d_flash_q_panel"
CONTROL = "raw-nz-brcb"
PANEL = "q-k2560-w2"
PREFETCH = "q-k2560-w2-prefetch"
MODES = (CONTROL, PANEL, PREFETCH)
ABI = dict(
    inputs=[dict(name="x", dtype="float16", shape=[M, K])],
    outputs=[dict(name="y", dtype="float16", shape=[M, N])],
)
REF_ABI = dict(
    inputs=[dict(name="embedding", dtype="float16", shape=[1, M, K])],
    outputs=[*ABI["inputs"], *ABI["outputs"]],
)
BYTES = K * N + 2 * (K // 128) * N + 2 * M * (K + N)
FLOPS = 2 * M * K * N
FLOOR_MS = BYTES / 204.8e6


def header(mode):
    require(mode in (PANEL, PREFETCH), "unknown optimized Q mode")
    return (
        f"#define DFLASH_Q_RAW_LOOKAHEAD {int(mode==PREFETCH)}\n".encode() + (HERE / "q_spec.h").read_bytes()
    )


def host_source():
    s = (gate.HERE / "op_host/d_flash_gate_panel.cpp").read_text()
    return (
        s.replace('"panel_spec.h"', '"q_spec.h"')
        .replace(gate.STEM, STEM)
        .replace(gate.OP, OP)
        .replace("namespace gate_panel", "namespace q_panel")
        .replace("Gate panel SDK", "Q panel SDK")
    )


def build(out, cann, env, mode):
    out.mkdir()
    definition = c.read(OPS / "draft_gate_tile_nd/DFlashGateTileNd.json")
    definition[0]["op"] = OP
    spec = out / (OP + ".json")
    c.write(spec, definition)
    project = generate(out, spec, env)
    host, device = project / "op_host", project / "op_kernel"
    hdr = header(mode)
    for folder in (host, device):
        (folder / "q_spec.h").write_bytes(hdr)
    (host / (STEM + ".cpp")).write_text(host_source())
    shutil.copyfile(HERE / "op_kernel" / (STEM + ".cpp"), device / (STEM + ".cpp"))
    helper = gate.HERE / "panel_vector.h"
    shutil.copyfile(helper, device / helper.name)
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
        {"q_spec.h": sha(hdr), helper.name: c.sha(helper)},
        "PRIVATE_C16_Q_PANEL",
    )


def plan(q, mode, original):
    p, s = q["platform"], q["schedule"]
    require(
        mode in MODES
        and p["availableCores"] == p["blockDim"] == 8
        and p["coreLimit"] == 0
        and p["workspace"] == 2097152
        and p["capacity"]["UB"] == 262144,
        "Q ATC platform/workspace changed",
    )
    require(
        {k: v for k, v in p.items() if k != "host_dso"}
        == {k: v for k, v in original["platform"].items() if k != "host_dso"}
        and s["baseline_assignments"]
        == [list(range(i, N // 64, p["blockDim"])) for i in range(p["blockDim"])],
        "different Q platform or incomplete tile assignment",
    )
    raw = bytes.fromhex(s["tiling_hex"])
    require(len(raw) == 224 and sha(raw) == q["tiling_sha256"], "Q tiling serialization differs")
    cube = {f["name"]: struct.unpack_from("<I", raw, 24 + f["offset"])[0] for f in schema()}
    if mode == CONTROL:
        require(
            q["tiling_sha256"] == original["tiling_sha256"]
            and s["resources"] == original["schedule"]["resources"],
            "control differs from sealed full Draft Q tiling/resources",
        )
        steps = K // 256
    else:
        banks = 2 if mode == PREFETCH else 1
        codes = banks * K * 16
        weight = 2 * 2 * (K * 16)
        scale = 2 * (K // 128) * 64
        user = codes + weight + scale + 8192
        scratch = p["capacity"]["UB"] - user
        require(struct.unpack_from("<4IQ", raw) == (M, K, N, K, scratch), "wrong Q parent/UB layout")
        fixed = dict(
            singleCoreM=M,
            singleCoreN=64,
            singleCoreK=K,
            baseM=M,
            baseN=64,
            baseK=256,
            stepKa=10,
            stepKb=10,
            isBias=0,
            dbL0C=1,
        )
        require(
            all(cube[k] == v for k, v in fixed.items())
            and min(cube[k] for k in ("depthA1", "depthB1", "dbL0A", "dbL0B")) > 0
            and 0 < cube["transLength"] <= scratch,
            "invalid Q Cube plan",
        )
        resources = dict(
            raw_banks=banks,
            tile_k=K,
            codes_bytes=codes,
            scale_bytes=scale,
            weight_bytes=weight,
            user_ub=user,
            scratch=scratch,
            workspace=p["workspace"],
            l1_used=2 * K * (M + 64) + 8192 * cube["depthA1"] + 32768 * cube["depthB1"],
            l0a_used=8192 * cube["dbL0A"],
            l0b_used=32768 * cube["dbL0B"],
            l0c_used=4096 * cube["dbL0C"],
        )
        require(
            s["resources"] == resources
            and all(
                resources[a] <= p["capacity"][b]
                for a, b in (("l1_used", "L1"), ("l0a_used", "L0A"), ("l0b_used", "L0B"), ("l0c_used", "L0C"))
            ),
            "Q per-core allocations differ from ATC",
        )
        steps = 1
    return dict(
        block_dim=p["blockDim"],
        cube=cube,
        cube_sha256=sha(raw[24:]),
        resources=s["resources"],
        iterates_per_n64=steps,
    )


def copy_panel_air(out, control, files):
    """Preserve even the original protobuf map-entry order across separate ATC processes."""
    packing = control["packing"]
    require(
        control["mode"] == PANEL
        and control["op"] == OP
        and control["abi"] == ABI
        and packing["panel_k"] == K
        and packing["original_nz"] == files["w_nz.bin"]
        and packing["original_gn"] == files["s_gn.bin"],
        "different Q AIR/weight binding",
    )
    require(
        [tuple(shape) for shape in packing["physical_shapes"]]
        == [(N // 64, 1, K * 64), (N // 64, K // 128, 64)]
        and len(packing["constants"]) == 2,
        "different packed Q shape",
    )
    for item in packing["constants"]:
        c.verify(item)
    air = c.verify(control["air"])
    out.mkdir()
    shutil.copyfile(air, out / "q.air")
    require(c.sha(out / "q.air") == control["air"]["sha256"], "Q AIR copy changed")
    return packing


def compile_q(out, types, pkg, env, argv, mode, source, *, packed_control=None):
    files = source["weights"]
    require(not (mode == CONTROL and packed_control is not None), "raw Q requires its NZ AIR")
    if mode == CONTROL:
        out.mkdir()
        parts = [files["w_nz.bin"], files["s_gn.bin"]]
        proto = model_proto(
            types, *(c.verify(f).read_bytes() for f in parts), projection_shape=SHAPE, name="c16_q"
        )
        graph = proto.graph[0]
        name = source["q_node"]
        for node in graph.op:
            if node.name == OLD_OP:
                node.name = name
            for i, v in enumerate(node.input):
                if v == OLD_OP + ":0":
                    node.input[i] = name + ":0"
        graph.output[:] = [name + ":0"]
        (out / "q.air").write_bytes(proto.SerializeToString())
        packing = dict(constants=parts, physical_shapes=((K // 32, N // 16, 16, 32), (K // 128, N)))
    elif packed_control is not None:
        packing = copy_panel_air(out, packed_control, files)
    else:
        packing = gate.make_projection_air(
            out, types, files, K, shape=SHAPE, op_type=OP, model_name="q", graph_name="c16_q"
        )
    typ = OLD_OP if mode == CONTROL else OP
    args, om, dump, query, node, image = compile_image(out, pkg, env, argv, typ, shape=SHAPE, model_name="q")
    contract = plan(query, mode, source["query"])
    selected = embedded(om, [image])[node["name"]]
    if mode == CONTROL:
        require(
            selected["executable_sections"] == source["compiled_q"]["executable_sections"],
            "raw Q machine text differs from measured full OM",
        )
    with om.open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as blob:
        offsets = []
        for d, fmt, shape, bound in zip(
            node["input_desc"][1:],
            ("FRACTAL_NZ" if mode == CONTROL else "ND", "ND"),
            packing["physical_shapes"],
            packing["constants"],
        ):
            require(
                d["layout"] == fmt and tuple(map(int, d["shape"]["dim"])) == tuple(shape),
                "Q OM constant layout differs",
            )
            offset = blob.find(c.verify(bound).read_bytes())
            require(offset >= 0, "Q constant payload missing in OM")
            offsets.append(offset)
    return dict(
        mode=mode,
        op=typ,
        abi=ABI,
        om=c.identity(om),
        air=c.identity(out / "q.air"),
        dump=c.identity(dump),
        selected=selected,
        query=query,
        contract=contract,
        argv=args,
        packing=packing,
        offsets=offsets,
    )
