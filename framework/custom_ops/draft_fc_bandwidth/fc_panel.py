"""Private C16/C64 feature-compression FC operators and actual ATC resource contracts."""

from pathlib import Path
import mmap
import shutil
import struct
import sys

HERE = Path(__file__).resolve().parent
OPS = HERE.parent
sys.path.insert(0, str(OPS / "draft_q_bandwidth"))
import q_panel as q
from q_panel import c, require, OLD_OP, OLD_STEM, schema
from bench import compile_image, embedded
from package import generate, install_project, write_preset
from layout import once
from tile_gate_graph import model_proto

K, N = 12800, 2560
OP, STEM = "DFlashFcPanel", "d_flash_fc_panel"
CONTROL = "raw-nz-brcb"
LOCAL = "fc-a1-k2560-next"
GLOBAL = "fc-gm-k6400-next"
MODES = (CONTROL, LOCAL, GLOBAL)
GEARS = (16, 64)
# (stepKa,stepKb,depthA1,depthB1) from the host CANN 9.0 MatmulApiTiling query.
SDK = {
    (16, LOCAL): (10, 1, 10, 2),
    (64, LOCAL): (1, 10, 2, 10),
    (16, GLOBAL): (5, 1, 10, 2),
    (64, GLOBAL): (1, 1, 2, 2),
}


def abi(m):
    require(m in GEARS, "unsupported FC gear")
    return dict(
        inputs=[dict(name="x", dtype="float16", shape=[m, K])],
        outputs=[dict(name="y", dtype="float16", shape=[m, N])],
    )


def logical_bytes(m):
    return K * N + 2 * (K // 128) * N + 2 * m * (K + N)


def panel_k(mode):
    require(mode in (LOCAL, GLOBAL), "unknown FC panel")
    return 2560 if mode == LOCAL else 6400


def header(mode):
    return f"#define DFLASH_FC_PANEL_K {panel_k(mode)}U\n".encode() + (HERE / "fc_spec.h").read_bytes()


def host_source():
    src = (q.gate.HERE / "op_host/d_flash_gate_panel.cpp").read_text()
    src = (
        src.replace(q.gate.STEM, STEM)
        .replace(q.gate.OP, OP)
        .replace("namespace gate_panel", "namespace fc_panel")
    )
    src = once(src, '"panel_spec.h"', '"fc_spec.h"')
    src = once(
        src,
        "    const auto* out = ctx->GetOutputShape(0);",
        """    const auto* x = ctx->GetInputShape(0);
    if (!x || x->GetStorageShape().GetDimNum()!=2) return ge::GRAPH_FAILED;
    const int64_t rows = x->GetStorageShape().GetDim(0);
    if (rows!=16 && rows!=64) return ge::GRAPH_FAILED;
    const uint32_t M = static_cast<uint32_t>(rows);
    const auto* out = ctx->GetOutputShape(0);""",
    )
    src = once(
        src,
        "    auto* y=ctx->GetOutputShape(0);",
        """    const auto* x=ctx->GetInputShape(0);
    if (!x || x->GetDimNum()!=2) return GRAPH_FAILED;
    const int64_t M=x->GetDim(0);
    if (M!=16 && M!=64) return GRAPH_FAILED;
    auto* y=ctx->GetOutputShape(0);""",
    )
    src = src.replace(
        "FullL1 ? TPosition::A1 : TPosition::GM, FullL1 ? CubeFormat::NZ : CubeFormat::ND",
        "StreamA ? TPosition::GM : TPosition::A1, StreamA ? CubeFormat::ND : CubeFormat::NZ",
    )
    src = once(src, "FullL1 ? TPosition::B1 : TPosition::VECOUT", "TPosition::B1")
    src = once(src, "mm.SetOrgShape(M, N, K, PanelK);", "mm.SetOrgShape(M, N, StreamA ? K : PanelK, PanelK);")
    src = once(
        src,
        "FullL1 ? static_cast<int>(l1Capacity-ResidentL1) : -1",
        "static_cast<int>(l1Capacity-ResidentL1)",
    )
    src = src.replace("ResidentL1", "ResidentL1(M)")
    src = once(
        src,
        "cube.get_stepKa() != PanelK/256 || cube.get_stepKb() != PanelK/256",
        "cube.get_stepKa() != (M==64 ? 1 : (StreamA ? 5 : 10)) || cube.get_stepKb() != ((!StreamA && M==64) ? 10 : 1)",
    )
    return src.replace("Gate panel SDK plan rejected", "FC panel SDK plan rejected")


def build(out, cann, env, mode):
    definition = c.read(OPS / "draft_gate_tile_nd/DFlashGateTileNd.json")
    definition[0]["op"] = OP
    spec = out / (OP + ".json")
    c.write(spec, definition)
    project = generate(out, spec, env)
    host, dev = project / "op_host", project / "op_kernel"
    config = header(mode)
    for folder in (host, dev):
        (folder / "fc_spec.h").write_bytes(config)
    (host / (STEM + ".cpp")).write_text(host_source())
    shutil.copyfile(HERE / "op_kernel" / (STEM + ".cpp"), dev / (STEM + ".cpp"))
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
        {"fc_spec.h": q.sha(config), helper.name: c.sha(helper)},
        "PRIVATE_FC_PANELS",
    )


def plan(query, original, m, mode):
    p, s = query["platform"], query["schedule"]
    require(
        m in GEARS
        and mode in MODES
        and p["availableCores"] == p["blockDim"] == 8
        and p["coreLimit"] == 0
        and p["workspace"] == 2097152
        and p["capacity"]["UB"] == 262144,
        "FC ATC device/resource view changed",
    )
    require(
        {k: v for k, v in p.items() if k != "host_dso"}
        == {k: v for k, v in original["platform"].items() if k != "host_dso"}
        and s["baseline_assignments"] == [list(range(i, N // 64, 8)) for i in range(8)],
        "FC platform or N64 coverage differs",
    )
    raw = bytes.fromhex(s["tiling_hex"])
    require(len(raw) == 224 and q.sha(raw) == query["tiling_sha256"], "invalid FC parent serialization")
    cube = {f["name"]: struct.unpack_from("<I", raw, 24 + f["offset"])[0] for f in schema()}
    if mode == CONTROL:
        require(
            query["tiling_sha256"] == original["tiling_sha256"]
            and s["resources"] == original["schedule"]["resources"],
            "FC control differs from measured Draft",
        )
    else:
        pk = panel_k(mode)
        chunk = 1280 if mode == GLOBAL else 2560
        codes = 16 * chunk
        scales = 2 * (pk // 128) * 64
        user = 5 * codes + scales + 8192
        scratch = 262144 - user
        local_k = 0 if mode == GLOBAL else K if m == 16 else pk
        sa, sb, da, db = SDK[m, mode]
        fixed = dict(
            singleCoreM=m,
            singleCoreN=64,
            singleCoreK=pk,
            baseM=m,
            baseN=64,
            baseK=256,
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
            struct.unpack_from("<4IQ", raw) == (m, K, N, pk, scratch)
            and all(cube[k] == v for k, v in fixed.items())
            and 0 < cube["transLength"] <= scratch,
            "FC Cube plan or scratch changed",
        )
        expected = dict(
            user_ub=user,
            scratch=scratch,
            raw_banks=1,
            tile_k=pk,
            codes_bytes=codes,
            scale_bytes=scales,
            weight_bytes=4 * codes,
            workspace=p["workspace"],
            l1_used=2 * (m * local_k + 64 * pk) + 2 * 256 * (m * da + 64 * db),
            l0a_used=2 * m * 256 * 2,
            l0b_used=2 * 64 * 256 * 2,
            l0c_used=4 * m * 64,
        )
        require(
            s["resources"] == expected
            and all(
                expected[k] <= p["capacity"][cap]
                for k, cap in (
                    ("l1_used", "L1"),
                    ("l0a_used", "L0A"),
                    ("l0b_used", "L0B"),
                    ("l0c_used", "L0C"),
                )
            ),
            "FC allocation mismatch",
        )
    return dict(
        block_dim=p["blockDim"],
        cube=cube,
        resources=s["resources"],
        iterates_per_n64=K // s["resources"]["tile_k"],
    )


def compile_fc(out, types, pkg, env, argv, m, mode, src):
    files = src["weights"]
    shape = (m, K, N)
    if mode == CONTROL:
        out.mkdir()
        consts = [files["w_nz.bin"], files["s_gn.bin"]]
        proto = model_proto(
            types, *(c.verify(f).read_bytes() for f in consts), projection_shape=shape, name=f"c{m}_fc"
        )
        g = proto.graph[0]
        name = src["node"]
        for node in g.op:
            if node.name == OLD_OP:
                node.name = name
            for i, edge in enumerate(node.input):
                if edge == OLD_OP + ":0":
                    node.input[i] = name + ":0"
        g.output[:] = [name + ":0"]
        (out / "fc.air").write_bytes(proto.SerializeToString(deterministic=True))
        packing = dict(constants=consts, physical_shapes=((K // 32, N // 16, 16, 32), (K // 128, N)))
    else:
        packing = q.gate.make_projection_air(
            out, types, files, panel_k(mode), shape=shape, op_type=OP, model_name="fc", graph_name=f"c{m}_fc"
        )
    typ = OLD_OP if mode == CONTROL else OP
    args, om, dump, query, node, image = compile_image(out, pkg, env, argv, typ, shape=shape, model_name="fc")
    proof = plan(query, src["gears"][str(m)]["query"], m, mode)
    selected = embedded(om, [image])[node["name"]]
    require(
        (selected["executable_sections"] == src["gears"][str(m)]["kernel"]["executable_sections"])
        == (mode == CONTROL),
        "FC machine text differs from expected implementation",
    )
    offsets = []
    with om.open("rb") as fd, mmap.mmap(fd.fileno(), 0, access=mmap.ACCESS_READ) as blob:
        for desc, fmt, shape, item in zip(
            node["input_desc"][1:],
            ("FRACTAL_NZ" if mode == CONTROL else "ND", "ND"),
            packing["physical_shapes"],
            packing["constants"],
        ):
            require(
                desc["layout"] == fmt and tuple(map(int, desc["shape"]["dim"])) == tuple(shape),
                "FC constant layout differs",
            )
            offset = blob.find(c.verify(item).read_bytes())
            require(offset >= 0, "compiled OM lost FC constant")
            offsets.append(offset)
    return dict(
        mode=mode,
        op=typ,
        abi=abi(m),
        air=c.identity(out / "fc.air"),
        om=c.identity(om),
        dump=c.identity(dump),
        selected=selected,
        query=query,
        contract=proof,
        packing=packing,
        offsets=offsets,
        argv=args,
    )
