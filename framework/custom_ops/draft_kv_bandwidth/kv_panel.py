"""Full-K Layer0 KV for C16 and C64, with three ways to supply the same A1 tensor."""

import mmap
from pathlib import Path
import shutil
import struct
import sys

HERE = Path(__file__).resolve().parent
OPS = HERE.parent
sys.path[:0] = [str(OPS / "draft_o_bandwidth"), str(OPS / "draft_fc_bandwidth")]
import packed_panel
from q_panel import c, require, sha, gate, OLD_OP, OLD_STEM, schema
from bench import compile_graph, selected_image, embedded
from layout import once
from package import generate, install_project, write_preset
from tile_gate_graph import model_proto

K, N, WIDTH = 2560, 2048, 64
OP, STEM, ROLE = "DFlashKvPanel", "d_flash_kv_panel", "layer0.kv"
RAW = "raw-nz-brcb"
DIRECT, STAGED, PACKED = "kv-k2560-next-n", "kv-k2560-ub-stage", "kv-k2560-packed-a"
CANDIDATES = (DIRECT, STAGED, PACKED)
INPUT_PATH = {DIRECT: 0, STAGED: 1, PACKED: 2}
SCOPE = "supplied Layer0 KV projection reference checked in process"


def shape(gear):
    require(gear in (16, 64), "unsupported KV gear")
    return gear + 16, K, N


def abi(gear):
    m, k, n = shape(gear)
    return dict(
        inputs=[dict(name="x", dtype="float16", shape=[m, k])],
        outputs=[dict(name="y", dtype="float16", shape=[m, n])],
    )


def logical_bytes(gear):
    m, k, n = shape(gear)
    return k * n + 2 * (k // 128) * n + 2 * m * (k + n)


def header(mode):
    return f"#define DFLASH_KV_INPUT {INPUT_PATH[mode]}\n".encode() + (HERE / "kv_spec.h").read_bytes()


def device_source():
    # Use the actual measured Q stripe pipeline, including the barriers and failed-Iterate cleanup.
    source = (OPS / "draft_q_bandwidth/op_kernel/next_n.cpp").read_text()
    source = once(source, '"q_spec.h"', '"kv_activation.h"')
    source = once(source, "namespace q_panel", "namespace kv_panel")
    a = source.index("__aicore__ inline void CacheActivation(")
    b = source.index("__aicore__ inline void ScaleStripe(", a)
    source = source[:a] + source[b:]
    a = source.index('extern "C" __global__ __aicore__ void d_flash_q_panel(')
    b = source.index("    const uint32_t block", a)
    source = source[:a] + """template<uint32_t M, class Tiling>
__aicore__ inline void Project(GM_ADDR x, GM_ADDR w_tiles, GM_ADDR s, GM_ADDR y, Tiling& td)
{
""" + source[b:]
    source = once(source, "CacheActivation(a, xgm);", "LoadActivation<M>(a, xgm, weights);")
    return source + """
extern "C" __global__ __aicore__ void d_flash_kv_panel(GM_ADDR x, GM_ADDR w_tiles,
    GM_ADDR s, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(td, tiling);
    if (td.globalM == 32) Project<32>(x, w_tiles, s, y, td);
    else if (td.globalM == 80) Project<80>(x, w_tiles, s, y, td);
}
"""


def build(out, cann, env, mode):
    rows = c.read(OPS / "draft_gate_tile_nd/DFlashGateTileNd.json")
    rows[0]["op"] = OP
    spec = out / (OP + ".json")
    c.write(spec, rows)
    project = generate(out, spec, env)
    host, dev = project / "op_host", project / "op_kernel"
    config = header(mode)
    for d in (host, dev):
        (d / "kv_spec.h").write_bytes(config)
    shutil.copyfile(HERE / "d_flash_kv_panel_host.cpp", host / (STEM + ".cpp"))
    (dev / (STEM + ".cpp")).write_text(device_source())
    helpers = [HERE / "kv_activation.h", gate.HERE / "panel_vector.h"]
    for h in helpers:
        shutil.copyfile(h, dev / h.name)
    (host / (STEM + "_tiling.h")).write_text(
        (c.QUANT / "op_host" / (OLD_STEM + "_tiling.h")).read_text().replace(OLD_OP, OP)
    )
    shutil.copyfile(c.QUANT / "op_host" / (OLD_STEM + "_contract.h"), host / (OLD_STEM + "_contract.h"))
    for name in ("scheduler_evidence.h", "platform_diag.h"):
        shutil.copyfile(OPS / "draft_a_core_scheduler_flow/op_host" / name, host / name)
    preset = project / "CMakePresets.json"
    write_preset(preset, preset, cann)
    pins = {h.name: c.sha(h) for h in helpers}
    return install_project(
        out, project, cann, env, OP, STEM, dict(pins, **{"kv_spec.h": sha(config)}), "PRIVATE_KV_PANEL"
    )


def plan(query, original, gear, mode):
    m, k, n = shape(gear)
    p, s = query["platform"], query["schedule"]
    require(
        mode in (RAW, *CANDIDATES)
        and p["blockDim"] == p["availableCores"] == 8
        and p["coreLimit"] == 0
        and p["workspace"] == 2097152
        and p["capacity"]["UB"] == 262144
        and {a: b for a, b in p.items() if a != "host_dso"}
        == {a: b for a, b in original["platform"].items() if a != "host_dso"}
        and s["baseline_assignments"] == [list(range(i, n // WIDTH, 8)) for i in range(8)],
        "KV platform, workspace or N64 ownership differs",
    )
    raw = bytes.fromhex(s["tiling_hex"])
    require(len(raw) == 224 and sha(raw) == query["tiling_sha256"], "invalid serialized KV tiling")
    cube = {f["name"]: struct.unpack_from("<I", raw, 24 + f["offset"])[0] for f in schema()}
    if mode == RAW:
        require(
            query["tiling_sha256"] == original["tiling_sha256"]
            and s["resources"] == original["schedule"]["resources"],
            "original KV Cube/UB changed",
        )
    else:
        bk, sa, sb, da, db = (256, 10, 10, 10, 10) if m == 32 else (160, 4, 2, 8, 4)
        fixed = dict(
            singleCoreM=m,
            singleCoreN=WIDTH,
            singleCoreK=k,
            baseM=m,
            baseN=WIDTH,
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
        r = dict(
            tile_k=k,
            raw_banks=1,
            codes_bytes=40960,
            scale_bytes=2560,
            weight_bytes=163840,
            user_ub=215552,
            scratch=46592,
            workspace=p["workspace"],
            l1_used=2 * (m + WIDTH) * k + 2 * bk * (m * da + WIDTH * db),
            l0a_used=4 * m * bk,
            l0b_used=4 * WIDTH * bk,
            l0c_used=4 * m * WIDTH,
        )
        require(
            struct.unpack_from("<4IQ", raw) == (m, k, n, k, r["scratch"])
            and all(cube[key] == value for key, value in fixed.items())
            and 0 < cube["transLength"] <= r["scratch"]
            and r == s["resources"],
            "KV SDK full-K geometry or allocation differs",
        )
        require(
            all(
                r[a] <= p["capacity"][b]
                for a, b in (("l1_used", "L1"), ("l0a_used", "L0A"), ("l0b_used", "L0B"), ("l0c_used", "L0C"))
            ),
            "KV on-chip capacity exceeded",
        )
    return dict(
        block_dim=p["blockDim"],
        cube=cube,
        resources=s["resources"],
        iterates_per_n64=k // s["resources"]["tile_k"],
    )


def compile_model(out, types, pkg, env, argv, gear, mode, src):
    m, k, n = shape(gear)
    files = src["weights"]
    nz, gn = (c.verify(files[name]).read_bytes() for name in ("w_nz.bin", "s_gn.bin"))
    out.mkdir()
    model = model_proto(types, nz, gn, projection_shape=(m, k, n), name=f"c{gear}_layer0_kv")
    if mode == RAW:
        typ = OLD_OP
        expected = [("FRACTAL_NZ", (k // 32, n // 16, 16, 32), nz), ("ND", (k // 128, n), gn)]
    else:
        typ = OP
        w, s, _ = gate.pack_cube_panels(nz, gn, k, k=k, n=n)
        packed_panel.add_pack_graph(
            model, WIDTH, k, w, s, m=m, k=k, n=n, op_type=OP, prefix="kv_a", pack_activation=mode == PACKED
        )
        expected = [("ND", (n // WIDTH, 1, k * WIDTH), w), ("ND", (n // WIDTH, k // 128, WIDTH), s)]
    g = model.graph[0]
    name = src["kv_node"]
    for node in g.op:
        if node.name == OLD_OP:
            node.name = name
        for idx, edge in enumerate(node.input):
            if edge == OLD_OP + ":0":
                node.input[idx] = name + ":0"
    g.output[:] = [name + ":0"]
    air = out / "kv.air"
    air.write_bytes(model.SerializeToString(deterministic=True))
    args, om, dump, query, nodes = compile_graph(out, pkg, env, argv, shape=(m, k, n), model_name="kv")
    proof = plan(query, src["query"], gear, mode)
    (node,) = (q for q in nodes if q["type"] == typ)
    prep = set()
    if mode == PACKED:
        prep = packed_panel.activation_nodes(nodes, node, m, k)
    else:
        require(
            all(q is node or q["type"] in ("Const", "FileConstant", "Data", "NetOutput") for q in nodes)
            and list(map(int, node["input_desc"][0]["shape"]["dim"])) == [m, k]
            and node["input_desc"][0]["layout"] == "ND",
            "unexpected KV compute or activation ABI",
        )
    require(
        len(node["input_desc"]) == len(node["inputs"]) == 3
        and [a["value"] for a in node["attributes"] if a["key"] == "custom_op_file_path"]
        == [{"s": str(Path(pkg["opp"]["install"]) / "vendors/customize")}],
        "different KV implementation",
    )
    selected = embedded(om, [selected_image(node, proof["block_dim"], typ)])[node["name"]]
    require(
        (selected["executable_sections"] == src["compiled_kv"]["executable_sections"]) == (mode == RAW),
        "KV control must match the measured full OM machine text",
    )
    constants = []
    with om.open("rb") as fp, mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ) as blob:
        for desc, (fmt, dims, data) in zip(node["input_desc"][1:], expected):
            require(
                desc["layout"] == fmt and tuple(map(int, desc["shape"]["dim"])) == dims,
                "KV constant descriptor drift",
            )
            pos = blob.find(data)
            require(pos >= 0, "KV packed constant bytes missing from compiled model")
            constants.append(dict(sha256=sha(data), bytes=len(data), offset=pos))
    return dict(
        om=c.identity(om),
        dump=c.identity(dump),
        air=c.identity(air),
        argv=args,
        query=query,
        contract=proof,
        selected=selected,
        constants=constants,
        abi=abi(gear),
        op=typ,
        mode=mode,
        prep_nodes=sorted(prep),
        compute_names={q["name"]: q["type"] for q in nodes if q["name"] in prep or q is node},
    )
